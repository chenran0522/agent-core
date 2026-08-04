# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""RealSense RGB-D capture + MobileSAM-only object segmentation for the SO-101 backend.

Two independent pieces, matching the two tools they back:

- :class:`So101Camera` -- ``capture_photo``. Ported near-verbatim from
  ``robotic_arm.vendors.so101.rekep.executor.So101RekepExecutor``'s RealSense
  I/O (``_read_frame``/``reset_camera``/``_retry_capture``), unchanged logic.
- :class:`So101Segmenter` -- ``segment_scene``. Deliberately much simpler than
  ``robotic_arm.vendors.so101.rekep.keypoint_proposal.KeypointProposer``: this
  drops DINOv2 dense-feature clustering entirely and returns exactly ONE
  representative pixel per detected MobileSAM mask (its centroid, snapped to
  the nearest in-mask pixel for concave shapes). The old pipeline needed up
  to 5 candidate points per object because its downstream VLM could only
  reference DINOv2/SAM-detected keypoints; here the model can call
  ``pixel_to_3d`` on ANY pixel it picks by eye from the photo, so
  ``segment_scene``'s only job is "what distinct objects are there and
  roughly where" -- not "give the VLM enough candidates to write geometric
  constraints against". This also drops the torch/transformers DINOv2
  dependency and the sklearn PCA/KMeans/MeanShift clustering entirely; only
  ``mobile_sam`` (itself torch-based) remains. Each mask is still returned
  (not just its centroid) so ``So101ArmBackend.get_object_geometry`` --
  ``get_object_geometry`` -- can run PCA over the object's own depth points
  (already cached per frame, see ``backend.py``) to report real bounding
  dimensions and a principal-axis direction, instead of the model having to
  guess an object's shape from the photo alone.
"""

from __future__ import annotations

import subprocess
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.exception.errors import ToolError
from openjiuwen.core.common.logging import tool_logger

_DEPTH_VALID_PIXEL_THRESHOLD = 800_000


class So101Camera:
    """RealSense RGB-D capture with retry + hardware-reset, ported from ``rekep/executor.py``."""

    def __init__(
        self,
        *,
        color_width: int = 1280,
        color_height: int = 720,
        fps: int = 6,
        frame_timeout_ms: int = 5000,
        pre_capture_hook: Any | None = None,
    ) -> None:
        self._color_width = color_width
        self._color_height = color_height
        self._fps = fps
        self._frame_timeout_ms = frame_timeout_ms
        self._pre_capture_hook = pre_capture_hook

    def retry_capture(self, max_retry_attempt: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Returns ``(rgb, depth_raw)``; raises :class:`ToolError` after exhausting retries."""
        for attempt in range(1, max_retry_attempt + 1):
            tool_logger.info("[So101Camera] capture attempt %s/%s", attempt, max_retry_attempt)
            self.reset_camera()
            rgb, depth = self._read_frame()
            if rgb is not None:
                return rgb, depth
            tool_logger.warning("[So101Camera] capture attempt %s failed, retrying", attempt)
            time.sleep(1)
        raise ToolError(
            StatusCode.TOOL_EXECUTION_ERROR,
            reason=f"could not capture a valid RGB-D frame after {max_retry_attempt} attempts",
        )

    def _read_frame(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self._pre_capture_hook is not None:
            self._pre_capture_hook()

        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise ImportError(
                f"pyrealsense2 is not installed; run `pip install 'openjiuwen[robotic-arm-so101]'` ({e})"
            ) from e
        try:
            import cv2
        except ImportError as e:
            raise ImportError(
                f"opencv-python is not installed; run `pip install 'openjiuwen[robotic-arm-so101-rekep]'` ({e})"
            ) from e

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self._color_width, self._color_height, rs.format.bgra8, self._fps)
        config.enable_stream(rs.stream.depth, self._color_width, self._color_height, rs.format.z16, self._fps)
        try:
            pipeline.start(config)
        except Exception as e:  # noqa: BLE001 -- best-effort capture, caller retries on any failure
            tool_logger.warning("[So101Camera] pipeline start failed: %s", e)
            return None, None

        align = rs.align(rs.stream.color)
        try:
            for i in range(50):
                frames = align.process(pipeline.wait_for_frames(self._frame_timeout_ms))
                depth = np.asarray(frames.get_depth_frame().get_data())
                valid = int(np.sum((depth > 0) & (depth < 65535)))
                tool_logger.debug("[So101Camera] frame %s: %s valid pixels", i, valid)
                if valid > _DEPTH_VALID_PIXEL_THRESHOLD:
                    rgb = cv2.cvtColor(np.asarray(frames.get_color_frame().get_data()), cv2.COLOR_BGRA2RGB)
                    pipeline.stop()
                    return rgb, depth
            pipeline.stop()
            return None, None
        except Exception as e:  # noqa: BLE001 -- best-effort capture, caller retries on any failure
            tool_logger.warning("[So101Camera] capture failed: %s", e)
            try:
                pipeline.stop()
            except Exception:  # noqa: BLE001 -- cleanup after an already-failed capture, nothing to recover
                tool_logger.warning("[So101Camera] pipeline stop after failed capture also failed")
            return None, None

    @staticmethod
    def reset_camera() -> None:
        """Kill macOS processes that can hold the RealSense camera open, then hardware-reset it."""
        subprocess.run(["/usr/bin/killall", "VDCAssistant"], capture_output=True, check=False)
        subprocess.run(["/usr/bin/killall", "AppleCameraAssistant"], capture_output=True, check=False)
        try:
            import pyrealsense2 as rs

            ctx = rs.context()
            if len(ctx.devices) > 0:
                ctx.devices[0].hardware_reset()
                time.sleep(3)
        except Exception:  # noqa: BLE001 -- best-effort reset, missing/busy hardware is not fatal
            tool_logger.warning("[So101Camera] camera hardware reset failed", exc_info=True)


class So101Segmenter:
    """MobileSAM automatic mask generation -> one centroid pixel per mask (no DINOv2)."""

    @staticmethod
    def _select_device(device: str | None, torch_module: Any) -> str:
        if device is not None:
            return device
        if torch_module.cuda.is_available():
            return "cuda"
        if torch_module.backends.mps.is_available():
            return "mps"
        return "cpu"

    def __init__(
        self,
        *,
        sam_checkpoint_path: str,
        device: str | None = None,
        min_mask_pixels: int = 300,
    ) -> None:
        try:
            import torch
            from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry
        except ImportError as e:
            raise ImportError(
                f"torch/mobile_sam are not installed; run `pip install 'openjiuwen[robotic-arm-so101-rekep]'` ({e})"
            ) from e

        self.device = self._select_device(device, torch)
        self.min_mask_pixels = min_mask_pixels

        tool_logger.info("[So101Segmenter] loading MobileSAM from %s onto %s", sam_checkpoint_path, self.device)
        sam = sam_model_registry["vit_t"](checkpoint=sam_checkpoint_path)
        sam.to(self.device)
        sam.eval()
        self.sam_generator = SamAutomaticMaskGenerator(sam)

    def _get_masks(self, rgb: np.ndarray) -> list[np.ndarray]:
        results = self.sam_generator.generate(rgb)
        masks = []
        for r in results:
            m = np.asarray(r["segmentation"]).astype(bool)
            if m.sum() >= self.min_mask_pixels:
                masks.append(m)
        return masks

    @staticmethod
    def _mask_centroid_pixel(mask: np.ndarray) -> tuple[int, int]:
        """Mean pixel of the mask, snapped to the nearest in-mask pixel for concave shapes."""
        ys, xs = np.where(mask)
        cy, cx = float(ys.mean()), float(xs.mean())
        if mask[round(cy), round(cx)]:
            return round(cx), round(cy)
        dist_sq = (ys - cy) ** 2 + (xs - cx) ** 2
        nearest = int(np.argmin(dist_sq))
        return int(xs[nearest]), int(ys[nearest])

    def segment(self, rgb: np.ndarray) -> list[tuple[tuple[int, int], np.ndarray]]:
        """Returns one ``((x, y), mask)`` pair per detected object mask. No overlay, no 3D --
        the caller (``So101ArmBackend.segment``) decides which survive a workspace filter
        before numbering/drawing the overlay it sends to the model. The mask is kept (not
        just the centroid pixel) so ``get_object_geometry`` can later run PCA over exactly
        the depth points belonging to this object.
        """
        masks = self._get_masks(rgb)
        tool_logger.info("[So101Segmenter] %s masks above size threshold", len(masks))
        return [(self._mask_centroid_pixel(m), m) for m in masks]


def draw_numbered_overlay(rgb: np.ndarray, pixels: list[tuple[int, int]]) -> np.ndarray:
    """Draw numbered red circles at ``pixels`` (index == the id the model sees)."""
    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except Exception:  # noqa: BLE001 -- any font-loading failure just falls back to the default font
        font = ImageFont.load_default()
    for i, (x, y) in enumerate(pixels):
        r = 10
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 0, 0), outline=(255, 255, 255), width=2)
        draw.text((x + r + 2, y - r), str(i), fill=(255, 255, 0), font=font)
    return np.array(img)


__all__ = ["So101Camera", "So101Segmenter", "draw_numbered_overlay"]
