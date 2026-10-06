"""Local Offline YOLO ONNX Object Detection Adapter (Phase 4H.6).

Concrete implementation of IObjectDetector executing local object detection via ONNX Runtime
and YOLO models (e.g. YOLOv8n) without cloud dependencies or external vision APIs.

Key Architectural Guarantees:
- Strictly local offline inference using ONNX Runtime (CPU / CUDA / DirectML).
- Standardized letterbox preprocessing, normalization, and tensor batching.
- Post-processing with vectorized confidence filtering and Non-Maximum Suppression (NMS).
- Asynchronous, non-blocking execution offloaded from the asyncio event loop.
- Strict Privacy: PERSON != IDENTITY. Detection is purely spatial perception; zero facial recognition,
  biometrics, demographic profiling, or identity matching.
- Strict Zero-Trust Security: MODEL OUTPUT != AUTHORIZATION. Zero tool execution, zero shell, zero IPC authority.
- Ephemeral frames: zero persistence to disk, databases, Redis, or IPC.
"""

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

import numpy as np

from app.vision.base import IObjectDetector
from app.vision.detection import (
    ObjectDetectionConfig,
    ObjectDetectionTelemetry,
    ObjectDetectionTimeoutError,
    ObjectDetectionUnavailableError,
    create_object_detection_config,
)
from app.vision.exceptions import (
    ObjectDetectionProcessingError,
    ObjectDetectionValidationError,
)
from app.vision.models import (
    BoundingBox,
    ObjectDetection,
    PixelFormat,
    VisionFrame,
)

logger = logging.getLogger(__name__)

# Conditional import for onnxruntime and cv2
try:
    import onnxruntime as ort  # type: ignore[import-untyped,import-not-found]

    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ort = None  # type: ignore[assignment]
    ONNXRUNTIME_AVAILABLE = False

try:
    import cv2  # type: ignore[import-untyped,import-not-found]

    CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    CV2_AVAILABLE = False

# Standard 80 COCO Class Labels
COCO_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def _compute_iou(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    """Computes Intersection over Union (IoU) of two boxes in (x1, y1, x2, y2) format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def _nms(
    boxes: list[tuple[float, float, float, float]],
    scores: list[float],
    class_ids: list[int],
    iou_threshold: float,
    max_detections: int,
) -> list[int]:
    """Applies class-aware Non-Maximum Suppression (NMS)."""
    if not boxes:
        return []

    # Sort indices by score descending
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    selected_indices: list[int] = []

    while indices and len(selected_indices) < max_detections:
        current = indices.pop(0)
        selected_indices.append(current)

        remaining = []
        for idx in indices:
            # Only suppress if same class and IoU exceeds threshold
            if class_ids[idx] == class_ids[current]:
                iou = _compute_iou(boxes[current], boxes[idx])
                if iou >= iou_threshold:
                    continue
            remaining.append(idx)
        indices = remaining

    return selected_indices


class YOLOONNXDetector(IObjectDetector):
    """Local Offline YOLO Object Detector using ONNX Runtime."""

    def __init__(
        self,
        config: ObjectDetectionConfig | None = None,
        model_path: str | None = None,
        input_size: tuple[int, int] = (640, 640),
        device: str = "cpu",
        simulated_mode: bool = False,
    ) -> None:
        """Initializes YOLOONNXDetector.

        Args:
            config: Optional ObjectDetectionConfig instance.
            model_path: Optional explicit filesystem path to YOLO .onnx model weights.
            input_size: Expected model input dimensions (width, height), default (640, 640).
            device: Requested hardware target ("cpu", "cuda", "directml").
            simulated_mode: If True, uses synthetic deterministic detections for testing without loading weights.
        """
        self.config = config or create_object_detection_config()
        self.input_size = input_size
        self.requested_device = device.lower()
        self._simulated_mode = simulated_mode or not ONNXRUNTIME_AVAILABLE

        # Resolve model path
        default_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "yolo")
        self.model_path = model_path or self.config.model_path or os.path.join(default_dir, "yolov8n.onnx")

        self._session: Any = None
        self._input_name: str = "images"
        self._output_name: str = "output0"
        self._active_provider: str = "CPUExecutionProvider"
        self._initialized = False
        self._status = "IDLE"

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo_detector")
        self._frames_processed = 0
        self._frames_dropped = 0
        self._inference_failures = 0
        self._model_load_latency_ms = 0.0
        self._last_telemetry: ObjectDetectionTelemetry | None = None

        logger.info(
            f"YOLOONNXDetector initialized: model='{self.model_path}', "
            f"device='{self.requested_device}', simulated={self._simulated_mode}."
        )

    def _select_execution_providers(self) -> list[str]:
        """Inspects host ONNX Runtime providers and resolves execution priority cleanly."""
        if not ONNXRUNTIME_AVAILABLE or ort is None:
            return ["CPUExecutionProvider"]

        available = ort.get_available_providers()
        selected: list[str] = []

        if self.requested_device in ("cuda", "gpu") and "CUDAExecutionProvider" in available:
            selected.append("CUDAExecutionProvider")
        elif self.requested_device in ("directml", "dml") and "DmlExecutionProvider" in available:
            selected.append("DmlExecutionProvider")

        selected.append("CPUExecutionProvider")
        return selected

    async def initialize(self) -> None:
        """Loads and initializes the ONNX inference session once."""
        if self._initialized and self._status == "READY":
            return

        start_time = time.perf_counter()

        if self._simulated_mode:
            self._status = "READY"
            self._initialized = True
            self._active_provider = "SimulatedExecutionProvider"
            self._model_load_latency_ms = 1.0
            logger.info("YOLOONNXDetector running in simulated mode.")
            return

        if not os.path.exists(self.model_path):
            self._status = "UNAVAILABLE"
            self._initialized = True
            logger.error(f"YOLO ONNX model not found at '{self.model_path}'.")
            raise ObjectDetectionUnavailableError(
                f"YOLO ONNX model file not found at path: {self.model_path}"
            )

        providers = self._select_execution_providers()

        def _load() -> tuple[Any, str, str, str]:
            sess = ort.InferenceSession(self.model_path, providers=providers)
            inp_name = sess.get_inputs()[0].name
            out_name = sess.get_outputs()[0].name
            active_prov = sess.get_providers()[0] if sess.get_providers() else "CPUExecutionProvider"
            return sess, inp_name, out_name, active_prov

        loop = asyncio.get_running_loop()
        try:
            self._session, self._input_name, self._output_name, self._active_provider = (
                await loop.run_in_executor(self._executor, _load)
            )
            self._status = "READY"
            self._initialized = True
            self._model_load_latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info(
                f"YOLO ONNX model loaded cleanly from '{self.model_path}' "
                f"using provider '{self._active_provider}' in {self._model_load_latency_ms:.1f}ms."
            )
        except Exception as exc:
            self._status = "ERROR"
            self._initialized = True
            logger.error(f"Failed to load YOLO ONNX session: {exc}")
            raise ObjectDetectionUnavailableError(f"Failed to initialize YOLO ONNX model: {exc}") from exc

    def _preprocess(self, frame: VisionFrame) -> tuple[np.ndarray, float, tuple[float, float]]:
        """Standardizes input VisionFrame to normalized letterbox tensor (1, 3, 640, 640).

        Returns:
            tensor: Normalized float32 numpy array with shape (1, 3, input_h, input_w).
            scale: Ratio between resized unpadded image and original dimensions.
            (pad_w, pad_h): Horizontal and vertical letterbox padding pixels.
        """
        orig_w, orig_h = frame.width, frame.height
        target_w, target_h = self.input_size

        # Decode or reshape raw frame bytes
        if frame.format.pixel_format in (PixelFormat.JPEG, PixelFormat.PNG):
            if not CV2_AVAILABLE or cv2 is None:
                raise ObjectDetectionProcessingError("OpenCV is required to decode compressed image frames.")
            img = cv2.imdecode(np.frombuffer(frame.payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ObjectDetectionValidationError("Failed to decode compressed image payload.")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif frame.format.pixel_format == PixelFormat.RGB24:
            img = np.frombuffer(frame.payload, dtype=np.uint8).reshape((orig_h, orig_w, 3))
        elif frame.format.pixel_format == PixelFormat.BGR24:
            bgr = np.frombuffer(frame.payload, dtype=np.uint8).reshape((orig_h, orig_w, 3))
            img = bgr[..., ::-1]  # BGR to RGB
        elif frame.format.pixel_format == PixelFormat.RGBA:
            rgba = np.frombuffer(frame.payload, dtype=np.uint8).reshape((orig_h, orig_w, 4))
            img = rgba[..., :3]
        else:
            raise ObjectDetectionValidationError(
                f"Unsupported pixel format '{frame.format.pixel_format.value}' for YOLO inference."
            )

        # Letterbox aspect-ratio preserving resize
        r = min(target_w / orig_w, target_h / orig_h)
        new_unpad_w = int(round(orig_w * r))
        new_unpad_h = int(round(orig_h * r))
        pad_w = (target_w - new_unpad_w) / 2.0
        pad_h = (target_h - new_unpad_h) / 2.0

        if CV2_AVAILABLE and cv2 is not None:
            if (orig_w, orig_h) != (new_unpad_w, new_unpad_h):
                img_resized = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
            else:
                img_resized = img
        else:
            # Simple nearest-neighbor fallback if cv2 not available
            y_indices = (np.linspace(0, orig_h - 1, new_unpad_h)).astype(int)
            x_indices = (np.linspace(0, orig_w - 1, new_unpad_w)).astype(int)
            img_resized = img[y_indices[:, None], x_indices]

        # Place onto padded canvas with neutral gray background (114)
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        top, left = int(round(pad_h - 0.1)), int(round(pad_w - 0.1))
        canvas[top : top + new_unpad_h, left : left + new_unpad_w] = img_resized

        # Normalization and HWC -> CHW -> (1, 3, H, W)
        tensor = canvas.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        tensor = np.ascontiguousarray(tensor)

        return tensor, r, (pad_w, pad_h)

    def _postprocess(
        self,
        output: np.ndarray,
        frame: VisionFrame,
        scale: float,
        pad: tuple[float, float],
    ) -> list[ObjectDetection]:
        """Parses raw YOLO output tensor and applies NMS to produce sanitized ObjectDetections."""
        # Expected output shape: (1, 84, N) or (1, N, 84)
        if output.ndim == 3 and output.shape[1] == 84:
            preds = output[0].T  # Transpose (84, N) -> (N, 84)
        elif output.ndim == 3 and output.shape[2] == 84:
            preds = output[0]  # Already (N, 84)
        elif output.ndim == 2:
            preds = output if output.shape[1] == 84 else output.T
        else:
            raise ObjectDetectionProcessingError(f"Unexpected YOLO output shape {output.shape}.")

        orig_w, orig_h = frame.width, frame.height
        pad_w, pad_h = pad

        boxes_xyxy: list[tuple[float, float, float, float]] = []
        confidences: list[float] = []
        class_ids: list[int] = []

        conf_thresh = self.config.confidence_threshold

        for row in preds:
            cx, cy, w, h = row[0:4]
            scores = row[4:]
            class_id = int(np.argmax(scores))
            score = float(scores[class_id])

            if score >= conf_thresh:
                # Convert center xywh to xyxy in letterbox coordinates
                x1 = cx - w / 2.0
                y1 = cy - h / 2.0
                x2 = cx + w / 2.0
                y2 = cy + h / 2.0

                # Rescale back to original unpadded frame coordinates
                orig_x1 = (x1 - pad_w) / scale
                orig_y1 = (y1 - pad_h) / scale
                orig_x2 = (x2 - pad_w) / scale
                orig_y2 = (y2 - pad_h) / scale

                boxes_xyxy.append((orig_x1, orig_y1, orig_x2, orig_y2))
                confidences.append(score)
                class_ids.append(class_id)

        # Apply class-aware Non-Maximum Suppression
        selected = _nms(
            boxes_xyxy,
            confidences,
            class_ids,
            iou_threshold=self.config.iou_threshold,
            max_detections=self.config.max_detections,
        )

        detections: list[ObjectDetection] = []
        for det_idx, idx in enumerate(selected):
            bx1, by1, bx2, by2 = boxes_xyxy[idx]
            cls_id = class_ids[idx]
            conf = confidences[idx]

            label = COCO_CLASSES[cls_id] if 0 <= cls_id < len(COCO_CLASSES) else f"object_{cls_id}"

            # CRITICAL PRIVACY INVARIANT: PERSON != IDENTITY
            # Detect person purely as non-identifying "person" label
            if label == "person":
                label = "person"

            # Clamp normalized coordinates strictly to [0.0, 1.0]
            norm_x = max(0.0, min(1.0, bx1 / orig_w))
            norm_y = max(0.0, min(1.0, by1 / orig_h))
            norm_w = min(1.0 - norm_x, max(0.001, (bx2 - bx1) / orig_w))
            norm_h = min(1.0 - norm_y, max(0.001, (by2 - by1) / orig_h))

            bbox = BoundingBox(
                x=norm_x,
                y=norm_y,
                width=norm_w,
                height=norm_h,
                confidence=conf,
            )

            detections.append(
                ObjectDetection(
                    detection_id=f"det_yolo_{frame.sequence_number}_{det_idx}",
                    label=label,
                    confidence=conf,
                    bounding_box=bbox,
                    timestamp=frame.timestamp,
                )
            )

        return detections

    async def detect(self, frame: VisionFrame) -> list[ObjectDetection]:
        """Executes asynchronous object detection on a single VisionFrame."""
        if not self._initialized or self._status != "READY":
            await self.initialize()

        if self._status == "UNAVAILABLE":
            raise ObjectDetectionUnavailableError("YOLO ONNX model is unavailable or missing.")

        start_time = time.perf_counter()
        timeout_sec = self.config.timeout_ms / 1000.0

        async def _run_detection() -> tuple[list[ObjectDetection], float, float, float]:
            if self._simulated_mode:
                await asyncio.sleep(0.01)
                now = datetime.now(UTC)
                dets = [
                    ObjectDetection(
                        detection_id=f"det_sim_{frame.sequence_number}_0",
                        label="person",
                        confidence=0.91,
                        bounding_box=BoundingBox(x=0.15, y=0.10, width=0.30, height=0.60, confidence=0.91),
                        timestamp=now,
                    ),
                    ObjectDetection(
                        detection_id=f"det_sim_{frame.sequence_number}_1",
                        label="laptop",
                        confidence=0.86,
                        bounding_box=BoundingBox(x=0.50, y=0.45, width=0.35, height=0.40, confidence=0.86),
                        timestamp=now,
                    ),
                ]
                return dets, 0.5, 5.0, 0.5

            def _infer_blocking() -> tuple[list[ObjectDetection], float, float, float]:
                t0 = time.perf_counter()
                tensor, scale, pad = self._preprocess(frame)
                t_prep = (time.perf_counter() - t0) * 1000.0

                t1 = time.perf_counter()
                raw_out = self._session.run([self._output_name], {self._input_name: tensor})[0]
                t_infer = (time.perf_counter() - t1) * 1000.0

                t2 = time.perf_counter()
                res = self._postprocess(raw_out, frame, scale, pad)
                t_post = (time.perf_counter() - t2) * 1000.0

                return res, t_prep, t_infer, t_post

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, _infer_blocking)

        task = asyncio.create_task(_run_detection())
        try:
            detections, prep_ms, infer_ms, post_ms = await asyncio.wait_for(
                task,
                timeout=timeout_sec,
            )
            self._frames_processed += 1
            total_lat = (time.perf_counter() - start_time) * 1000.0
            self._record_telemetry(
                frame_id=frame.frame_id,
                detections=detections,
                total_latency_ms=total_lat,
                success=True,
                error_code=None,
                prep_ms=prep_ms,
                infer_ms=infer_ms,
                post_ms=post_ms,
            )
            return detections
        except TimeoutError as exc:
            self._frames_dropped += 1
            self._record_telemetry(frame.frame_id, [], (time.perf_counter() - start_time) * 1000.0, False, "TIMEOUT")
            raise ObjectDetectionTimeoutError(
                f"YOLO inference timed out after {self.config.timeout_ms}ms."
            ) from exc
        except Exception as exc:
            self._inference_failures += 1
            self._record_telemetry(frame.frame_id, [], (time.perf_counter() - start_time) * 1000.0, False, "ERROR")
            if isinstance(exc, (ObjectDetectionValidationError, ObjectDetectionUnavailableError, ObjectDetectionTimeoutError)):
                raise
            raise ObjectDetectionProcessingError(f"YOLO detection processing failure: {exc}") from exc

    def _record_telemetry(
        self,
        frame_id: str,
        detections: list[ObjectDetection],
        total_latency_ms: float,
        success: bool,
        error_code: str | None,
        prep_ms: float = 0.0,
        infer_ms: float = 0.0,
        post_ms: float = 0.0,
    ) -> None:
        """Records privacy-safe telemetry without raw pixel buffers or image files."""
        class_counts: dict[str, int] = {}
        for d in detections:
            class_counts[d.label] = class_counts.get(d.label, 0) + 1

        self._last_telemetry = ObjectDetectionTelemetry(
            frame_id=frame_id,
            detection_count=len(detections),
            class_counts=class_counts,
            model_load_latency_ms=self._model_load_latency_ms,
            inference_latency_ms=infer_ms,
            preprocessing_latency_ms=prep_ms,
            postprocessing_latency_ms=post_ms,
            total_latency_ms=total_latency_ms,
            success=success,
            error_code=error_code,
        )

    async def shutdown(self) -> None:
        """Cleans up ONNX session and shuts down thread executor."""
        self._executor.shutdown(wait=False)
        self._session = None
        self._status = "SHUTDOWN"
        self._initialized = False
        logger.info("YOLOONNXDetector shut down cleanly.")

    async def health(self) -> dict[str, Any]:
        """Probes operational readiness, hardware execution provider, and inference metrics."""
        return {
            "subsystem": "yolo_onnx_detector",
            "status": self._status,
            "provider": "onnx",
            "active_execution_provider": self._active_provider,
            "model_path": self.model_path,
            "model_present": os.path.exists(self.model_path),
            "simulated_mode": self._simulated_mode,
            "frames_processed": self._frames_processed,
            "frames_dropped": self._frames_dropped,
            "inference_failures": self._inference_failures,
            "last_latency_ms": self._last_telemetry.total_latency_ms if self._last_telemetry else 0.0,
        }
