"""Run a YOLO model on a webcam feed.

Examples:
    python webcam_yolo_test.py --model ../runs/sign_detector/weights/best.pt
    python webcam_yolo_test.py --model yolo26n.pt
    python webcam_yolo_test.py --model yolov8s.pt --imgsz 416 --conf 0.25

Press `q` to quit the preview window.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT.parent / "runs" / "sign_detector" / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a YOLO model on a webcam feed.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to a YOLO .pt file. Defaults to the trained robot model.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index to open.")
    parser.add_argument("--imgsz", type=int, default=416, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use, for example 0, cpu, or cuda:0. Defaults to auto selection.",
    )
    return parser.parse_args()


def resolve_device(device: str | None) -> str:
    if device:
        return device
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def describe_primary_detection(result: object) -> tuple[str, tuple[int, int, int, int] | None]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return "no sign", None

    best_idx = int(boxes.conf.argmax().item())
    best_class = int(boxes.cls[best_idx].item())
    label = result.names[best_class]

    x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()
    return label, (int(x1), int(y1), int(x2), int(y2))


def main() -> int:
    args = parse_args()
    model_path = args.model.expanduser().resolve()

    if not model_path.exists():
        print(f"Model file not found: {model_path}", file=sys.stderr)
        return 1

    device = resolve_device(args.device)
    model = YOLO(str(model_path))

    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(args.camera)

    if not capture.isOpened():
        print(f"Could not open camera {args.camera}", file=sys.stderr)
        return 1

    print(f"Loaded model: {model_path}")
    print(f"Using device: {device}")
    print("Press q in the window to quit.")

    while True:
        ok, frame = capture.read()
        if not ok:
            print("Failed to read from webcam.", file=sys.stderr)
            break

        results = model.predict(frame, device=device, imgsz=args.imgsz, conf=args.conf, verbose=False)
        status, box = describe_primary_detection(results[0])

        annotated = frame.copy()
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                status,
                (x1, max(24, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                annotated,
                "no sign",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("YOLO Webcam Test", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())