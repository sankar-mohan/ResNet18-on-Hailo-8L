"""Live camera classification on Raspberry Pi 5 + AI HAT (Hailo-8L) with resnet18.hef.

Grabs frames from a Pi Camera (Picamera2) or USB webcam (OpenCV), classifies the centre of each
frame on the Hailo chip, and shows the video in an OpenCV window with the top predictions,
FPS and Hailo inference time. Runs until 'q' / Esc is pressed or the window is closed.

Usage:
  python3 camera_classify.py --camera picam
  python3 camera_classify.py --camera usb --device 0 --threshold 0.3 --topk 3

Preprocessing matches prep_data.py / infer_pi.py: centre-crop (224/256 of the short side),
resize to 224x224, RGB uint8. Normalization runs on the chip.

Steps on the Pi

1. Copy the files (from WSL on your PC):
cd /mnt/d/eai_apps/ml_apps/model_study/resnet18
scp script/camera_classify.py pihat@sankar-pihat1:~/study/hailo-apps-8l/resnet18/scripts/

2. Install OpenCV and Picamera2 (on the Pi, from a terminal in the VNC desktop):
sudo apt install -y python3-opencv python3-picamera2
Your venv venv_hailo_apps needs to see these system packages. Check:
python -c "import cv2, picamera2, hailo_platform; print('ok', cv2.__version__)"
If that fails, use the system python3 instead of the venv's python. hailo_platform already loads from /usr/lib/python3/dist-packages, judging by your traceback.

3. Check the camera:
rpicam-hello --list-cameras      # Pi Camera
ls /dev/video*                   # USB webcam (usually /dev/video0)

4. Run it from the VNC desktop's terminal, not from a plain SSH session, which can't open windows:
cd ~/study/hailo-apps-8l/resnet18
# Pi Camera
python scripts/camera_classify.py --hef hf_out/resnet18.hef --labels hf_out/imagenet_labels.txt --camera picam
ls /dev/video*                   # USB webcam (usually /dev/video0)

4. Run it from the VNC desktop's terminal, not from a plain SSH session, which can't open windows:
cd ~/study/hailo-apps-8l/resnet18
# Pi Camera
python scripts/camera_classify.py --hef hf_out/resnet18.hef --labels hf_out/imagenet_labels.txt --camera picam
# USB webcam
python scripts/camera_classify.py --hef hf_out/resnet18.hef --labels hf_out/imagenet_labels.txt --camera usb --device 0

Options:

┌────────────────────┬───────────┬─────────────────────────────────────────┐
│        Flag        │  Default  │                 Meaning                 │
├────────────────────┼───────────┼─────────────────────────────────────────┤
│ --threshold        │ 0.2       │ Hide predictions below this probability │
├────────────────────┼───────────┼─────────────────────────────────────────┤
│ --topk             │ 3         │ How many predictions to show            │
├────────────────────┼───────────┼─────────────────────────────────────────┤
│ --width / --height │ 640 / 480 │ Capture size                            │
├────────────────────┼───────────┼─────────────────────────────────────────┤
│ --device           │ 0         │ USB camera number (/dev/videoN)         │
└────────────────────┴───────────┴─────────────────────────────────────────┘

"""
import argparse
import time

import cv2
import numpy as np
from hailo_platform import (HEF, ConfigureParams, FormatType, HailoStreamInterface, InferVStreams,
                            InputVStreamParams, OutputVStreamParams, VDevice)

CROP_FRACTION = 224 / 256   # same as "resize short side to 256, centre-crop 224"
WINDOW = "ResNet18 on Hailo-8L"


class PiCamera:
    def __init__(self, width, height):
        from picamera2 import Picamera2  # only needed for the Pi Camera
        self.cam = Picamera2()
        # Picamera2's "RGB888" is stored B,G,R in memory: already OpenCV's BGR order.
        config = self.cam.create_preview_configuration(main={"size": (width, height), "format": "RGB888"})
        self.cam.configure(config)
        self.cam.start()

    def read(self):
        return self.cam.capture_array("main")

    def close(self):
        self.cam.stop()


class UsbCamera:
    def __init__(self, device, width, height):
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open USB camera {device}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


def centre_crop_box(frame):
    h, w = frame.shape[:2]
    side = int(min(h, w) * CROP_FRACTION)
    x0, y0 = (w - side) // 2, (h - side) // 2
    return x0, y0, side


def preprocess(frame_bgr, box):
    x0, y0, side = box
    crop = frame_bgr[y0:y0 + side, x0:x0 + side]
    rgb = cv2.cvtColor(cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    # HailoRT needs a C-contiguous, writeable uint8 buffer with a batch dimension.
    return np.array(rgb[None], dtype=np.uint8, order="C")


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def draw_text(img, text, org, scale=0.6, color=(255, 255, 255)):
    # Dark outline keeps text readable on any background.
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_overlay(frame, box, preds, fps, infer_ms, threshold):
    x0, y0, side = box
    cv2.rectangle(frame, (x0, y0), (x0 + side, y0 + side), (0, 200, 255), 2)

    y = 28
    if preds:
        for rank, (name, prob) in enumerate(preds, 1):
            draw_text(frame, f"{rank}. {name}  {prob:.2f}", (10, y), 0.7, (80, 255, 80) if rank == 1 else (255, 255, 255))
            y += 28
    else:
        draw_text(frame, f"no prediction above {threshold:.2f}", (10, y), 0.7, (0, 200, 255))

    h = frame.shape[0]
    draw_text(frame, f"FPS {fps:5.1f} | Hailo {infer_ms:4.1f} ms | q to quit", (10, h - 12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", default="resnet18.hef")
    ap.add_argument("--labels", default="imagenet_labels.txt")
    ap.add_argument("--camera", choices=["picam", "usb"], default="picam")
    ap.add_argument("--device", type=int, default=0, help="USB camera index (/dev/videoN)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--threshold", type=float, default=0.2, help="hide predictions below this probability")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    try:
        names = open(args.labels).read().splitlines()
    except OSError:
        print(f"labels file {args.labels} not found; showing class indices")
        names = None

    hef = HEF(args.hef)
    in_info = hef.get_input_vstream_infos()[0]
    out_info = hef.get_output_vstream_infos()[0]

    camera = PiCamera(args.width, args.height) if args.camera == "picam" else \
        UsbCamera(args.device, args.width, args.height)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    try:
        with VDevice() as device:
            params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
            network_group = device.configure(hef, params)[0]
            in_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
            out_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

            with InferVStreams(network_group, in_params, out_params) as pipeline, \
                    network_group.activate(network_group.create_params()):
                print(f"running: camera={args.camera} {args.width}x{args.height}, press q in the window to quit")
                fps, last = 0.0, time.perf_counter()

                while True:
                    frame = camera.read()
                    if frame is None:
                        print("camera returned no frame; stopping")
                        break

                    box = centre_crop_box(frame)
                    t0 = time.perf_counter()
                    logits = pipeline.infer({in_info.name: preprocess(frame, box)})[out_info.name].reshape(-1)
                    infer_ms = (time.perf_counter() - t0) * 1000

                    probs = softmax(logits)
                    top = probs.argsort()[::-1][:args.topk]
                    preds = [(names[i] if names else str(i), float(probs[i])) for i in top if probs[i] >= args.threshold]

                    now = time.perf_counter()
                    fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6)) if fps else 1.0 / max(now - last, 1e-6)
                    last = now

                    draw_overlay(frame, box, preds, fps, infer_ms, args.threshold)
                    cv2.imshow(WINDOW, frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
