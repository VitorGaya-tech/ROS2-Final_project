# Sign Detection Project

This folder contains the traffic-sign detection project for the robot.
## Project Layout

`jpg/`
: Source images for the project. These are the original pictures collected.

  Current class folders:

  - `oneway/`
  - `roadClossed/`
  - `stop/`
  - `no_sign/`

  The first three folders are the sign classes. `no_sign` is for background images with no target sign visible.

`yolo_signs/`
: Generated YOLO training data created by the notebook. This folder is rebuilt when you run the dataset preparation step.

  - `images/train` and `images/val` contain copied training images.
  - `labels/train` and `labels/val` contain YOLO label files.
  - `data.yaml` is the dataset config used by Ultralytics YOLO.

`runs/`
: Training outputs saved by Ultralytics. The current training run is in `runs/sign_detector/`.

  Useful files inside that folder include:

  - `weights/best.pt` for the best saved model
  - `weights/last.pt` for the last epoch
  - `results.csv` and `results.png` for training metrics
  - confusion-matrix and curve images for evaluation

`torch_repo/`
: Scripts and notebooks for training and testing.

  - `SignDetectTorch.ipynb` prepares the dataset, trains the model, and can export or test it.
  - `webcam_yolo_test.py` runs a trained YOLO model on the webcam.
  - `yolov8s.pt` and `yolo26n.pt` are starter model weights.

## What You Need

You need a Python environment with the following packages installed:

`torch`
: Deep learning library used by the notebook and model runtime.

`ultralytics`
: YOLO training and inference package.

`opencv-python`
: Used by the webcam tester.

`pyyaml`
: Used to write the YOLO dataset config.

`pillow`
: Used for image handling in the notebook.

`ipykernel`
: Lets the environment show up as a notebook kernel in VS Code.

If you do not already have a Python environment, create one with your preferred tool and then install the packages above.

Example using a virtual environment:

```bash
python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install torch ultralytics opencv-python pyyaml pillow ipykernel
```

If you want GPU acceleration on an NVIDIA card, install a CUDA-enabled PyTorch build that matches your system. On this machine, the notebook was set up for the RTX 4070 on my laptop.

## How To Use It

1. Put your source images in `jpg/`.
2. If you want background training data, add images with no sign visible to `jpg/no_sign/`.
3. Open `torch_repo/SignDetectTorch.ipynb`.
4. Select the Python kernel that has the required packages installed.
5. Run the notebook from top to bottom.

The notebook will:

- detect the project root
- rebuild `yolo_signs/` from the images in `jpg/`
- write the YOLO dataset config
- train the model if training is enabled
- save results into `runs/sign_detector/`

## Webcam Testing

After training, use `torch_repo/webcam_yolo_test.py` to test the saved model on your webcam.

Example:

```bash
python torch_repo/webcam_yolo_test.py --model runs/sign_detector/weights/best.pt
```

The webcam tester shows only one primary detection at a time. If nothing is detected, it shows `no sign`.

## Important Notes

- `jpg/` is the source of truth for your raw images.
- `yolo_signs/` is generated data and can be recreated from the notebook.
- `runs/` contains training results and model weights.
- Do not edit the generated YOLO labels by hand unless you know you need to.
- If you add or remove classes, make sure the image folders and label list match.
- If you want the robot to treat nothing visible as a decision, use the `no_sign` folder with real background images.

## Suggested Workflow

1. Collect new photos.
2. Sort them into the correct folder under `jpg/`.
3. Add background-only images to `jpg/no_sign/` if you want the model to learn empty scenes.
4. Run the notebook to regenerate the YOLO dataset.
5. Train the model.
6. Test with the webcam script.
7. Copy `runs/sign_detector/weights/best.pt` into your robot project when you are ready to deploy.

## Troubleshooting

- If the notebook says a class is missing, check that the folder names in `jpg/` match the class names expected by the notebook.
- If the webcam script cannot open the camera, try a different camera index with `--camera 1` or `--camera 2`.
- If PyTorch cannot see the GPU, make sure you installed a CUDA-enabled build of PyTorch and that your NVIDIA drivers are working.
- If you only want to test the model and not retrain it, set the notebook's training flag to off before running the last cell.
