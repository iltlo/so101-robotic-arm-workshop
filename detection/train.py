# YOLOv8 Ultralytics training script
# Trains a YOLO model using a dataset in YOLO format under dataset/generated
# Generates utils as needed for dataset config and training

import os
from ultralytics import YOLO

DATASET_DIR = os.environ.get('DATASET_DIR', os.path.join(os.path.dirname(__file__), '../dataset/screw.v2i.yolov8-obb'))

# Default model names / overrides
# If your ultralytics install provides official OBB weights, set YOLO_OBB_MODEL accordingly
YOLO_OBB_MODEL = os.environ.get('YOLO_OBB_MODEL', 'yolov8m-obb.pt')
YOLO_DEFAULT_MODEL = os.environ.get('YOLO_MODEL', 'yolo11m.pt')

def detect_label_format(dataset_path: str):
    """Inspect label files to determine whether labels are OBB (8 coords) or bbox (4 coords).

    Returns: 'obb', 'bbox', or None if unknown/not found.
    """
    candidates = [
        os.path.join(dataset_path, 'train', 'labels'),
        os.path.join(dataset_path, 'valid', 'labels'),
        os.path.join(dataset_path, 'test', 'labels'),
    ]
    for labels_dir in candidates:
        if not os.path.isdir(labels_dir):
            continue
        for fname in sorted(os.listdir(labels_dir)):
            if not fname.lower().endswith('.txt'):
                continue
            path = os.path.join(labels_dir, fname)
            try:
                with open(path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) <= 1:
                            continue
                        nums = parts[1:]
                        if len(nums) == 8:
                            return 'obb'
                        if len(nums) == 4:
                            return 'bbox'
            except Exception:
                continue
    return None

# Download dataset from Roboflow if not already present
def download_dataset():
    # Use DATASET_DIR (either from env or default relative path)
    dataset_dir = os.path.abspath(DATASET_DIR)

    # If the dataset_dir itself contains data.yaml, treat it as the dataset root
    data_yaml = os.path.join(dataset_dir, 'data.yaml')
    if os.path.isfile(data_yaml):
        return dataset_dir

    # Otherwise, look for a subdirectory that contains data.yaml
    if os.path.isdir(dataset_dir):
        for entry in sorted(os.listdir(dataset_dir)):
            candidate = os.path.join(dataset_dir, entry)
            if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, 'data.yaml')):
                return candidate

    # Not found — give a helpful error explaining how to point to a local dataset
    raise FileNotFoundError(
        f"No dataset found in '{dataset_dir}'.\n"
        "Place your dataset directory (the folder that contains 'data.yaml') inside this path,\n"
        "or set the DATASET_DIR environment variable to point directly at your dataset folder.\n"
        "Example: export DATASET_DIR=$(pwd)/../dataset/screw.v2i.yolov8-obb"
    )

# Main training function
def main():
    dataset_path = download_dataset()

    # Setup some sensible defaults and paths
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    yolo_dir = os.path.join(repo_dir, "yolo")
    os.makedirs(yolo_dir, exist_ok=True)

    # data yaml path expected by Ultralytics YOLO - require it to exist
    data_yaml_path = os.path.join(dataset_path, "data.yaml")

    # Train YOLOv11 (medium) model, change model as needed
    # detect label format and select model
    label_format = detect_label_format(dataset_path)
    if label_format == 'obb':
        model_path = YOLO_OBB_MODEL
        print(f"Detected OBB-style labels; using OBB model: {model_path}")
    elif label_format == 'bbox':
        model_path = YOLO_DEFAULT_MODEL
        print(f"Detected bbox labels; using standard model: {model_path}")
    else:
        # fallback: prefer OBB model because dataset folder name suggests OBB, but warn
        model_path = YOLO_OBB_MODEL
        print(f"Could not detect label format. Defaulting to OBB model: {model_path}. \n"
              "If this is incorrect, set YOLO_MODEL or YOLO_OBB_MODEL env vars.")

    model = YOLO(model_path)
    results = model.train(
        data=data_yaml_path,
        epochs=30,
        imgsz=640,
        batch=4,
        workers=16,
        project=os.path.join(repo_dir, 'model'),
        name='yolo11m-screw',
        save_period=10,  # Save every 50 epochs
    )

if __name__ == "__main__":
    main()