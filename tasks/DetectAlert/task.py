[sys.path.append(os.path.join(os.getcwd(), folder)) for folder in variables.get("dependent_modules_folders").split(",")]
import proactive_helper as ph
import os
import cv2
import numpy as np
import torch
from typing import Tuple, List


def load_model(model_path: str) -> torch.nn.Module:
    """
    Loads the YOLOv5 model from the specified path.
    :param model_path: Path to the YOLOv5 model file (.pt)
    :return: Loaded YOLOv5 model
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    model.conf = 0.3
    model.to(device)
    print(f"Model classes: {model.names}")
    return model


def is_image_file(filename: str) -> bool:
    """
    Checks if a file is an image based on its extension.
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    ext = os.path.splitext(filename)[1].lower()
    return ext in image_extensions


def is_video_file(filename: str) -> bool:
    """
    Checks if a file is a video based on its extension.
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    ext = os.path.splitext(filename)[1].lower()
    return ext in video_extensions


def detect_fire(model: torch.nn.Module, image: np.ndarray, visualize: bool = False) -> Tuple[bool, List[Tuple[str, float]]]:
    """
    Detects fire in the given image using the YOLOv5 model.
    :param model: Loaded YOLOv5 model
    :param image: Image to analyze (NumPy array)
    :param visualize: If True, saves the annotated image with detections
    :return: Tuple containing a boolean indicating fire detection and a list of detected objects
    """
    try:
        results = model(image)
        labels = results.names
        detections = results.pred[0]

        fire_detected = False
        detected_objects = []

        for detection in detections:
            confidence = float(detection[4])
            class_id = int(detection[5])
            class_name = labels[class_id].lower()
            detected_objects.append((class_name, confidence))
            if class_name == 'fire':
                fire_detected = True

        if visualize and (fire_detected or detected_objects):
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                label = f"{labels[int(cls)]} {conf:.2f}"
                cv2.rectangle(image, (int(x1), int(y1)),(int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            output_path = "detection_result.jpg"
            cv2.imwrite(output_path, image)
            print(f"Detections visualized and saved to {output_path}")

        return fire_detected, detected_objects

    except Exception as e:
        print(f"Error during fire detection: {e}")
        return False, []


def process_video(model: torch.nn.Module, video_path: str, frame_sampling: int = 30) -> Tuple[bool, List[Tuple[str, float]]]:
    """
    Processes a video file by extracting frames and detecting fire.
    :param model: Loaded YOLOv5 model
    :param video_path: Path to the video file
    :param frame_sampling: Sample every Nth frame (1 for every frame)
    :return: Tuple containing a boolean indicating fire detection and a list of detected objects
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return False, []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fire_detected = False
    detected_objects_total = []

    print(f"Processing video: {video_path}")
    print(f"Total number of frames: {frame_count}, Sampling every {frame_sampling} frames.")

    for frame_num in range(0, frame_count, frame_sampling):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: cannot read frame {frame_num}")
            continue
        detected, objects = detect_fire(model, frame)
        print(f"Frame {frame_num}: {objects}")
        detected_objects_total.extend(objects)
        if detected:
            print(f"Fire detected in frame {frame_num} of video {video_path}")
            fire_detected = True
            detect_fire(model, frame, visualize=False)
            # Do not stop, continue analyzing all frames
            # Comment the following line to continue analysis
            break

    cap.release()
    return fire_detected, detected_objects_total



# ------------------------------
# Part 3: Local Processing
# ------------------------------

def process_local_files(model, directory: str, user_id: str = "user123"):
    """
    Randomly selects a file from the directory and processes it locally.
    :param model: Loaded YOLOv5 model
    :param directory: Path to the directory containing images/videos
    :param user_id: User identifier
    :return: inference_time (float) if image processed, else None
    """
    import random
    # List all image and video files in the directory
    all_files = [f for f in os.listdir(directory) if is_image_file(f) or is_video_file(f)]
    if not all_files:
        print(f"No image or video files found in directory '{directory}'.")
        return None

    # Randomly select a file
    selected_file = random.choice(all_files)
    selected_path = os.path.join(directory, selected_file)
    print(f"Selected file for processing: {selected_file}")

    if is_image_file(selected_file):
        try:
            import time
            img = cv2.imread(selected_path)
            if img is None:
                raise ValueError("Failed to read image.")
            start_time = time.time()
            fire_detected, objects = detect_fire(model, img, visualize=False)
            end_time = time.time()
            inference_time = end_time - start_time
            print(f"User: {user_id} | Image: {selected_file} | Fire detected: {fire_detected} | Objects: {objects}")
            print(f"Inference time: {inference_time:.4f} seconds")
            return inference_time
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    elif is_video_file(selected_file):
        try:
            fire_detected, objects = process_video(model, selected_path, frame_sampling=10)
            print(f"User: {user_id} | Video: {selected_file} | Fire detected: {fire_detected} | Objects: {objects}")
        except Exception as e:
            print(f"Error processing video: {e}")
        return None
    else:
        print(f"Unsupported file type: {selected_file}")
        return None


# ------------------------------
# Part 4: Combined Execution
# ------------------------------

if __name__ == '__main__':
    import time
    import random
    model_file = variables.get("ModelFile")
    model = load_model(model_file + "/best.pt")
    directory = model_file + "/data"
    user_id = "operator"  # User identifier
    nb_requests = 1  # Number of files to process

    inference_time = None
    for i in range(nb_requests):
        print(f"Processing file {i+1}/{nb_requests}")
        inference_time = process_local_files(model, directory, user_id=user_id)
        if(i < nb_requests - 1):
            time.sleep(1)  # Pause between requests

    if inference_time is not None:
        resultMap.put("INFERENCE_TIME", f"{inference_time:.4f} seconds")
