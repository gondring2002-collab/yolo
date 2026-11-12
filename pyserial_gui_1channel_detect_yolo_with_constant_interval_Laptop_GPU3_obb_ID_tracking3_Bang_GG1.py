import cv2
import numpy as np
import time
import csv
import threading
import queue
from datetime import datetime
from collections import Counter, deque
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
from PIL import Image, ImageTk
import sys
import os

# --- TRIK BYPASS INSTALASI ---
#PATH_KE_CLONE_LOKAL = r"D:\ultralytics"
#sys.path.insert(0, PATH_KE_CLONE_LOKAL)

from ultralytics import YOLO


# Try to import YOLO
try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: ultralytics not installed. Run 'pip install ultralytics'")
    YOLO_AVAILABLE = False

# Configuration
MODEL_PATH = r"C:\Users\gigin\Documents\python\best1409.pt"
#MODEL_PATH = r"D:\YOLO_LenganPhantom\runs\obb\yolo11s-obb_nibp_10092026_600epoch\weights\best.pt"
#MODEL_PATH = r"D:\YOLO_LenganPhantom\runs\obb\yolo11s-obb-CSFI_Attn_13092026\weights\best.pt"
OUTPUT_DIR = "yolo_detection_results"
VIDEO_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "videos")
FRAMES_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "frames")
ANNOTATED_FRAMES_DIR = os.path.join(OUTPUT_DIR, "annotated_frames")
DATASET_FRAMES_DIR = os.path.join(OUTPUT_DIR, "dataset_frames")
RESULTS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "results")

# Detection settings
DETECTION_INTERVAL = 0.1
MIN_CONFIDENCE = 0.1

# Temporal Filter settings
TEMPORAL_FILTER_ENABLED = True
TEMPORAL_WINDOW_SIZE = 4
MIN_STABLE_COUNT = 2

# Pre-filter settings
PRE_FILTER_ENABLED = True
MAX_NEIGHBOR_CHANGE = 40

# Performance settings
USE_GPU = True
CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = 640
VIDEO_QUALITY = 95
USE_OBB = True

# Save EVERY detection frame's data (except removed outliers)
SAVE_EVERY_DETECTION = True

# Skip saving frames for removed outliers (keeps frame/label count 1-to-1)
SKIP_FRAMES_FOR_REMOVED = True

# ⭐ Auto-start recording when a video file is loaded
AUTO_START_RECORDING_FOR_VIDEO = True

# ⭐ Auto-stop recording when video file ends
AUTO_STOP_RECORDING_FOR_VIDEO = True

# Create output directories
for dir_path in [OUTPUT_DIR, VIDEO_OUTPUT_DIR, FRAMES_OUTPUT_DIR, ANNOTATED_FRAMES_DIR,
                 DATASET_FRAMES_DIR, RESULTS_OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Colors for different digits
color_map = {
    '0': (0, 255, 0), '1': (255, 0, 0), '2': (0, 0, 255),
    '3': (255, 255, 0), '4': (255, 0, 255), '5': (0, 255, 255),
    '6': (255, 165, 0), '7': (128, 0, 128), '8': (0, 128, 128),
    '9': (128, 128, 0),
}


class YOLODetector:
    """Class for YOLO-based digit detection with Temporal & Outlier filtering"""

    def __init__(self, model_path, camera_index=0, frame_width=640, frame_height=480, detection_interval=0.1):
        self.model_path = model_path
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.detection_interval = detection_interval

        self.model = None
        self.cap = None
        self.video_writer = None
        self.video_fps = 30
        self.video_frame_count = 0
        self.last_frame_time = 0

        self.detections = []
        self.removed_detections = []
        self.frame_count = 0
        self.frames_saved = 0
        self.annotated_frames_saved = 0
        self.dataset_frames_saved = 0
        self.previous_combined_string = ""

        self.temporal_buffer = deque(maxlen=TEMPORAL_WINDOW_SIZE)
        self.previous_valid_reading = None
        self.last_raw_reading = None

        self.start_time = None
        self.end_time = None
        self.is_running = False
        self.is_recording = False
        self.save_original_frames = False
        self.save_annotated_frames = False
        self.preview_running = False
        self.preview_thread = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.device = None
        self.model_loaded = False
        self.actual_fps = 0
        self.inference_time = 0
        self.fps_start_time = time.time()
        self.fps_frame_count = 0

        self.last_detection_time = 0
        self.last_detection_frame = None
        self.last_detection_data = None

        self.total_detections_raw = 0
        self.removed_count = 0

        # ⭐ Track if current source is a video file
        self.is_video_file_source = False

        # ⭐ Callback when video ends (used by GUI to auto-stop recording)
        self.on_video_end_callback = None

        # ⭐ Flag to signal video has ended
        self.video_ended = False

    def check_gpu(self):
        if torch.cuda.is_available():
            self.device = 'cuda:0'
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            return True
        else:
            self.device = 'cpu'
            print("No GPU detected, using CPU")
            return False

    def load_model(self):
        if not YOLO_AVAILABLE:
            print("Error: ultralytics not installed")
            return False

        if not os.path.exists(self.model_path):
            print(f"Error: Model file not found at {self.model_path}")
            return False

        try:
            print("Loading YOLO model...")
            start_time = time.time()
            self.check_gpu()

            if USE_OBB:
                self.model = YOLO(self.model_path, task='obb')
            else:
                self.model = YOLO(self.model_path)

            print("Warming up model...")
            dummy_input = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            self.model(dummy_input, conf=CONFIDENCE_THRESHOLD, verbose=False, device=self.device)

            print(f"Model loaded in {time.time() - start_time:.2f}s")
            self.model_loaded = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def initialize_camera(self, video_path=None):
        if video_path and os.path.exists(video_path):
            self.cap = cv2.VideoCapture(video_path)
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.video_fps <= 0 or self.video_fps > 120:
                self.video_fps = 30
            self.is_video_file_source = True
            print(f"Video file loaded: {video_path}")
            print(f"Video FPS: {self.video_fps}")
        else:
            self.is_video_file_source = False
            for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
                self.cap = cv2.VideoCapture(self.camera_index, backend)
                if self.cap.isOpened():
                    break

            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_index}")
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.video_fps = 30

        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frame_width = int(actual_width)
        self.frame_height = int(actual_height)
        return True

    def initialize_video_writer(self, output_path):
        codecs = [
            ('avc1', cv2.VideoWriter_fourcc(*'avc1')),
            ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
            ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
        ]
        for codec_name, fourcc in codecs:
            try:
                self.video_writer = cv2.VideoWriter(
                    output_path, fourcc, self.video_fps, (self.frame_width, self.frame_height)
                )
                if self.video_writer.isOpened():
                    return True
            except:
                continue
        return False

    def process_frame_with_detection(self, frame):
        inference_start = time.time()
        results = self.model(
            frame, conf=CONFIDENCE_THRESHOLD, verbose=False,
            device=self.device, imgsz=IMAGE_SIZE, half=False
        )
        self.inference_time = time.time() - inference_start

        detection_list = []
        annotated_frame = frame.copy()

        if len(results) > 0:
            result = results[0]
            if USE_OBB:
                obb = getattr(result, 'obb', None)
                if obb is not None and len(obb) > 0:
                    try:
                        xyxyxyxy = obb.xyxyxyxy.cpu().numpy()
                        confs = obb.conf.cpu().numpy()
                        clss = obb.cls.cpu().numpy()

                        for i in range(len(xyxyxyxy)):
                            pts = xyxyxyxy[i].astype(np.int32)
                            confidence = float(confs[i])
                            class_id = int(clss[i])
                            class_name = result.names[class_id]

                            if class_name in '0123456789' and confidence >= MIN_CONFIDENCE:
                                xs, ys = pts[:, 0], pts[:, 1]
                                x1, y1 = int(xs.min()), int(ys.min())
                                x2, y2 = int(xs.max()), int(ys.max())

                                detection_list.append({
                                    'label': class_name, 'confidence': confidence,
                                    'bbox': (x1, y1, x2, y2), 'poly': pts
                                })

                                color = color_map.get(class_name, (255, 255, 255))
                                cv2.polylines(annotated_frame, [pts], True, color, 3)
                                cv2.putText(annotated_frame, f"{class_name} {confidence:.2f}",
                                            (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    except Exception as e:
                        print(f"OBB parse error: {e}")
            else:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = result.names[class_id]

                        if class_name in '0123456789' and confidence >= MIN_CONFIDENCE:
                            detection_list.append({
                                'label': class_name, 'confidence': confidence,
                                'bbox': (int(x1), int(y1), int(x2), int(y2))
                            })
                            color = color_map.get(class_name, (255, 255, 255))
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                            cv2.putText(annotated_frame, f"{class_name} {confidence:.2f}",
                                        (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        detection_list.sort(key=lambda d: d['bbox'][0])
        detection_data = None
        should_remove = False
        display_text = ""
        text_color = (255, 255, 255)
        reading_value = None

        if detection_list:
            combined_number = "".join([d['label'] for d in detection_list])
            avg_confidence = np.mean([d['confidence'] for d in detection_list])
            bbox_count = len(detection_list)
            self.total_detections_raw += 1

            if combined_number.isdigit() and len(combined_number) <= 3:
                raw_reading_value = int(combined_number)
                reading_value = raw_reading_value

                # ⭐ Compute change from previous RAW reading (for logging)
                change_from_prev = 0
                if self.last_raw_reading is not None:
                    change_from_prev = abs(raw_reading_value - self.last_raw_reading)

                # --- 1. TEMPORAL FILTER ---
                if TEMPORAL_FILTER_ENABLED:
                    if self.last_raw_reading is not None and \
                            raw_reading_value != self.last_raw_reading:
                        self.temporal_buffer.clear()

                    self.temporal_buffer.append(raw_reading_value)

                    if len(self.temporal_buffer) < TEMPORAL_WINDOW_SIZE:
                        reading_value = raw_reading_value
                        should_remove = False
                        display_text = f"Reading: {reading_value} (buffering)"
                        text_color = (0, 255, 255)
                    else:
                        counter = Counter(self.temporal_buffer)
                        most_common_val, count = counter.most_common(1)[0]

                        if count >= MIN_STABLE_COUNT and \
                                most_common_val == raw_reading_value:
                            reading_value = most_common_val
                        elif self.temporal_buffer.count(raw_reading_value) >= 2:
                            reading_value = raw_reading_value
                        else:
                            reading_value = raw_reading_value
                            should_remove = True
                            display_text = f"Unstable Transition: {raw_reading_value}"
                            text_color = (0, 165, 255)

                # ============================================================
                # --- 2. OUTLIER FILTER (⭐ FIXED) ---
                # Use last_raw_reading as primary reference to track the
                # RUNNING BASELINE instead of a stale "last valid" value.
                # This prevents the entire sweep from being rejected.
                # ============================================================
                if not should_remove:
                    reference = self.last_raw_reading
                    if reference is None:
                        reference = self.previous_valid_reading

                    if PRE_FILTER_ENABLED and reference is not None:
                        change_from_valid = abs(reading_value - reference)
                        if change_from_valid > MAX_NEIGHBOR_CHANGE:
                            should_remove = True
                            display_text = f"Removed: {reading_value} (change {change_from_valid})"
                            text_color = (0, 0, 255)

                    if not should_remove and display_text == "":
                        display_text = f"Reading: {reading_value}"
                        text_color = (0, 255, 0)
                        self.previous_valid_reading = reading_value

                # ⭐ Update last_raw_reading ONLY when NOT removed
                # This keeps the reference clean for the next frame's comparison
                if not should_remove:
                    self.last_raw_reading = raw_reading_value

                # Log removed items only for outliers
                if should_remove and "Removed:" in display_text:
                    self.removed_count += 1
                    self.removed_detections.append({
                        'timestamp': datetime.now(),
                        'reading': reading_value,
                        'raw_reading': raw_reading_value,
                        'confidence': avg_confidence,
                        'bbox_count': bbox_count,
                        'frame_number': self.video_frame_count,
                        'change_from_prev': change_from_prev
                    })

                cv2.putText(annotated_frame, display_text, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                cv2.putText(annotated_frame, display_text, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 2)

                # ============================================================
                # SAVE LOGIC - Only save if NOT removed
                # ============================================================
                if self.is_recording:
                    now = datetime.now()

                    if should_remove:
                        save_status = "removed"
                    elif "buffering" in display_text:
                        save_status = "buffering"
                    else:
                        save_status = "valid"

                    if SAVE_EVERY_DETECTION and save_status != "removed":
                        self.previous_combined_string = str(reading_value) if reading_value else ""

                        detection_data = {
                            'timestamp': now,
                            'reading': reading_value if reading_value else 0,
                            'raw_reading': raw_reading_value,
                            'confidence': avg_confidence,
                            'bbox_count': bbox_count,
                            'frame_number': self.video_frame_count,
                            'change_from_prev': change_from_prev,
                            'status': save_status
                        }

        elif self.is_recording and SAVE_EVERY_DETECTION:
            detection_data = {
                'timestamp': datetime.now(),
                'reading': 0,
                'raw_reading': 0,
                'confidence': 0.0,
                'bbox_count': 0,
                'frame_number': self.video_frame_count,
                'change_from_prev': 0,
                'status': 'no_detection'
            }

        cv2.putText(annotated_frame, datetime.now().strftime('%H:%M:%S'),
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Frame: {self.video_frame_count}",
                    (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(annotated_frame, f"Removed (Outlier): {self.removed_count}",
                    (self.frame_width - 300, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.putText(annotated_frame, f"Saved: {len(self.detections)}",
                    (self.frame_width - 300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if self.is_recording:
            cv2.circle(annotated_frame, (self.frame_width - 40, 40), 15, (0, 0, 255), -1)

        return annotated_frame, detection_data

    def save_original_frame(self, frame, timestamp):
        if not self.save_original_frames:
            return
        try:
            timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f"original_{self.dataset_frames_saved:06d}_{timestamp_str}.jpg"
            filepath = os.path.join(DATASET_FRAMES_DIR, filename)
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, VIDEO_QUALITY])
            self.dataset_frames_saved += 1
        except Exception as e:
            print(f"Error saving original frame: {e}")

    def save_annotated_frame(self, frame, timestamp):
        if not self.save_annotated_frames:
            return
        try:
            timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f"annotated_{self.annotated_frames_saved:06d}_{timestamp_str}.jpg"
            filepath = os.path.join(ANNOTATED_FRAMES_DIR, filename)
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, VIDEO_QUALITY])
            self.annotated_frames_saved += 1
        except Exception as e:
            print(f"Error saving annotated frame: {e}")

    def start_preview(self, video_path=None):
        if not self.model_loaded and not self.load_model():
            return False
        if not self.cap or not self.cap.isOpened():
            if not self.initialize_camera(video_path):
                return False

        self.preview_running = True
        self.is_running = True
        self.video_ended = False
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.last_detection_time = 0
        self.last_detection_frame = None
        self.last_detection_data = None
        self.video_frame_count = 0
        self.last_frame_time = time.time()

        self.total_detections_raw = 0
        self.removed_count = 0
        self.previous_valid_reading = None
        self.last_raw_reading = None
        self.removed_detections = []

        self.temporal_buffer.clear()
        self.previous_combined_string = ""

        self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
        self.preview_thread.start()
        return True

    def _preview_loop(self):
        frame_interval = 1.0 / self.video_fps if self.video_fps > 0 else 1.0/30
        self.last_frame_time = time.time()
        consecutive_failures = 0

        while self.preview_running:
            ret, frame = self.cap.read()

            if not ret:
                if self.is_video_file_source:
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        print("\n" + "="*60)
                        print("📹 VIDEO FILE ENDED")
                        print("="*60)
                        self.video_ended = True

                        if self.on_video_end_callback is not None:
                            try:
                                print("Triggering auto-stop callback...")
                                self.on_video_end_callback()
                            except Exception as e:
                                print(f"Error in video end callback: {e}")
                        break
                    else:
                        time.sleep(0.05)
                        continue
                else:
                    print("Camera stream ended")
                    break

            consecutive_failures = 0

            self.video_frame_count += 1
            current_time = time.time()
            expected_time = self.last_frame_time + frame_interval

            if current_time < expected_time:
                time.sleep(expected_time - current_time)

            self.last_frame_time = time.time()
            should_detect = (current_time - self.last_detection_time) >= self.detection_interval

            if should_detect:
                annotated_frame, detection_data = self.process_frame_with_detection(frame)
                self.last_detection_time = time.time()
                self.last_detection_frame = annotated_frame
                self.last_detection_data = detection_data
                should_save_frame = True
            else:
                if self.last_detection_frame is not None:
                    annotated_frame = self.last_detection_frame.copy()
                    cv2.putText(annotated_frame, datetime.now().strftime('%H:%M:%S'),
                                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    annotated_frame = frame.copy()

                detection_data = self.last_detection_data
                should_save_frame = False

            self.fps_frame_count += 1
            elapsed = time.time() - self.fps_start_time
            if elapsed >= 1.0:
                self.actual_fps = self.fps_frame_count / elapsed
                self.fps_frame_count = 0
                self.fps_start_time = time.time()

            with self.frame_lock:
                self.current_frame = annotated_frame.copy()

            if self.is_recording:
                current_timestamp = datetime.now()
                if self.video_writer:
                    self.video_writer.write(frame)

                if should_save_frame:
                    if detection_data:
                        if self.save_original_frames:
                            self.save_original_frame(frame, current_timestamp)
                        if self.save_annotated_frames:
                            self.save_annotated_frame(annotated_frame, current_timestamp)
                        self.detections.append(detection_data)
                    elif not SKIP_FRAMES_FOR_REMOVED:
                        if self.save_original_frames:
                            self.save_original_frame(frame, current_timestamp)
                        if self.save_annotated_frames:
                            self.save_annotated_frame(annotated_frame, current_timestamp)

        print("Preview loop stopped.")

    def start_recording(self):
        if not self.preview_running and not self.start_preview():
            return False

        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.recording_timestamp = timestamp_str
        video_filename = f"output_{timestamp_str}.mp4"
        video_path = os.path.join(VIDEO_OUTPUT_DIR, video_filename)

        if not self.initialize_video_writer(video_path):
            return False

        self.detections = []
        self.removed_detections = []
        self.frames_saved = 0
        self.annotated_frames_saved = 0
        self.dataset_frames_saved = 0
        self.previous_combined_string = ""
        self.previous_valid_reading = None
        self.last_raw_reading = None
        self.total_detections_raw = 0
        self.removed_count = 0
        self.temporal_buffer.clear()

        self.is_recording = True
        self.start_time = datetime.now()

        print("\n" + "="*60)
        print("Recording Started - Save All EXCEPT Removed Outliers")
        print(f"Video FPS: {self.video_fps:.2f}")
        print(f"Detection Interval: {self.detection_interval*1000:.0f}ms")
        print(f"Pre-filter: Remove changes > {MAX_NEIGHBOR_CHANGE} mmHg")
        print(f"Temporal Filter: Window {TEMPORAL_WINDOW_SIZE}, Min {MIN_STABLE_COUNT}")
        print(f"Skip frames for removed: {SKIP_FRAMES_FOR_REMOVED}")
        print("="*60)
        return True

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        self.end_time = datetime.now()

        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        self.save_results()
        print("\n" + "="*60)
        print(f"Total Detections: {self.total_detections_raw}")
        print(f"Removed (Outliers, NOT saved): {self.removed_count}")
        print(f"Saved Detection Records: {len(self.detections)}")
        print(f"Saved Annotated Frames: {self.annotated_frames_saved}")
        print(f"Total Frames Processed: {self.video_frame_count}")
        print("="*60)

    def stop_preview(self):
        self.preview_running = False
        self.is_running = False
        if self.is_recording:
            self.stop_recording()
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    def get_current_frame(self):
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None

    def save_results(self):
        """
        Save results to THREE files:
        1. results_*.txt  - CLEAN ML-ready data (no removed outliers)
        2. results_*.csv  - Same clean data in CSV format
        3. removed_*.txt  - Debug file with removed outliers only
        """
        if not self.detections:
            print("No detections to save")
            return

        timestamp_str = getattr(self, 'recording_timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))
        result_path = os.path.join(RESULTS_OUTPUT_DIR, f"results_{timestamp_str}.txt")
        csv_path = os.path.join(RESULTS_OUTPUT_DIR, f"results_{timestamp_str}.csv")
        removed_path = os.path.join(RESULTS_OUTPUT_DIR, f"removed_{timestamp_str}.txt")

        try:
            with open(result_path, 'w') as f:
                f.write("# YOLO NIBP Detection Results (Temporal + Outlier Filtered)\n")
                f.write(f"# Detection Interval: {self.detection_interval*1000:.0f}ms\n")
                f.write(f"# Temporal Filter: Stable required {MIN_STABLE_COUNT}/{TEMPORAL_WINDOW_SIZE}\n")
                f.write(f"# Pre-filter threshold: >{MAX_NEIGHBOR_CHANGE} mmHg change\n")
                f.write(f"# Total Raw Detections: {self.total_detections_raw}\n")
                f.write(f"# Removed (Outliers, excluded from this file): {self.removed_count}\n")
                f.write(f"# Saved Detection Records: {len(self.detections)}\n")
                f.write(f"# Annotated Frames Saved: {self.annotated_frames_saved}\n")
                f.write("#" + "="*59 + "\n")
                f.write("Timestamp,Reading,Raw_Reading,Confidence,BBox_Count,Frame_Number,Change,Status\n")

                for detection in self.detections:
                    f.write(f"{detection['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]},"
                           f"{detection['reading']},"
                           f"{detection.get('raw_reading', detection['reading'])},"
                           f"{detection['confidence']:.3f},"
                           f"{detection['bbox_count']},"
                           f"{detection['frame_number']},"
                           f"{detection.get('change_from_prev', 0)},"
                           f"{detection.get('status', 'valid')}\n")

            with open(csv_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Timestamp', 'Reading', 'Raw_Reading', 'Confidence',
                                     'BBox_Count', 'Frame_Number', 'Change', 'Status'])
                for detection in self.detections:
                    csv_writer.writerow([
                        detection['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                        detection['reading'],
                        detection.get('raw_reading', detection['reading']),
                        f"{detection['confidence']:.3f}",
                        detection['bbox_count'],
                        detection['frame_number'],
                        detection.get('change_from_prev', 0),
                        detection.get('status', 'valid')
                    ])

            if self.removed_detections:
                with open(removed_path, 'w') as f:
                    f.write("# Removed Pre-filtered Outliers (for debugging only)\n")
                    f.write(f"# Threshold: >{MAX_NEIGHBOR_CHANGE} mmHg change from previous valid reading\n")
                    f.write(f"# Total removed: {len(self.removed_detections)}\n")
                    f.write("#" + "="*59 + "\n")
                    f.write("Timestamp,Reading,Raw_Reading,Confidence,BBox_Count,Frame_Number,Change\n")
                    for detection in self.removed_detections:
                        f.write(f"{detection['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]},"
                               f"{detection['reading']},"
                               f"{detection.get('raw_reading', detection['reading'])},"
                               f"{detection['confidence']:.3f},"
                               f"{detection['bbox_count']},"
                               f"{detection['frame_number']},"
                               f"{detection.get('change_from_prev', 0)}\n")
                print(f"Removed outliers logged to: {removed_path}")

            print(f"\nResults saved to: {result_path}")
            print(f"CSV results saved to: {csv_path}")
            print(f"Detection records saved: {len(self.detections)}")
            print(f"Annotated frames saved: {self.annotated_frames_saved}")
            print(f"Removed outliers: {self.removed_count} (in separate file, NOT in main dataset)")

        except Exception as e:
            print(f"Error saving results: {e}")


class YOLOGUI:
    def __init__(self, root, detector):
        self.root = root
        self.detector = detector
        self.root.title("YOLO NIBP Detection - Temporal & Outlier Filter")
        self.root.geometry("1000x850")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.create_widgets()
        self.update_preview()

        # ⭐ Register callback for video end
        self.detector.on_video_end_callback = self._on_video_end

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        title_label = ttk.Label(main_frame, text="YOLO NIBP Detection - Filter Transisi & Outlier", font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 10))

        video_frame = ttk.LabelFrame(main_frame, text="Camera Preview", padding="5")
        video_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)

        self.video_label = tk.Label(video_frame, bg='black')
        self.video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.create_placeholder()

        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        button_frame1 = ttk.Frame(control_frame)
        button_frame1.pack(fill=tk.X, pady=5)

        ttk.Label(button_frame1, text="Source:").pack(side=tk.LEFT, padx=(0, 5))
        self.source_var = tk.StringVar(value="Camera 0")
        source_combo = ttk.Combobox(button_frame1, textvariable=self.source_var,
                                    values=["Camera 0", "Camera 1", "Video File"], width=15)
        source_combo.pack(side=tk.LEFT, padx=(0, 10))

        self.load_model_btn = ttk.Button(button_frame1, text="Load Model", command=self.load_model)
        self.load_model_btn.pack(side=tk.LEFT, padx=5)

        self.start_preview_btn = ttk.Button(button_frame1, text="Start Preview", command=self.start_preview, state=tk.DISABLED)
        self.start_preview_btn.pack(side=tk.LEFT, padx=5)

        self.stop_preview_btn = ttk.Button(button_frame1, text="Stop Preview", command=self.stop_preview, state=tk.DISABLED)
        self.stop_preview_btn.pack(side=tk.LEFT, padx=5)

        interval_frame = ttk.LabelFrame(control_frame, text="Detection Interval", padding="5")
        interval_frame.pack(fill=tk.X, pady=5)

        ttk.Label(interval_frame, text="Interval (ms):").pack(side=tk.LEFT, padx=(5, 5))
        self.interval_var = tk.StringVar(value="100")
        interval_entry = ttk.Entry(interval_frame, textvariable=self.interval_var, width=10)
        interval_entry.pack(side=tk.LEFT, padx=(0, 10))

        intervals = [("50ms", 50), ("100ms", 100), ("200ms", 200), ("500ms", 500), ("1s", 1000)]
        for label, ms in intervals:
            btn = ttk.Button(interval_frame, text=label, width=6, command=lambda m=ms: self.set_interval(m))
            btn.pack(side=tk.LEFT, padx=2)

        self.apply_interval_btn = ttk.Button(interval_frame, text="Apply", command=self.apply_interval)
        self.apply_interval_btn.pack(side=tk.LEFT, padx=(10, 5))

        save_frame = ttk.LabelFrame(control_frame, text="Save Options", padding="5")
        save_frame.pack(fill=tk.X, pady=5)

        self.save_original_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(save_frame, text="Save Original Frames", variable=self.save_original_var,
                       command=self.update_save_options).pack(side=tk.LEFT, padx=10)

        self.save_annotated_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(save_frame, text="Save Annotated Frames", variable=self.save_annotated_var,
                       command=self.update_save_options).pack(side=tk.LEFT, padx=10)

        button_frame2 = ttk.Frame(control_frame)
        button_frame2.pack(fill=tk.X, pady=5)

        self.start_record_btn = ttk.Button(button_frame2, text="● Start Recording", command=self.start_recording, state=tk.DISABLED)
        self.start_record_btn.pack(side=tk.LEFT, padx=5)

        self.stop_record_btn = ttk.Button(button_frame2, text="■ Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_record_btn.pack(side=tk.LEFT, padx=5)

        status_frame = ttk.Frame(control_frame)
        status_frame.pack(fill=tk.X, pady=5)

        self.model_status_var = tk.StringVar(value="● Model: Not Loaded")
        ttk.Label(status_frame, textvariable=self.model_status_var, foreground="red", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)

        self.preview_status_var = tk.StringVar(value="● Preview: OFF")
        ttk.Label(status_frame, textvariable=self.preview_status_var, foreground="red", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)

        self.record_status_var = tk.StringVar(value="● Recording: OFF")
        ttk.Label(status_frame, textvariable=self.record_status_var, foreground="red", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)

        self.removed_count_var = tk.StringVar(value="Removed: 0")
        ttk.Label(status_frame, textvariable=self.removed_count_var, font=("Arial", 10), foreground="orange").pack(side=tk.LEFT, padx=5)

        self.saved_count_var = tk.StringVar(value="Saved: 0")
        ttk.Label(status_frame, textvariable=self.saved_count_var, font=("Arial", 10), foreground="green").pack(side=tk.LEFT, padx=5)

        self.fps_var = tk.StringVar(value="FPS: 0")
        ttk.Label(status_frame, textvariable=self.fps_var, font=("Arial", 10, "bold"), foreground="green").pack(side=tk.LEFT, padx=5)

        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="5")
        log_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=6, wrap=tk.WORD, font=("Courier", 9))
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.log_text.insert(tk.END, "Log...\n")

        ttk.Button(main_frame, text="Exit", command=self.on_closing).grid(row=4, column=0, pady=10)

    def create_placeholder(self):
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Camera Preview", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        img = Image.fromarray(placeholder)
        img = img.resize((640, 480), Image.Resampling.LANCZOS)
        self.placeholder_img = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=self.placeholder_img)

    def set_interval(self, ms):
        self.interval_var.set(str(ms))
        self.apply_interval()

    def apply_interval(self):
        try:
            ms = int(self.interval_var.get())
            ms = max(10, min(ms, 10000))
            self.detector.detection_interval = ms / 1000.0
            self.log_message(f"Interval set to {ms}ms")
        except ValueError:
            messagebox.showerror("Error", "Invalid interval")

    def update_save_options(self):
        self.detector.save_original_frames = self.save_original_var.get()
        self.detector.save_annotated_frames = self.save_annotated_var.get()

    def load_model(self):
        self.load_model_btn.config(state=tk.DISABLED)
        self.model_status_var.set("● Model: Loading...")
        thread = threading.Thread(target=self._load_model_thread, daemon=True)
        thread.start()

    def _load_model_thread(self):
        if self.detector.load_model():
            self.root.after(0, self._model_loaded_success)
        else:
            self.root.after(0, self._model_loaded_fail)

    def _model_loaded_success(self):
        self.model_status_var.set(f"● Model: Loaded ({self.detector.device})")
        self.start_preview_btn.config(state=tk.NORMAL)

    def _model_loaded_fail(self):
        self.model_status_var.set("● Model: Failed")
        self.load_model_btn.config(state=tk.NORMAL)
        messagebox.showerror("Error", "Failed to load model")

    def start_preview(self):
        source = self.source_var.get()
        video_path = None
        is_video_file = False

        if source == "Video File":
            video_path = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
            )
            if not video_path:
                return
            self.detector.camera_index = -1
            is_video_file = True
        else:
            camera_idx = int(source.split()[-1])
            self.detector.camera_index = camera_idx
            is_video_file = False

        self.apply_interval()
        self.update_save_options()

        if is_video_file and AUTO_START_RECORDING_FOR_VIDEO:
            self._start_video_with_recording(video_path)
        else:
            if self.detector.start_preview(video_path):
                self.start_preview_btn.config(state=tk.DISABLED)
                self.stop_preview_btn.config(state=tk.NORMAL)
                self.start_record_btn.config(state=tk.NORMAL)
                self.preview_status_var.set("● Preview: ON")
                self.log_message(f"Preview started - FPS: {self.detector.video_fps:.2f}")

    def _start_video_with_recording(self, video_path):
        """
        ⭐ Start video file processing with recording enabled from frame 1.
        """
        self.log_message("📹 Video mode: Starting with recording from frame 1...")

        if not self.detector.model_loaded:
            self.log_message("Loading model...")
            if not self.detector.load_model():
                messagebox.showerror("Error", "Failed to load model")
                return

        if not self.detector.initialize_camera(video_path):
            messagebox.showerror("Error", "Failed to open video file")
            return

        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.detector.recording_timestamp = timestamp_str
        video_filename = f"output_{timestamp_str}.mp4"
        video_out_path = os.path.join(VIDEO_OUTPUT_DIR, video_filename)

        if not self.detector.initialize_video_writer(video_out_path):
            messagebox.showerror("Error", "Failed to initialize video writer")
            return

        # Reset all detection state
        self.detector.detections = []
        self.detector.removed_detections = []
        self.detector.frames_saved = 0
        self.detector.annotated_frames_saved = 0
        self.detector.dataset_frames_saved = 0
        self.detector.previous_combined_string = ""
        self.detector.previous_valid_reading = None
        self.detector.last_raw_reading = None
        self.detector.total_detections_raw = 0
        self.detector.removed_count = 0
        self.detector.temporal_buffer.clear()

        # Activate recording mode
        self.detector.is_recording = True
        self.detector.start_time = datetime.now()

        # Initialize preview state
        self.detector.preview_running = True
        self.detector.is_running = True
        self.detector.video_ended = False
        self.detector.fps_start_time = time.time()
        self.detector.fps_frame_count = 0
        self.detector.last_detection_time = 0
        self.detector.last_detection_frame = None
        self.detector.last_detection_data = None
        self.detector.video_frame_count = 0
        self.detector.last_frame_time = time.time()

        # Start preview thread
        self.detector.preview_thread = threading.Thread(
            target=self.detector._preview_loop, daemon=True
        )
        self.detector.preview_thread.start()

        # Update GUI state
        self.start_preview_btn.config(state=tk.DISABLED)
        self.stop_preview_btn.config(state=tk.NORMAL)
        self.start_record_btn.config(state=tk.DISABLED)
        self.stop_record_btn.config(state=tk.NORMAL)
        self.preview_status_var.set("● Preview: ON")
        self.record_status_var.set("● Recording: ON")

        self.log_message(f"✓ Video FPS: {self.detector.video_fps:.2f}")
        self.log_message(f"✓ Video output: {video_filename}")
        self.log_message("✓ Recording active from frame 1 - ZERO frames lost")
        self.log_message("⏳ Processing video (auto-stop when finished)...")

    def _auto_start_recording(self):
        """Auto-trigger recording after video file preview starts (legacy fallback)."""
        try:
            if self.detector.start_recording():
                self.start_record_btn.config(state=tk.DISABLED)
                self.stop_record_btn.config(state=tk.NORMAL)
                self.record_status_var.set("● Recording: ON")
                self.log_message("✓ Auto-recording started (video file mode)")
            else:
                self.log_message("⚠ Auto-start recording failed")
        except Exception as e:
            self.log_message(f"⚠ Auto-start recording error: {e}")

    def _on_video_end(self):
        """Called from the detector thread when video file ends."""
        self.root.after(0, self._handle_video_end)

    def _handle_video_end(self):
        """Handle video end on the main thread."""
        self.log_message("📹 Video ended - auto-stopping recording...")

        try:
            if self.detector.is_recording and AUTO_STOP_RECORDING_FOR_VIDEO:
                self.detector.stop_recording()
                self.start_record_btn.config(state=tk.NORMAL)
                self.stop_record_btn.config(state=tk.DISABLED)
                self.record_status_var.set("● Recording: OFF")
                self.log_message("✓ Auto-recording stopped (video ended)")

            self.detector.preview_running = False
            self.detector.is_running = False
            if self.detector.cap:
                self.detector.cap.release()
                self.detector.cap = None

            self.start_preview_btn.config(state=tk.NORMAL)
            self.stop_preview_btn.config(state=tk.DISABLED)
            self.start_record_btn.config(state=tk.DISABLED)
            self.stop_record_btn.config(state=tk.DISABLED)
            self.preview_status_var.set("● Preview: OFF")
            self.record_status_var.set("● Recording: OFF")
            self.create_placeholder()

            self.log_message("✅ Video processing complete. Ready for next file.")

        except Exception as e:
            self.log_message(f"⚠ Error during auto-stop: {e}")

    def stop_preview(self):
        self.detector.stop_preview()
        self.start_preview_btn.config(state=tk.NORMAL)
        self.stop_preview_btn.config(state=tk.DISABLED)
        self.start_record_btn.config(state=tk.DISABLED)
        self.stop_record_btn.config(state=tk.DISABLED)
        self.preview_status_var.set("● Preview: OFF")
        self.record_status_var.set("● Recording: OFF")
        self.create_placeholder()

    def start_recording(self):
        self.apply_interval()
        self.update_save_options()
        if self.detector.start_recording():
            self.start_record_btn.config(state=tk.DISABLED)
            self.stop_record_btn.config(state=tk.NORMAL)
            self.record_status_var.set("● Recording: ON")

    def stop_recording(self):
        self.detector.stop_recording()
        self.start_record_btn.config(state=tk.NORMAL)
        self.stop_record_btn.config(state=tk.DISABLED)
        self.record_status_var.set("● Recording: OFF")

    def update_preview(self):
        if self.detector.preview_running:
            frame = self.detector.get_current_frame()
            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.config(image=imgtk)
                self.video_label.image = imgtk

            self.fps_var.set(f"FPS: {self.detector.actual_fps:.1f}")
            self.removed_count_var.set(f"Removed: {self.detector.removed_count}")
            self.saved_count_var.set(f"Saved: {len(self.detector.detections)}")
        self.root.after(30, self.update_preview)

    def log_message(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

    def on_closing(self):
        self.detector.stop_preview()
        self.root.quit()
        self.root.destroy()


def main():
    print("="*60)
    print("YOLO NIBP Detection - ML-Ready Dataset Generator")
    print("="*60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected, using CPU")

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    if not YOLO_AVAILABLE:
        print("Error: ultralytics not installed")
        return

    detector = YOLODetector(
        model_path=MODEL_PATH,
        camera_index=0,
        frame_width=640,
        frame_height=480,
        detection_interval=DETECTION_INTERVAL
    )

    root = tk.Tk()
    app = YOLOGUI(root, detector)
    root.mainloop()
    print("\nApplication closed!")


if __name__ == "__main__":
    main()