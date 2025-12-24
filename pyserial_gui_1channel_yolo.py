import serial
import serial.tools.list_ports
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import queue
import time
import csv
import os
from datetime import datetime
import re
import cv2
import supervision as sv
from ultralytics import YOLO
import torch
import numpy as np
from PIL import Image, ImageTk

class CombinedDataLogger:
    def __init__(self, root):
        self.root = root
        self.root.title("Serial + YOLO Data Acquisition")
        self.root.geometry("1400x800")
        
        # Serial connection variables
        self.serial_port = None
        self.serial_thread = None
        self.data_queue = queue.Queue()
        self.running = False
        
        # YOLO variables
        self.yolo_running = False
        self.yolo_thread = None
        self.video_queue = queue.Queue(maxsize=1)
        self.detection_queue = queue.Queue(maxsize=10)
        
        # Data storage for 3 channels from serial
        self.channel_data = [[], [], []]  # Separate lists for each channel
        self.timestamps = [[], [], []]    # Separate timestamps for each channel
        
        # YOLO detection storage
        self.yolo_detections = []
        self.yolo_timestamps = []
        self.previous_combined_string = ""
        
        # File logging - For all 3 serial channels
        self.csv_file = None
        self.csv_writer = None
        self.txt_file = None
        
        self.max_data_points = 500
        self.channel_counter = 0  # Counter to track which channel we're on for serial
        
        # YOLO model setup
        self.setup_yolo_model()
        
        # GUI variables
        self.create_widgets()
        
        # Auto-detect COM ports
        self.detect_com_ports()
        
    def setup_yolo_model(self):
        """Setup YOLO model for detection"""
        # Check if CUDA is available and use the GPU if it is
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load the YOLOv11 model and move it to the GPU (if available)
        self.model = YOLO('best_tensi_bppump_prosim_17122025.pt').to(self.device)
        
        # Get class names from the model
        self.class_names = self.model.names
        
        # Initialize the webcam
        self.cap = cv2.VideoCapture(0)
        
        # Check if webcam is opened successfully
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            self.yolo_available = False
        else:
            self.yolo_available = True
        
        # Define the confidence threshold
        self.confidence_threshold = 0.5
        
        # Create BoxAnnotator for YOLO
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,  # Box thickness
            color_lookup=sv.ColorLookup.CLASS  # Color by class
        )
        
        # Create directory for saving results if it doesn't exist
        self.output_dir = "detection_results"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)  # More weight for YOLO
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Serial + YOLO Data Acquisition System", 
                                font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # Connection frame
        conn_frame = ttk.LabelFrame(main_frame, text="System Controls", padding="10")
        conn_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # COM Port selection
        ttk.Label(conn_frame, text="COM Port:").grid(row=0, column=0, padx=(0, 5))
        self.com_port_var = tk.StringVar(value="COM3")
        self.com_port_combo = ttk.Combobox(conn_frame, textvariable=self.com_port_var, width=10)
        self.com_port_combo.grid(row=0, column=1, padx=(0, 10))
        
        # Baud rate selection
        ttk.Label(conn_frame, text="Baud Rate:").grid(row=0, column=2, padx=(0, 5))
        self.baud_rate_var = tk.StringVar(value="9600")
        baud_rates = ["300", "600", "1200", "2400", "4800", "9600", "14400", 
                      "19200", "38400", "57600", "115200", "230400", "460800", "921600"]
        self.baud_rate_combo = ttk.Combobox(conn_frame, textvariable=self.baud_rate_var, 
                                            values=baud_rates, width=10)
        self.baud_rate_combo.grid(row=0, column=3, padx=(0, 10))
        
        # Refresh ports button
        refresh_btn = ttk.Button(conn_frame, text="Refresh Ports", command=self.detect_com_ports)
        refresh_btn.grid(row=0, column=4, padx=(10, 5))
        
        # Connect/Disconnect buttons for Serial
        self.connect_btn = ttk.Button(conn_frame, text="Connect Serial", command=self.connect_serial)
        self.connect_btn.grid(row=0, column=5, padx=5)
        
        self.disconnect_btn = ttk.Button(conn_frame, text="Disconnect Serial", 
                                         command=self.disconnect_serial, state=tk.DISABLED)
        self.disconnect_btn.grid(row=0, column=6, padx=5)
        
        # YOLO Start/Stop buttons
        self.start_yolo_btn = ttk.Button(conn_frame, text="Start YOLO", command=self.start_yolo)
        self.start_yolo_btn.grid(row=0, column=7, padx=5)
        
        self.stop_yolo_btn = ttk.Button(conn_frame, text="Stop YOLO", 
                                        command=self.stop_yolo, state=tk.DISABLED)
        self.stop_yolo_btn.grid(row=0, column=8, padx=5)
        
        # Status indicators
        self.serial_status_label = ttk.Label(conn_frame, text="Serial: Disconnected", foreground="red")
        self.serial_status_label.grid(row=0, column=9, padx=(20, 5))
        
        self.yolo_status_label = ttk.Label(conn_frame, text="YOLO: Stopped", foreground="red")
        self.yolo_status_label.grid(row=0, column=10, padx=(0, 0))
        
        # Data logging frame
        log_frame = ttk.LabelFrame(main_frame, text="Data Logging (3 Serial Channels)", padding="10")
        log_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # File name entry
        ttk.Label(log_frame, text="File Name:").grid(row=0, column=0, padx=(0, 5))
        self.file_name_var = tk.StringVar(value=f"serial_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        file_name_entry = ttk.Entry(log_frame, textvariable=self.file_name_var, width=30)
        file_name_entry.grid(row=0, column=1, padx=(0, 10))
        
        # Start/Stop logging buttons
        self.start_log_btn = ttk.Button(log_frame, text="Start Logging Serial", command=self.start_logging)
        self.start_log_btn.grid(row=0, column=2, padx=5)
        
        self.stop_log_btn = ttk.Button(log_frame, text="Stop Logging", 
                                       command=self.stop_logging, state=tk.DISABLED)
        self.stop_log_btn.grid(row=0, column=3, padx=5)
        
        # Logging status
        self.log_status_var = tk.StringVar(value="Not Logging")
        log_status_label = ttk.Label(log_frame, textvariable=self.log_status_var, foreground="blue")
        log_status_label.grid(row=0, column=4, padx=(20, 0))
        
        # Channel display frames (3 columns)
        channel_frames = []
        channel_names = ['Channel 1 - Serial', 'Channel 2 - YOLO Video & Detection', 'Channel 3 - Serial']
        channel_colors = ['blue', 'red', 'green']
        
        for i in range(3):
            # Create frame for each channel
            chan_frame = ttk.LabelFrame(main_frame, text=channel_names[i], padding="5")
            
            if i == 1:  # YOLO channel gets more space (2 columns)
                chan_frame.grid(row=3, column=i, columnspan=1, sticky=(tk.W, tk.E, tk.N, tk.S), 
                               padx=(5, 5), pady=(0, 10))
                main_frame.columnconfigure(i, weight=2)  # More weight for YOLO
            else:
                chan_frame.grid(row=3, column=i, sticky=(tk.W, tk.E, tk.N, tk.S), 
                               padx=(5, 5), pady=(0, 10))
                main_frame.columnconfigure(i, weight=1)
            
            main_frame.rowconfigure(3, weight=1)
            
            if i == 1:  # YOLO channel - video display
                # Video frame display
                self.video_label = ttk.Label(chan_frame)
                self.video_label.pack(expand=True, fill="both")
                
                # Current detection display below video
                detection_frame = ttk.Frame(chan_frame)
                detection_frame.pack(fill="x", pady=(5, 0))
                
                current_var = tk.StringVar(value="No Detection")
                current_label = ttk.Label(detection_frame, textvariable=current_var,
                                         font=("Arial", 16, "bold"),
                                         foreground="red")
                current_label.pack(pady=(5, 5))
                
                # Create text widget for detection history
                text_widget = scrolledtext.ScrolledText(detection_frame, width=40, height=8)
                text_widget.pack(fill=tk.BOTH, expand=True)
                text_widget.insert(tk.END, f"YOLO Detection History:\n")
                text_widget.insert(tk.END, "=" * 40 + "\n")
                
                # Store references
                channel_frames.append({
                    'frame': chan_frame,
                    'current_var': current_var,
                    'text_widget': text_widget,
                    'is_yolo': True
                })
                
            else:  # Serial channels (1 and 3)
                # Current value display (large font)
                current_var = tk.StringVar(value="No Data")
                current_label = ttk.Label(chan_frame, textvariable=current_var,
                                         font=("Arial", 20, "bold"),
                                         foreground=channel_colors[i])
                current_label.pack(pady=(5, 10))
                
                # Create text widget for channel data
                text_widget = scrolledtext.ScrolledText(chan_frame, width=35, height=20)
                text_widget.pack(fill=tk.BOTH, expand=True)
                text_widget.insert(tk.END, f"{channel_names[i]} History:\n")
                text_widget.insert(tk.END, "=" * 40 + "\n")
                
                # Store references
                channel_frames.append({
                    'frame': chan_frame,
                    'current_var': current_var,
                    'text_widget': text_widget,
                    'is_yolo': False
                })
        
        # Store channel frames
        self.channel_frames = channel_frames
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(main_frame, text="Statistics", padding="10")
        stats_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Statistics for each channel
        self.channel_stats = []
        for i in range(3):
            stat_frame = ttk.Frame(stats_frame)
            stat_frame.grid(row=0, column=i, padx=(10, 10), sticky=(tk.W, tk.E))
            
            # Channel label
            chan_label = ttk.Label(stat_frame, text=f"{channel_names[i]} Statistics:", 
                                  font=("Arial", 10, "bold"))
            chan_label.grid(row=0, column=0, columnspan=2, pady=(0, 5))
            
            # Data points
            points_var = tk.StringVar(value="Points: 0")
            ttk.Label(stat_frame, textvariable=points_var).grid(row=1, column=0, sticky=tk.W)
            
            # Average value
            avg_var = tk.StringVar(value="Average: N/A")
            ttk.Label(stat_frame, textvariable=avg_var).grid(row=2, column=0, sticky=tk.W)
            
            # Min value
            min_var = tk.StringVar(value="Min: N/A")
            ttk.Label(stat_frame, textvariable=min_var).grid(row=3, column=0, sticky=tk.W)
            
            # Max value
            max_var = tk.StringVar(value="Max: N/A")
            ttk.Label(stat_frame, textvariable=max_var).grid(row=4, column=0, sticky=tk.W)
            
            # Latest value
            latest_var = tk.StringVar(value="Latest: N/A")
            ttk.Label(stat_frame, textvariable=latest_var).grid(row=5, column=0, sticky=tk.W)
            
            self.channel_stats.append({
                'points_var': points_var,
                'avg_var': avg_var,
                'min_var': min_var,
                'max_var': max_var,
                'latest_var': latest_var
            })
        
        # Control buttons frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=5, column=0, columnspan=3, pady=(10, 0))
        
        # Clear all data button
        clear_btn = ttk.Button(control_frame, text="Clear All Data", command=self.clear_data)
        clear_btn.grid(row=0, column=0, padx=5)
        
        # Save Serial data button
        save_serial_btn = ttk.Button(control_frame, text="Save Serial Data", command=self.save_serial_data)
        save_serial_btn.grid(row=0, column=1, padx=5)
        
        # Save YOLO detections button
        save_yolo_btn = ttk.Button(control_frame, text="Save YOLO Detections", command=self.save_yolo_detections)
        save_yolo_btn.grid(row=0, column=2, padx=5)
        
        # Export all data button
        export_all_btn = ttk.Button(control_frame, text="Export All Data", command=self.export_all_data)
        export_all_btn.grid(row=0, column=3, padx=5)
        
        # Exit button
        exit_btn = ttk.Button(control_frame, text="Exit", command=self.on_closing)
        exit_btn.grid(row=0, column=4, padx=5)
        
        # Connection time display
        self.connection_time_var = tk.StringVar(value="Connection Time: 00:00:00")
        ttk.Label(control_frame, textvariable=self.connection_time_var).grid(row=0, column=5, padx=(20, 5))
        
        # Start time tracking
        self.start_time = None
        self.update_stats()
    
    def detect_com_ports(self):
        """Detect available COM ports and update the dropdown"""
        ports = serial.tools.list_ports.comports()
        port_names = [port.device for port in ports]
        self.com_port_combo['values'] = port_names
        
        if port_names and "COM3" in port_names:
            self.com_port_var.set("COM3")
        elif port_names:
            self.com_port_var.set(port_names[0])
        else:
            self.com_port_var.set("")
    
    def connect_serial(self):
        """Connect to the selected serial port"""
        port = self.com_port_var.get()
        baud = int(self.baud_rate_var.get())
        
        if not port:
            messagebox.showerror("Error", "Please select a COM port")
            return
        
        try:
            self.serial_port = serial.Serial(
                port=port,
                baudrate=baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1
            )
            
            # Reset channel counter
            self.channel_counter = 0
            
            # Start serial reading thread
            self.running = True
            self.serial_thread = threading.Thread(target=self.read_serial_data, daemon=True)
            self.serial_thread.start()
            
            # Update UI
            self.connect_btn.config(state=tk.DISABLED)
            self.disconnect_btn.config(state=tk.NORMAL)
            self.serial_status_label.config(text="Serial: Connected", foreground="green")
            self.start_time = time.time()
            
            messagebox.showinfo("Success", f"Connected to {port} at {baud} baud")
            
        except serial.SerialException as e:
            messagebox.showerror("Connection Error", f"Failed to connect to {port}:\n{str(e)}")
    
    def disconnect_serial(self):
        """Disconnect from the serial port"""
        self.running = False
        
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        
        # Update UI
        self.connect_btn.config(state=tk.NORMAL)
        self.disconnect_btn.config(state=tk.DISABLED)
        self.serial_status_label.config(text="Serial: Disconnected", foreground="red")
        
        # Stop logging if active
        if self.csv_file:
            self.stop_logging()
        
        messagebox.showinfo("Disconnected", "Serial connection closed")
    
    def start_yolo(self):
        """Start YOLO detection"""
        if not self.yolo_available:
            messagebox.showerror("YOLO Error", "Webcam not available for YOLO detection")
            return
        
        self.yolo_running = True
        self.yolo_thread = threading.Thread(target=self.yolo_detection_thread, daemon=True)
        self.yolo_thread.start()
        
        # Update UI
        self.start_yolo_btn.config(state=tk.DISABLED)
        self.stop_yolo_btn.config(state=tk.NORMAL)
        self.yolo_status_label.config(text="YOLO: Running", foreground="green")
        
        messagebox.showinfo("YOLO Started", "YOLO detection started")
    
    def stop_yolo(self):
        """Stop YOLO detection"""
        self.yolo_running = False
        
        if self.yolo_thread:
            self.yolo_thread.join(timeout=2)
        
        # Update UI
        self.start_yolo_btn.config(state=tk.NORMAL)
        self.stop_yolo_btn.config(state=tk.DISABLED)
        self.yolo_status_label.config(text="YOLO: Stopped", foreground="red")
        
        messagebox.showinfo("YOLO Stopped", "YOLO detection stopped")
    
    # YOLO Helper Functions
    def filter_overlapping_detections(self, detection_info, overlap_threshold=0.5):
        """Filter out overlapping detections of the same class."""
        if len(detection_info) <= 1:
            return detection_info
        
        # Sort by confidence (highest first)
        detection_info.sort(key=lambda x: x['confidence'], reverse=True)
        
        filtered_detections = []
        used_indices = set()
        
        for i in range(len(detection_info)):
            if i in used_indices:
                continue
                
            current_det = detection_info[i]
            filtered_detections.append(current_det)
            used_indices.add(i)
            
            # Check for overlapping detections with the same class
            for j in range(i + 1, len(detection_info)):
                if j in used_indices:
                    continue
                    
                other_det = detection_info[j]
                
                # Only check overlap for same class
                if current_det['class_name'] != other_det['class_name']:
                    continue
                
                # Calculate IoU (Intersection over Union)
                x1_i = max(current_det['x1'], other_det['x1'])
                y1_i = max(current_det['y1'], other_det['y1'])
                x2_i = min(current_det['x2'], other_det['x2'])
                y2_i = min(current_det['y2'], other_det['y2'])
                
                # Calculate intersection area
                intersection_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
                
                # Calculate union area
                area_current = (current_det['x2'] - current_det['x1']) * (current_det['y2'] - current_det['y1'])
                area_other = (other_det['x2'] - other_det['x1']) * (other_det['y2'] - other_det['y1'])
                union_area = area_current + area_other - intersection_area
                
                # Calculate IoU
                iou = intersection_area / union_area if union_area > 0 else 0
                
                # If overlap is significant, mark as used (skip this detection)
                if iou > overlap_threshold:
                    used_indices.add(j)
        
        # Sort filtered detections back by x position
        filtered_detections.sort(key=lambda x: x['center_x'])
        return filtered_detections
    
    def is_reasonable_reading(self, combined_number, max_digits=3, max_value=300):
        """Check if the combined number is reasonable."""
        # Check length
        if len(combined_number) > max_digits:
            return False
        
        # Check if all characters are digits
        if not combined_number.isdigit():
            return False
        
        # Additional checks can be added here
        # Check if value is within range
        try:
            value = int(combined_number)
            if value > max_value:
                return False
        except ValueError:
            return False
        
        return True
    
    def yolo_detection_thread(self):
        """Thread function for YOLO detection"""
        while self.yolo_running:
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture image from webcam.")
                break
            
            frame_timestamp = datetime.now()
            
            # Perform YOLO object detection on the current frame
            results = self.model(frame)[0]
            
            # Convert YOLO detections to Supervision format
            detections = sv.Detections.from_ultralytics(results)
            
            # Filter detections by confidence threshold
            detections = detections[detections.confidence > self.confidence_threshold]
            
            if len(detections) > 0:
                # Combine detections from left to right
                detection_info = []
                for i in range(len(detections.class_id)):
                    x1, y1, x2, y2 = detections.xyxy[i]
                    center_x = (x1 + x2) / 2
                    class_index = int(detections.class_id[i])
                    class_name = self.class_names[class_index]
                    confidence = float(detections.confidence[i])
                    
                    detection_info.append({
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2),
                        'center_x': float(center_x),
                        'class_id': class_index,
                        'class_name': class_name,
                        'confidence': confidence
                    })
                
                # Filter out overlapping detections
                detection_info = self.filter_overlapping_detections(detection_info, overlap_threshold=0.3)
                
                # Sort detections from left to right based on center_x
                detection_info.sort(key=lambda x: x['center_x'])
                
                # Create combined strings
                class_names_only = [det['class_name'] for det in detection_info]
                combined_confidences = [det['confidence'] for det in detection_info]
                
                # Create the combined number string (e.g., "169")
                combined_number_string = "".join(class_names_only)
                
                # Check if the reading is reasonable
                if self.is_reasonable_reading(combined_number_string, max_digits=3, max_value=300):
                    # Check if combined string has changed
                    string_changed = combined_number_string != self.previous_combined_string
                    self.previous_combined_string = combined_number_string
                    
                    if string_changed:
                        # Create the detailed string with confidences
                        combined_detail_string = " | ".join([
                            f"{cls} ({conf:.2f})" 
                            for cls, conf in zip(class_names_only, combined_confidences)
                        ])
                        
                        # Store YOLO detection
                        yolo_data = {
                            'timestamp': frame_timestamp,
                            'combined_string': combined_number_string,
                            'details': combined_detail_string,
                            'num_detections': len(detection_info),
                            'detection_info': detection_info
                        }
                        
                        self.yolo_detections.append(yolo_data)
                        self.yolo_timestamps.append(time.time())
                        
                        # Keep buffer at max size
                        if len(self.yolo_detections) > self.max_data_points:
                            self.yolo_detections.pop(0)
                            self.yolo_timestamps.pop(0)
                        
                        # Send to GUI
                        self.detection_queue.put({
                            'timestamp': frame_timestamp,
                            'reading': combined_number_string,
                            'details': combined_detail_string
                        })
                    
                    # Annotate frame
                    annotated_frame = self.box_annotator.annotate(
                        scene=frame.copy(),
                        detections=detections
                    )
                    
                    # Add reading text to frame
                    y_offset = 30
                    cv2.putText(
                        annotated_frame,
                        f"Reading: {combined_number_string}",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
                    cv2.putText(
                        annotated_frame,
                        f"Time: {frame_timestamp.strftime('%H:%M:%S.%f')[:-3]}",
                        (10, y_offset + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        2
                    )
                    
                    frame_to_show = annotated_frame
                else:
                    frame_to_show = frame
            else:
                frame_to_show = frame
            
            # Convert frame for display
            frame_to_show = cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB)
            frame_to_show = cv2.resize(frame_to_show, (640, 480))
            
            # Send frame to GUI thread
            try:
                self.video_queue.put(frame_to_show, timeout=0.1)
            except queue.Full:
                pass
            
            # Small delay
            cv2.waitKey(1)
    
    def extract_numeric_value(self, raw_data):
        """Extract numeric value from raw data string"""
        try:
            # Try to convert directly to float
            return float(raw_data.strip())
        except ValueError:
            # If that fails, try to find a number in the string
            match = re.search(r'[-+]?\d*\.\d+|\d+', raw_data)
            if match:
                return float(match.group())
            else:
                return None
    
    def read_serial_data(self):
        """Read data from serial port in a separate thread"""
        while self.running and self.serial_port and self.serial_port.is_open:
            try:
                if self.serial_port.in_waiting:
                    # Read line from serial
                    line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
                    
                    if line:
                        # Extract numeric value
                        numeric_value = self.extract_numeric_value(line)
                        
                        if numeric_value is not None:
                            # Add timestamp
                            timestamp = time.time()
                            
                            # Create data entry
                            data_entry = {
                                'timestamp': timestamp,
                                'datetime': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f'),
                                'raw_data': line,
                                'numeric_value': numeric_value,
                                'channel': self.channel_counter  # Current channel (0, 1, 2)
                            }
                            
                            # Put data in queue for GUI thread
                            self.data_queue.put(data_entry)
                            
                            # Print to terminal
                            print(f"CH{self.channel_counter + 1}: {data_entry['datetime']}: {numeric_value}")
                            
                            # Increment channel counter (0, 1, 2, then back to 0)
                            self.channel_counter = (self.channel_counter + 1) % 3
                        
            except (serial.SerialException, OSError):
                break
            except Exception as e:
                print(f"Error reading serial data: {e}")
        
        # If we get here, the connection was lost
        self.root.after(0, self.disconnect_serial)
    
    def process_data(self):
        """Process data from all queues in the main thread"""
        # Process serial data
        try:
            while not self.data_queue.empty():
                data_entry = self.data_queue.get_nowait()
                
                channel_idx = data_entry['channel']
                numeric_value = data_entry['numeric_value']
                
                # Skip channel 1 (index 0) for YOLO, use serial for CH1 and CH3 only
                # Map: Serial CH0 -> Display CH1, Serial CH1 -> Display CH3, Serial CH2 -> Not used
                display_channel = channel_idx * 2  # 0->0, 1->2, 2->4 (skip 1 for YOLO)
                if display_channel >= 3:  # Skip if out of range
                    continue
                
                # Add to appropriate channel data buffer
                self.channel_data[display_channel].append(numeric_value)
                self.timestamps[display_channel].append(data_entry['timestamp'])
                
                # Keep buffer at max size
                if len(self.channel_data[display_channel]) > self.max_data_points:
                    self.channel_data[display_channel].pop(0)
                    self.timestamps[display_channel].pop(0)
                
                # Log to files if logging is active
                if self.csv_file:
                    self.log_serial_data(data_entry)
                
                # Update channel display (skip channel 1 which is for YOLO)
                if display_channel != 1:  # Skip YOLO channel
                    self.update_serial_display(display_channel, data_entry)
                    
                    # Update statistics
                    self.update_channel_stats(display_channel)
                
        except queue.Empty:
            pass
        
        # Process YOLO detection data
        try:
            while not self.detection_queue.empty():
                detection_data = self.detection_queue.get_nowait()
                
                # Update YOLO channel display (Channel 1)
                self.update_yolo_display(detection_data)
                
                # Update YOLO statistics
                self.update_yolo_stats()
                
        except queue.Empty:
            pass
        
        # Update video feed
        self.update_video_feed()
        
        # Schedule next update
        self.root.after(100, self.process_data)
    
    def update_serial_display(self, channel_idx, data_entry):
        """Update display for serial channels (CH1 and CH3)"""
        # Skip channel 1 (YOLO)
        if channel_idx == 1:
            return
            
        # Update current value display
        self.channel_frames[channel_idx]['current_var'].set(f"{data_entry['numeric_value']:.4f}")
        
        # Get text widget for this channel
        text_widget = self.channel_frames[channel_idx]['text_widget']
        
        # Add new data line
        display_line = f"{data_entry['datetime']}: {data_entry['numeric_value']:.4f}\n"
        text_widget.insert(tk.END, display_line)
        text_widget.see(tk.END)
        
        # Keep only last 30 lines in text widget
        lines = text_widget.get(1.0, tk.END).split('\n')
        if len(lines) > 32:  # 30 data lines + 2 header lines
            text_widget.delete(1.0, tk.END)
            # Keep header and last 30 data lines
            header = lines[0] + "\n" + lines[1] + "\n"
            data_lines = lines[2:][-30:]
            text_widget.insert(tk.END, header + "\n".join(data_lines) + "\n")
    
    def update_yolo_display(self, detection_data):
        """Update display for YOLO channel"""
        # Update current value display
        self.channel_frames[1]['current_var'].set(f"{detection_data['reading']}")
        
        # Get text widget for this channel
        text_widget = self.channel_frames[1]['text_widget']
        
        # Add new detection line
        timestamp_str = detection_data['timestamp'].strftime('%H:%M:%S.%f')[:-3]
        display_line = f"{timestamp_str}: {detection_data['reading']} ({detection_data['details']})\n"
        text_widget.insert(tk.END, display_line)
        text_widget.see(tk.END)
        
        # Keep only last 20 lines in text widget
        lines = text_widget.get(1.0, tk.END).split('\n')
        if len(lines) > 22:  # 20 data lines + 2 header lines
            text_widget.delete(1.0, tk.END)
            # Keep header and last 20 data lines
            header = lines[0] + "\n" + lines[1] + "\n"
            data_lines = lines[2:][-20:]
            text_widget.insert(tk.END, header + "\n".join(data_lines) + "\n")
    
    def update_video_feed(self):
        """Update the video feed from YOLO"""
        try:
            frame = self.video_queue.get_nowait()
            
            # Convert numpy array to PhotoImage
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update label
            self.video_label.config(image=imgtk)
            self.video_label.image = imgtk
            
        except queue.Empty:
            pass
    
    def update_channel_stats(self, channel_idx):
        """Update statistics for serial channels"""
        if self.channel_data[channel_idx]:
            data = self.channel_data[channel_idx]
            
            # Calculate statistics
            avg_value = sum(data) / len(data)
            min_value = min(data)
            max_value = max(data)
            latest_value = data[-1]
            
            # Update display variables
            stats = self.channel_stats[channel_idx]
            stats['points_var'].set(f"Points: {len(data)}")
            stats['avg_var'].set(f"Average: {avg_value:.4f}")
            stats['min_var'].set(f"Min: {min_value:.4f}")
            stats['max_var'].set(f"Max: {max_value:.4f}")
            stats['latest_var'].set(f"Latest: {latest_value:.4f}")
    
    def update_yolo_stats(self):
        """Update statistics for YOLO detections"""
        if self.yolo_detections:
            # Update detection count
            self.channel_stats[1]['points_var'].set(f"Detections: {len(self.yolo_detections)}")
            
            # Count unique readings
            unique_readings = set()
            for detection in self.yolo_detections:
                unique_readings.add(detection['combined_string'])
            
            # Use avg_var for unique readings count
            self.channel_stats[1]['avg_var'].set(f"Unique Readings: {len(unique_readings)}")
            
            # Update latest reading
            latest_reading = self.yolo_detections[-1]['combined_string']
            self.channel_stats[1]['latest_var'].set(f"Latest: {latest_reading}")
            
            # Update last detection time
            last_time = self.yolo_detections[-1]['timestamp'].strftime('%H:%M:%S')
            self.channel_stats[1]['min_var'].set(f"Last Detection: {last_time}")
    
    def start_logging(self):
        """Start logging Serial data to CSV and TXT files"""
        if not self.serial_port or not self.serial_port.is_open:
            messagebox.showwarning("Not Connected", "Please connect to a serial port first")
            return
        
        file_name = self.file_name_var.get()
        if not file_name:
            messagebox.showwarning("File Name Required", "Please enter a file name")
            return
        
        try:
            # Create CSV file
            csv_file_path = f"{file_name}.csv"
            self.csv_file = open(csv_file_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['Timestamp', 'DateTime', 'Channel', 'Numeric_Value', 'Raw_Data'])
            
            # Create TXT file
            txt_file_path = f"{file_name}.txt"
            self.txt_file = open(txt_file_path, 'w')
            self.txt_file.write("3-Channel Serial Data Log\n")
            self.txt_file.write("="*50 + "\n")
            self.txt_file.write("Format: Timestamp, Channel, Value, Raw_Data\n")
            self.txt_file.write("="*50 + "\n")
            
            # Update UI
            self.start_log_btn.config(state=tk.DISABLED)
            self.stop_log_btn.config(state=tk.NORMAL)
            self.log_status_var.set(f"Logging Serial to {file_name}")
            
            success_msg = f"Started logging Serial data to:\n{csv_file_path}\n{txt_file_path}"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [SUCCESS] {success_msg}")
            messagebox.showinfo("Logging Started", success_msg)
            
        except Exception as e:
            error_msg = f"Failed to start logging: {str(e)}"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [ERROR] {error_msg}")
            messagebox.showerror("Logging Error", error_msg)
    
    def stop_logging(self):
        """Stop logging data to files"""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [INFO] CSV file closed")
        
        if self.txt_file:
            self.txt_file.close()
            self.txt_file = None
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [INFO] TXT file closed")
        
        # Update UI
        self.start_log_btn.config(state=tk.NORMAL)
        self.stop_log_btn.config(state=tk.DISABLED)
        self.log_status_var.set("Not Logging")
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [INFO] Serial logging stopped")
        messagebox.showinfo("Logging Stopped", "Serial data logging has been stopped")
    
    def log_serial_data(self, data_entry):
        """Log Serial data to both CSV and TXT files"""
        if self.csv_writer:
            self.csv_writer.writerow([
                data_entry['timestamp'], 
                data_entry['datetime'], 
                f"CH{data_entry['channel'] + 1}",
                data_entry['numeric_value'],
                data_entry['raw_data']
            ])
        
        if self.txt_file:
            self.txt_file.write(f"{data_entry['datetime']}, CH{data_entry['channel'] + 1}, {data_entry['numeric_value']:.4f}, {data_entry['raw_data']}\n")
    
    def clear_data(self):
        """Clear all data buffers and displays"""
        # Clear serial data
        for i in range(3):
            self.channel_data[i].clear()
            self.timestamps[i].clear()
            
            # Clear text widgets for serial channels
            if i != 1:  # Skip YOLO channel
                text_widget = self.channel_frames[i]['text_widget']
                text_widget.delete(1.0, tk.END)
                if i == 0:
                    text_widget.insert(tk.END, "Channel 1 (Serial) History:\n")
                else:
                    text_widget.insert(tk.END, "Channel 3 (Serial) History:\n")
                text_widget.insert(tk.END, "=" * 40 + "\n")
                self.channel_frames[i]['current_var'].set("No Data")
        
        # Clear YOLO data
        self.yolo_detections.clear()
        self.yolo_timestamps.clear()
        self.previous_combined_string = ""
        
        # Clear YOLO display
        if self.channel_frames[1]['text_widget']:
            text_widget = self.channel_frames[1]['text_widget']
            text_widget.delete(1.0, tk.END)
            text_widget.insert(tk.END, "YOLO Detection History:\n")
            text_widget.insert(tk.END, "=" * 40 + "\n")
        self.channel_frames[1]['current_var'].set("No Detection")
        
        # Reset all statistics
        for i in range(3):
            stats = self.channel_stats[i]
            stats['points_var'].set("Points: 0")
            stats['avg_var'].set("Average: N/A")
            stats['min_var'].set("Min: N/A")
            stats['max_var'].set("Max: N/A")
            stats['latest_var'].set("Latest: N/A")
        
        messagebox.showinfo("Data Cleared", "All data has been cleared")
    
    def save_serial_data(self):
        """Save only Serial data to a file"""
        # Check if any serial data exists
        has_serial_data = any(self.channel_data[0]) or any(self.channel_data[2])
        if not has_serial_data:
            messagebox.showwarning("No Data", "No Serial data to save")
            return
        
        file_name = f"serial_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        try:
            with open(file_name, 'w') as f:
                f.write("Serial Data Export (Channels 1 & 3)\n")
                f.write("="*60 + "\n")
                f.write(f"Exported at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                # Write statistics for each serial channel
                for i in [0, 2]:  # Only CH1 and CH3
                    if self.channel_data[i]:
                        data = self.channel_data[i]
                        f.write(f"\nChannel {i+1} Statistics:\n")
                        f.write(f"  Data Points: {len(data)}\n")
                        f.write(f"  Average: {sum(data)/len(data):.4f}\n")
                        f.write(f"  Minimum: {min(data):.4f}\n")
                        f.write(f"  Maximum: {max(data):.4f}\n")
                
                f.write("\n" + "="*60 + "\n")
                f.write("Timestamp, Channel1, Channel3\n")
                f.write("="*60 + "\n")
                
                # Find the maximum length among serial channels
                max_len = max(len(self.channel_data[0]), len(self.channel_data[2]))
                
                # Write data in rows
                for j in range(max_len):
                    timestamp_str = ""
                    values = ["", ""]
                    
                    # Get CH1 data
                    if j < len(self.timestamps[0]):
                        timestamp_str = datetime.fromtimestamp(self.timestamps[0][j]).strftime('%Y-%m-%d %H:%M:%S.%f')
                        values[0] = f"{self.channel_data[0][j]:.4f}"
                    
                    # Get CH3 data
                    if j < len(self.timestamps[2]):
                        if not timestamp_str:  # Use CH3 timestamp if no CH1 timestamp
                            timestamp_str = datetime.fromtimestamp(self.timestamps[2][j]).strftime('%Y-%m-%d %H:%M:%S.%f')
                        values[1] = f"{self.channel_data[2][j]:.4f}"
                    
                    if timestamp_str:
                        f.write(f"{timestamp_str}, {values[0]}, {values[1]}\n")
            
            messagebox.showinfo("Data Saved", f"Serial data saved to {file_name}")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save data:\n{str(e)}")
    
    def save_yolo_detections(self):
        """Save YOLO detections to a file"""
        if not self.yolo_detections:
            messagebox.showwarning("No Data", "No YOLO detections to save")
            return
        
        file_name = f"yolo_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        try:
            with open(file_name, 'w') as f:
                f.write("YOLO Detection Data Export\n")
                f.write("="*60 + "\n")
                f.write(f"Exported at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Detection Events: {len(self.yolo_detections)}\n")
                
                # Write unique readings
                unique_readings = set()
                for detection in self.yolo_detections:
                    unique_readings.add(detection['combined_string'])
                
                f.write(f"Unique Readings: {len(unique_readings)}\n")
                f.write("="*60 + "\n\n")
                f.write("Detection History:\n")
                f.write("="*60 + "\n")
                f.write("Timestamp, Reading, Details\n")
                f.write("="*60 + "\n")
                
                for detection in self.yolo_detections:
                    timestamp_str = detection['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')
                    f.write(f"{timestamp_str}, {detection['combined_string']}, {detection['details']}\n")
            
            messagebox.showinfo("Data Saved", f"YOLO detections saved to {file_name}")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save data:\n{str(e)}")
    
    def export_all_data(self):
        """Export all data to CSV format"""
        # Check if any data exists
        has_data = any(any(self.channel_data[i]) for i in [0, 2]) or bool(self.yolo_detections)
        if not has_data:
            messagebox.showwarning("No Data", "No data to export")
            return
        
        file_name = f"all_data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            with open(file_name, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Channel1_Serial', 'Channel3_Serial', 'YOLO_Reading', 'YOLO_Details'])
                
                # Find the maximum length among all data sources
                max_len = max(len(self.channel_data[0]), len(self.channel_data[2]), len(self.yolo_detections))
                
                # Write data in rows
                for i in range(max_len):
                    timestamp_str = ""
                    ch1_value = ""
                    ch3_value = ""
                    yolo_reading = ""
                    yolo_details = ""
                    
                    # Get CH1 data
                    if i < len(self.channel_data[0]):
                        timestamp_str = datetime.fromtimestamp(self.timestamps[0][i]).strftime('%Y-%m-%d %H:%M:%S.%f')
                        ch1_value = f"{self.channel_data[0][i]:.4f}"
                    
                    # Get CH3 data
                    if i < len(self.channel_data[2]):
                        if not timestamp_str:  # Use CH3 timestamp if no CH1 timestamp
                            timestamp_str = datetime.fromtimestamp(self.timestamps[2][i]).strftime('%Y-%m-%d %H:%M:%S.%f')
                        ch3_value = f"{self.channel_data[2][i]:.4f}"
                    
                    # Get YOLO data if available
                    if i < len(self.yolo_detections):
                        if not timestamp_str:  # Use YOLO timestamp if no serial timestamp
                            timestamp_str = self.yolo_detections[i]['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')
                        yolo_reading = self.yolo_detections[i]['combined_string']
                        yolo_details = self.yolo_detections[i]['details']
                    
                    # Write row if we have at least one value
                    if timestamp_str or ch1_value or ch3_value or yolo_reading:
                        writer.writerow([timestamp_str, ch1_value, ch3_value, yolo_reading, yolo_details])
            
            messagebox.showinfo("Export Complete", f"All data exported to {file_name}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data:\n{str(e)}")
    
    def update_stats(self):
        """Update connection time display"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            self.connection_time_var.set(f"Connection Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # Schedule next update
        self.root.after(1000, self.update_stats)
    
    def on_closing(self):
        """Handle application closing"""
        self.running = False
        self.yolo_running = False
        
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        
        if self.cap and self.cap.isOpened():
            self.cap.release()
        
        if self.csv_file:
            self.csv_file.close()
        
        if self.txt_file:
            self.txt_file.close()
        
        cv2.destroyAllWindows()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = CombinedDataLogger(root)
    
    # Start processing data
    root.after(100, app.process_data)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    root.mainloop()

if __name__ == "__main__":
    main()
