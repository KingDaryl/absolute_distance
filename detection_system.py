#!/usr/bin/env python3
"""
Pure Computer Vision Vehicle Detection System
- YOLO (yolov8s.pt) for vehicle detection
- Custom YOLO (best.pt) for cone detection
- MiDaS for depth estimation
- Distance calibration using 2 cones at 5m and 10m
- Three states: RED <4m, YELLOW 4-8m, GREEN >8m
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from enum import Enum
import argparse
import os

class PatchState(Enum):
    RED = 0      # <4m - Immediate threat
    YELLOW = 1   # 4-8m - Caution
    GREEN = 2    # >8m - Safe

class VehicleDetectionSystem:
    def __init__(self):
        print("Initializing Vehicle Detection System...")
        
        # Load YOLO models
        self.vehicle_yolo = YOLO("yolov8s.pt")
        print("Vehicle detection model loaded: yolov8s.pt")
        
        # Load custom cone detection model
        try:
            self.cone_yolo = YOLO("best.pt")
            print("Cone detection model loaded: best.pt")
        except Exception as e:
            print(f"ERROR: Could not load cone model 'best.pt'!")
            print(f"Make sure 'best.pt' is in the same folder as this script!")
            exit(1)
        
        # Load MiDaS depth estimation
        self._init_midas()
        
        # Reference distances and thresholds
        self.ref_distances = [5.0, 10.0]  # Cone reference distances (meters)
        self.thresholds = [4.0, 8.0]      # [RED_limit, YELLOW_limit]
        
        # Calibration state
        self.calibrated = False
        self.depth_to_distance = None
        
        print("Vehicle Detection System ready!")
    
    def _init_midas(self):
        """Initialize MiDaS depth estimation model"""
        print("Loading MiDaS...")
        import torch.hub
        
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").default_transform
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas.to(self.device).eval()
        
        print(f"MiDaS loaded on {self.device}")
    
    def get_depth_map(self, image):
        """Generate depth map from RGB image using MiDaS"""
        input_tensor = self.transform(image).to(self.device)
        
        with torch.no_grad():
            depth = self.midas(input_tensor)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1), 
                size=image.shape[:2], 
                mode="bicubic", 
                align_corners=False
            ).squeeze()
        
        return depth.cpu().numpy()
    
    def detect_vehicles(self, image):
        """Detect vehicles using YOLO"""
        results = self.vehicle_yolo(image, verbose=False)
        vehicles = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    conf = box.conf[0].cpu().numpy()
                    cls_id = int(box.cls[0].cpu().numpy())
                    
                    # Vehicle classes: car=2, motorcycle=3, bus=5, truck=7
                    if cls_id in [2, 3, 5, 7] and conf > 0.5:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detection = {
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                            'conf': float(conf),
                            'class': result.names[cls_id]
                        }
                        vehicles.append(detection)
        
        return vehicles
    
    def detect_cones(self, image):
        """Detect cones using custom model"""
        results = self.cone_yolo(image, verbose=False)
        detected_cones = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    conf = box.conf[0].cpu().numpy()
                    
                    if conf > 0.3:  # Cone confidence threshold
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detection = {
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                            'conf': float(conf),
                            'class': 'cone'
                        }
                        detected_cones.append(detection)
        
        return detected_cones
    
    def calibrate_with_cones(self, depth_map, cone_detections):
        """Calibrate distance measurement using 2 reference cones"""
        if len(cone_detections) < 2:
            return False
        
        # Sort cones by y-position (closer cone has higher y value)
        cone_detections.sort(key=lambda c: c['center'][1], reverse=True)
        
        # Get depth values at cone centers
        depths = []
        for cone in cone_detections[:2]:
            x, y = cone['center']
            if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
                depth_val = depth_map[y, x]
                depths.append(depth_val)
        
        if len(depths) < 2 or abs(depths[1] - depths[0]) < 1e-6:
            return False
        
        # Create linear mapping: real_distance = slope * depth + intercept
        slope = (self.ref_distances[1] - self.ref_distances[0]) / (depths[1] - depths[0])
        intercept = self.ref_distances[0] - slope * depths[0]
        
        self.depth_to_distance = lambda d: slope * d + intercept
        self.calibrated = True
        
        print(f"Calibrated with 2 cones at {self.ref_distances}m")
        return True
    
    def get_distance(self, depth_map, position):
        """Get real-world distance for a pixel position"""
        if not self.calibrated:
            return None
        
        x, y = position
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            depth_val = depth_map[y, x]
            distance = self.depth_to_distance(depth_val)
            return max(0.1, distance)
        return None
    
    def distance_to_state(self, distance):
        """Convert distance to threat level state"""
        if distance < self.thresholds[0]:  # <4m
            return PatchState.RED
        elif distance < self.thresholds[1]:  # 4-8m
            return PatchState.YELLOW
        else:  # >8m
            return PatchState.GREEN
    
    def process_frame(self, image):
        """Process a single frame and return state, distance, and detections"""
        # Generate depth map
        depth_map = self.get_depth_map(image)
        
        # Detect objects
        vehicles = self.detect_vehicles(image)
        cones = self.detect_cones(image)
        
        # Calibrate if cones found and not yet calibrated
        if cones and not self.calibrated:
            self.calibrate_with_cones(depth_map, cones)
        
        # Calculate distances for vehicles
        min_distance = float('inf')
        for vehicle in vehicles:
            distance = self.get_distance(depth_map, vehicle['center'])
            vehicle['distance'] = distance
            if distance and distance < min_distance:
                min_distance = distance
        
        # Determine system state
        if min_distance == float('inf'):
            current_state = PatchState.GREEN
            min_distance = -1
        else:
            current_state = self.distance_to_state(min_distance)
        
        return current_state, min_distance, {'vehicles': vehicles, 'cones': cones}
    
    def visualize_frame(self, image, state, distance, detections):
        """Create visualization overlay on frame"""
        vis = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # State colors (BGR format)
        colors = {
            PatchState.RED: (0, 0, 255),
            PatchState.YELLOW: (0, 255, 255),
            PatchState.GREEN: (0, 255, 0)
        }
        
        # Draw vehicle detections
        for vehicle in detections['vehicles']:
            x1, y1, x2, y2 = vehicle['bbox']
            distance = vehicle.get('distance')
            
            if distance:
                vehicle_state = self.distance_to_state(distance)
                color = colors[vehicle_state]
                label = f"{vehicle['class']}: {distance:.1f}m"
            else:
                color = (128, 128, 128)
                label = f"{vehicle['class']}: No calibration"
            
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
            
            # Label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(vis, (x1, y1-30), (x1+label_size[0]+10, y1), color, -1)
            cv2.putText(vis, label, (x1+5, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        
        # Draw cone detections
        for cone in detections['cones']:
            x1, y1, x2, y2 = cone['bbox']
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 165), 2)
            cone_label = f"Cone ({cone['conf']:.2f})"
            cv2.putText(vis, cone_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 165), 2)
        
        # System status
        state_text = f"STATUS: {state.name}"
        cv2.putText(vis, state_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, colors[state], 3)
        
        if distance > 0:
            dist_text = f"CLOSEST: {distance:.1f}m"
            cv2.putText(vis, dist_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colors[state], 2)
        
        # Calibration status
        cal_text = f"CALIBRATED ({len(detections['cones'])} cones)" if self.calibrated else f"CALIBRATING... ({len(detections['cones'])} cones)"
        cal_color = (0, 255, 0) if self.calibrated else (255, 255, 0)
        cv2.putText(vis, cal_text, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cal_color, 2)
        
        return vis

def run_camera(camera_id=0):
    """Run with camera input"""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    detector = VehicleDetectionSystem()
    print("Detection System Running!")
    print("Place 2 cones at 5m and 10m from camera for calibration")
    print("Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            state, distance, detections = detector.process_frame(rgb_frame)
            vis_frame = detector.visualize_frame(rgb_frame, state, distance, detections)
            
            cv2.imshow('Vehicle Detection System', vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

def run_video(video_path):
    """Run with video input"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    detector = VehicleDetectionSystem()
    print(f"Processing video: {video_path}")
    print("Press 'q' to quit, 'space' to pause")
    
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video")
                    break
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                state, distance, detections = detector.process_frame(rgb_frame)
                vis_frame = detector.visualize_frame(rgb_frame, state, distance, detections)
                
                cv2.imshow('Vehicle Detection System - Video', vis_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Vehicle Detection System')
    parser.add_argument('--input', choices=['camera', 'video'], default='camera',
                       help='Input source: camera or video')
    parser.add_argument('--camera-id', type=int, default=0,
                       help='Camera ID for webcam input')
    parser.add_argument('--video-path', type=str,
                       help='Path to video file')
    
    args = parser.parse_args()
    
    print("=== Vehicle Detection System ===")
    print("Thresholds: RED <4m | YELLOW 4-8m | GREEN >8m")
    print("Requires: best.pt (cone model), yolov8s.pt (downloads automatically)")
    
    if not os.path.exists("best.pt"):
        print("ERROR: 'best.pt' not found!")
        return
    
    if args.input == 'video':
        if not args.video_path:
            print("Error: --video-path required for video input")
            return
        run_video(args.video_path)
    else:
        run_camera(args.camera_id)

if __name__ == "__main__":
    main()