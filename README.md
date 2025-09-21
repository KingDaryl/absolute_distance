# Absolute_Distance_Estimation

A real-time vehicle detection and distance monitoring system using computer vision. Detects vehicles and provides three-tier alerts (RED/YELLOW/GREEN) based on proximity using cone-based distance calibration.

## Project Overview

**Main Project**: Pure computer vision system for camera/video input

**Optional**: CARLA simulation integration for testing

## Installation & Setup

### 1. Prerequisites

- **Python 3.8+** (Python 3.9 recommended)
- **Webcam** (for camera mode) or **video files** (for video mode)
- **Your trained cone model** (`best.pt` file)

### 2. Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt
```

**Optional: Create virtual environment first (recommended)**
```bash
python -m venv vehicle_detection
# Windows:
vehicle_detection\Scripts\activate
# Linux/Mac:
source vehicle_detection/bin/activate
# Then: pip install -r requirements.txt
```

### 3. Download CARLA (Optional - for simulation mode)

```bash
pip install -r requirements.txt
```

**What gets downloaded automatically on first run:**
- `yolov8s.pt` (~25MB) - Vehicle detection model
- MiDaS depth model (~100MB) - Depth estimation

### 3. Download CARLA (Optional - for simulation mode)

If you want to use CARLA simulation:

1. **Download CARLA 0.9.13+** from [https://carla.org/](https://carla.org/)
2. **Extract** to a folder like `C:\CARLA_0.9.15\` 
3. **Add to PATH** (optional): Add CARLA Python API to environment variables

### 4. Project Structure Setup

Create this folder structure:
```
vehicle_detection_system/
├── detection_system.py           # Main script
├── best.pt                       # Your cone model (REQUIRED)
├── requirements.txt              # Dependencies
├── README.md                     # This file
└── video/                        # Video files (optional)
    ├── your_video1.mp4
    └── your_video2.mp4
```

**Important**: Place your `best.pt` file in the main folder alongside `detection_system.py`

## Usage

### Camera Mode (Live Detection)

```bash
# Default camera (usually built-in webcam)
python detection_system.py

# External USB camera
python detection_system.py --camera-id 1

# Try different camera IDs if needed
python detection_system.py --camera-id 2
```

### Video Mode

```bash
# Specific video file
python detection_system.py --input video --video-path path/to/your/video.mp4

# Video from video folder
python detection_system.py --input video --video-path video/test_video.mp4

# Auto-find first video in video folder (if video/ exists)
python detection_system.py --input video
```

## Calibration Setup

**Required**: Place 2 traffic cones at exactly **5 meters** and **10 meters** from your camera

1. Measure and place first cone 5m from camera
2. Measure and place second cone 10m from camera
3. Ensure both cones are visible in camera view
4. System auto-calibrates when both cones detected

## Alert System

- 🔴 **RED**: Vehicle closer than 4 meters (immediate threat)
- 🟡 **YELLOW**: Vehicle 4-8 meters away (caution zone)  
- 🟢 **GREEN**: Vehicle farther than 8 meters (safe)

## Controls

- **'q'**: Quit application
- **'space'**: Pause/resume (video mode only)

## Required Models

**You provide:**
- `best.pt` - Your custom trained cone detection model

**Auto-downloaded:**
- `yolov8s.pt` - YOLO vehicle detection (cars, trucks, buses, motorcycles)
- MiDaS depth model - Depth estimation via torch.hub

## Troubleshooting

### "ERROR: Could not load cone model 'best.pt'"
- Ensure `best.pt` is in same folder as `detection_system.py`
- Check file name is exactly `best.pt`

### "Error: Could not open camera"
- Try different camera IDs: `--camera-id 1`, `--camera-id 2`
- Close other apps using the camera
- Check camera permissions

### Poor Detection Performance
- Ensure good lighting
- Position cones clearly visible
- Clean camera lens
- Try lower resolution for better performance

### Video Issues
- Check video file path and format
- Supported: .mp4, .avi, .mov, .mkv, .wmv

## Command Line Options

```bash
# Camera input options
python detection_system.py                                    # Default camera
python detection_system.py --camera-id 1                      # External camera

# Video input options  
python detection_system.py --input video                      # Auto-find video
python detection_system.py --input video --video-path video/test.mp4    # Specific video
```

## Performance Tips

- **GPU**: Install CUDA PyTorch for faster processing
- **Resolution**: Lower camera resolution if performance is slow
- **Lighting**: Ensure good lighting for better cone detection

---

## CARLA Simulation (Optional)

For testing with CARLA simulator instead of real camera/video.

### Additional Setup for CARLA

1. **Install CARLA 0.9.13+** from [carla.org](https://carla.org/)

2. **Add CARLA files** to your project:
```
vehicle_detection_system/
├── carla_files/
│   ├── detection_system_carla.py
│   └── carla_opendrive_world.py
```

3. **Add CARLA Python API** to environment:
```bash
# Windows
set PYTHONPATH=%PYTHONPATH%;C:\CARLA_0.9.15\WindowsNoEditor\PythonAPI\carla\dist\carla-0.9.13-py3.7-win-amd64.egg

# Linux  
export PYTHONPATH=$PYTHONPATH:/path/to/carla/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg
```

### Running CARLA Mode

**Terminal 1**: Start CARLA server
```bash
cd C:\CARLA_0.9.15\WindowsNoEditor\
CarlaUE4.exe -quality-level=Low -windowed -ResX=640 -ResY=480
```

**Terminal 2**: Run detection system
```bash
cd carla_files
python detection_system_carla.py
```

## System Requirements

**Minimum:**
- Python 3.8+
- 4GB RAM
- Any USB camera or video files
- Your `best.pt` model file

**Recommended:**
- Python 3.9
- 8GB RAM  
- NVIDIA GPU with CUDA support
- Good lighting conditions

## Dependencies

Core libraries (in requirements.txt):
```
torch>=2.0.0
torchvision>=0.15.0  
opencv-python>=4.8.0
ultralytics>=8.0.0
numpy>=1.24.0
pillow>=9.5.0
```

## Quick Start Summary

1. **Install Python 3.9**
2. **Create environment**: `conda create -n vehicle_detection python=3.9`
3. **Activate**: `conda activate vehicle_detection`  
4. **Install deps**: `pip install -r requirements.txt`
5. **Add your model**: Place `best.pt` in main folder
6. **Place cones**: 5m and 10m from camera
7. **Run**: `python detection_system.py`

