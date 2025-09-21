# Sleep-detection-system
 "Advanced Sleep Detection System using Computer Vision"
# Advanced Sleep Detection System

A computer vision-based sleep detection system that monitors user sleep states and controls music volume accordingly. Built as a final year college project demonstrating real-time AI/ML applications.

##  Features

- **Real-time Face Detection**: Uses MediaPipe for accurate facial landmark detection
- **Multi-modal Sleep Classification**: Combines eye tracking, breathing analysis, and movement detection
- **5-State Sleep Detection**: AWAKE → TIRED → RESTING → DROWSY → DEEP_SLEEP
- **Intelligent Music Control**: Automatically adjusts volume based on detected sleep state
- **Advanced Confidence Scoring**: Addresses "eyes closed ≠ asleep" problem
- **Professional Interface**: Real-time metrics and visualization

## 🛠 Technologies Used

- **Computer Vision**: OpenCV, MediaPipe
- **Machine Learning**: Eye Aspect Ratio (EAR) algorithm, Statistical analysis
- **Audio Processing**: Pygame
- **Data Analysis**: NumPy, SciPy
- **Programming Language**: Python 3.10+

##  Requirements

- Python 3.10 or higher
- Webcam/Camera device
- Audio output device
- Music file (MP3 format recommended)

##  Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/sleep-detection-system.git
   cd sleep-detection-system
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add music file (optional):**
   - Place an MP3 file named `relaxing_music.mp3` in the project directory
   - Or modify the music file list in the code

##  Usage

### Basic Version:
```bash
python complete_sleep_system.py
```

### Advanced Version (Recommended):
```bash
python advanced_sleep_system.py
```

### Test Components:
```bash
python music_debug.py  # Test audio system
```

##  How It Works

### 1. **Eye Aspect Ratio (EAR) Calculation**
- Monitors eye landmarks to calculate openness ratio
- EAR < 0.25: Eyes closed
- EAR < 0.18: Deep sleep threshold

### 2. **Multi-Modal Detection**
- **Breathing Analysis**: Tracks nose movement patterns
- **Micro-movement Detection**: Monitors facial stillness
- **Time Consistency**: Validates sustained states
- **Confidence Scoring**: Weighted combination of all factors

### 3. **Sleep State Classification**
```
AWAKE (EAR > 0.25) → Music: Normal volume
TIRED (Low blink rate) → Music: Slight reduction
RESTING_EYES (Eyes closed briefly) → Music: Gentle reduction
DROWSY (Sustained closure) → Music: Significant reduction
DEEP_SLEEP (High confidence) → Music: Stopped
```

##  Key Innovations

### Addresses "Eyes Closed ≠ Asleep" Problem:
- **Conservative Classification**: Requires 85%+ confidence for sleep detection
- **Multi-factor Validation**: Combines eye, breathing, and movement data
- **Immediate Recovery**: Any movement returns to awake state
- **Progressive States**: Gradual transition instead of binary classification

##  Performance Metrics

- **Processing Speed**: 30 FPS real-time processing
- **Detection Accuracy**: Multi-modal confidence scoring
- **Response Time**: < 100ms state transitions
- **Memory Usage**: Optimized with circular buffers

##  Interface Features

- **Real-time State Display**: Current sleep state with color coding
- **Confidence Visualization**: Live confidence score bar
- **Detailed Metrics**: EAR values, breathing patterns, movement scores
- **Session Statistics**: Comprehensive end-of-session reporting

##  Configuration

### Adjust Sensitivity:
```python
# In advanced_sleep_system.py
EAR_THRESHOLD = 0.25          # Eye closure threshold
SLEEP_FRAMES = 240            # Frames required for sleep (8 seconds)
SLEEP_EAR_THRESHOLD = 0.18    # Deep sleep threshold
```

### Custom Music:
```python
# Add your music file to this list
music_files = ["your_music.mp3", "relaxing_music.mp3", "music.mp3"]
```

##  Project Structure

```
sleep-detection-system/
├── advanced_sleep_system.py    # Advanced multi-modal version
├── complete_sleep_system.py    # Basic functional version
├── music_debug.py              # Audio system testing
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── demo/                       # Screenshots and demo videos
```

##  Academic Context

This project demonstrates:
- **Computer Vision Applications**: Real-time face and eye tracking
- **Machine Learning Techniques**: Feature extraction and classification
- **Signal Processing**: Breathing pattern analysis
- **Human-Computer Interaction**: Responsive audio control
- **Software Engineering**: Modular, documented, testable code

##  Technical Approach

### Eye Aspect Ratio (EAR) Algorithm:
```
EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
```
Where p1-p6 are eye landmark coordinates.

### Confidence Calculation:
```python
confidence = (eye_closure * 0.3) + (breathing * 0.25) + 
             (stillness * 0.25) + (time_consistency * 0.15) + 
             (stability * 0.05)
```

##  Limitations & Future Work

### Current Limitations:
- Requires good lighting conditions
- Single-user detection only  
- Dependent on facial visibility

### Future Enhancements:
- Audio-based breathing detection
- Multiple user support
- Mobile app integration
- Cloud-based analytics

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📄 License

This project is created for educational purposes. Feel free to use and modify for academic projects.

##  Author

**Nigranth Shah** - Final Year Artificial Intelligence and Data Science Student


##  Acknowledgments

- MediaPipe team for facial landmark detection
- OpenCV community for computer vision tools
- Academic advisors and project mentors

##  Demo

### System Interface:
![System Interface](demo/screenshot.png)

### State Transitions:
```
AWAKE → TIRED → RESTING_EYES → DROWSY → DEEP_SLEEP
```

### Performance Example:
```
Session Duration: 120.5 seconds
Frames Processed: 3614
Processing Rate: 30.0 FPS
Final Sleep Confidence: 0.87
State Distribution: AWAKE(45%), DROWSY(30%), DEEP_SLEEP(25%)
```
