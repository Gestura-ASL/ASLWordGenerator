#  ASL Word Generator — Real-Time Sign Language Recognition

An **AI-powered American Sign Language (ASL) Word Generator** that recognizes hand gestures in real-time and predicts the most probable ASL words using a **TFLite deep learning model**.  
The system captures video input via webcam, extracts hand landmarks using **MediaPipe**, processes frames through a trained **TensorFlow Lite** model, and visualizes predictions dynamically.

---

##  Overview

This project aims to bridge the communication gap between ASL users and text-based systems by providing an **end-to-end ASL-to-Word recognition framework**.

-  Captures real-time ASL gestures from webcam/video  
-  Uses **TFLite** for efficient inference  
-  Displays **top-N predicted words with confidence levels**  
-  Supports **frame-by-frame analysis**  
-  Provides graphical visualization of predictions (via Matplotlib)  

---

##  Tech Stack

| Component | Technology |
|------------|-------------|
| Programming Language | Python 3.10 |
| Landmark Detection | [MediaPipe Hands + Pose](https://developers.google.com/mediapipe) |
| Model Framework | TensorFlow Lite |
| Visualization | Matplotlib, Pandas |
| Video Processing | OpenCV |
| Data Handling | NumPy, PyArrow |
| Interface | Jupyter / CLI |
| ML Utilities | scikit-learn, TensorFlow Addons FastAPI |

---

##  Installation

###  Clone the Repository
```bash
git clone https://github.com/<your-username>/ASL-WordGenerator.git
cd ASL-WordGenerator

```
### Environment
```bash
use python 3.10.19

python3 -m venv .venv
source .venv/bin/activate  # Linux / macOS
.venv\Scripts\activate     # Windows

pip install -r requirements.txt
```
##  Contributing

### Development Workflow
1. Review architecture and coding guidelines
2. Create feature branch from main
3. Implement following established patterns
4. Update documentation and tests
5. Submit pull request with comprehensive description

### Code Quality Standards
- Follow existing module structure and naming conventions
- Maintain comprehensive test coverage
- Update documentation for any API changes
- Ensure security best practices are followed







