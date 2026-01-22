# Real-Time Face Blurring with MediaPipe

A Python application for privacy protection that detects faces in real-time and applies an adaptive blur effect. This project combines Deep Learning for detection (MediaPipe) with Classical Computer Vision for image processing (OpenCV).

## 📋 Context

This repository is part of my portfolio in **Robotics and Computer Vision**. While object detection is crucial for robot navigation, privacy preservation is equally important in real-world deployments (e.g., surveillance drones or service robots). This script demonstrates how to integrate pre-trained inference models into a video processing pipeline.

## 🛠️ Techniques & Concepts

The pipeline operates in the following stages:

1.  **Face Detection (Deep Learning):**
    Utilizes **Google MediaPipe's Face Detector** (specifically the `blaze_face_short_range` model). This is a lightweight model optimized for mobile and real-time inference, capable of detecting faces even with partial occlusion or rotation.

2.  **Region of Interest (ROI) Extraction:**
    Once a face is detected, the bounding box coordinates are extracted and clamped to the image dimensions. The face area is isolated as a specific Region of Interest (ROI) for processing.

3.  **Adaptive Gaussian Blur:**
    Instead of a fixed blur intensity, the script calculates the kernel size dynamically based on the face's width.
    * **Logic:** `k_size = width * 0.15`
    * **Benefit:** This ensures consistent anonymization. Faces closer to the camera (larger) receive a stronger blur, while faces further away receive a lighter blur, maintaining visual consistency.


## 🚀 How to Run

### Prerequisites
* Python 3.x installed
* A webcam connected

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/bernardo-sabino/face_anonymizer_opencv.git 
    cd face_anonymizer_opencv
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the script:**
    ```bash
    python face_anonymizer.py
    ```

Press **`q`** to quit the application.

## 👤 Author

**Bernardo Sabino**

* **Education:** Control and Automation Engineering Student at UFMG | Industrial Automation Technician (SENAI-MG)
* **Interests:** Robotics, Computer Vision, ROS/ROS 2, and Embedded Systems.
* **Connect with me:** [LinkedIn](https://www.linkedin.com/in/bernardosab/) | [GitHub](https://github.com/bernardo-sabino)

---
*Developed as part of my studies in Computer Vision algorithms.*

## 📄 License & Acknowledgments

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Acknowledgments:**
* The face detection model (`blaze_face_short_range.tflite`) is provided by **Google MediaPipe** and is subject to the **Apache License 2.0**.
