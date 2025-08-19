Face Anti-Spoofing System
This project implements a face anti-spoofing system using deep learning. It detects whether a face in an image or video is real or spoofed (e.g., printed photo, replay attack). The system uses a MobileNetV2-based model and supports real-time detection.

Project Structure
face.py: Main script for face detection and anti-spoofing.
spoof.py: Contains core logic for spoof detection.
face_anti_spoofing_model.h5, best_model.h5: Pre-trained model files.
dataset: Contains labeled samples for training and evaluation (ignored by git).
system_architecture.gv: Graphviz file describing the system architecture.
Dataset
The dataset is organized in dataset with labeled samples for real and spoofed faces.
Each sample is stored in a separate folder with extracted frames.
System Architecture
The system follows this pipeline:

Input Data: Video or image files.
Frame Extraction: Extract frames using OpenCV.
Labeling: Annotate frames as real or spoof.
Preprocessing: Data augmentation, resizing, normalization.
Model Training: Train MobileNetV2-based model.
Evaluation: Confusion matrix, classification report, accuracy metrics.
Deployment: Save and deploy the model for real-time detection.
See system_architecture.gv for a detailed diagram.

Usage
Install dependencies
pip install -r requirements.txt
Run detection
python face.py
Train model
Modify and run spoof.py as needed.

Notes
The dataset and large files are ignored by git (see .gitignore).
Pre-trained models are required for detection.
License
This project is for educational purposes.
