# 🧠 Computer Vision: Brain Tumor Detection using ResNet50

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange.svg)](https://tensorflow.org)
[![Model](https://img.shields.io/badge/Model-ResNet50-green.svg)](https://arxiv.org/abs/1512.03385)

An end-to-end computer vision project designed to classify brain tumor types from MRI scans using deep transfer learning with the **ResNet50** architecture. This repository contains the training pipelines, exploratory data analysis, and scripts necessary to train and evaluate the classification model.

---

## 📌 Project Overview
Brain tumors require fast and highly accurate diagnosis to assist medical professionals in treatment planning. This project utilizes automated image classification powered by Deep Learning to detect and categorize brain abnormalities from MRI imaging.

### Key Features:
* **Deep Transfer Learning:** Leverages a pre-trained `ResNet50` network fine-tuned for specialized medical image classification.
* **Streamlined Pipeline:** Includes both a Jupyter Notebook (`.ipynb`) for experimentation and a production-ready Python script (`.py`) for automated training execution.
* **Training Insights:** Tracks model training progress with serialized history logs (`history.pkl`) for easy post-training evaluation.

---

## 📁 Repository Structure

```text
├── class_indices.json                               # Mapping of class numbers to tumor labels
├── cv-rm-prj03-image-classification-brain-tumor.ipynb  # Interactive development & EDA notebook
├── cv_rm_prj03_image_classification_brain_tumor.py   # Production training & execution script
├── history.pkl                                      # Saved training history metrics
└── README.md                                        # Project documentation
