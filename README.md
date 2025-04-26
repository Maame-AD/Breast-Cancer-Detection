# Breast Cancer Detection and Tumour Localisation on Mammographic Images using YOLOv9 and Deep Learning

## Project Overview 📄

This repository documents my **MSc Data Science thesis** at **Kingston University London**, which focuses on enhancing **Early detection and localisation of breast cancer tumours** using **mammographic images**. The project integrates **object detection** with **image classification** to support radiologists in identifying and categorising breast abnormalities more accurately.

The core of the project leverages **YOLOv9**, a state-of-the-art object detection model, to **localise tumour regions** within mammograms. This is complemented by **classification models** (e.g., CNNs) to categorise detected tumours as **benign** or **malignant**.

---

## Research Objectives 🎯 

- **Apply YOLOv9 for object detection** to **localise tumour regions** in mammographic images.  
- **Develop classification models** (e.g., CNNs) to categorise mammograms (benign vs malignant).  
- **Compare the performance of classification models** with and without **dimensionality reduction techniques** like **Principal Component Analysis (PCA)**.  
- **Evaluate models** using metrics suited for both tasks:  
  - **Object detection**: Mean Average Precision (**mAP**), Intersection over Union (**IoU**).  
  - **Classification**: Accuracy, Precision, Recall, F1-score, AUC-ROC.

---

## Methodology

1. **Data Acquisition & Preprocessing:**  
   - Utilising publicly available mammographic datasets (e.g., **MIAS**, **DDSM**).  
   - Preprocessing includes resizing, normalisation, annotation for detection, and augmentation.

2. **Object Detection with YOLOv9:**  
   - Implementing and fine-tuning **YOLOv9** for detecting tumour regions.  
   - Annotating datasets with bounding boxes for detection tasks.

3. **Classification Modelling:**  
   - Developing **CNN-based classifiers** for image classification (benign vs malignant).  
   - Applying **dimensionality reduction (PCA)** to assess its effect on model performance.

4. **Evaluation & Validation:**  
   - Using **mAP** and **IoU** for detection model evaluation.  
   - Using **accuracy**, **precision**, **recall**, **F1-score**, and **AUC-ROC** for classification evaluation.  
   - Applying **Grad-CAM** for model interpretability.

---

## Tools & Technologies 🛠️

- **Languages:** Python  
- **Frameworks:** PyTorch, TensorFlow, Keras, Ultralytics YOLOv9  
- **Libraries:** OpenCV, Scikit-learn, NumPy, Pandas  
- **Techniques:** Object Detection (YOLOv9), Convolutional Neural Networks (CNNs), Dimensionality Reduction (PCA), Transfer Learning

---

## Current Progress

- ✅ Completed literature review on **object detection** and **classification** in medical imaging.  
- ✅ Data annotation and preprocessing in progress.  
- 🔄 Fine-tuning **YOLOv9** for tumour localisation.  
- 🔄 Developing and comparing **classification models**.

---

## 🧠 Anticipated Contributions

- Demonstrate the feasibility of using **YOLOv9** for **accurate tumour localisation** in mammographic images.  
- Provide comparative insights into **classification model performance** with and without **dimensionality reduction**.  
- Contribute to advancements in **AI-assisted diagnostics** in breast cancer detection.

---

*This research merges the power of deep learning with healthcare, aiming to enhance diagnostic accuracy and support radiologists in the fight against breast cancer.*
