# **Iris and Pupil Detection using YOLOv5 and OpenCV**

This project focuses on detecting and segmenting the iris and pupil using **YOLOv5** and **OpenCV** with **Haar Cascade classifiers**. It includes real-time eye detection, enhancement (CLAHE), and specular reflection removal.

## **1. Project Overview**
- Detect face and eyes using **Haar Cascade Classifier**.
- Manually select the eye region.
- Apply **CLAHE enhancement** and **specular reflection removal**.
- Detect iris and pupil using a **fine-tuned YOLOv5s model**.
- Generate bounding boxes for the iris and pupil.
- Evaluate results using **precision, recall, F1-score, and mAP**.

## **2. Repository Structure**
```
📂 Project Root
 ├── 📂 yolov5              # Cloned YOLOv5 repo (contains trained models in runs/)
 ├── 📂 weights             # Trained YOLOv5s model (best.pt, last.pt)
 ├── 📂 processed_eyes      # Final output images with bounding boxes
 ├── 📂 results             # Graphs, performance metrics, and classification reports
 ├── iris_pupil.ipynb       # Notebook to run eye detection and YOLOv5 inference
 ├── README.md              # Project documentation
 ├── Classification_Report.pdf  # YOLOv5 performance metrics
 ├── Table(iris_pupil).pdf  # Iris-to-pupil ratio detection results
```

## **3. Installation & Setup**
### **Step 1: Clone the YOLOv5 Repository**
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

### **Step 2: Download the Dataset**
Download the **Iris-Pupil dataset** from [Roboflow](https://universe.roboflow.com/iris-annotation/irispupille) and place it inside the `yolov5` folder.

### **Step 3: Train the YOLOv5s Model**
```bash
python train.py --img 640 --batch 4 --epochs 100 --data irispupille/data.yaml --weights yolov5s.pt --name iris_pupil_detection
```
- The trained model weights (`best.pt`, `last.pt`) will be stored in `runs/train/iris_pupil_detection/`.

### **Step 4: Run the Detection Script**
```bash
jupyter notebook iris_pupil.ipynb
```
- Open the notebook and run the cells to:
  - Detect and select the eyes.
  - Enhance images using CLAHE.
  - Detect the **iris and pupil** with YOLOv5.

## **4. Results & Evaluation**
- The **processed images** with bounding boxes are stored in `processed_eyes/`.
- The **performance metrics and graphs** are available in `results/`.
- Access the **classification report** and detection results:
  - [📄 Classification Report_(YOLOv5)](./Classification_Report.pdf)
  - [📄 Iris-Pupil Detection Table](./Table(iris,pupil).pdf)

## **5. Suggested Approach & Methodology**
1. **Face & Eye Detection**  
   - Used **Haar Cascade Classifier** to detect face and eyes.  
   - Allowed **manual selection** of the eye pair.

2. **Image Enhancement**  
   - Applied **CLAHE (Contrast Limited Adaptive Histogram Equalization)**.  
   - Removed **specular reflections** to enhance clarity.

3. **YOLOv5 Model for Iris-Pupil Detection**  
   - Trained **YOLOv5s** on **310 training images, 90 validation images, 45 test images**.  
   - Achieved **mAP@0.5: 0.98664** and **F1-score: 0.96866**.

4. **Evaluation Metrics**  
   - **Precision**: 98.87%  
   - **Recall**: 94.93%  
   - **mAP@0.5**: 98.66%  

## **6. Conclusion**
The project successfully detects the iris and pupil with high accuracy. The trained **YOLOv5s model** generalizes well across various eye conditions. The approach can be extended for **biometric authentication, medical imaging, and gaze tracking applications**.
