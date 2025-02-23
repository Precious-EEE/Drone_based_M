# 🌽 Maize Tassel Detection Using YOLOv5  

## Overview  
This project focuses on detecting maize tassels using **YOLOv5**, a state-of-the-art object detection model. The goal is to automate tassel detection from aerial images captured by drones, aiding in maize growth monitoring and yield estimation.  

## Features  
✅ **Dataset Collection**: Captured aerial images of maize fields using a **Tello drone**.  
✅ **Data Annotation**: Labeled maize tassels using **LabelImg** for YOLO format.  
✅ **Model Training**: Trained **YOLOv5s** on annotated images to detect tassels accurately.  
✅ **Evaluation Metrics**: Achieved **Precision = 0.801, Recall = 0.514, mAP50 = 0.577, mAP50-95 = 0.182**.  
✅ **Optimization**: Implemented techniques to handle **zero variances** and improve detection performance.  

## Installation  
Clone the repository and install dependencies:  
```sh
git clone https://github.com/your-repo/maize-tassel-detection.git
cd maize-tassel-detection
pip install -r requirements.txt
```

## Training the Model  
To train YOLOv5 on your dataset:  
```sh
python train.py --img 640 --batch 16 --epochs 50 --data dataset.yaml --weights yolov5s.pt --device 0
```

## Running Inference  
Use the trained model to detect maize tassels:  
```sh
python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.4 --source test_images/
```

## Results  
The trained model successfully detects maize tassels, providing bounding boxes around each detected tassel in aerial images.  

## Future Improvements  
🔹 Enhance model accuracy with more annotated data.  
🔹 Deploy the model using **Flask API or TensorFlow Serving**.  
🔹 Optimize inference speed for real-time drone applications.  

