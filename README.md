# Critiscan's Fruit Quality Assessment Model
![image](https://github.com/user-attachments/assets/1c57d228-e99e-48a0-ae29-17906734b350)
## *Overview*

Critiscan's Fruit Quality Assessment Model is an advanced deep learning solution for detecting the freshness of fruits and vegetables using computer vision. Developed for the Flipkart Robotics Challenge Hackathon, this model provides an innovative approach to quality assessment in agricultural produce.

- [Link to Website Repository](https://github.com/aanushkaguptaa/critiscan)
- [Link to OCR Detection Model](https://github.com/tsu-ki/ocr-script-freshness-model)
- [Link to Item Counting and Brand Detection](https://github.com/tsu-ki/FMCGDetectron)

## *Key Features*

- **High-Accuracy Freshness Classification**: Achieves an impressive 99.69% accuracy in detecting fruit and vegetable quality
- **Multi-Class Detection**: Classifies produce across six fruit types with granular quality states
- **Robust Deep Learning Architecture**: Utilizes state-of-the-art EfficientNetB5 model with advanced regularization techniques
## *Dataset*

- **Source**: [FruitNet Indian Fruits Dataset with Quality](https://www.kaggle.com/datasets/shashwatwork/fruitnet-indian-fruits-dataset-with-quality/)
- **Fruits Covered**: Apple, Banana, Guava, Lime, Orange, Pomegranate
- **Quality States**: Good, Bad, Mixed

---
## *Technical Specifications*

#### **1. Model Architecture**

- **Base Model**: EfficientNetB5 (pre-trained on ImageNet).
- **Input Shape**: `(img_size[0], img_size[1], 3)`
- **Modifications**:
    - Batch normalization layers added for regularization.
    - Two dense layers (1024 and 128 neurons) with L1/L2 regularization.
    - Dropout layers with rates of 30% and 45% to prevent overfitting.
    - Final output layer with `softmax` activation for multi-class classification.
- **Optimizer**: Adamax with an initial learning rate of 0.001.
- **Loss Function**: Categorical Crossentropy.
#### **2. Training and Validation**

- **Epochs**: 40
- **Batch Size**: 20
- **Data Augmentation**: (Mention if used and details, e.g., rotation, zoom, flip.)
- **Callbacks**:
    - **ReduceLROnPlateau**: Reduces learning rate by a factor of 0.5 if validation loss stagnates for 2 epochs.
    - **EarlyStopping**: Stops training after 4 epochs of no improvement in validation loss.
    - **ASK Callback**: Interactive callback to query user on training continuation after every `ask_epoch` (5 epochs in this case).
#### **3. Performance Metrics**

| Metric           | Value  |
| ---------------- | ------ |
| Overall Accuracy | 99.69% |
| Precision        | 0.9970 |
| Recall           | 0.9969 |
| F1-Score         | 0.9969 |
#### **4. Deployment**

- **Framework**: Flask
- **Containerization**: Docker
- **Cloud Deployment**: AWS EC2
- **Inference Time**: 3-4 seconds per prediction
---
## *Getting Started*

#### **1. Prerequisites**

- Python 3.8+
- pip
- Docker (optional)

#### **2. Installation**
```
#Clone the repository:    
    git clone https://github.com/tsu-ki/Freshness-model
    
#Install dependencies:
    pip install -r requirements.txt
    
#Run the application:
   python app.py
```

---
## *Future Roadmap*

- Expand dataset diversity
- Include packaged food freshness detection
- Optimize real-time inference capabilities

## *References*

- Dataset: [FruitNet: Indian Fruits Dataset with Quality](https://www.kaggle.com/datasets/shashwatwork/fruitnet-indian-fruits-dataset-with-quality/)
- [EfficientNet Research Paper](https://arxiv.org/abs/1905.11946)
