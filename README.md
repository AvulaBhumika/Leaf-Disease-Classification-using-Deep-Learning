# 🌿 Leaf Disease Classification using Deep Learning

This project focuses on classifying rice leaf diseases—**Bacterial leaf blight**, **Brown spot**, and **Leaf smut**—from images using Convolutional Neural Networks. Two models were implemented:

1. **Pure CNN**: A baseline custom Convolutional Neural Network.
2. **VGG16-based Transfer Learning**: Leveraging pretrained weights from ImageNet for enhanced performance.

---

## 🧠 Objective

To accurately classify rice leaf images into one of the following categories:

* Bacterial leaf blight
* Brown spot
* Leaf smut

The project aims to demonstrate the benefit of transfer learning in improving classification performance over a vanilla CNN architecture.

---

## 📊 Dataset

* Images are categorized into three classes.
* Train/Validation/Test split applied.
* Images were resized to `(224, 224, 3)` for compatibility with VGG16 input format.

---

## 🏗️ Model Architectures

### 🔹 1. Pure CNN (Baseline)

A basic CNN with:

* Conv2D + MaxPooling layers
* Flatten → Dense → Dropout → Output

Performance:

```
Accuracy       : 65%
Macro F1-score : 65%
Recall (avg)   : 66%
```

### 🔹 2. VGG16 (Transfer Learning)

Using:

```python
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
```

Performance:

```
Accuracy       : 83%
Macro F1-score : 82%
Recall (avg)   : 83%
```

📈 **Significant improvement** in all metrics compared to pure CNN.

---

## 📈 Evaluation Metrics

Metrics considered:

* **Precision**
* **Recall**
* **F1-Score**
* **Accuracy**
* Confusion Matrix (optional: include plot if available)

These metrics were calculated for each class, and as macro and weighted averages.

---

## 🧪 Training Strategy

* Optimizer: Adam
* Loss: Categorical Crossentropy
* EarlyStopping and ModelCheckpoint used
* Augmentation: Horizontal Flip, Rotation, Zoom, etc.

---

## 🧰 Technologies Used

* Python
* TensorFlow / Keras
* NumPy, Pandas
* Matplotlib / Seaborn (for visualization)
* Scikit-learn (metrics)

---

## 🚀 Results & Insights

| Metric         | Pure CNN | VGG16 |
| -------------- | -------- | ----- |
| Accuracy       | 65%      | 83%   |
| Macro F1-score | 65%      | 82%   |
| Recall (avg)   | 66%      | 83%   |

* **VGG16 Transfer Learning model significantly outperformed the custom CNN**, especially in recall and F1-score.
* Highlights the value of pretrained features when dataset size is small to medium.

---

## 📌 Conclusion

This project validates the **effectiveness of transfer learning** in image classification tasks, particularly when data availability is limited. Fine-tuning a pretrained VGG16 model provided higher accuracy and robustness in classifying rice leaf diseases.

---

## ✅ Future Work

* Fine-tune the top VGG16 layers for even better performance.
* Experiment with other architectures: ResNet, EfficientNet.
* Build a Flask/Streamlit web app for real-time predictions.
* Convert model to ONNX/TFLite for edge deployment.


**~AvB**
