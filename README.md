# Saudi Currency Classifier

A real-time computer vision application that **recognizes Saudi banknotes and coins from a webcam feed** using **PyTorch, ResNet-18, and OpenCV**.

The application captures frames from a webcam, classifies the detected currency denomination, displays the prediction and confidence score, and tracks the detected Sum and Total in real time.

**~84% Validation Accuracy | 12 Currency Classes | Real-Time Inference**

## Why This Project?

Recognizing currency automatically can simplify cash handling by reducing the need for manual identification and calculation. This project explores how **deep learning and computer vision** can be applied to recognize different Saudi currency denominations and provide real-time results from a webcam.

---

## Features

* Classifies **12 Saudi banknote and coin classes**
* Performs **real-time inference** through a webcam
* Uses a **custom ResNet-18-based classification model**
* Displays the prediction and confidence score
* Applies a **75% confidence threshold**
* Calculates and displays the detected **Sum and Total**
* Includes model evaluation using **accuracy, precision, recall, F1-score, and confusion matrix**

---

## How It Works

```text
Webcam
   ↓
Frame Capture
   ↓
Image Preprocessing
   ↓
ResNet-18 Classification
   ↓
Confidence Check
   ↓
Currency Prediction
   ↓
Sum / Total
   ↓
Display Result
```

Each webcam frame is resized to **224 × 224**, converted from BGR to RGB, normalized, and passed to the trained model.

If the prediction confidence is below **75%**, the application displays:

```text
No class detected
```

---

## Supported Classes

The model recognizes the following 12 classes.

### Coins

* HALALA 10
* HALALA 25
* HALALA 50
* RIYAL 1
* RIYAL 2

### Banknotes

* RIYAL 5 NOTE
* RIYAL 10 NOTE
* RIYAL 20 NOTE
* RIYAL 50 NOTE
* RIYAL 100 NOTE
* RIYAL 200 NOTE
* RIYAL 500 NOTE

---

## Model

The project uses a **custom ResNet-18-based convolutional neural network implemented in PyTorch**.

The network is built using residual `BasicBlock` layers with the following ResNet-18 configuration:

```text
[2, 2, 2, 2]
```

### Training Configuration

| Parameter         | Value              |
| ----------------- | ------------------ |
| Model             | ResNet-18          |
| Framework         | PyTorch            |
| Number of Classes | 12                 |
| Input Size        | 224 × 224          |
| Batch Size        | 32                 |
| Epochs            | 10                 |
| Optimizer         | Adam               |
| Learning Rate     | 0.001              |
| Loss Function     | Cross-Entropy Loss |

The trained model weights are stored in:

```text
ptmodel/currency_classifier.pt
```

---

## Results

The model achieved approximately **84% validation accuracy**.

| Metric              | Result |
| ------------------- | -----: |
| Validation Accuracy |   ~84% |
| Macro Precision     |   ~83% |
| Macro Recall        |   ~84% |
| Macro F1-Score      |   ~82% |
| Weighted Precision  |   ~85% |
| Weighted Recall     |   ~84% |
| Weighted F1-Score   |   ~83% |

Performance varied between classes, particularly for visually similar coin denominations.

---

## Dataset

The original dataset contained **2,223 images across 12 classes**.

The data was split into:

* **80% training**
* **20% validation**

using a fixed random seed of `1337`.

The dataset is not included in this repository.

---

## Preprocessing

Images are processed using:

* Resize to `224 × 224`
* BGR to RGB conversion
* Tensor conversion
* Normalization

The normalization values are:

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

---

## Data Augmentation

The project includes an augmentation script using **Albumentations**.

The script includes transformations such as:

* Brightness and contrast adjustment
* Hue and saturation adjustment
* RGB shifting
* Horizontal flipping
* Rotation
* Shift and scale transformations
* Gaussian blur
* Gaussian noise

---

## Real-Time Application

The main application is located in:

```text
app.py
```

It captures webcam frames and performs classification in real time. The result is displayed together with the confidence score and the running **Sum and Total**.

Example:

```text
RIYAL 10 NOTE (0.99)
```

```text
Sum: 15 Total: 15
```

The current application uses:

```python
cv2.VideoCapture(1)
```

If the webcam is assigned to a different camera index, this value can be changed in `app.py`.

---

## Technologies

| Category          | Technologies                         |
| ----------------- | ------------------------------------ |
| Programming       | Python                               |
| Deep Learning     | PyTorch, Torchvision                 |
| Computer Vision   | OpenCV                               |
| Data Processing   | NumPy, Pandas                        |
| Data Augmentation | Albumentations                       |
| Development       | Visual Studio Code, Jupyter Notebook |


---

## Project Structure

```text
Saudi-Currency-Classifier/
│
├── app.py
├── data-splitter.py
├── Data_augumentation.py
├── requirements.txt
├── Resnet18CustomModel.ipynb
│
├── ptmodel/
│   ├── currency_classifier.pt
│   ├── class_names.pkl
│   ├── resnet18_model.pth
│   └── resnet_model.py
│
└── Dataset/
    └── Not included in repository
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Rana-Aljarboua/Saudi-Currency-Classifier.git
cd Saudi-Currency-Classifier
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```
