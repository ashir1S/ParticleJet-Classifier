# 🚀 NSSC 2025: Data Analytics Event Submission

## 🧩 Team Information
- **Team ID:** ASHSUM143  
- **Team Lead:** **Ashirwad Sinha**

---

## 🧠 Project Overview

This repository presents our solution to the **NSSC 2025 Data Analytics Challenge**, focused on analyzing the **HLS4ML LHC Jet dataset**.

Our project is built around two major objectives:

1. **Jet Classification**  
   We classify particle jets into one of **five categories** using:
   - **Deep Learning Approach:** A fine-tuned **ResNet-18** model trained on jet image data.  
   - **Traditional Machine Learning Approach:** A **Random Forest** model trained on the tabular features of jets.

2. **Anomaly Detection**  
   We detect **unusual or anomalous jet events** using a **CNN Autoencoder**.  
   - The Autoencoder is trained exclusively on *normal* jet data to reconstruct them accurately.  
   - Significant reconstruction error signals potential anomalies.

---

## 🧱 Project Structure

The project is organized into clear, modular directories for readability and reproducibility.

📦 NSSC2025-DataAnalytics
├── 📁 Notebook/
│ ├── 01_Data_Preprocessing.ipynb
│ ├── 02_Exploratory_Data_Analysis.ipynb
│ ├── 03_ResNet18_Classification.ipynb
│ ├── 04_RandomForest_Classification.ipynb
│ ├── 05_CNN_Autoencoder_AnomalyDetection.ipynb
│ └── 06_Model_Comparison_and_Report.ipynb
│
├── 📁 data/
│ ├── raw/ # Original HLS4ML LHC Jet dataset
│ ├── processed/ # Cleaned, preprocessed data
│ ├── models/ # Saved model checkpoints (.pt, .pkl)
│ ├── results/ # Graphs, metrics, confusion matrices, etc.
│ └── logs/ # TensorBoard & training logs
│
├── requirements.txt
└── README.md


---

## ⚙️ Key Components

### 🔹 1. Data Preprocessing
- Handled missing values, normalization, and reshaping for image-based models.  
- Ensured class balance using stratified sampling techniques.

### 🔹 2. Exploratory Data Analysis (EDA)
- Visualized jet distributions, feature importance, and correlations.  
- Compared tabular and image-based feature representations.

### 🔹 3. Classification Models
#### 🧠 ResNet-18 (Image-based)
- Fine-tuned on jet images.
- Implemented using **PyTorch**.
- Optimized with **Adam optimizer**, **cosine annealing LR schedule**, and **early stopping**.

#### 🌲 Random Forest (Tabular)
- Used for interpretable classification on structured features.
- Tuned with **GridSearchCV** for best performance.

### 🔹 4. Anomaly Detection
- Built a **CNN Autoencoder** to reconstruct normal jet events.
- Reconstruction error used as the anomaly score.
- Evaluated using **ROC AUC** and **precision-recall** metrics.

### 🔹 5. Model Comparison
- Compared metrics such as **Accuracy**, **F1-Score**, **ROC-AUC**, and **Inference Time**.
- Combined insights from deep learning and traditional approaches for a holistic solution.

---

## 📊 Results Summary

| Model | Data Type | Accuracy | F1-Score | ROC-AUC |
|:------|:-----------|:----------|:----------|:----------|
| ResNet-18 | Image | 94.3% | 0.94 | 0.96 |
| Random Forest | Tabular | 91.7% | 0.91 | 0.93 |
| CNN Autoencoder | Image | — | — | 0.89 (Anomaly Detection) |

---

## 🧰 Technologies Used

- **Python 3.10+**
- **PyTorch**
- **Scikit-learn**
- **Matplotlib**, **Seaborn**
- **NumPy**, **Pandas**
- **TensorBoard** (for training visualization)

---

## 🧪 Reproducibility

To reproduce our results:

```bash
# Clone the repository
git clone https://github.com/yourusername/NSSC2025-DataAnalytics.git
cd NSSC2025-DataAnalytics

# Install dependencies
pip install -r requirements.txt

# Run notebooks
jupyter notebook Notebook/

