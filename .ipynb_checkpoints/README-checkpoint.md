## 🛡️ Network Anomaly Detection System

A Machine Learning-based web application that detects whether network traffic is normal or malicious using the NSL-KDD dataset, with built-in explainability using SHAP.

---

## Overview

This system analyzes network traffic and classifies it as:

- Normal Traffic
- Attack / Malicious Traffic

It also explains why a prediction was made using:

- Feature importance
- SHAP (Explainable AI)
- Human-readable insights

---

## Features

- Detects network intrusions using Machine Learning
- Displays top influencing features
- Uses SHAP for model explainability
- Provides human-friendly explanations
- Interactive UI built with Streamlit
- Real-time prediction with confidence score

---

## How It Works

1. Input network traffic sample
2. Model predicts Normal or Attack
3. SHAP explains feature contribution
4. UI displays:
   - Key influencing factors
   - Feature insights
   - Explanation of prediction

---

## Screenshots

🔴 Attack Detection
![Attack Prediction](attack_prediction.png)
![Attack Features](attack_features.png)
![Attack Explanation](attack_explanation.png)

---

🟢 Normal Traffic
![Normal Prediction](normal_prediction.png)
![Normal Features](normal_features.png)
![Normal Explanation](normal_explanation.png)

---

## Tech Stack
- Python
- Scikit-learn
- Pandas & NumPy
- Streamlit
- SHAP (Explainable AI)
- Matplotlib

---

## Dataset

- NSL-KDD Dataset
- Used for training and testing intrusion detection models
- Contains labeled network traffic data

---

## Installation & Setup

1. Clone the repository

git clone https://github.com/your-username/network-anomaly-detection.git
cd network-anomaly-detection

2. Install dependencies

pip install -r requirements.txt

3. Run the application

streamlit run app.py

---

## Output

The system provides:

- Prediction (Normal / Attack)
- Confidence score
- Status (Safe / Suspicious)
- Key influencing features
- SHAP-based explanation
- Human-readable interpretation

---

## 👩‍💻 Author

Divyadarshini

---

⭐ If you like this project

Give it a ⭐ on GitHub!