# 🛡️ Network Anomaly Detection System

A Machine Learning-based web application that detects whether network traffic is **normal or malicious** using the **NSL-KDD dataset**, with built-in explainability using SHAP.

---

## Overview

This project analyzes network traffic features and classifies them into:
- ✅ Normal Traffic  
- 🚨 Attack / Anomalous Traffic  

It also provides **interpretable insights** into why a prediction was made using:
- Feature Importance  
- SHAP (Explainable AI)  
- Human-readable explanations  

---

## Features

- 🔍 Detects network intrusions using Machine Learning  
- 📊 Displays top influencing features  
- 🧠 Uses SHAP for model explainability  
- 💡 Provides human-friendly explanations  
- 🖥️ Interactive UI built with Streamlit  
- ⚡ Real-time prediction with confidence score  

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

- **NSL-KDD Dataset**
- Used for training and testing intrusion detection models  
- Contains labeled network traffic data  

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/network-anomaly-detection.git
cd network-anomaly-detection