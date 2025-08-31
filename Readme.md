# 🎙️ TrueVoxAI – Speech Emotion Recognition

TrueVoxAI is a **Speech Emotion Recognition (SER)** project that analyzes voice recordings to detect emotions such as **Happy, Sad, Angry, Neutral, Fear, etc.**  
It combines **Machine Learning + Audio Signal Processing + Web Deployment** into one complete project.

---

## 🚀 Features
- 🎤 Record or Upload Speech through the web interface  
- 🧠 ML Model (MLP Classifier) trained on speech datasets (RAVDESS/CREMA-D)  
- 🎼 Feature Extraction using MFCC, Chroma, Spectral Contrast  
- 🌐 Flask-based Web App with a simple UI  
- ☁️ Deployment-ready with Vercel + Hugging Face Spaces  

---

## 🏗️ Tech Stack
- **Python** (librosa, scikit-learn, numpy, pandas, matplotlib)  
- **Flask** (backend + UI rendering)  
- **MLPClassifier** from scikit-learn for classification  
- **Vercel / Hugging Face Spaces** for deployment  

---

## 📂 Repository Structure
- `app.py` – Flask App  
- `requirements.txt` – Python Dependencies  
- `vercel.json` – Vercel Deployment Config  
- `models/` – Trained ML artifacts  
  - `mlp_emotion_model.pkl` – Trained Model  
  - `scaler.pkl` – Feature Normalizer  
  - `label_encoder.pkl` – Label Mapping  
- `templates/` – HTML UI  
- `static/` – CSS/JS Assets  
- `Notebooks/TrueVox.ipynb` – Training Notebook  

---

## ⚙️ How It Works
1. User uploads/records speech 🎙️  
2. Audio features extracted using **librosa** 🎼  
3. Features normalized using **scaler.pkl** 📊  
4. Model predicts emotion using **mlp_emotion_model.pkl** 🧠  
5. Result displayed on UI 🎉  

---

## ▶️ Running Locally
```
# clone repo
git clone https://github.com/tannuiscoding/TrueVoxAI.git
cd TrueVoxAI

# install dependencies
pip install -r requirements.txt

# run Flask app
python app.py
```
## 🧑‍🤝‍🧑 Contributors

This project was built with ❤️ by:

| **Mansi** | **Tannu Choudhary** |
|------------------|----------------------|
| <img src="assets/mansi.jpg" width="150" style="border-radius:10px;"/> | <img src="assets/tannu.jpg" width="150" style="border-radius:10px;"/> |


