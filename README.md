# CardioAI — Heart Disease Predictor
## Setup & Run Instructions

### 1. مجلد المشروع
ضع كل الملفات دي في مجلد واحد:
```
cardioai/
├── app.py              ← الـ Streamlit app
├── heart_model.pkl     ← الموديل المدرب
├── heart_scaler.pkl    ← الـ scaler
└── requirements.txt    ← المتطلبات
```

### 2. تنصيب المتطلبات
```bash
pip install -r requirements.txt
```

### 3. تشغيل الـ App
```bash
streamlit run app.py
```

ثم افتح المتصفح على: **http://localhost:8501**

---
## ملاحظات على الموديل
- **Dataset:** UCI Heart Disease (Cleveland)
- **Algorithm:** Random Forest (300 trees)
- **Accuracy:** ~98% على test set
- **Features (13):** age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
