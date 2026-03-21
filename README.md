# 🧠 HealthMind AI

> **Intelligent Early Health Assessment powered by Machine Learning**

HealthMind AI is a full-stack web application that predicts probable health conditions from symptoms using a trained Random Forest model, rule-based medical logic, and an explainable AI interface.

---

## 🌟 Features

| Feature | Description |
|---|---|
| 🔍 **Smart Symptom Input** | Search bar + category browser with tag-based selection |
| 🤖 **AI Prediction** | Random Forest model trained on 15 diseases × 100 samples each |
| 📊 **Risk Score (0–100)** | Blended score based on probability, disease severity & symptom count |
| 🚨 **Emergency Alert** | Rule-based detection of critical symptom combinations |
| 💡 **Explainability** | Bar chart showing which symptoms influenced the prediction most |
| 🏥 **Top 3 Conditions** | Each with probability, risk level, description & recommendations |

---

## 🗂️ Project Structure

```
healthmind-ai/
├── app.py                  # Flask backend & API routes
├── train_model.py          # ML model training script
├── requirements.txt        # Python dependencies
│
├── data/
│   └── disease_symptoms.py # Disease knowledge base (15 diseases, 45 symptoms)
│
├── model/
│   └── healthmind_model.pkl  # Saved model (auto-generated on first run)
│
├── templates/
│   └── index.html          # Main HTML page
│
└── static/
    ├── css/style.css       # All styling
    └── js/main.js          # Frontend logic & Chart.js visualisations
```

---

## 🚀 Quick Start

### Step 1 – Clone the repository
```bash
git clone https://github.com/Yug1010/AI-Health-prediction-system.git
cd healthmind-ai
```

### Step 2 – Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 3 – Run the app
```bash
python app.py
```

Then open your browser and go to: **http://127.0.0.1:5000**

> ℹ️ The ML model trains automatically on the first run (~10 seconds). After that it loads instantly.

---

## 🧪 How to Use

1. **Search or browse** symptoms by category (Fever, Respiratory, Digestive, etc.)
2. **Select all symptoms** you are experiencing — minimum 2 required
3. Click **Analyse Symptoms**
4. View your results:
   - 🚨 Emergency alert (if critical combination detected)
   - 📊 Risk Score gauge (0–100)
   - 🔬 Top 3 probable conditions with probabilities
   - 💡 Symptom contribution chart (explainability)

---

## 🤖 How the AI Works

```
Symptoms Selected
      │
      ▼
Feature Vector (1/0 for each of 45 symptoms)
      │
      ▼
Random Forest Classifier (300 trees, trained on 1500 samples)
      │
      ├──► Top 3 predicted conditions with probability scores
      │
      ├──► Feature importances → Explainability chart
      │
      └──► Rule-based emergency check (6 critical combinations)
```

**Model Details:**
- Algorithm: `RandomForestClassifier` (scikit-learn)
- Trees: 300
- Training samples: 1,500 (100 per disease × 15 diseases)
- Features: 45 binary symptom features
- Classes: 15 health conditions

---

## 🏥 Supported Conditions

| # | Condition | Risk Level |
|---|-----------|------------|
| 1 | Common Cold | 🟢 Low |
| 2 | Influenza (Flu) | 🟡 Medium |
| 3 | COVID-19 | 🔴 High |
| 4 | Pneumonia | 🔴 High |
| 5 | Dengue Fever | 🔴 High |
| 6 | Malaria | 🔴 High |
| 7 | Typhoid | 🔴 High |
| 8 | Diabetes (Type 2) | 🟡 Medium |
| 9 | Hypertension | 🔴 High |
| 10 | Heart Attack | 🔴 Critical |
| 11 | Appendicitis | 🔴 Critical |
| 12 | Asthma | 🟡 Medium |
| 13 | Migraine | 🟢 Low |
| 14 | Food Poisoning | 🟡 Medium |
| 15 | Anemia | 🟡 Medium |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Machine Learning | scikit-learn (Random Forest) |
| Data | NumPy, custom knowledge base |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Visualisation | Chart.js |

---

## ⚠️ Disclaimer

> HealthMind AI is designed for **early health awareness and educational purposes only**.
> It is **not** a medical device and does **not** replace professional medical advice,
> diagnosis, or treatment. Always consult a qualified healthcare provider.

---

## 📄 License

MIT License — free to use and modify.

---

