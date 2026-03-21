# 🧠 HealthMind AI

> **Dual-module intelligent health assessment platform powered by Machine Learning**

HealthMind AI is a full-stack web application with two independent modules — a **Physical Health** disease predictor and a **Mental Health** self-assessment tool — built with Python, Flask, and scikit-learn.

---

## 🌟 Features

### 🫀 Physical Health Module
| Feature | Description |
|---|---|
| 🔍 **Smart Symptom Input** | Search bar + category browser with tag-based selection across 9 categories |
| 👤 **Patient Profile** | Age, gender, and 14 pre-existing conditions personalise every prediction |
| 🩺 **Live Vitals Input** | Heart rate, blood pressure, temperature, SpO2, respiratory rate with instant feedback |
| 🤖 **AI Prediction** | Random Forest model (98% accuracy) trained on 15 diseases × 100 samples |
| 📊 **Risk Score (0–100)** | Blended score using ML probability + disease severity + age + vitals + comorbidities |
| 🚨 **Emergency Alert** | Rule-based detection of critical symptom and vitals combinations |
| 💡 **Explainability** | Bar chart showing how much each symptom influenced the prediction |
| 🏥 **Top 3 Conditions** | Each with probability bar, risk level, description, and recommendations |

### 🧠 Mental Health Module
| Feature | Description |
|---|---|
| 📋 **20-Question Assessment** | Clinically inspired questions across 6 domains: Mood, Anxiety, Sleep, Stress, Social, Physical |
| 📊 **Wellness Score (0–100)** | Scored on a 5-level scale: Minimal → Mild → Moderate → High → Severe |
| 🕸️ **Radar Chart** | Visual breakdown of scores across all 6 domains simultaneously |
| 📈 **Domain Progress Bars** | Per-domain scores shown clearly with colour-coded indicators |
| 💡 **Personalised Recommendations** | Tailored suggestions based on your assessment level |
| 🆘 **Crisis Detection** | Dedicated alert with helpline numbers if self-harm thoughts are indicated |
| ⏱️ **Progress Tracker** | Real-time answered/total counter as you complete the questionnaire |

---

## 🗂️ Project Structure

```
healthmind-ai/
├── app.py                        # Flask backend — all routes & scoring logic
├── train_model.py                # ML model training script (auto-runs on first launch)
├── requirements.txt              # Python dependencies
│
├── data/
│   └── disease_symptoms.py       # Full knowledge base:
│                                 #   15 diseases, 45 symptoms, 9 categories,
│                                 #   14 preexisting conditions, 6 vitals,
│                                 #   6 emergency rules
│
├── model/
│   └── healthmind_model.pkl      # Trained Random Forest (auto-generated)
│
├── templates/
│   ├── index.html                # Physical Health page
│   └── mental_health.html        # Mental Health page
│
└── static/
    ├── css/
    │   ├── style.css             # Shared base styles (navy/teal/mint theme)
    │   └── mental_health.css     # Mental health module styles
    └── js/
        ├── main.js               # Physical health frontend logic
        └── mental_health.js      # Mental health frontend logic
```

---

## 🚀 Quick Start

### Step 1 — Clone the repository
```bash
git clone https://github.com/Yug1010/AI-Health-prediction-system.git
cd healthmind-ai
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the app
```bash
python app.py
```

Open your browser at: **http://127.0.0.1:5000**

> ℹ️ The ML model trains automatically on the very first run (~10 seconds). After that it loads instantly from the saved file.

---

## 🧪 How to Use

### 🫀 Physical Health Assessment
1. Enter your **age** (required) and optionally gender
2. Select any **pre-existing conditions** you have (e.g. Diabetes, Heart Disease)
3. Optionally enter your current **vitals** — live colour feedback shows normal/abnormal instantly
4. **Search or browse** symptoms by category and select all you are experiencing
5. Click **Analyse Now** to see:
   - 🚨 Emergency alert if critical combinations are detected
   - 📊 Personalised risk score gauge (0–100)
   - 🩺 Vitals status cards
   - ⚕️ Preexisting condition impact notes
   - 🔬 Top 3 probable conditions with probability bars
   - 💡 Symptom contribution chart

### 🧠 Mental Health Assessment
1. Navigate to **Mental Health** using the header button or go to `/mental-health`
2. Optionally enter your age and gender
3. Answer all **20 questions** across 6 domains using the 0–3 frequency scale
4. Watch the progress bar fill as you answer
5. Click **Analyse My Responses** to see:
   - 🆘 Crisis alert with helplines (if applicable)
   - 📊 Animated wellness score with level badge
   - 🕸️ Radar chart across all 6 domains
   - 📈 Domain-by-domain progress bars
   - 💡 Personalised recommendations

---

## 🤖 How the AI Works

### Physical Health — Machine Learning Pipeline

```
Patient Input (symptoms + age + vitals + preexisting conditions)
        │
        ▼
Feature Vector — 45 binary symptom features
        │
        ▼
Random Forest Classifier (300 trees, 1,500 training samples)
        │
        ├──► Raw probabilities for all 15 conditions
        │
        ├──► Age group modifier applied  (child / adult / senior multipliers)
        │
        ├──► Preexisting condition boost  (disease-specific risk multipliers)
        │
        ├──► Vitals analysis  (6 vital signs → status + emergency flags)
        │
        ├──► Emergency rule check  (6 critical symptom/vitals combinations)
        │
        └──► Final blended Risk Score (0–100) + Top 3 predictions + Explainability
```

**Model Details:**
- Algorithm: `RandomForestClassifier` (scikit-learn)
- Trees: 300 — Max depth: 20
- Training samples: 1,500 (100 per disease × 15 diseases)
- Features: 45 binary symptom inputs
- Cross-validation accuracy: **~98%**

### Mental Health — Scoring Engine

```
20 answers (0–3 each) across 6 domains
        │
        ▼
Raw score summed (max = 57, excluding crisis question)
        │
        ▼
Normalised to 0–100 scale
        │
        ├──► Level assigned  (Minimal / Mild / Moderate / High / Severe)
        │
        ├──► Per-domain scores calculated  (Mood, Anxiety, Sleep, Stress, Social, Physical)
        │
        ├──► Crisis flag checked  (question mh_20)
        │
        └──► Recommendations selected based on level
```

---

## 🏥 Supported Physical Conditions

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
| 10 | Heart Attack | 🚨 Critical |
| 11 | Appendicitis | 🚨 Critical |
| 12 | Asthma | 🟡 Medium |
| 13 | Migraine | 🟢 Low |
| 14 | Food Poisoning | 🟡 Medium |
| 15 | Anemia | 🟡 Medium |

## 🧠 Mental Health Assessment Domains

| Domain | Questions | What It Measures |
|---|---|---|
| 😔 Mood | 4 | Depression, hopelessness, loss of interest |
| 😰 Anxiety | 4 | Nervousness, worry, irritability, fear |
| 😴 Sleep | 3 | Insomnia, fatigue, unrefreshing sleep |
| 😤 Stress | 3 | Overwhelm, decision difficulty, coping ability |
| 🤝 Social | 3 | Isolation, withdrawal, loneliness |
| 🫁 Physical | 2 | Appetite changes, stress-linked physical symptoms |
| 🆘 Crisis | 1 | Self-harm ideation (triggers dedicated alert) |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10+, Flask |
| Machine Learning | scikit-learn (Random Forest), NumPy |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Visualisation | Chart.js 4 (doughnut gauge, bar chart, radar chart) |
| Templating | Jinja2 |

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Physical Health assessment page |
| `GET` | `/mental-health` | Mental Health assessment page |
| `GET` | `/api/meta` | Returns symptoms, categories, preexisting options |
| `POST` | `/api/predict` | Physical health prediction (symptoms + age + vitals + preexisting) |
| `POST` | `/predict-mental` | Mental health scoring (20 question answers) |

---

## ⚠️ Disclaimer

> HealthMind AI is designed for **early health awareness and educational purposes only**.
> It is **not** a medical device and does **not** replace professional medical advice,
> diagnosis, or treatment. The mental health module is a self-awareness tool and is
> **not** a clinical diagnostic instrument. Always consult a qualified healthcare
> or mental health professional for any medical concerns.

---

## 📄 License

MIT License — free to use and modify.

---

*Built for hackathon submission — HealthMind AI Team*
