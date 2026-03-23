# 🧠 HealthMind AI

> **Dual-module intelligent health assessment platform powered by Machine Learning**

HealthMind AI is a full-stack web application with two independent modules — a **Physical Health** disease predictor and a **Mental Health** self-assessment tool — built with Python, Flask, and scikit-learn.

---

## 🌟 Features

### 🫀 Physical Health Module

| Feature | Description |
|---|---|
| 👤 **Patient Profile** | Age, gender, and 14 pre-existing conditions personalise every prediction |
| 🧬 **Family History** | 12 hereditary conditions apply evidence-based risk multipliers |
| 📋 **Medical Reports** | Upload a PDF or manually enter 12 blood test values — both contribute directly to prediction |
| 🩺 **Live Vitals Input** | Heart rate, blood pressure, SpO2, temperature, respiratory rate with instant colour feedback |
| 🔍 **Smart Symptom Input** | Search bar + category browser across 9 categories and 45 symptoms |
| 🤖 **AI Prediction** | Random Forest model (98% accuracy) trained on 15 diseases × 100 samples |
| 📊 **Risk Score (0–100)** | Blended score combining ML probability + disease severity + age + vitals + preexisting + family history + report findings |
| 🚨 **Emergency Alert** | Rule-based detection of critical symptom and vitals combinations |
| 💡 **Explainability** | Bar chart showing how much each symptom influenced the prediction |
| 🏥 **Top 3 Conditions** | Each with probability bar, risk level, description, and personalised recommendations |

### 🧠 Mental Health Module

| Feature | Description |
|---|---|
| 📋 **20-Question Assessment** | Clinically inspired questions across 6 domains: Mood, Anxiety, Sleep, Stress, Social, Physical |
| 📊 **Wellness Score (0–100)** | 5-level scale: Minimal → Mild → Moderate → High → Severe |
| 🕸️ **Radar Chart** | Visual domain breakdown across all 6 areas simultaneously |
| 📈 **Domain Progress Bars** | Per-domain scores with colour-coded indicators |
| 💡 **Personalised Recommendations** | Tailored suggestions based on assessment level |
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
│                                 #   15 diseases · 45 symptoms · 9 categories
│                                 #   14 preexisting conditions
│                                 #   12 family history conditions
│                                 #   28 report parameter flags
│                                 #   6 vital sign ranges · 6 emergency rules
│
├── model/
│   └── healthmind_model.pkl      # Trained Random Forest (auto-generated on first run)
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

> ℹ️ The ML model trains automatically on the very first run (~10 seconds). After that it loads instantly.

---

## 🧪 How to Use

### 🫀 Physical Health Assessment

**Step 1 — Patient Profile**
- Enter your **age** (required) and optionally gender
- Select any **pre-existing conditions** (e.g. Diabetes, Heart Disease, Asthma)
- Select **family history** of diseases (e.g. if a parent had heart disease or diabetes)

**Step 2 — Medical Reports** *(optional but improves accuracy)*

Two ways to add your report data:

| Method | How |
|---|---|
| 📤 **Upload PDF** | Drag & drop or browse a PDF blood test report — values are auto-extracted |
| ✏️ **Enter Values** | Type in your numbers directly (Hb, WBC, Blood Sugar, Cholesterol, etc.) |

Extracted or entered findings are shown as colour-coded chips (⬆️ high / ⬇️ low / ✅ normal) and are fed directly into the prediction engine.

**Step 3 — Current Vitals** *(optional)*
- Enter Heart Rate, Blood Pressure, Temperature, SpO2, Respiratory Rate
- Each field shows live colour feedback (green = normal, yellow = warning, red = critical)

**Step 4 — Symptoms**
- Search or browse symptoms by category
- Select all symptoms you are currently experiencing (minimum 2)

**Click Analyse Now — results show:**
- 🚨 Emergency alert if critical combinations are detected
- 📊 Personalised risk score gauge (0–100)
- 🩺 Vitals status cards
- ⚕️ Preexisting condition impact notes
- 🧬 Family history risk factor notes
- 📋 Report findings impact notes
- 🔬 Top 3 probable conditions with probability bars and recommendations
- 💡 Symptom contribution chart (explainability)

---

### 🧠 Mental Health Assessment

1. Click **🧠 Mental Health** in the header to navigate to `/mental-health`
2. Optionally enter your age and gender
3. Answer all **20 questions** using the 0–3 frequency scale:
   - 0 = Not at all
   - 1 = Several days
   - 2 = More than half the days
   - 3 = Nearly every day
4. Watch the progress bar fill as you answer
5. Click **Analyse My Responses** — results show:
   - 🆘 Crisis alert with helplines (if applicable)
   - 📊 Animated wellness score with level badge
   - 🕸️ Radar chart across all 6 domains
   - 📈 Domain-by-domain progress bars
   - 💡 Personalised recommendations

---

## 🤖 How the AI Works

### Physical Health — Full Pipeline

```
Patient Input
  ├── Symptoms (45 binary features)
  ├── Age + Gender
  ├── Pre-existing Conditions (14 options)
  ├── Family History (12 options)
  ├── Medical Report (PDF auto-extract or 12 manual fields)
  └── Vitals (6 measurements)
         │
         ▼
Random Forest Classifier (300 trees · 1,500 training samples)
         │
         ├──► Raw probabilities for all 15 conditions
         ├──► Age group modifier applied  (child / adult / senior)
         ├──► Pre-existing condition boosts  (up to ×2.5)
         ├──► Family history boosts  (up to ×1.5)
         ├──► Medical report finding boosts  (up to ×2.4)
         ├──► Vitals analysis  (status + emergency flags)
         ├──► Emergency rule check  (6 critical combinations)
         └──► Final Risk Score (0–100) + Top 3 Predictions + Explainability
```

**Model Details:**
- Algorithm: `RandomForestClassifier` (scikit-learn)
- Trees: 300 · Max depth: 20
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
         ├──► Normalised to 0–100 scale
         ├──► Level assigned  (Minimal / Mild / Moderate / High / Severe)
         ├──► Per-domain scores  (Mood, Anxiety, Sleep, Stress, Social, Physical)
         ├──► Crisis flag checked  (self-harm question)
         └──► Personalised recommendations generated
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

---

## 🧬 Mental Health Assessment Domains

| Domain | Questions | What It Measures |
|---|---|---|
| 😔 Mood | 4 | Depression, hopelessness, loss of interest |
| 😰 Anxiety | 4 | Nervousness, worry, irritability, fear |
| 😴 Sleep | 3 | Insomnia, fatigue, unrefreshing sleep |
| 😤 Stress | 3 | Overwhelm, decision difficulty, coping ability |
| 🤝 Social | 3 | Isolation, withdrawal, loneliness |
| 🫁 Physical | 2 | Appetite changes, stress-linked physical symptoms |
| 🆘 Crisis | 1 | Self-harm ideation — triggers dedicated helpline alert |

---

## 📋 Report Parameter Flags (28 total)

### Blood Test Flags
`anemia_low_hb` · `low_rbc` · `high_wbc_infection` · `low_platelets_dengue` · `high_blood_sugar` · `high_hba1c` · `high_cholesterol` · `high_ldl` · `high_triglycerides` · `high_creatinine` · `high_liver_enzymes` · `high_tsh` · `low_tsh` · `high_esr_crp` · `high_uric_acid`

### Radiology / Imaging Flags
`pneumonia_consolidation` · `pleural_effusion` · `cardiomegaly` · `splenomegaly` · `hepatomegaly` · `fatty_liver` · `kidney_stone` · `gallstone`

### Manual Entry Fields (12)
Haemoglobin · WBC Count · Platelets · Fasting Blood Sugar · HbA1c · Total Cholesterol · LDL · Triglycerides · Creatinine · SGPT/ALT · TSH · Uric Acid

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Physical Health assessment page |
| `GET` | `/mental-health` | Mental Health assessment page |
| `GET` | `/api/meta` | Symptoms, categories, preexisting, family history options |
| `POST` | `/api/predict` | Full physical health prediction |
| `POST` | `/api/analyze-report` | Extract findings from uploaded PDF |
| `POST` | `/api/process-manual` | Process manually entered lab values |
| `POST` | `/predict-mental` | Mental health scoring (20 answers) |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10+, Flask |
| Machine Learning | scikit-learn (Random Forest), NumPy |
| Report Parsing | pdfplumber (PDF text extraction), regex thresholds |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Visualisation | Chart.js 4 (gauge, bar chart, radar chart) |
| Templating | Jinja2 |

---

## ⚠️ Disclaimer

> HealthMind AI is designed for **early health awareness and educational purposes only**.
> It is **not** a medical device and does **not** replace professional medical advice,
> diagnosis, or treatment. The mental health module is a self-awareness tool and is
> **not** a clinical diagnostic instrument. Always consult a qualified healthcare
> or mental health professional for any medical concern.

---

## 📄 License

MIT License — free to use and modify.

---
