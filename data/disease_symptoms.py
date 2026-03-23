# HealthMind AI – Disease & Symptoms Knowledge Base (v2)

# ── All symptoms (ML feature columns) ───────────────────────────────────────
SYMPTOMS = [
    "fever", "high_fever", "mild_fever", "chills", "sweating", "cold_sweats",
    "cough", "dry_cough", "shortness_of_breath", "wheezing",
    "chest_pain", "chest_tightness", "congestion",
    "headache", "severe_headache", "dizziness", "blurred_vision",
    "body_ache", "joint_pain", "muscle_pain", "back_pain",
    "neck_stiffness", "weakness", "fatigue",
    "nausea", "vomiting", "diarrhea", "abdominal_pain", "loss_of_appetite",
    "runny_nose", "sore_throat", "sneezing",
    "loss_of_taste", "loss_of_smell",
    "rash", "itching", "pale_skin", "skin_rash",
    "frequent_urination", "excessive_thirst", "slow_healing_wounds",
    "arm_pain", "rapid_heartbeat", "nosebleed", "palpitations",
]

SYMPTOM_DISPLAY = {s: s.replace("_", " ").title() for s in SYMPTOMS}

SYMPTOM_CATEGORIES = {
    "Fever & Temperature":  ["fever", "high_fever", "mild_fever", "chills", "sweating", "cold_sweats"],
    "Respiratory":          ["cough", "dry_cough", "shortness_of_breath", "wheezing",
                             "chest_pain", "chest_tightness", "congestion"],
    "Head & Neurological":  ["headache", "severe_headache", "dizziness", "blurred_vision"],
    "Body & Muscles":       ["body_ache", "joint_pain", "muscle_pain", "back_pain",
                             "neck_stiffness", "weakness", "fatigue"],
    "Digestive":            ["nausea", "vomiting", "diarrhea", "abdominal_pain", "loss_of_appetite"],
    "Nose & Throat":        ["runny_nose", "sore_throat", "sneezing", "loss_of_taste", "loss_of_smell"],
    "Skin":                 ["rash", "itching", "pale_skin", "skin_rash"],
    "Metabolic":            ["frequent_urination", "excessive_thirst", "slow_healing_wounds"],
    "Cardiac":              ["arm_pain", "rapid_heartbeat", "nosebleed", "palpitations"],
}

# ── Disease knowledge base ────────────────────────────────────────────────────
DISEASES = {
    "Common Cold": {
        "symptoms":       ["mild_fever", "runny_nose", "sore_throat", "sneezing", "cough", "congestion", "fatigue"],
        "risk_level":     "low",
        "description":    "A viral upper-respiratory infection. Common in children & elderly. Usually self-limiting within 7–10 days.",
        "recommendation": "Rest, stay hydrated, use OTC decongestants. No antibiotics needed.",
        "emergency":      False,
        "icon":           "🤧",
        "age_risk":       {"child": 1.2, "adult": 1.0, "senior": 1.3},
    },
    "Influenza (Flu)": {
        "symptoms":       ["high_fever", "chills", "body_ache", "fatigue", "cough", "headache", "sweating", "sore_throat"],
        "risk_level":     "medium",
        "description":    "Contagious respiratory illness. Significantly more dangerous in seniors (65+) and young children.",
        "recommendation": "Rest, fluids, antivirals within 48 hrs. Seek care if symptoms worsen rapidly.",
        "emergency":      False,
        "icon":           "🤒",
        "age_risk":       {"child": 1.3, "adult": 1.0, "senior": 1.6},
    },
    "COVID-19": {
        "symptoms":       ["fever", "dry_cough", "fatigue", "loss_of_taste", "loss_of_smell",
                           "shortness_of_breath", "body_ache", "headache"],
        "risk_level":     "high",
        "description":    "SARS-CoV-2 infection. High risk in seniors and those with preexisting respiratory/cardiac conditions.",
        "recommendation": "Isolate immediately, get tested, monitor SpO2. Seek care if SpO2 drops below 94%.",
        "emergency":      False,
        "icon":           "🦠",
        "age_risk":       {"child": 0.7, "adult": 1.0, "senior": 2.0},
    },
    "Pneumonia": {
        "symptoms":       ["high_fever", "cough", "chest_pain", "shortness_of_breath", "fatigue", "chills", "sweating"],
        "risk_level":     "high",
        "description":    "Lung infection serious in seniors, children under 5, and immunocompromised patients.",
        "recommendation": "Seek medical attention promptly. Requires prescribed antibiotics or antivirals.",
        "emergency":      False,
        "icon":           "🫁",
        "age_risk":       {"child": 1.5, "adult": 1.0, "senior": 2.2},
    },
    "Dengue Fever": {
        "symptoms":       ["high_fever", "severe_headache", "joint_pain", "muscle_pain", "rash", "fatigue", "nausea", "vomiting"],
        "risk_level":     "high",
        "description":    "Mosquito-borne viral infection. Risk of severe dengue higher in children and adults with previous infection.",
        "recommendation": "See a doctor immediately. Stay hydrated. Use paracetamol only — avoid aspirin/ibuprofen.",
        "emergency":      False,
        "icon":           "🦟",
        "age_risk":       {"child": 1.4, "adult": 1.0, "senior": 1.3},
    },
    "Malaria": {
        "symptoms":       ["high_fever", "chills", "sweating", "headache", "nausea", "vomiting", "body_ache", "fatigue"],
        "risk_level":     "high",
        "description":    "Parasite infection via Anopheles mosquitoes. Severe outcomes more likely in children and pregnant women.",
        "recommendation": "Immediate medical attention required. Blood tests and antimalarial prescription needed.",
        "emergency":      False,
        "icon":           "🦟",
        "age_risk":       {"child": 1.5, "adult": 1.0, "senior": 1.3},
    },
    "Typhoid": {
        "symptoms":       ["high_fever", "abdominal_pain", "weakness", "headache", "loss_of_appetite", "nausea", "fatigue", "sweating"],
        "risk_level":     "high",
        "description":    "Bacterial infection via contaminated food/water. More severe in children and immunocompromised patients.",
        "recommendation": "Consult a doctor for antibiotics. Strict hygiene. Stay hydrated.",
        "emergency":      False,
        "icon":           "🧫",
        "age_risk":       {"child": 1.4, "adult": 1.0, "senior": 1.2},
    },
    "Diabetes (Type 2)": {
        "symptoms":       ["frequent_urination", "excessive_thirst", "fatigue", "blurred_vision", "slow_healing_wounds", "weakness"],
        "risk_level":     "medium",
        "description":    "Chronic metabolic condition. Risk increases significantly after age 40 and with obesity.",
        "recommendation": "Consult a doctor for blood sugar testing. Diet, exercise, and medication may be needed.",
        "emergency":      False,
        "icon":           "🩸",
        "age_risk":       {"child": 0.4, "adult": 1.0, "senior": 1.5},
    },
    "Hypertension": {
        "symptoms":       ["headache", "dizziness", "chest_pain", "shortness_of_breath", "nosebleed", "rapid_heartbeat", "fatigue"],
        "risk_level":     "high",
        "description":    "Elevated blood pressure. Risk increases with age, obesity, and family history.",
        "recommendation": "See a doctor for BP measurement. May require medication and lifestyle changes.",
        "emergency":      False,
        "icon":           "💊",
        "age_risk":       {"child": 0.3, "adult": 1.0, "senior": 1.8},
    },
    "Heart Attack": {
        "symptoms":       ["chest_pain", "shortness_of_breath", "arm_pain", "cold_sweats", "nausea", "dizziness", "rapid_heartbeat", "palpitations"],
        "risk_level":     "critical",
        "description":    "Medical emergency. Risk sharply increases in men over 45 and women over 55, and in those with prior heart disease.",
        "recommendation": "🚨 CALL EMERGENCY SERVICES (102/112) IMMEDIATELY. Do not drive yourself.",
        "emergency":      True,
        "icon":           "❤️",
        "age_risk":       {"child": 0.1, "adult": 1.0, "senior": 2.5},
    },
    "Appendicitis": {
        "symptoms":       ["abdominal_pain", "nausea", "vomiting", "fever", "loss_of_appetite", "weakness"],
        "risk_level":     "critical",
        "description":    "Appendix inflammation — surgical emergency. Most common in ages 10–30.",
        "recommendation": "🚨 SEEK EMERGENCY CARE IMMEDIATELY. Do not eat, drink, or take pain relievers.",
        "emergency":      True,
        "icon":           "🏥",
        "age_risk":       {"child": 1.3, "adult": 1.2, "senior": 1.0},
    },
    "Asthma": {
        "symptoms":       ["shortness_of_breath", "wheezing", "chest_tightness", "cough", "fatigue"],
        "risk_level":     "medium",
        "description":    "Chronic airway inflammation. Often starts in childhood; worsens with allergens and pollution.",
        "recommendation": "Use prescribed rescue inhaler. Identify triggers. Seek emergency care if breathing is severely compromised.",
        "emergency":      False,
        "icon":           "💨",
        "age_risk":       {"child": 1.6, "adult": 1.0, "senior": 1.2},
    },
    "Migraine": {
        "symptoms":       ["severe_headache", "nausea", "vomiting", "dizziness", "fatigue", "blurred_vision"],
        "risk_level":     "low",
        "description":    "Neurological condition. More prevalent in adults 25–55, especially women.",
        "recommendation": "Rest in a dark quiet room. Take prescribed medication. Track triggers.",
        "emergency":      False,
        "icon":           "🧠",
        "age_risk":       {"child": 0.7, "adult": 1.2, "senior": 0.9},
    },
    "Food Poisoning": {
        "symptoms":       ["nausea", "vomiting", "diarrhea", "abdominal_pain", "fever", "weakness", "fatigue"],
        "risk_level":     "medium",
        "description":    "Contaminated food/water illness. More dangerous in children, seniors, and immunocompromised patients.",
        "recommendation": "Stay hydrated. BRAT diet. Seek help if symptoms persist >48 hrs or dehydration occurs.",
        "emergency":      False,
        "icon":           "🤢",
        "age_risk":       {"child": 1.4, "adult": 1.0, "senior": 1.5},
    },
    "Anemia": {
        "symptoms":       ["fatigue", "weakness", "pale_skin", "shortness_of_breath", "dizziness", "headache", "rapid_heartbeat"],
        "risk_level":     "medium",
        "description":    "Low red blood cell count. Common in women of reproductive age, children, and seniors.",
        "recommendation": "Blood tests needed. Iron supplements, dietary changes, or treatment of underlying cause.",
        "emergency":      False,
        "icon":           "🩺",
        "age_risk":       {"child": 1.3, "adult": 1.0, "senior": 1.4},
    },
}

# ── Emergency symptom combinations ───────────────────────────────────────────
EMERGENCY_COMBINATIONS = [
    (["chest_pain", "arm_pain"],
     "Chest pain radiating to the arm may indicate a heart attack. Call emergency services immediately."),
    (["chest_pain", "shortness_of_breath", "cold_sweats"],
     "This symptom combination is a medical emergency. Call 102/112 now."),
    (["chest_pain", "rapid_heartbeat", "dizziness"],
     "These cardiac symptoms require immediate emergency attention."),
    (["abdominal_pain", "high_fever", "vomiting"],
     "Severe abdominal symptoms may indicate appendicitis or another emergency. Seek immediate care."),
    (["shortness_of_breath", "rapid_heartbeat", "chest_pain"],
     "Breathing and cardiac symptoms together are a serious emergency. Call 102/112 immediately."),
    (["chest_pain", "palpitations", "shortness_of_breath"],
     "Possible cardiac emergency. Do not delay — call emergency services now."),
]

RISK_COLORS = {
    "low":      "#02C39A",
    "medium":   "#F7B731",
    "high":     "#E67E22",
    "critical": "#E63946",
}

# ══════════════════════════════════════════════════════════════════════════════
#  Family History
# ══════════════════════════════════════════════════════════════════════════════

# Diseases with known hereditary / familial risk factors
FAMILY_HISTORY_CONDITIONS = {
    "fh_heart_disease":  {"label": "Heart Disease",       "icon": "❤️"},
    "fh_diabetes":       {"label": "Diabetes",            "icon": "🩸"},
    "fh_hypertension":   {"label": "Hypertension",        "icon": "💊"},
    "fh_cancer":         {"label": "Cancer",              "icon": "🔬"},
    "fh_asthma":         {"label": "Asthma",              "icon": "💨"},
    "fh_migraine":       {"label": "Migraine",            "icon": "🧠"},
    "fh_anemia":         {"label": "Anemia",              "icon": "🩺"},
    "fh_thyroid":        {"label": "Thyroid Disorder",    "icon": "🦋"},
    "fh_kidney_disease": {"label": "Kidney Disease",      "icon": "🫘"},
    "fh_mental_illness": {"label": "Mental Illness",      "icon": "🧬"},
    "fh_obesity":        {"label": "Obesity",             "icon": "⚖️"},
    "fh_stroke":         {"label": "Stroke",              "icon": "⚡"},
}

# Risk multipliers: family_history_id → { disease_name: multiplier }
# Kept lower than CONDITION_DISEASE_RISK (personal history) since these are
# inherited predispositions, not current diagnoses.
FAMILY_HISTORY_RISK = {
    "fh_heart_disease":  {
        "Heart Attack":      1.45,
        "Hypertension":      1.25,
        "Anemia":            1.10,
    },
    "fh_diabetes":       {
        "Diabetes (Type 2)": 1.50,
        "Hypertension":      1.20,
        "Heart Attack":      1.20,
    },
    "fh_hypertension":   {
        "Hypertension":      1.40,
        "Heart Attack":      1.25,
        "Anemia":            1.10,
    },
    "fh_cancer":         {
        "Anemia":            1.20,
        "Pneumonia":         1.15,
    },
    "fh_asthma":         {
        "Asthma":            1.40,
        "Pneumonia":         1.15,
        "Common Cold":       1.10,
    },
    "fh_migraine":       {
        "Migraine":          1.50,
    },
    "fh_anemia":         {
        "Anemia":            1.40,
    },
    "fh_thyroid":        {
        "Diabetes (Type 2)": 1.10,
        "Anemia":            1.15,
        "Hypertension":      1.10,
    },
    "fh_kidney_disease": {
        "Hypertension":      1.20,
        "Anemia":            1.25,
        "Diabetes (Type 2)": 1.15,
    },
    "fh_mental_illness": {},   # no physical disease overlap — handled by mental module
    "fh_obesity":        {
        "Diabetes (Type 2)": 1.25,
        "Hypertension":      1.20,
        "Heart Attack":      1.15,
        "Asthma":            1.10,
    },
    "fh_stroke":         {
        "Heart Attack":      1.35,
        "Hypertension":      1.30,
    },
}

# ══════════════════════════════════════════════════════════════════════════════
#  Medical Report Findings → Disease Risk
# ══════════════════════════════════════════════════════════════════════════════

# Each flag (returned by Claude when it analyses a report) maps to disease risk
# multipliers. These represent objective lab/imaging evidence — so multipliers
# are stronger than family history but similar to personal preexisting conditions.
REPORT_PARAMETER_FLAGS = {

    # ── Blood test flags ──────────────────────────────────────────────────────
    "anemia_low_hb": {
        "label":  "Low Haemoglobin",
        "icon":   "🩸",
        "color":  "#E63946",
        "boosts": {"Anemia": 2.2, "Heart Attack": 1.2, "Pneumonia": 1.1},
    },
    "low_rbc": {
        "label":  "Low RBC Count",
        "icon":   "🩸",
        "color":  "#E67E22",
        "boosts": {"Anemia": 1.8, "Malaria": 1.3},
    },
    "high_wbc_infection": {
        "label":  "Elevated WBC (Infection)",
        "icon":   "🦠",
        "color":  "#E67E22",
        "boosts": {"Pneumonia": 1.6, "Typhoid": 1.4, "Influenza (Flu)": 1.3,
                   "COVID-19": 1.3, "Appendicitis": 1.4},
    },
    "low_platelets_dengue": {
        "label":  "Low Platelets",
        "icon":   "🔬",
        "color":  "#E63946",
        "boosts": {"Dengue Fever": 2.0, "Malaria": 1.5, "Anemia": 1.3},
    },
    "high_blood_sugar": {
        "label":  "High Blood Sugar",
        "icon":   "🩸",
        "color":  "#E67E22",
        "boosts": {"Diabetes (Type 2)": 2.0, "Hypertension": 1.3, "Heart Attack": 1.2},
    },
    "high_hba1c": {
        "label":  "High HbA1c",
        "icon":   "🩸",
        "color":  "#E63946",
        "boosts": {"Diabetes (Type 2)": 2.4, "Hypertension": 1.3, "Heart Attack": 1.3},
    },
    "high_cholesterol": {
        "label":  "High Total Cholesterol",
        "icon":   "💊",
        "color":  "#E67E22",
        "boosts": {"Heart Attack": 1.4, "Hypertension": 1.2},
    },
    "high_ldl": {
        "label":  "High LDL Cholesterol",
        "icon":   "💊",
        "color":  "#E67E22",
        "boosts": {"Heart Attack": 1.5, "Hypertension": 1.2},
    },
    "high_triglycerides": {
        "label":  "High Triglycerides",
        "icon":   "💊",
        "color":  "#F7B731",
        "boosts": {"Diabetes (Type 2)": 1.3, "Heart Attack": 1.3, "Hypertension": 1.1},
    },
    "high_creatinine": {
        "label":  "High Creatinine (Kidney)",
        "icon":   "🫘",
        "color":  "#E67E22",
        "boosts": {"Hypertension": 1.3, "Diabetes (Type 2)": 1.2, "Anemia": 1.2},
    },
    "high_liver_enzymes": {
        "label":  "High Liver Enzymes (SGOT/SGPT)",
        "icon":   "🫁",
        "color":  "#E67E22",
        "boosts": {"Typhoid": 1.5, "Malaria": 1.4, "Dengue Fever": 1.3,
                   "Food Poisoning": 1.3},
    },
    "high_tsh": {
        "label":  "High TSH (Hypothyroidism)",
        "icon":   "🦋",
        "color":  "#F7B731",
        "boosts": {"Anemia": 1.3, "Hypertension": 1.1},
    },
    "low_tsh": {
        "label":  "Low TSH (Hyperthyroidism)",
        "icon":   "🦋",
        "color":  "#F7B731",
        "boosts": {"Hypertension": 1.2, "Heart Attack": 1.1},
    },
    "high_esr_crp": {
        "label":  "Elevated ESR / CRP (Inflammation)",
        "icon":   "🔬",
        "color":  "#E67E22",
        "boosts": {"Pneumonia": 1.4, "Typhoid": 1.3, "Dengue Fever": 1.2,
                   "COVID-19": 1.3, "Malaria": 1.2},
    },
    "high_uric_acid": {
        "label":  "High Uric Acid",
        "icon":   "🔬",
        "color":  "#F7B731",
        "boosts": {"Hypertension": 1.1, "Diabetes (Type 2)": 1.1},
    },

    # ── Radiology / Imaging flags ─────────────────────────────────────────────
    "pneumonia_consolidation": {
        "label":  "Lung Consolidation (X-Ray/CT)",
        "icon":   "🫁",
        "color":  "#E63946",
        "boosts": {"Pneumonia": 2.0, "COVID-19": 1.6, "Asthma": 1.2},
    },
    "pleural_effusion": {
        "label":  "Pleural Effusion",
        "icon":   "🫁",
        "color":  "#E63946",
        "boosts": {"Pneumonia": 1.8, "Heart Attack": 1.4, "Malaria": 1.3},
    },
    "cardiomegaly": {
        "label":  "Cardiomegaly (Enlarged Heart)",
        "icon":   "❤️",
        "color":  "#E63946",
        "boosts": {"Heart Attack": 1.9, "Hypertension": 1.6, "Anemia": 1.3},
    },
    "splenomegaly": {
        "label":  "Splenomegaly (Enlarged Spleen)",
        "icon":   "🔬",
        "color":  "#E67E22",
        "boosts": {"Malaria": 1.9, "Dengue Fever": 1.6, "Typhoid": 1.5, "Anemia": 1.4},
    },
    "hepatomegaly": {
        "label":  "Hepatomegaly (Enlarged Liver)",
        "icon":   "🫁",
        "color":  "#E67E22",
        "boosts": {"Malaria": 1.6, "Typhoid": 1.5, "Dengue Fever": 1.4},
    },
    "fatty_liver": {
        "label":  "Fatty Liver (Sonography)",
        "icon":   "🫁",
        "color":  "#F7B731",
        "boosts": {"Diabetes (Type 2)": 1.4, "Hypertension": 1.2},
    },
    "kidney_stone": {
        "label":  "Kidney Stone (Sonography)",
        "icon":   "🫘",
        "color":  "#F7B731",
        "boosts": {"Hypertension": 1.1},
    },
    "gallstone": {
        "label":  "Gallstone (Sonography)",
        "icon":   "🔬",
        "color":  "#F7B731",
        "boosts": {"Food Poisoning": 1.2, "Appendicitis": 1.1},
    },

    # ── Catch-all ─────────────────────────────────────────────────────────────
    "normal_finding": {
        "label":  "Within Normal Range",
        "icon":   "✅",
        "color":  "#02C39A",
        "boosts": {},
    },
    "other": {
        "label":  "Other Finding",
        "icon":   "📋",
        "color":  "#64748B",
        "boosts": {},
    },
}

# ══════════════════════════════════════════════════════════════════════════════
#  NEW: Pre-existing conditions
# ══════════════════════════════════════════════════════════════════════════════

PREEXISTING_CONDITIONS = {
    "diabetes":       {"label": "Diabetes",           "icon": "🩸"},
    "hypertension":   {"label": "Hypertension",       "icon": "💊"},
    "heart_disease":  {"label": "Heart Disease",      "icon": "❤️"},
    "asthma":         {"label": "Asthma / COPD",      "icon": "💨"},
    "obesity":        {"label": "Obesity",            "icon": "⚖️"},
    "kidney_disease": {"label": "Kidney Disease",     "icon": "🫘"},
    "liver_disease":  {"label": "Liver Disease",      "icon": "🫁"},
    "cancer":         {"label": "Cancer",             "icon": "🔬"},
    "hiv_aids":       {"label": "HIV / AIDS",         "icon": "🧬"},
    "thyroid":        {"label": "Thyroid Disorder",   "icon": "🦋"},
    "autoimmune":     {"label": "Autoimmune Disease", "icon": "🛡️"},
    "anemia":         {"label": "Anemia",             "icon": "🩺"},
    "stroke_history": {"label": "Stroke History",     "icon": "🧠"},
    "smoking":        {"label": "Smoking",            "icon": "🚬"},
}

# Risk boost multipliers: preexisting → disease
CONDITION_DISEASE_RISK = {
    "diabetes":       {"Diabetes (Type 2)": 2.5, "Hypertension": 1.4, "Heart Attack": 1.6,
                       "Pneumonia": 1.3, "COVID-19": 1.5, "Anemia": 1.2},
    "hypertension":   {"Hypertension": 2.0, "Heart Attack": 1.8, "Anemia": 1.1},
    "heart_disease":  {"Heart Attack": 2.5, "Hypertension": 1.5, "Anemia": 1.3,
                       "COVID-19": 1.4, "Pneumonia": 1.3},
    "asthma":         {"Asthma": 2.0, "Pneumonia": 1.4, "COVID-19": 1.3,
                       "Common Cold": 1.2, "Influenza (Flu)": 1.3},
    "obesity":        {"Diabetes (Type 2)": 1.5, "Hypertension": 1.4,
                       "Heart Attack": 1.3, "Asthma": 1.2},
    "kidney_disease": {"Hypertension": 1.4, "Anemia": 1.5, "Diabetes (Type 2)": 1.3},
    "liver_disease":  {"Anemia": 1.4, "Typhoid": 1.3, "Food Poisoning": 1.3},
    "cancer":         {"Pneumonia": 1.5, "COVID-19": 1.6, "Anemia": 1.5, "Influenza (Flu)": 1.4},
    "hiv_aids":       {"Pneumonia": 1.7, "COVID-19": 1.6, "Typhoid": 1.4, "Malaria": 1.3},
    "smoking":        {"Asthma": 1.5, "Pneumonia": 1.4, "COVID-19": 1.3,
                       "Hypertension": 1.2, "Heart Attack": 1.3},
    "anemia":         {"Anemia": 2.0},
    "stroke_history": {"Heart Attack": 1.8, "Hypertension": 1.5},
    "autoimmune":     {"Anemia": 1.3, "Pneumonia": 1.2, "COVID-19": 1.3},
    "thyroid":        {"Anemia": 1.2, "Diabetes (Type 2)": 1.1, "Hypertension": 1.1},
}

# ══════════════════════════════════════════════════════════════════════════════
#  NEW: Vitals reference ranges
# ══════════════════════════════════════════════════════════════════════════════

VITALS_RANGES = {
    "heart_rate": {
        "label": "Heart Rate", "unit": "bpm",
        "ranges": [
            {"label": "Critical Low",        "min": 0,   "max": 39,
                "status": "critical", "color": "#E63946"},
            {"label": "Low",                 "min": 40,  "max": 59,
                "status": "warning",  "color": "#F7B731"},
            {"label": "Normal",              "min": 60,  "max": 100,
                "status": "normal",   "color": "#02C39A"},
            {"label": "Elevated",            "min": 101, "max": 120,
                "status": "warning",  "color": "#F7B731"},
            {"label": "Critical High",       "min": 121, "max": 9999,
                "status": "critical", "color": "#E63946"},
        ],
    },
    "systolic_bp": {
        "label": "Systolic BP", "unit": "mmHg",
        "ranges": [
            {"label": "Hypotension",         "min": 0,   "max": 89,
                "status": "warning",  "color": "#F7B731"},
            {"label": "Normal",              "min": 90,  "max": 119,
                "status": "normal",   "color": "#02C39A"},
            {"label": "Elevated",            "min": 120, "max": 129,
                "status": "warning",  "color": "#F7B731"},
            {"label": "High – Stage 1",      "min": 130, "max": 139,
                "status": "warning",  "color": "#E67E22"},
            {"label": "High – Stage 2",      "min": 140, "max": 179,
                "status": "danger",   "color": "#E63946"},
            {"label": "Hypertensive Crisis", "min": 180, "max": 9999,
                "status": "critical", "color": "#E63946"},
        ],
    },
    "diastolic_bp": {
        "label": "Diastolic BP", "unit": "mmHg",
        "ranges": [
            {"label": "Hypotension",         "min": 0,   "max": 59,
                "status": "warning",  "color": "#F7B731"},
            {"label": "Normal",              "min": 60,  "max": 79,
                "status": "normal",   "color": "#02C39A"},
            {"label": "Elevated",            "min": 80,  "max": 89,
                "status": "warning",  "color": "#E67E22"},
            {"label": "High",                "min": 90,  "max": 119,
                "status": "danger",   "color": "#E63946"},
            {"label": "Hypertensive Crisis", "min": 120, "max": 9999,
                "status": "critical", "color": "#E63946"},
        ],
    },
    "temperature": {
        "label": "Body Temperature", "unit": "°F",
        "ranges": [
            {"label": "Hypothermia",         "min": 0,    "max": 95.9,
                "status": "critical", "color": "#E63946"},
            {"label": "Low Normal",          "min": 96.0, "max": 97.9,
                "status": "warning",  "color": "#F7B731"},
            {"label": "Normal",              "min": 98.0, "max": 99.0,
                "status": "normal",   "color": "#02C39A"},
            {"label": "Low Fever",           "min": 99.1, "max": 100.3,
                "status": "warning",  "color": "#F7B731"},
            {"label": "Fever",               "min": 100.4,
                "max": 103.0, "status": "danger",   "color": "#E67E22"},
            {"label": "High Fever",          "min": 103.1, "max": 9999,
                "status": "critical", "color": "#E63946"},
        ],
    },
    "spo2": {
        "label": "Oxygen Saturation", "unit": "%",
        "ranges": [
            {"label": "Critical",            "min": 0,   "max": 89,
                "status": "critical", "color": "#E63946"},
            {"label": "Low",                 "min": 90,  "max": 93,
                "status": "danger",   "color": "#E67E22"},
            {"label": "Below Normal",        "min": 94,  "max": 95,
                "status": "warning",  "color": "#F7B731"},
            {"label": "Normal",              "min": 96,  "max": 100,
                "status": "normal",   "color": "#02C39A"},
        ],
    },
    "respiratory_rate": {
        "label": "Respiratory Rate", "unit": "breaths/min",
        "ranges": [
            {"label": "Low",                 "min": 0,   "max": 11,
                "status": "warning",  "color": "#F7B731"},
            {"label": "Normal",              "min": 12,  "max": 20,
                "status": "normal",   "color": "#02C39A"},
            {"label": "Elevated",            "min": 21,  "max": 24,
                "status": "warning",  "color": "#E67E22"},
            {"label": "High",                "min": 25,  "max": 9999,
                "status": "critical", "color": "#E63946"},
        ],
    },
}

# Vitals that trigger immediate emergency alerts
# (vital_key, direction, threshold, message)
VITALS_EMERGENCY = [
    ("spo2",             "max", 89,
     "Oxygen saturation critically low (<90%). Risk of hypoxia — seek emergency care immediately."),
    ("spo2",             "max", 93,
     "Oxygen saturation below safe level (<94%). Monitor closely and consider immediate medical attention."),
    ("systolic_bp",      "min", 180,
     "Blood pressure dangerously high (hypertensive crisis ≥180). Call emergency services immediately."),
    ("diastolic_bp",     "min", 120,
     "Diastolic BP critically high (≥120 mmHg). Hypertensive crisis — seek emergency care now."),
    ("temperature",      "min", 103.1,
     "Dangerously high fever (>103°F / 39.5°C). Seek immediate medical attention."),
    ("heart_rate",       "max", 39,
     "Heart rate critically low (<40 bpm). Possible bradycardia — seek emergency care."),
    ("heart_rate",       "min", 150,
     "Heart rate dangerously high (>150 bpm). Seek emergency care immediately."),
    ("respiratory_rate", "min", 30,
     "Respiratory rate severely elevated (≥30). Possible respiratory distress — seek emergency care."),
]
