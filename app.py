"""
HealthMind AI – Flask Backend (v2)
Start:  python app.py
Open:   http://127.0.0.1:5000
"""

import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

from data.disease_symptoms import (
    SYMPTOMS, SYMPTOM_DISPLAY, SYMPTOM_CATEGORIES,
    DISEASES, EMERGENCY_COMBINATIONS, RISK_COLORS,
    PREEXISTING_CONDITIONS, CONDITION_DISEASE_RISK,
    VITALS_RANGES, VITALS_EMERGENCY,
)

app       = Flask(__name__)
MODEL_PATH = "model/healthmind_model.pkl"


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    if not os.path.exists(MODEL_PATH):
        print("No trained model found — training now (~10 seconds)…")
        from train_model import train_and_save
        return train_and_save(MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model_data = load_model()


# ══════════════════════════════════════════════════════════════════════════════
#  Helper – Age
# ══════════════════════════════════════════════════════════════════════════════

def age_group(age: int) -> str:
    if age <= 17:  return "child"
    if age <= 59:  return "adult"
    return "senior"


def apply_age_modifier(predictions: list, age: int | None) -> list:
    """Adjust prediction probabilities based on patient age."""
    if age is None:
        return predictions
    group = age_group(age)
    for p in predictions:
        info  = DISEASES.get(p["condition"], {})
        mults = info.get("age_risk", {})
        mult  = mults.get(group, 1.0)
        p["probability"]       = round(min(99.9, p["probability"] * mult), 1)
        p["age_modifier"]      = mult
        p["age_modifier_label"] = (
            f"+{round((mult-1)*100)}% risk (age group: {group})" if mult > 1.05
            else f"-{round((1-mult)*100)}% risk (age group: {group})" if mult < 0.95
            else f"Age has minimal effect on this condition"
        )
    predictions.sort(key=lambda x: x["probability"], reverse=True)
    return predictions


# ══════════════════════════════════════════════════════════════════════════════
#  Helper – Pre-existing conditions
# ══════════════════════════════════════════════════════════════════════════════

def apply_preexisting_modifiers(predictions: list, conditions: list) -> tuple[list, list]:
    """
    Boost prediction probabilities for diseases linked to the patient's
    pre-existing conditions.  Returns updated predictions + a list of
    human-readable impact notes.
    """
    impact_notes = []
    for p in predictions:
        disease    = p["condition"]
        total_mult = 1.0
        reasons    = []
        for cond_id in conditions:
            boosts = CONDITION_DISEASE_RISK.get(cond_id, {})
            mult   = boosts.get(disease, 1.0)
            if mult > 1.0:
                label = PREEXISTING_CONDITIONS[cond_id]["label"]
                reasons.append(f"{label} (+{round((mult-1)*100)}%)")
                total_mult *= mult
        if total_mult > 1.0:
            p["probability"] = round(min(99.9, p["probability"] * total_mult), 1)
            note = f"{disease} risk elevated due to: {', '.join(reasons)}"
            impact_notes.append(note)
        p["preexisting_boost"] = round(total_mult, 2)

    predictions.sort(key=lambda x: x["probability"], reverse=True)
    return predictions, impact_notes


# ══════════════════════════════════════════════════════════════════════════════
#  Helper – Vitals analysis
# ══════════════════════════════════════════════════════════════════════════════

def _get_vital_status(key: str, value: float) -> dict:
    """Return the label, status, and color for a given vital value."""
    ranges = VITALS_RANGES.get(key, {}).get("ranges", [])
    for r in ranges:
        if r["min"] <= value <= r["max"]:
            return {"label": r["label"], "status": r["status"], "color": r["color"]}
    return {"label": "Unknown", "status": "normal", "color": "#64748B"}


def analyze_vitals(vitals: dict) -> dict:
    """
    Analyse each provided vital sign.
    Returns:
      • summary   – per-vital status cards
      • alerts    – list of emergency/warning messages
      • risk_bump – extra points to add to the final risk score
    """
    summary    = {}
    alerts     = []
    risk_bump  = 0

    for key, value in vitals.items():
        if value is None:
            continue
        meta   = VITALS_RANGES.get(key, {})
        status = _get_vital_status(key, value)
        summary[key] = {
            "label":  meta.get("label", key),
            "unit":   meta.get("unit", ""),
            "value":  value,
            **status,
        }
        # Bump risk score for abnormal vitals
        if status["status"] == "warning":  risk_bump += 5
        if status["status"] == "danger":   risk_bump += 12
        if status["status"] == "critical": risk_bump += 20

    # Check vitals emergency thresholds
    for vital_key, direction, threshold, message in VITALS_EMERGENCY:
        val = vitals.get(vital_key)
        if val is None:
            continue
        triggered = (direction == "max" and val <= threshold) or \
                    (direction == "min" and val >= threshold)
        if triggered:
            alerts.append(message)

    return {"summary": summary, "alerts": alerts, "risk_bump": min(risk_bump, 30)}


# ══════════════════════════════════════════════════════════════════════════════
#  Helper – Risk score & label
# ══════════════════════════════════════════════════════════════════════════════

def calculate_risk_score(top: dict, n_symptoms: int,
                         vitals_bump: int = 0,
                         n_preexisting: int = 0,
                         age: int | None = None) -> int:
    prob         = top["probability"] / 100
    risk_map     = {"low": 0.25, "medium": 0.50, "high": 0.75, "critical": 1.0}
    risk_weight  = risk_map.get(top["risk_level"], 0.5)
    sym_weight   = min(n_symptoms / 8, 1.0)

    # Age modifier: children and seniors get a slight bump
    age_bump = 0
    if age is not None:
        if age <= 12 or age >= 70: age_bump = 8
        elif age <= 17 or age >= 60: age_bump = 4

    # Preexisting bump
    prex_bump = min(n_preexisting * 4, 16)

    score = (prob * 0.50 + risk_weight * 0.35 + sym_weight * 0.15) * 100
    score += vitals_bump + age_bump + prex_bump
    return min(100, round(score))


def get_risk_label(score: int) -> str:
    if score >= 75: return "Critical"
    if score >= 50: return "High"
    if score >= 25: return "Medium"
    return "Low"


# ══════════════════════════════════════════════════════════════════════════════
#  Helper – Symptom emergency check
# ══════════════════════════════════════════════════════════════════════════════

def check_symptom_emergency(symptoms: list) -> dict | None:
    for combo, message in EMERGENCY_COMBINATIONS:
        if all(s in symptoms for s in combo):
            return {"triggered": True, "message": message, "combo": combo}
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/meta")
def api_meta():
    """Return symptoms, categories, and preexisting condition options."""
    categories = {}
    for cat_name, sym_list in SYMPTOM_CATEGORIES.items():
        categories[cat_name] = [
            {"id": s, "label": SYMPTOM_DISPLAY[s]}
            for s in sym_list if s in SYMPTOM_DISPLAY
        ]
    preexisting = [
        {"id": k, "label": v["label"], "icon": v["icon"]}
        for k, v in PREEXISTING_CONDITIONS.items()
    ]
    return jsonify({
        "all_symptoms": [{"id": s, "label": SYMPTOM_DISPLAY[s]} for s in SYMPTOMS],
        "categories":   categories,
        "preexisting":  preexisting,
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Accepts:
      {
        "symptoms":       ["fever", "cough", ...],
        "age":            35,                         # optional int
        "gender":         "male",                     # optional
        "preexisting":    ["diabetes", "hypertension"], # optional list
        "vitals": {                                    # all optional
          "heart_rate":       72,
          "systolic_bp":      120,
          "diastolic_bp":     80,
          "temperature":      98.6,
          "spo2":             98,
          "respiratory_rate": 16
        }
      }
    """
    body       = request.get_json(force=True)
    selected   = body.get("symptoms",    [])
    age        = body.get("age",         None)
    gender     = body.get("gender",      None)
    preexisting= body.get("preexisting", [])
    vitals_raw = body.get("vitals",      {})

    # ── Validation ────────────────────────────────────────────────────────────
    if not selected:
        return jsonify({"error": "Please select at least one symptom."}), 400
    if len(selected) < 2:
        return jsonify({"error": "Please select at least 2 symptoms for a reliable prediction."}), 400

    # Parse vitals (cast strings to float, skip blanks)
    vitals = {}
    for k, v in vitals_raw.items():
        try:
            if v not in (None, ""):
                vitals[k] = float(v)
        except (ValueError, TypeError):
            pass

    # ── ML prediction ─────────────────────────────────────────────────────────
    feat    = np.array([[1 if s in selected else 0 for s in SYMPTOMS]], dtype=np.float32)
    clf     = model_data["model"]
    classes = model_data["classes"]
    proba   = clf.predict_proba(feat)[0]
    top_idx = np.argsort(proba)[::-1][:5]

    predictions = []
    for idx in top_idx:
        name = classes[idx]
        info = DISEASES.get(name, {})
        prob = round(float(proba[idx]) * 100, 1)
        if prob < 0.5:
            continue
        predictions.append({
            "condition":        name,
            "probability":      prob,
            "risk_level":       info.get("risk_level", "unknown"),
            "risk_color":       RISK_COLORS.get(info.get("risk_level",""), "#aaa"),
            "description":      info.get("description", ""),
            "recommendation":   info.get("recommendation", ""),
            "emergency":        info.get("emergency", False),
            "icon":             info.get("icon", "🏥"),
            "age_modifier":     1.0,
            "age_modifier_label": "",
            "preexisting_boost":  1.0,
        })

    if not predictions:
        return jsonify({"error": "Could not generate a prediction. Please try selecting more symptoms."}), 422

    # ── Apply age modifiers ───────────────────────────────────────────────────
    if age is not None:
        predictions = apply_age_modifier(predictions, int(age))

    # ── Apply pre-existing condition modifiers ────────────────────────────────
    preexisting_impact = []
    if preexisting:
        predictions, preexisting_impact = apply_preexisting_modifiers(predictions, preexisting)

    # Keep top 3 after re-sorting
    predictions = predictions[:3]

    # ── Vitals analysis ───────────────────────────────────────────────────────
    vitals_analysis = analyze_vitals(vitals)

    # ── Risk score (now uses all factors) ─────────────────────────────────────
    risk_score = calculate_risk_score(
        predictions[0],
        len(selected),
        vitals_bump    = vitals_analysis["risk_bump"],
        n_preexisting  = len(preexisting),
        age            = int(age) if age else None,
    )
    risk_label = get_risk_label(risk_score)

    # ── Emergency checks ──────────────────────────────────────────────────────
    sym_emergency    = check_symptom_emergency(selected)
    vitals_alerts    = vitals_analysis["alerts"]
    any_emergency    = bool(
        (sym_emergency and sym_emergency["triggered"])
        or vitals_alerts
        or any(p["emergency"] for p in predictions[:1])
    )
    emergency_messages = []
    if sym_emergency and sym_emergency["triggered"]:
        emergency_messages.append(sym_emergency["message"])
    emergency_messages.extend(vitals_alerts)

    # ── Explainability ────────────────────────────────────────────────────────
    importances   = clf.feature_importances_
    contributions = [
        {"symptom": SYMPTOM_DISPLAY[s], "contribution": round(float(importances[i]) * 100, 2)}
        for i, s in enumerate(SYMPTOMS) if s in selected
    ]
    contributions.sort(key=lambda x: x["contribution"], reverse=True)

    # ── Confidence label ──────────────────────────────────────────────────────
    top_prob = predictions[0]["probability"]
    if top_prob >= 70:
        confidence_label, confidence_color = "High Confidence",    "#02C39A"
    elif top_prob >= 40:
        confidence_label, confidence_color = "Moderate Confidence","#F7B731"
    else:
        confidence_label, confidence_color = "Low Confidence",     "#E67E22"

    # ── Patient profile summary ───────────────────────────────────────────────
    patient_profile = {
        "age":          age,
        "age_group":    age_group(int(age)) if age else None,
        "gender":       gender,
        "preexisting":  [
            {"id": c, "label": PREEXISTING_CONDITIONS[c]["label"],
             "icon": PREEXISTING_CONDITIONS[c]["icon"]}
            for c in preexisting if c in PREEXISTING_CONDITIONS
        ],
    }

    return jsonify({
        "predictions":         predictions,
        "risk_score":          risk_score,
        "risk_label":          risk_label,
        "emergency":           any_emergency,
        "emergency_messages":  emergency_messages,
        "symptom_count":       len(selected),
        "contributions":       contributions[:8],
        "confidence_label":    confidence_label,
        "confidence_color":    confidence_color,
        "vitals_summary":      vitals_analysis["summary"],
        "preexisting_impact":  preexisting_impact,
        "patient_profile":     patient_profile,
    })


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  🧠 HealthMind AI (v2) is starting…")
    print("  Open:  http://127.0.0.1:5000\n")
    app.run(debug=True)
