"""
HealthMind AI – Model Training Script
Run this once to train and save the ML model:  python train_model.py
"""

import numpy as np
import pickle
import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from data.disease_symptoms import SYMPTOMS, DISEASES


def generate_training_data():
    """
    Generates a synthetic labelled dataset from the disease-symptom knowledge
    base.  Each disease gets ~100 training samples with realistic variation
    (random subsets of core symptoms + small amounts of noise).
    """
    symptom_index = {s: i for i, s in enumerate(SYMPTOMS)}
    X, y = [], []

    for disease_name, disease_info in DISEASES.items():
        core = disease_info["symptoms"]

        for _ in range(100):
            vec = [0] * len(SYMPTOMS)

            # Include 65–100 % of the disease's core symptoms
            n_core = max(3, int(len(core) * np.random.uniform(0.65, 1.0)))
            chosen = np.random.choice(core, n_core, replace=False)
            for s in chosen:
                if s in symptom_index:
                    vec[symptom_index[s]] = 1

            # Add 0–2 random noise symptoms to simulate real-world messiness
            noise = np.random.randint(0, 3)
            for _ in range(noise):
                vec[np.random.randint(len(SYMPTOMS))] = 1

            X.append(vec)
            y.append(disease_name)

    return np.array(X, dtype=np.float32), np.array(y)


def train_and_save(model_path: str = "model/healthmind_model.pkl"):
    print("=" * 55)
    print("  HealthMind AI – Training ML Model")
    print("=" * 55)

    X, y = generate_training_data()
    print(f"  Training samples : {len(X)}")
    print(f"  Symptoms (features): {len(SYMPTOMS)}")
    print(f"  Disease classes  : {len(DISEASES)}")

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X, y)

    # Quick cross-validation estimate
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"\n  Cross-val accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model_data = {
        "model":    clf,
        "classes":  clf.classes_.tolist(),
        "symptoms": SYMPTOMS,
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"\n  ✅ Model saved → {model_path}")
    print("=" * 55)
    return model_data


if __name__ == "__main__":
    train_and_save()
