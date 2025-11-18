#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime

DATASET_PATH = Path("/Users/rcabello/Documents/GitHub/LedgerAI/medical_sft_dataset_enhanced.json")
BACKUP_PATH = DATASET_PATH.with_suffix(".backup." + datetime.utcnow().strftime("%Y%m%d%H%M%S") + ".json")

def load_dataset(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list of conversations.")
    return data

def save_dataset(path: Path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def make_scoring_example(title: str, chief_complaint: str, element: str, answer: str, deltas: dict):
    """
    Create a compact scoring example that teaches JSON-only scoring behavior for a specific OLD CARTS element.
    The assistant output is JSON-only mapping condition -> delta.
    """
    system = (
        "You are a medical expert with extensive training in clinical reasoning.\n"
        "Return ONLY valid JSON with condition deltas for scoring in the exact format:\n"
        "{\"Condition A\": 0.2, \"Condition B\": -0.1}\n"
        "No explanations, no extra text. Values must be between -0.3 and +0.3."
    )
    user = (
        f"Chief complaint: {chief_complaint}\n"
        f"OLD CARTS element: {element}\n"
        f"Patient's answer: '{answer}'\n"
        "Evaluate how this answer affects the likelihood of the listed conditions.\n"
        "Return ONLY JSON mapping condition to score change (delta)."
    )
    assistant = json.dumps(deltas, ensure_ascii=False)
    return {
        "meta": {"title": title, "type": "json_scoring_example"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }

def main():
    data = load_dataset(DATASET_PATH)
    # Backup
    save_dataset(BACKUP_PATH, data)

    examples = []
    # Character: burning (GERD positive, cardiac negative)
    examples.append(make_scoring_example(
        title="GERD character burning upscore; cardiac downscore",
        chief_complaint="chest pain",
        element="character",
        answer="burning",
        deltas={
            "Gastroesophageal Reflux Disease (GERD)": 0.2,
            "Acute Myocardial Infarction (Heart Attack)": -0.1,
            "Unstable Angina": -0.1,
            "Pericarditis": -0.1,
            "Aortic Dissection": -0.2
        }
    ))
    # Aggravating: lying down after meals (GERD strong positive, cardiac negative)
    examples.append(make_scoring_example(
        title="GERD worse lying down after meals strong upscore",
        chief_complaint="chest pain",
        element="aggravating",
        answer="laying down after meals",
        deltas={
            "Gastroesophageal Reflux Disease (GERD)": 0.3,
            "Acute Myocardial Infarction (Heart Attack)": -0.2,
            "Unstable Angina": -0.2,
            "Pericarditis": -0.1,
            "Aortic Dissection": -0.2
        }
    ))
    # Relieving: sitting up (GERD positive, cardiac neutral/negative)
    examples.append(make_scoring_example(
        title="GERD better with sitting up upscore",
        chief_complaint="chest pain",
        element="relieving",
        answer="sitting up",
        deltas={
            "Gastroesophageal Reflux Disease (GERD)": 0.2,
            "Acute Myocardial Infarction (Heart Attack)": -0.1,
            "Unstable Angina": -0.1,
            "Pericarditis": 0.0,
            "Aortic Dissection": -0.1
        }
    ))
    # Relieving: antacids (GERD positive, cardiac negative)
    examples.append(make_scoring_example(
        title="GERD better with antacids upscore",
        chief_complaint="chest pain",
        element="relieving",
        answer="antacids help",
        deltas={
            "Gastroesophageal Reflux Disease (GERD)": 0.3,
            "Acute Myocardial Infarction (Heart Attack)": -0.2,
            "Unstable Angina": -0.2,
            "Pericarditis": -0.1
        }
    ))
    # Aggravating: exertion (cardiac positive, GERD negative)
    examples.append(make_scoring_example(
        title="Cardiac worse with exertion upscore; GERD downscore",
        chief_complaint="chest pain",
        element="aggravating",
        answer="worse with exertion",
        deltas={
            "Acute Myocardial Infarction (Heart Attack)": 0.2,
            "Unstable Angina": 0.2,
            "Stable Angina": 0.2,
            "Gastroesophageal Reflux Disease (GERD)": -0.2,
            "Pericarditis": 0.0
        }
    ))
    # Character: pressure/heaviness (cardiac positive, GERD negative)
    examples.append(make_scoring_example(
        title="Cardiac pressure/heaviness upscore; GERD downscore",
        chief_complaint="chest pain",
        element="character",
        answer="pressure and heaviness",
        deltas={
            "Acute Myocardial Infarction (Heart Attack)": 0.2,
            "Unstable Angina": 0.2,
            "Stable Angina": 0.2,
            "Gastroesophageal Reflux Disease (GERD)": -0.2,
            "Aortic Dissection": 0.0
        }
    ))
    # Aggravating: deep breath/cough (pleuritic positive, others negative)
    examples.append(make_scoring_example(
        title="Pleuritic pain worse with deep breath upscore",
        chief_complaint="chest pain",
        element="aggravating",
        answer="worse with deep breaths and coughing",
        deltas={
            "Pleuritis": 0.3,
            "Pneumonia": 0.2,
            "Pulmonary Embolism": 0.1,
            "Acute Myocardial Infarction (Heart Attack)": -0.2,
            "Gastroesophageal Reflux Disease (GERD)": -0.1
        }
    ))
    # Radiation: to the throat after meals (GERD positive, aortic dissection negative)
    examples.append(make_scoring_example(
        title="GERD radiation to throat after meals upscore",
        chief_complaint="chest pain",
        element="radiation",
        answer="travels up to my throat after meals",
        deltas={
            "Gastroesophageal Reflux Disease (GERD)": 0.2,
            "Acute Myocardial Infarction (Heart Attack)": 0.0,
            "Unstable Angina": 0.0,
            "Aortic Dissection": -0.2
        }
    ))

    data.extend(examples)
    save_dataset(DATASET_PATH, data)
    print(f"Added {len(examples)} targeted JSON-scoring examples.")
    print(f"Backup saved to: {BACKUP_PATH}")
    print(f"Updated dataset: {DATASET_PATH}")

if __name__ == "__main__":
    main()


