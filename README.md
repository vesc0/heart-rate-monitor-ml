# Stress detection from heart-rate variability

Trains the stress classifier used by the Heart Rate Monitor app. Input is a 60-second window of beat-to-beat intervals — the only thing a phone camera can measure, and the output is a stress probability.

Data: [WESAD](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/)

## Contents
- [Results](#results)
- [Preprocessing](#preprocessing)
- [Usage](#usage)
- [Layout](#layout)
- [Limitations](#limitations)

## Results

Leave-One-Subject-Out cross-validation, so every score is on a person the model has never seen. 1396 windows, 22.3% stress.

| Model | ROC AUC | Stress recall | Macro F1 | Accuracy |
|---|---|---|---|---|
| **Random Forest** | **0.909** | **0.788** | **0.803** | **0.852** |
| Extra Trees | 0.907 | 0.791 | 0.778 | 0.828 |
| SVM (RBF) | 0.887 | 0.579 | 0.786 | 0.865 |
| Logistic Regression | 0.866 | 0.768 | 0.776 | 0.829 |

Predicting "never stressed" would score 0.777 accuracy, so accuracy alone is a poor guide — ROC AUC and stress recall are more important metrics. Random k-fold is deliberately not reported: windows overlap by 50% and subjects would appear in both train and test, which inflates results without measuring generalization.

## Preprocessing

Labels and ECG both come from `SX.pkl`, which ships already time-aligned. Two other routes were tried and rejected:

- **`SX_quest.csv` + E4 `IBI.csv`** — the wristband clock starts 18–26 minutes before the protocol, and the offset differs per subject. Applying the quest timings to it mislabels every window. This inverted the physiology: "stress" windows landed on quiet sitting and "baseline" on the sensor setup, so the model learned that a *low* heart rate meant stress.
- **Wrist PPG** — the stress task involves standing and speaking, and the motion makes HRV unusable there. The E4's own beat detector keeps 0–13% of beats during it.

Chest ECG is clean across all conditions. The resulting features show the expected response — under stress heart rate rises and variability falls:

| condition | mean HR | RMSSD | LF/HF |
|---|---|---|---|
| baseline | 73.0 | 49.6 | 2.9 |
| stress | 98.7 | 31.7 | 4.5 |
| amusement | 73.1 | 49.1 | 5.1 |
| meditation | 70.0 | 66.5 | 5.7 |

Demographics (age, height, weight, gender) are not used. Under LOSO they add 0.003 F1, and because 15 subjects have 15 unique demographic rows they act as a subject fingerprint that destabilizes predictions on identical heart data.

## Usage

```bash
pip install -r requirements.txt
python src/prepare.py --data-dir ~/Documents/WESAD   # -> data/hrv_features.csv
python src/train.py                                  # -> models/, outputs/
```

`models/` is gitignored. The API loads `all_artifacts.joblib`; copy it across after retraining.

## Layout

```
src/features.py   19 HRV features from RR intervals
src/prepare.py    WESAD -> data/hrv_features.csv
src/train.py      LOSO evaluation, model selection, artifacts and plots
notebooks/        exploratory figures
outputs/          generated plots
```

## Limitations

- 15 subjects, one lab protocol. Stress here means the Trier Social Stress Test, not everyday stress.
- Trained on chest ECG intervals. The app measures fingertip PPG, which is a cleaner signal than a moving wrist but still a different sensor — real-device accuracy is unverified.
- Not a medical device.
