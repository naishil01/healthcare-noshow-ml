# Healthcare No-Show Prediction

ML pipeline to predict whether patients will miss their healthcare appointments — built on a real-world dataset of 110,000+ patient records from Brazil. Includes class imbalance analysis, multiple model comparisons, and an interactive Power BI dashboard for clinic staff.

---

## The core problem

Predicting no-shows sounds simple until you look at the data. 80% of patients show up, 20% don't. A naive model can hit **80% accuracy by predicting everyone shows up** — without learning anything useful. This project identifies that failure, fixes it, and builds a model that actually catches no-shows.

---

## Model performance

| Model | Accuracy | No-show Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|
| Decision Tree (baseline) | 80% | ~0% | 0.00 | 0.58 |
| Decision Tree + `class_weight='balanced'` | 66% | 73% | 0.46 | 0.68 |
| Decision Tree + SMOTE oversampling | 52% | 85% | 0.42 | 0.64 |
| **Random Forest + `class_weight='balanced'`** | **67%** | **71%** | **0.46** | **0.77** |

**Why accuracy drops — and why that's correct:**
The baseline model predicted "show up" for every patient and scored 80%. It caught 9 no-shows out of 4,391. The fixed model catches 3,168 — 350x more — at the cost of some accuracy. For a clinic, catching real no-shows is the entire point of building this model.

**Key metric: No-show Recall and ROC-AUC, not accuracy.**

---

## What I found and fixed

**Problem:** Severe class imbalance (80:20 ratio). The Decision Tree learned the lazy strategy — predict "show up" every time — and scored well on accuracy while being completely useless for no-show detection (F1 = 0.00, Recall ≈ 0%).

**Fix 1 — `class_weight='balanced'`:** One parameter change. Tells the model to treat each no-show as 4x more important during training. No-show recall jumped from ~0% to 73%.

**Fix 2 — SMOTE oversampling:** Synthetically generates no-show examples so the model trains on a balanced 50/50 dataset. Highest recall (85%) but more false alarms.

**Fix 3 — Random Forest + balanced (final model):** Ensemble of 100 trees with balanced class weights. Best ROC-AUC (0.77), strongest overall generalisation. This is the production-ready model.

---

## Project workflow

**1. Data cleaning and feature engineering**
- Imported 110,000+ patient records from KaggleV2 dataset
- Converted scheduled and appointment dates; engineered `AppointmentWeekday` and `ScheduledToAppointmentDays`
- Filtered invalid ages; mapped target column to binary (0 = showed up, 1 = no-show)

**2. Exploratory data analysis**
- Identified 80:20 class distribution as the core challenge
- Analysed feature impact: SMS reminders, waiting days, age, scholarship status
- Found that longer wait times and no SMS reminder strongly correlate with no-shows

**3. Baseline model (broken)**
- Trained `DecisionTreeClassifier(max_depth=5)`
- 80% accuracy, 0% no-show recall — misleading metric exposed

**4. Imbalance correction**
- Applied `class_weight='balanced'` to Decision Tree
- Applied SMOTE oversampling from `imbalanced-learn`
- Compared all three approaches on Recall, F1, and ROC-AUC

**5. Final model**
- `RandomForestClassifier(n_estimators=100, class_weight='balanced')`
- ROC-AUC: 0.77 | No-show Recall: 71%
- Feature importance extracted — `ScheduledToAppointmentDays` and `Age` are strongest predictors

**6. Power BI dashboard**
- Interactive dashboard for clinic staff to view high-risk appointment slots
- Integrates model predictions with patient demographics and scheduling data

---

## Tech stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| ML | Scikit-learn, imbalanced-learn (SMOTE) |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Power BI |
| Dataset | [KaggleV2 Medical Appointment No Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments) |

---

## Files

| File | Description |
|---|---|
| `noshow_fixed.py` | Full pipeline — baseline, all fixes, comparison summary |
| `Data analysis project.pdf` | Original analysis and EDA |
| `Dashboard.pbix` | Power BI dashboard file |

---

## Key takeaways

- **Accuracy is a misleading metric on imbalanced datasets.** Always check class distribution before choosing your evaluation metric.
- **F1-score and ROC-AUC** are the right metrics when false negatives have real-world cost.
- **`class_weight='balanced'`** is the fastest, most practical fix — one parameter, significant impact.
- **Random Forest outperforms a single Decision Tree** on generalisation (ROC-AUC 0.77 vs 0.68) while maintaining similar recall.
