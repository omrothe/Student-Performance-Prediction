# Week 6 Final AI Assignment: Student Performance Prediction

## Project overview
This project demonstrates an end-to-end supervised machine-learning workflow to estimate a student's final score from study habits, attendance, previous score, and selected categorical indicators.

**Important:** `data/student_performance.csv` is a synthetic dataset generated for this assignment. It contains no real student records and should not be used for grading, admissions, or other consequential decisions.

## Files
- `Student_Performance_Prediction.ipynb` — step-by-step Jupyter Notebook
- `src/student_performance_prediction.py` — runnable Python script
- `data/student_performance.csv` — raw synthetic dataset (1,012 rows before cleaning)
- `data/student_performance_cleaned.csv` — cleaned dataset
- `outputs/figures/` — generated visualizations
- `outputs/test_predictions.csv` — held-out test predictions
- `outputs/evaluation_metrics.csv` — regression evaluation metrics
- `report/Week6_Student_Performance_Prediction_Report.pdf` — submission report
- `report/Week6_Student_Performance_Prediction_Report.docx` — editable report source

## Dataset columns
- `hours_studied`: estimated study hours
- `attendance_pct`: attendance percentage
- `previous_score`: previous assessment score
- `sleep_hours`: typical sleep hours
- `tutoring`: tutoring participation (Yes/No)
- `parental_education`: category (Low/Medium/High)
- `extracurricular`: participation (Yes/No)
- `internet_access`: access (Yes/No)
- `final_score`: target score from 0 to 100

## Technologies
Python, Pandas, NumPy, Matplotlib, scikit-learn, Jupyter Notebook, ReportLab.

## Setup
1. Install Python 3.10 or newer.
2. Open this folder in VS Code.
3. Create a virtual environment:
   - Windows: `python -m venv .venv` then `.venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
4. Install packages: `pip install -r requirements.txt`
5. Open `Student_Performance_Prediction.ipynb` and run all cells from the project root.

## Run as a Python script
`python src/student_performance_prediction.py`

The script recreates charts and outputs under `outputs/`.

## Model and evaluation
Random Forest Regression, with an 80/20 train/test split. Metrics from the included reproducible run:
- MAE: 6.206
- MSE: 61.778
- RMSE: 7.860
- R²: 0.548

These results are based on a synthetic dataset and a single split. They do not establish real-world predictive validity.

## Project workflow
Problem definition → data collection/generation → cleaning → EDA → preprocessing → training → prediction → evaluation → conclusion.

## Academic submission checklist
- Review the code and run the notebook in your own environment.
- Add your name/institute details if required.
- Include the report and screenshots/figures in the submission.
- State clearly that the dataset is synthetic.
