# Student Performance Analysis System

A machine learning web application built with Python, Flask, and scikit-learn.

## Project Structure

```
student_performance/
├── app.py                  # Flask web application
├── model_training.py       # ML model training script
├── dataset.csv             # Student dataset (40 records)
├── model/
│   ├── student_model.pkl   # Trained Random Forest model
│   └── label_encoder.pkl   # Label encoder
├── templates/
│   └── index.html          # Frontend UI
├── static/
│   ├── css/
│   ├── js/
│   └── graphs/
└── README.md
```

## Setup & Run

### 1. Install dependencies
```bash
pip install flask scikit-learn matplotlib pandas numpy
```

### 2. Train the model
```bash
python model_training.py
```

### 3. Run the Flask app
```bash
python app.py
```

### 4. Open browser
Visit: http://localhost:5000

## Features
- Search student by Registration Number
- Display G1–G7 marks with grade badges
- Bar chart visualization (Matplotlib)
- ML-powered performance prediction (Good / Average / Poor)
- Pass/Fail classification
- Validation for unknown registration numbers

## ML Model
- Algorithm: Random Forest Classifier (100 trees)
- Accuracy: ~87.5%
- Classes: Good (≥75%), Average (60–74%), Poor (<60%)

## Sample Registration Numbers
- U05CG22A0001, U05CG22A0015, U05CG22A0017, U05CG22A0030, U05SK22A0051
