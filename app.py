from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import base64
from io import BytesIO

app = Flask(__name__)

# Load dataset
df = pd.read_csv("dataset.csv")
df["Registration_Number"] = df["Registration_Number"].astype(str).str.strip()

# Load model
with open("model/student_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

SUBJECTS = ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]
SUBJECT_LABELS = ["Subject 1", "Subject 2", "Subject 3", "Subject 4", "Subject 5", "Subject 6", "Subject 7"]

def get_performance_label(percentage):
    if percentage >= 75:
        return "Good"
    elif percentage >= 60:
        return "Average"
    else:
        return "Poor"

def get_grade(mark):
    if mark >= 90: return "O"
    elif mark >= 80: return "A+"
    elif mark >= 70: return "A"
    elif mark >= 60: return "B+"
    elif mark >= 50: return "B"
    elif mark >= 40: return "C"
    else: return "F"

def generate_chart(student_data):
    marks = [student_data[s] for s in SUBJECTS]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')

    colors = []
    for m in marks:
        if m >= 75: colors.append('#22d3ee')
        elif m >= 60: colors.append('#a78bfa')
        else: colors.append('#f87171')

    bars = ax.bar(SUBJECT_LABELS, marks, color=colors, width=0.55, zorder=3)

    # Value labels on bars
    for bar, mark in zip(bars, marks):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                str(mark), ha='center', va='bottom',
                color='white', fontsize=11, fontweight='bold',
                fontfamily='monospace')

    # Add max mark line
    ax.axhline(y=100, color='#334155', linewidth=1, linestyle='--', zorder=2)
    ax.axhline(y=50, color='#f87171', linewidth=0.8, linestyle=':', zorder=2, alpha=0.6)
    ax.axhline(y=75, color='#22d3ee', linewidth=0.8, linestyle=':', zorder=2, alpha=0.4)

    ax.set_ylim(0, 115)
    ax.set_ylabel("Marks", color='#94a3b8', fontsize=11)
    ax.set_xlabel("Subjects", color='#94a3b8', fontsize=11)
    ax.tick_params(colors='#94a3b8', labelsize=10)
    ax.spines['bottom'].set_color('#1e293b')
    ax.spines['left'].set_color('#1e293b')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, color='#1e293b', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    legend_elements = [
        mpatches.Patch(facecolor='#22d3ee', label='Good (≥75)'),
        mpatches.Patch(facecolor='#a78bfa', label='Average (60–74)'),
        mpatches.Patch(facecolor='#f87171', label='Poor (<60)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right',
              facecolor='#1e293b', edgecolor='#334155',
              labelcolor='#94a3b8', fontsize=9)

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor='#0f172a', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_b64

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    reg_no = request.form.get("reg_no", "").strip()

    row = df[df["Registration_Number"] == reg_no]
    if row.empty:
        return jsonify({"error": "Student not found. Please check the registration number."})

    student = row.iloc[0]
    marks = [int(student[s]) for s in SUBJECTS]
    percentage = float(student["Percentage"])
    performance = get_performance_label(percentage)

    feature_cols = SUBJECTS + ["Total", "Percentage"]
    features = pd.DataFrame([[student[s] for s in feature_cols]], columns=feature_cols)
    predicted_class = le.inverse_transform(model.predict(features))[0]

    grades = {s: get_grade(int(student[s])) for s in SUBJECTS}
    subject_data = [{"subject": SUBJECT_LABELS[i], "code": SUBJECTS[i],
                     "marks": marks[i], "grade": grades[SUBJECTS[i]]} for i in range(7)]

    chart_b64 = generate_chart(student)

    return jsonify({
        "reg_no": str(student["Registration_Number"]),
        "total": int(student["Total"]),
        "percentage": round(percentage, 2),
        "result": str(student["Result"]),
        "performance": performance,
        "ml_prediction": predicted_class,
        "subjects": subject_data,
        "chart": chart_b64
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
