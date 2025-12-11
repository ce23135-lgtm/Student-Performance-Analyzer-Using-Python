import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tabulate import tabulate

# 1) Generate Data for 10 Students Only
def create_student_data(num_students=10): 
    np.random.seed(42)
    
    data = {
        'student_id': range(1, num_students + 1),
        'study_hours': np.random.normal(15, 5, num_students).clip(5, 25),
        'attendance': np.random.normal(85, 10, num_students).clip(60, 100),
        'math_score': np.random.normal(75, 15, num_students).clip(0, 100),
        'science_score': np.random.normal(80, 12, num_students).clip(0, 100),
        'english_score': np.random.normal(70, 18, num_students).clip(0, 100),
        'previous_gpa': np.random.normal(3.0, 0.8, num_students).clip(1.0, 4.0),
        'extracurricular': np.random.randint(0, 3, num_students)
    }
    
    df = pd.DataFrame(data)
    df['average_score'] = df[['math_score', 'science_score', 'english_score']].mean(axis=1)
    df['status'] = np.where(df['average_score'] >= 60, 'Pass', 'Fail')

    return df

print("\n[1] Data Generation...")
students_df = create_student_data(30)
print(f"-> Generated {len(students_df)} student records.")

# 2) Descriptive Statistics
print("\n[2] Score Statistics:")
grades_cols = ['math_score', 'science_score', 'english_score', 'average_score']
grades_data = []

for col in grades_cols:
    grades_data.append([
        col,
        str(int(students_df[col].mean())),
        str(int(students_df[col].max())),
        str(int(students_df[col].min()))
    ])

print(tabulate(grades_data, 
               headers=["Subject", "Mean", "Max", "Min"], 
               tablefmt="fancy_grid"))

# Pass/Fail Summary
status_counts = students_df['status'].value_counts()
status_data = [
    ["Pass Count", status_counts.get('Pass', 0)],
    ["Fail Count", status_counts.get('Fail', 0)],
    ["Pass Rate", f"{int(status_counts.get('Pass', 0)/len(students_df)*100)}%"]
]

print("\nPass/Fail Summary:")
print(tabulate(status_data, 
               headers=["Metric", "Value"], 
               tablefmt="fancy_grid"))

# 3) Train Random Forest Model
print("\n[3] Training Random Forest Model...")

features = ['study_hours', 'attendance', 'previous_gpa', 'extracurricular']
X = students_df[features]
y = students_df['status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=5
)

model.fit(X_train, y_train)
print("-> Model trained successfully.")

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

model_results = [
    ["Training Accuracy", f"{int(accuracy_score(y_train, train_predictions)*100)}%"],
    ["Testing Accuracy", f"{int(accuracy_score(y_test, test_predictions)*100)}%"]
]

print(tabulate(model_results, 
               headers=["Accuracy Type", "Score"], 
               tablefmt="fancy_grid"))

# 4) Classification Report
print("\n[4] Classification Report:")
report = classification_report(y_test, test_predictions, target_names=['Fail', 'Pass'], output_dict=True)

report_data = []
for label in ['Fail', 'Pass']:
    metrics = report[label]
    report_data.append([
        label,
        str(int(metrics['precision'] * 100)),
        str(int(metrics['recall'] * 100)),
        str(int(metrics['f1-score'] * 100)),
        metrics['support']
    ])

print(tabulate(report_data, 
               headers=["Class", "Precision", "Recall", "F1-Score", "Support"], 
               tablefmt="fancy_grid"))

# 5) Visualization
print("\n[5] Generating Charts...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Student Performance Analysis', fontsize=14)

# Chart 1 – Score Distribution
ax1 = axes[0]
ax1.hist(students_df['average_score'], bins=5, edgecolor='black')
ax1.axvline(60, color='red', linestyle='--')
ax1.set_title('Average Score Distribution')

# Chart 2 – Feature Importance
importance = model.feature_importances_
fi = pd.DataFrame({
    'Feature': ['Study Hours', 'Attendance', 'Previous GPA', 'Extracurricular'],
    'Importance': importance
}).sort_values('Importance', ascending=True)

ax2 = axes[1]
ax2.barh(fi['Feature'], fi['Importance'])
ax2.set_title('Feature Importance')

plt.tight_layout()
plt.show()

print("\nProcess Completed Successfully.")


