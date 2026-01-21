import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# =========================
# 1. Load Dataset
# =========================
data = load_wine()
wine_df = pd.DataFrame(data.data, columns=data.feature_names)
wine_df['cultivar'] = data.target

# =========================
# 2. Feature Selection (ONLY 6)
# =========================
selected_features = [
    'alcohol',
    'malic_acid',
    'ash',
    'magnesium',
    'total_phenols',
    'flavanoids'
]

X = wine_df[selected_features]
y = wine_df['cultivar']

# =========================
# 3. Check Missing Values
# =========================
print("Missing values:\n", X.isnull().sum())

# =========================
# 4. Feature Scaling (MANDATORY)
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 5. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 6. Model Training (SVM)
# =========================
model = SVC(kernel='rbf', probability=True)
model.fit(X_train, y_train)

# =========================
# 7. Model Evaluation
# =========================
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# =========================
# 8. Save Model and Scaler
# =========================
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/wine_cultivar_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("Model and scaler saved successfully.")