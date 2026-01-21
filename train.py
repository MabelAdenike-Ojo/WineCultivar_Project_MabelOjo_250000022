import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# 1. LOAD DATA (Make sure wine.csv is in your folder!)
# df = pd.read_csv('wine.csv')
# ... (insert your X_train, y_train, and scaler logic here) ...

# 2. TRAIN (Using your SVM as an example since it's your best)
svm = SVC(probability=True) # probability=True is helpful for web apps
svm.fit(X_train, y_train)

# 3. SAVE (No more "files.download" needed!)
joblib.dump(svm, 'wine_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and Scaler saved locally!")