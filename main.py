import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

print("Loading dataset...")
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
print("Dataset loaded successfully!")
print(df.head())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining the Logistic Regression model...")
model = LogisticRegression(max_iter=500)
model.fit(X_train_scaled, y_train)
print("Model training complete!")

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/diabetes_model.joblib")
joblib.dump(scaler, "model/scaler.joblib")
print("Model saved inside 'model' folder.")

print("\nEvaluating model performance on test data...")
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nModel Evaluation Results:")
print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

sample = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]  
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)
print("\nSample Prediction (1=Diabetes, 0=No Diabetes):", int(prediction[0]))

print("\n Project Completed Successfully!")

