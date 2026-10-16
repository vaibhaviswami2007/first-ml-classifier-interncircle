# First Machine Learning Classifier - Vaibhavi Swami | IC-2026-3303
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error
import seaborn as sns, matplotlib.pyplot as plt

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# 1 Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# 2 Model training with Scikit-Learn
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 3 Evaluation metrics
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\nAccuracy: {acc*100:.2f}%")
print(f"MSE: {mse:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))
print("\nConfusion Matrix:\n", cm)

# Plot Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix - Decision Tree')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout(); plt.show()
