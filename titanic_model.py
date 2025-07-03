import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Titanic dataset
titanic = sns.load_dataset("titanic")

# Drop missing data
titanic = titanic.dropna(subset=['age', 'embarked'])

# Select features and target
X = titanic[['pclass', 'sex', 'age', 'sibsp', 'parch', 'embarked']]
y = titanic['survived']

# Convert categorical features
X = pd.get_dummies(X, columns=['sex', 'embarked'], drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


joblib.dump(model, "titanic_model.pkl")
