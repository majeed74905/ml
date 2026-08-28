import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv(r"C:\Users\majeed\Downloads\dataset\insurance_data.csv")

# Display first 5 rows
print(df.head())

# Scatter plot
plt.scatter(df.age, df.bought_insurance, marker='+', color='red')
plt.xlabel("Age")
plt.ylabel("Bought Insurance")
plt.title("Insurance Purchase by Age")
plt.show()

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    df[['age']],
    df['bought_insurance'],
    train_size=0.8,
    random_state=42
)

# Create model
model = LogisticRegression()

# Train model
model.fit(x_train, y_train)

# Predict
y_predicted = model.predict(x_test)

# Prediction probabilities
print("Prediction Probabilities:")
print(model.predict_proba(x_test))

# Accuracy
accuracy = model.score(x_test, y_test)
print("\nAccuracy:", accuracy)

# Classification Report
classification_rep = classification_report(y_test, y_predicted)
print("\nClassification Report:")
print(classification_rep)
