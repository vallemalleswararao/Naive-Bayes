import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv("C:/Users/LENOVO/Documents/Python Projects/Weather Report/weatherHistory.csv")  # Replace 'weather_data.csv' with your dataset path

# Data Preprocessing
# We will use the 'Summary', 'Precip Type', 'Temperature (C)', 'Humidity', and 'Wind Speed (km/h)' as features.
X = data[['Summary', 'Precip Type', 'Temperature (C)', 'Humidity', 'Wind Speed (km/h)']]
y = data['Daily Summary']  # Target variable

# Convert categorical data to numerical using one-hot encoding
X = pd.get_dummies(X, columns=['Summary', 'Precip Type'], drop_first=True)

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Gaussian Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:\n', report)
print('Confusion Matrix:\n', conf_matrix)
