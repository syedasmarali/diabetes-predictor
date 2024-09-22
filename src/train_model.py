from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from data_processing import load_data, preprocess_data, get_project_root
import os

# Load the processed dataframe
df = load_data()
df = preprocess_data(df)

# Prepare the features and target variable
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Print X columns to see the order of trained independent variables
print(X.columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Model Accuracy: {accuracy * 100:.2f}%")

# Save the model to a file using joblib
project_root = get_project_root()
model_dir = os.path.join(project_root, 'model')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'logistic_regression_model.pkl')
joblib.dump(model, model_path)

print("Model has been trained and saved!")