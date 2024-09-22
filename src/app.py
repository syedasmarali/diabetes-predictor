import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from data_processing import get_project_root, load_data, preprocess_data
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the trained model
project_root = get_project_root()
model_dir = os.path.join(project_root, 'model')
model_path = os.path.join(model_dir, 'logistic_regression_model.pkl')
model = joblib.load(model_path)

# Set page layout to wide
st.set_page_config(layout="wide")

# Streamlit app interface
st.markdown("<h1 style='text-align: center;'>Diabetes Predictor</h1>", unsafe_allow_html=True)

# Add a divider
st.divider()

# Define a placeholder for the message
message_placeholder = st.empty()
message_placeholder.markdown(
    "<h4 style='text-align: center;'>Select parameters and press the predict button to predict diabetes percentage.</h4>",
    unsafe_allow_html=True
)

# User input for prediction
st.sidebar.subheader('Select parameters for predicting the diabetes in a patient')

# Age input
age = st.sidebar.number_input('Age', min_value=1, max_value=90, value=30)

# Hypertension inout
hypertension = st.sidebar.radio('Has Patient Hypertension?', ['Yes', 'No'])

# Heart disease input
heart_disease = st.sidebar.radio('Has Patient Heart Disease?', ['Yes', 'No'])

# BMI input
bmi = st.sidebar.number_input('Patient BMI', min_value=10, max_value=95, value=32)

# HBA1C input
hba1c = st.sidebar.number_input('HBA1C Level', min_value=3.5, max_value=9.0, value=5.5, step=0.1, format="%.1f")

# Blood glucose input
blood_glucose = st.sidebar.number_input('Blood Glucose Level', min_value=80, max_value=500, value=120)

# Smoking history input
smoking_history = st.sidebar.selectbox("Smoking History", ["No Info", "Never Smoked", "Former Smoker",
                                                           "Currenlty Smoking", "Not Currently Smoking",
                                                           "Ever Smoked"])

# Gender input
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])

# Button to predict dropout
if st.sidebar.button('Predict'):
    # Clear the message
    message_placeholder.empty()

    # Encode inputs
    hypertension = 1 if hypertension == 'Yes' else 0
    heart_disease = 1 if heart_disease == 'Yes' else 0
    smoking_history_current = 1 if smoking_history == 'Currenlty Smoking' else 0
    smoking_history_ever = 1 if smoking_history == 'Ever Smoked' else 0
    smoking_history_former = 1 if smoking_history == 'Former Smoker' else 0
    smoking_history_never = 1 if smoking_history == 'Never Smoked' else 0
    smoking_history_not_current = 1 if smoking_history == 'Not Currently Smoking' else 0
    gender_male = 1 if gender == 'Male' else 0
    gender_other = 1 if gender == 'Other' else 0

    # Dataframe for the patient input (order should be same as the trained dataframe check X.columns)
    patient = pd.DataFrame([{
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'bmi': bmi,
        'HbA1c_level': hba1c,
        'blood_glucose_level': blood_glucose,
        'smoking_history_current': smoking_history_current,
        'smoking_history_ever': smoking_history_ever,
        'smoking_history_former': smoking_history_former,
        'smoking_history_never': smoking_history_never,
        'smoking_history_not current': smoking_history_not_current,
        'gender_Male': gender_male,
        'gender_Other': gender_other
    }])

    # Predict the diabetes probability
    diabetes_prob = model.predict_proba(patient)[0]

    # Convert probabilities to percentage and round to the nearest whole number
    prob_not_diabetic = round(diabetes_prob[0] * 100)
    prob_diabetic = round(diabetes_prob[1] * 100)

    # Display results
    st.markdown(
        f"<div style='text-align:center;'>Probability of NOT Diabetic: <strong style='color:green; font-size:20px;'>{prob_not_diabetic}%</strong></div>",
        unsafe_allow_html=True)
    st.markdown(
        f"<div style='text-align:center;'>Probability of Diabetic: <strong style='color:red; font-size:20px;'>{prob_diabetic}%</strong></div>",
        unsafe_allow_html=True)

    # Display the final prediction
    if diabetes_prob[1] > 0.5:
        st.error("The patient is predicted to be diabetic.")
    else:
        st.success("The patient is predicted NOT to be diabetic.")

# Add a divider
st.divider()

# Loading df for plots
df = load_data()
df = preprocess_data(df)
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# --- Model Performance Visualization ---
st.markdown("<h3 style='text-align: center;'>Model Performance", unsafe_allow_html=True)

# Columns for model performance charts
col1, col2, col3 = st.columns(3)  # Create two columns

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Predict on the test data using the pretrained model
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=['Not Diabetic', 'Diabetic'])

# Display the model accuracy
accuracy = accuracy_score(y_test, y_pred)
with col1:
    st.markdown("<h5 style='text-align: center;'>Model Accuracy", unsafe_allow_html=True)
    st.markdown(
        f"<div style='text-align:center;'><strong style='color:green; font-size:20px;'>{accuracy * 100:.2f}%</strong></div>",
        unsafe_allow_html=True)

# Plot the confusion matrix
with col2:
    fig_cm, ax_cm = plt.subplots()
    ax_cm.set_title('Confusion Matrix')
    cmd.plot(ax=ax_cm)
    st.pyplot(fig_cm)

# ROC Curve
with col3:
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

# Add a divider
st.divider()

# Columns charts
col1, col2 = st.columns(2)  # Create two columns

# Correlation between user selected variables
with col1:
    st.sidebar.header('Select Variables for Correlation Plot')
    columns = df.columns.tolist()
    variable_1 = st.sidebar.selectbox('Select the first variable', columns)
    variable_2 = st.sidebar.selectbox('Select the second variable', columns)
    if variable_1 != variable_2:
        # Plotting the scatter plot with regression line
        st.markdown(f"<h4 style='text-align: center;'>Correlation Between {variable_1} & {variable_2}</h4>",
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 5))  # Adjust the width and height as needed
        sns.regplot(x=df[variable_1], y=df[variable_2], ax=ax, scatter=False, line_kws={'color': 'red'})
        ax.set_xlabel(variable_1)
        ax.set_ylabel(variable_2)
        st.pyplot(fig)
    else:
        st.write("Please select different variables for the scatter plot.")

# Plot the heatmap
with col2:
    st.markdown(f"<h4 style='text-align: center;'>Correlation Heatmap</h4>",
                unsafe_allow_html=True)
    fig, ax = plt.subplots()
    corr = df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
    st.pyplot(fig)