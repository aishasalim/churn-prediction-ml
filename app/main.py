import streamlit as st
import pandas as pd
import pickle
import numpy as np
from openai import OpenAI
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

def create_gauge_chart(probability):
    if probability < 0.3:
        color = "green"
    elif probability < 0.6:
        color = "yellow"
    else:
        color = "red"

    fig = go.Figure(
        go.Indicator(mode="gauge+number",
                     value=probability * 100,
                     domain={
                         'x': [0, 1],
                         'y': [0, 1]
                     },
                     title={
                         'text': 'Churn Probability',
                         'font': {
                             'size': 24,
                             'color': 'white'
                         }
                     },
                     number={"font": {
                         'size': 40,
                         'color': 'white'
                     }},
                     gauge={
                         'axis': {
                             'range': [0, 100],
                             'tickwidth': 1,
                             'tickcolor': 'white'
                         },
                         'bar': {
                             'color': color
                         },
                         'bgcolor':
                         'rgba(0,0,0,0)',
                         'borderwidth':
                         2,
                         'bordercolor':
                         'white',
                         'steps': [{
                             'range': [0, 30],
                             'color': "rgba(0, 255, 0, 0.3)"
                         }, {
                             'range': [30, 60],
                             'color': "rgba(255, 255, 0, 0.3)"
                         }, {
                             'range': [60, 100],
                             'color': "rgba(255, 0, 0, 0.3)"
                         }],
                         'threshold': {
                             'line': {
                                 'color': "white",
                                 'width': 4
                             },
                             'thickness': 0.75,
                             'value': 100
                         }
                     }))

    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      font={'color': 'white'},
                      width=400,
                      height=300,
                      margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_model_probability_chart(probabilities):
    models = list(probabilities.keys())
    probs = list(probabilities.values())

    fig = go.Figure(data=[
        go.Bar(y=models,
               x=probs,
               orientation='h',
               text=[f'{p:.2%}' for p in probs],
               textposition='auto')
    ])

    fig.update_layout(title='Churn Probability by Model',
                      yaxis_title='Models',
                      xaxis_title="Probability",
                      xaxis=dict(tickformat='.0%', range=[0, 1]),
                      height=400,
                      margin=dict(l=20, r=20, t=40, b=20))
    return fig


client = OpenAI(
    base_url='https://api.groq.com/openai/v1',
    api_key=api_key)


def load_model(filename):
    # Get the absolute path of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the full path to the model
    model_path = os.path.join(current_dir, 'models', filename)
    
    # Verify the model file exists
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at path: {model_path}")
    
    # Load the model
    model = joblib.load(model_path)
    
    # Initialize variables
    model_name = os.path.splitext(os.path.basename(filename))[0]  # Extract model name without extension
    num_features = None
    
    # Check for scikit-learn models
    if hasattr(model, 'n_features_in_'):
        num_features = model.n_features_in_
    elif hasattr(model, 'coef_'):
        num_features = model.coef_.shape[1]
    
    # Check for XGBoost models
    if hasattr(model, 'get_booster'):
        booster = model.get_booster()
        feature_names = booster.feature_names
        if feature_names:
            num_features = len(feature_names)
        else:
            num_features = booster.num_features()
    
    print(f"Loaded model '{model_name}' expects {num_features} features.")
    return model

def load_data(filename):
    # Get the absolute path of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the full path to the data file
    data_path = os.path.join(current_dir, 'data', filename)
    
    # Verify the data file exists
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Data file not found at path: {data_path}")
    
    # Load the data
    df = pd.read_csv(data_path)
    return df


best_lr_model = load_model('best_lr_model.pkl')
stacking_model = load_model('stacking_model.pkl')
gbc_model = load_model('gbc_model.pkl')


# this function will prepare input data for the model
#  it will take user input data and make predictions with the models


def prepare_input(credit_score, location, gender, age, tenure, balance,
                  num_products, has_credit_card, is_active_member, estimated_salary):
    
    # Feature engineering: CLV, TenureAgeRatio, AgeGroup features
    clv = (balance + estimated_salary) / (age + 1)  # Example calculation for CLV
    tenure_age_ratio = tenure / (age + 1)

    age_group_middle_age = 1 if 30 <= age < 50 else 0
    age_group_senior = 1 if 50 <= age < 65 else 0
    age_group_elderly = 1 if age >= 65 else 0

    # Placeholder for additional missing features
    # Replace 'MissingFeature1', 'MissingFeature2', 'MissingFeature3' with actual feature names
    missing_feature1 = 0
    missing_feature2 = 0
    missing_feature3 = 0

    input_dict = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': int(has_credit_card),
        'IsActiveMember': int(is_active_member),
        'EstimatedSalary': estimated_salary,
        'Geography_France': 1 if location == "France" else 0,
        'Geography_Germany': 1 if location == "Germany" else 0,
        'Geography_Spain': 1 if location == "Spain" else 0,
        'Gender_Male': 1 if gender == "Male" else 0,
        'Gender_Female': 1 if gender == "Female" else 0,
        'CLV': clv,  # Adding CLV feature
        'TenureAgeRatio': tenure_age_ratio,  # Adding TenureAgeRatio feature
        'AgeGroup_MiddleAge': age_group_middle_age,  # Adding AgeGroup features
        'AgeGroup_Senior': age_group_senior,
        'AgeGroup_Elderly': age_group_elderly,
        'MissingFeature1': missing_feature1,  # Placeholder feature
        'MissingFeature2': missing_feature2,  # Placeholder feature
        'MissingFeature3': missing_feature3,  # Placeholder feature
    }

    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict


def make_predictions(input_df, input_dict):
    print("Input Features:", input_df.columns.tolist())
    print("Number of Features:", input_df.shape[1])

    probabilities = {
        'Logistic Regression': best_lr_model.predict_proba(input_df)[0][1],
        'Stacking Classifier': stacking_model.predict_proba(input_df)[0][1],
        'Gradient Boosting': gbc_model.predict_proba(input_df)[0][1]
    }
    
    avg_probability = np.mean(list(probabilities.values()))
    return avg_probability, probabilities


def explain_prediction(probability, input_dict, surname):
    prompt = f"""You are an expert data scientist at a bank, where you specialize in 
  interpreting and explaining predictions of machine learning models.

  Your machine learning model has predicted that a customer named {surname} has a 
  {round(probability * 100, 1)}% probability of churning, based on the information provided below.

  Here is the customer's information:
  {input_dict}

  Here are the machine learning model's top 10 most important features for predicting churn:

  Feature | Importance
  -----------------------
  NumOfProducts | 0.323888
  IsActiveMember | 0.164146
  Age | 0.109550
  Geography_Germany | 0.091373
  Balance | 0.052786
  Geography_France | 0.046463
  Gender_Female | 0.045283
  Geography_Spain | 0.036855
  CreditScore | 0.035005
  EstimatedSalary | 0.032655
  HasCrCard | 0.031940
  Tenure | 0.030054
  Gender_Male | 0.000000

  {pd.set_option('display.max_columns', None)}

  Here are summary statistics for churned customers:
  {df[df['Exited'] == 1].describe()}

  Here are summary statistics for non-churned customers:
  {df[df['Exited'] == 0].describe()}

  WORD RESTRICTION: 100-150 words!! IT IS VERY IMPORTANT. 

  - If the customer has over a 40% risk of churning, generate a 3 sentence explanation of why they are at risk of churning.
  - If the customer has less than a 40% risk of churning, generate a 3 sentence explanation of why they might not be at risk of churning.

  Your explanation should be based on the customer's information, the summary statistics of churned and non-churned customers, and the feature importances provided.

  Don't mention the probability of churning, or the machine learning model, or say anything like "Based on the machine learning model's prediction and top 10 most important features", just explain the prediction.
  """

    print("EXPLANATION PROMPT", prompt)

    raw_response = client.chat.completions.create(model="llama-3.1-8b-instant",
                                                  messages=[
                                                      {
                                                          "role": "user",
                                                          "content": prompt
                                                      },
                                                  ])

    return raw_response.choices[0].message.content


def generate_email(probability, input_dict, explanation, surname):
    prompt = f"""You are a manager at HS Bank. You are responsible for 
  ensuring customers stay with the bank and are incentivized with various offers.

    WORD RESTRICTION: 250-350 words!! IT IS VERY IMPORTANT. 

  You noticed a customer named {surname} has a {round(probability * 100, 1)}% probability of churning.

  Here is the customer's information:
  {input_dict}

  Here is some explanation as to why the customer might be at risk of churning:
  {explanation}

  Generate an email to the customer based on their information, asking them to stay if they are at risk of churning, or offering them incentives so that they become more loyal to the bank.
  Be specific about the incentives, and make sure to emphasize that the customer is not at risk of churning.
  Email should be straight forward, facts only 250-350 words restriction. 

  Make sure to list out a set of incentives to stay based on their information, in bullet point format. Don't ever mention the probability of churning, or the machine learning model to the customer.
  """

    raw_response = client.chat.completions.create(model="llama-3.1-8b-instant",
                                                  messages=[
                                                      {
                                                          "role": "user",
                                                          "content": prompt
                                                      },
                                                  ])

    print("\n\nEMAIL PROMPT", prompt)

    return raw_response.choices[0].message.content


# =========== UI ===========
st.title("Banking Analytics Dashboard")

# Create Tabs
tabs = st.tabs(["Customer Churn Prediction", "Fraud Analysis"])

# =======================
# Tab 1: Customer Churn Prediction
# =======================
with tabs[0]:
    df = load_data('churn.csv')

    customers = [
        f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()
    ]

    selected_customer_option = st.selectbox("Select a customer", customers)

    if selected_customer_option:
        selected_customer_id = int(selected_customer_option.split(" - ")[0])
        selected_customer_surname = selected_customer_option.split(" - ")[1]
        selected_customer = df.loc[df["CustomerId"] ==
                                selected_customer_id].iloc[0]

        # Prepare default input values from selected customer
        credit_score_default = int(selected_customer['CreditScore'])
        location_default = selected_customer['Geography']
        gender_default = selected_customer['Gender']
        age_default = int(selected_customer['Age'])
        tenure_default = int(selected_customer['Tenure'])
        balance_default = float(selected_customer['Balance'])
        num_products_default = int(selected_customer['NumOfProducts'])
        has_credit_card_default = bool(selected_customer['HasCrCard'])
        is_active_member_default = bool(selected_customer['IsActiveMember'])
        estimated_salary_default = float(selected_customer['EstimatedSalary'])

        # Prepare input data for prediction
        input_df, input_dict = prepare_input(
            credit_score_default, location_default, gender_default, age_default,
            tenure_default, balance_default, num_products_default,
            has_credit_card_default, is_active_member_default,
            estimated_salary_default)

        # Make predictions
        avg_probability, probabilities = make_predictions(input_df, input_dict)

        # Collect user inputs
        st.markdown("---")
        st.header("Customer Details")
        col1, col2 = st.columns(2)
        with col1:
            credit_score = st.number_input("Credit Score",
                                        min_value=300,
                                        max_value=850,
                                        value=credit_score_default)
            location = st.selectbox("Location", ["Spain", "France", "Germany"],
                                    index=["Spain", "France",
                                        "Germany"].index(location_default))
            gender = st.radio("Gender", ["Male", "Female"],
                            index=0 if gender_default == 'Male' else 1)
            age = st.number_input("Age",
                                min_value=18,
                                max_value=100,
                                value=age_default)
            tenure = st.number_input("Tenure (years)",
                                    min_value=0,
                                    max_value=50,
                                    value=tenure_default)
        with col2:
            balance = st.number_input("Balance",
                                    min_value=0.0,
                                    value=balance_default)
            num_products = st.number_input("Number of Products",
                                        min_value=1,
                                        max_value=10,
                                        value=num_products_default)
            has_credit_card = st.checkbox("Has Credit Card",
                                        value=has_credit_card_default)
            is_active_member = st.checkbox("Is Active Member",
                                        value=is_active_member_default)
            estimated_salary = st.number_input("Estimated Salary",
                                            min_value=0.0,
                                            value=estimated_salary_default)

        # Update predictions based on user inputs
        input_df, input_dict = prepare_input(credit_score, location, gender, age,
                                            tenure, balance, num_products,
                                            has_credit_card, is_active_member,
                                            estimated_salary)
        avg_probability, probabilities = make_predictions(input_df, input_dict)

        # Update the plots
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            fig = create_gauge_chart(avg_probability)
            st.plotly_chart(fig, use_container_width=True)
            st.write(
                f'The customer has a {avg_probability:.2%} probability of churning.'
            )
        with col2:
            fig_probs = create_model_probability_chart(probabilities)
            st.plotly_chart(fig_probs, use_container_width=True)

        # Explanation and email generation
        explanation = explain_prediction(avg_probability, input_dict,
                                        selected_customer["Surname"])
        st.markdown("---")
        st.subheader("Explanation of Prediction")
        st.markdown(explanation)
        email = generate_email(avg_probability, input_dict, explanation,
                            selected_customer["Surname"])
        st.markdown("---")
        st.subheader("Personalized Email")
        st.markdown(email)


# =======================
# Tab 2: Transaction Fraud Prediction
# =======================

# Load Fraud Detection Models
dtc_model_fraud = load_model('DecisionTreeClassifier.pkl')
rfc_model_fraud = load_model('RandomForestClassifier.pkl')
xgbc_model_fraud = load_model('XGBClassifier.pkl')

# =====================
# Load Training Data for Fraud Models
# =====================

fraud_train = load_data('balanced_fraud_sample.csv')

def load_scaler(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scaler_path = os.path.join(current_dir, 'models', filename)
    
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at path: {scaler_path}")
    
    scaler = joblib.load(scaler_path)
    print(f"Loaded scaler '{filename}'.")
    return scaler

# Preprocess the training data to get the same feature set as used in models
def preprocess_fraud_data(df):
    # Handle categorical variables: Only 'category' and 'gender'
    categorical_columns = ['category', 'gender']
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    # Extract datetime components if 'trans_date_trans_time' exists
    if 'trans_date_trans_time' in df_encoded.columns:
        df_encoded['trans_date_trans_time'] = pd.to_datetime(df_encoded['trans_date_trans_time'])
        df_encoded['trans_year'] = df_encoded['trans_date_trans_time'].dt.year
        df_encoded['trans_month'] = df_encoded['trans_date_trans_time'].dt.month
        df_encoded['trans_day'] = df_encoded['trans_date_trans_time'].dt.day
        df_encoded['trans_hour'] = df_encoded['trans_date_trans_time'].dt.hour
        
        # Drop the original datetime column
        df_encoded = df_encoded.drop('trans_date_trans_time', axis=1)
    
    # Drop unnecessary columns
    columns_to_drop = ['Unnamed: 0', 'trans_num', 'dob', 'cc_num', 'first', 'last', 
                       'street', 'city', 'state', 'zip', 'lat', 'long', 
                       'merchant', 'job', 'trans_date_trans_time']
    df_encoded = df_encoded.drop(columns=columns_to_drop, errors='ignore')
    
    # Scale numerical columns using the pre-fitted scaler
    numerical_cols = ['amt', 'merch_lat', 'merch_long', 'unix_time', 
                      'trans_year', 'trans_month', 'trans_day', 'trans_hour']
    existing_numerical_cols = [col for col in numerical_cols if col in df_encoded.columns]
    
    if existing_numerical_cols:
        # Load the saved scaler
        scaler = load_scaler('scaler_fraud.pkl')
        df_encoded[existing_numerical_cols] = scaler.transform(df_encoded[existing_numerical_cols])
    
    return df_encoded

X_fraud_encoded = preprocess_fraud_data(fraud_train).drop('is_fraud', axis=1)
y_fraud = fraud_train['is_fraud']

def prepare_transaction_input(amt, category, gender, state, city, job, trans_date_trans_time, merch_lat, merch_long):
    """
    Prepares input data for transaction fraud prediction models.
    
    Parameters:
        amt (float): Transaction amount.
        category (str): Transaction category.
        gender (str): Gender of the individual.
        state (str): State where the transaction occurred.
        city (str): City where the transaction occurred.
        job (str): Job title of the individual.
        trans_date_trans_time (datetime or str): Transaction datetime.
        merch_lat (float): Merchant latitude.
        merch_long (float): Merchant longitude.
    
    Returns:
        input_df_encoded (pd.DataFrame): Preprocessed and scaled input data.
        input_dict (dict): Dictionary of input features.
    """
    # Convert trans_date_trans_time to datetime if it's a string
    if isinstance(trans_date_trans_time, str):
        trans_date_trans_time = pd.to_datetime(trans_date_trans_time)
    
    # Ensure trans_date_trans_time is a datetime object
    if not isinstance(trans_date_trans_time, datetime):
        raise ValueError("trans_date_trans_time must be a datetime object or a string representing a datetime.")
    
    # Calculate unix_time from trans_date_trans_time
    unix_time = int(trans_date_trans_time.timestamp())
    
    # Create input dictionary with required features
    input_dict = {
        'amt': amt,
        'category': category,
        'gender': gender,
        'state': state,
        'city': city,
        'job': job,
        'merch_lat': merch_lat,
        'merch_long': merch_long,
        'unix_time': unix_time
    }
    
    # Convert input_dict to DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # One-hot encode categorical variables
    categorical_columns = ['category', 'gender']
    input_df_encoded = pd.get_dummies(input_df, columns=categorical_columns, drop_first=True)
    
    # Extract datetime features
    input_df_encoded['trans_year'] = trans_date_trans_time.year
    input_df_encoded['trans_month'] = trans_date_trans_time.month
    input_df_encoded['trans_day'] = trans_date_trans_time.day
    input_df_encoded['trans_hour'] = trans_date_trans_time.hour
    
    # List of numerical columns to scale
    numerical_cols_to_scale = ['amt', 'merch_lat', 'merch_long', 'unix_time',
                               'trans_year', 'trans_month', 'trans_day', 'trans_hour']
    
    # Scale numerical columns using the pre-fitted scaler
    scaler = load_model('scaler_fraud.pkl')
    input_df_encoded[numerical_cols_to_scale] = scaler.transform(input_df_encoded[numerical_cols_to_scale])
    
    # Ensure all model features are present
    model_features = X_fraud_encoded.columns
    for col in model_features:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0  # Add missing columns with default value 0
    
    # Reorder columns to match model's expected features
    input_df_encoded = input_df_encoded[model_features]

    print("Prepared Input DataFrame:")
    print(input_df_encoded.head())
    return input_df_encoded, input_dict

def make_transaction_predictions(input_df):
    probabilities = {}
    models = [
        ('Decision Tree Classifier', dtc_model_fraud),
        ('Random Forest Classifier', rfc_model_fraud),
        ('XGBoost Classifier', xgbc_model_fraud)
    ]
    
    for name, model in models:
        # Get feature names used during training
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
        else:
            # For XGBoost, you may need to use model.get_booster().feature_names
            feature_names = model.get_booster().feature_names
        
        # Ensure input_df has these features
        model_input_df = input_df.copy()
        
        # Add missing columns with zeros
        for col in feature_names:
            if col not in model_input_df.columns:
                model_input_df[col] = 0
        
        # Remove any extra columns not used in the model
        model_input_df = model_input_df[feature_names]
        
        # Predict probability
        y_pred = model.predict_proba(model_input_df)[0][1]
        print(f"Model: {name}, Predicted Fraud Probability: {y_pred}")
        probabilities[name] = y_pred
    
    avg_probability = np.mean(list(probabilities.values()))
    print(f"Average Fraud Probability: {avg_probability}")
    return avg_probability, probabilities


def get_feature_importances(models, feature_names):
    """
    Aggregates feature importances from multiple models.

    Parameters:
        models (list): List of trained models.
        feature_names (list): List of feature names.

    Returns:
        dict: Average feature importances.
    """
    importances = {feature: 0 for feature in feature_names}
    
    for model in models:
        if hasattr(model, 'feature_importances_'):
            model_importances = model.feature_importances_
            for feature, importance in zip(feature_names, model_importances):
                importances[feature] += importance
        elif hasattr(model, 'get_booster'):
            booster = model.get_booster()
            model_importances = booster.get_score(importance_type='weight')
            for feature, importance in model_importances.items():
                if feature in importances:
                    importances[feature] += importance
                else:
                    importances[feature] = importance
    # Average importances
    for feature in importances:
        importances[feature] /= len(models)
    
    # Sort by importance
    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
    
    # Get top 10
    top_10_importances = dict(list(sorted_importances.items())[:10])
    
    return top_10_importances

def explain_transaction(probability, input_dict, transaction_id):
    """
    Generates a natural language explanation for a transaction's fraud prediction.

    Parameters:
        probability (float): Predicted probability of fraud.
        input_dict (dict): Dictionary of input features.
        transaction_id (str/int): Unique identifier for the transaction.

    Returns:
        str: Generated explanation.
    """
    # Get feature importances
    models = [dtc_model_fraud, rfc_model_fraud, xgbc_model_fraud]
    feature_names = X_fraud_encoded.columns
    top_features = get_feature_importances(models, feature_names)
    
    # Create feature importance table
    feature_importance_table = "Feature | Importance\n-----------------------\n"
    for feature, importance in top_features.items():
        feature_importance_table += f"{feature} | {importance:.6f}\n"
    
    # Prepare summary statistics
    fraudulent_stats = fraud_train[fraud_train['is_fraud'] == 1].describe().to_dict()
    non_fraudulent_stats = fraud_train[fraud_train['is_fraud'] == 0].describe().to_dict()
    
    # Construct the prompt
    prompt = f"""You are an expert data scientist at a bank, specializing in 
interpreting and explaining predictions of machine learning models.

Your machine learning model has predicted that transaction ID {transaction_id} has a 
{round(probability * 100, 1)}% probability of being fraudulent, based on the information provided below.

Here is the transaction's information:
{input_dict}

Here are the machine learning model's top 10 most important features for predicting fraud:

{feature_importance_table}

Here are summary statistics for fraudulent transactions:
{fraud_train[fraud_train['is_fraud'] == 1].describe()}

Here are summary statistics for non-fraudulent transactions:
{fraud_train[fraud_train['is_fraud'] == 0].describe()}

WORD RESTRICTION: 150-200 words!! IT IS VERY IMPORTANT. DO NOT INCLUDE ID OR ANY OTHER SENSITIVE INFORMATION

- If the transaction has over a 40% probability of being fraudulent, generate a 3 sentence explanation of why it is likely fraudulent.
- If the transaction has less than a 40% probability of being fraudulent, generate a 3 sentence explanation of why it might not be fraudulent.

Your explanation should be based on the transaction's information, the summary statistics of fraudulent and non-fraudulent transactions, and the feature importances provided.

Don't mention the probability of being fraudulent, or the machine learning model, or say anything like "Based on the machine learning model's prediction and top 10 most important features", just explain the prediction.
"""

    print("EXPLANATION PROMPT", prompt)

    try:
        raw_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                },
            ]
        )
        return raw_response.choices[0].message.content
    except Exception as e:
        print(f"Error generating explanation: {e}")
        return "Unable to generate an explanation at this time."

# =======================
# Tab 2: Transaction Fraud Prediction
# =======================

# =======================
# Tab 2: Transaction Fraud Prediction
# =======================
with tabs[1]:
    st.header("Transaction Fraud Prediction")
    
    # Load Fraud Data
    fraud_data = load_data('balanced_fraud_sample.csv')
    fraud_data.reset_index(inplace=True)  # Ensure the index is a column if needed
    
    # Proceed only if data is loaded
    if not fraud_data.empty:
        # Create a list of transactions for selection
        transactions = [
            f"Transaction {index} - Amount: ${row['amt']} - Category: {row['category']}"
            for index, row in fraud_data.iterrows()
        ]
        
        selected_transaction_option = st.selectbox("Select a transaction", transactions)
        
        if selected_transaction_option:
            # Extract the transaction index from the selected option
            try:
                transaction_index = int(selected_transaction_option.split(" ")[1])
                selected_transaction = fraud_data.loc[transaction_index]
            except (IndexError, ValueError, KeyError) as e:
                st.error(f"Error selecting transaction: {e}")
                st.stop()
            
            st.markdown("---")
            st.header("Transaction Details")
            
            # Define which columns to include (exclude PII and irrelevant columns)
            included_columns = ['amt', 'category', 'gender', 'state', 'age', 'city', 'job']
            
            # Adjust 'age' based on dataset
            if 'age' not in fraud_data.columns:
                included_columns.remove('age')
            
            col1, col2 = st.columns(2)
            with col1:
                if 'amt' in included_columns:
                    amt = st.number_input(
                        "Transaction Amount",
                        min_value=0.0,
                        value=float(selected_transaction['amt'])
                    )
                if 'category' in included_columns:
                    try:
                        category_index = list(fraud_data['category'].unique()).index(selected_transaction['category'])
                        category = st.selectbox(
                            "Category",
                            fraud_data['category'].unique(),
                            index=category_index
                        )
                    except ValueError:
                        category = st.selectbox("Category", fraud_data['category'].unique())
                if 'gender' in included_columns:
                    try:
                        gender_index = list(fraud_data['gender'].unique()).index(selected_transaction['gender'])
                        gender = st.selectbox(
                            "Gender",
                            fraud_data['gender'].unique(),
                            index=gender_index
                        )
                    except ValueError:
                        gender = st.selectbox("Gender", fraud_data['gender'].unique())
                if 'state' in included_columns:
                    try:
                        state_index = list(fraud_data['state'].unique()).index(selected_transaction['state'])
                        state = st.selectbox(
                            "State",
                            fraud_data['state'].unique(),
                            index=state_index
                        )
                    except ValueError:
                        state = st.selectbox("State", fraud_data['state'].unique())
            
            with col2:
                if 'age' in included_columns:
                    try:
                        age = st.number_input(
                            "Age",
                            min_value=18,
                            max_value=100,
                            value=int(selected_transaction['age'])
                        )
                    except ValueError:
                        age = st.number_input("Age", min_value=18, max_value=100, value=30)
                if 'city' in included_columns:
                    try:
                        city_index = list(fraud_data['city'].unique()).index(selected_transaction['city'])
                        city = st.selectbox(
                            "City",
                            fraud_data['city'].unique(),
                            index=city_index
                        )
                    except ValueError:
                        city = st.selectbox("City", fraud_data['city'].unique())
                if 'job' in included_columns:
                    try:
                        job_index = list(fraud_data['job'].unique()).index(selected_transaction['job'])
                        job = st.selectbox(
                            "Job",
                            fraud_data['job'].unique(),
                            index=job_index
                        )
                    except ValueError:
                        job = st.selectbox("Job", fraud_data['job'].unique())
                # Add more features as needed based on included_columns
            
            # Prepare input data
            trans_date_trans_time = selected_transaction.get('trans_date_trans_time', pd.Timestamp.now())

            input_df, input_dict = prepare_transaction_input(
                amt=amt,
                category=category,
                gender=gender,
                state=state,
                city=city,
                job=job,
                trans_date_trans_time=trans_date_trans_time,
                merch_lat=selected_transaction['merch_lat'],
                merch_long=selected_transaction['merch_long']
            )

            # Make predictions
            avg_probability, probabilities = make_transaction_predictions(input_df)
            
            # Generate explanation
            transaction_id = selected_transaction['trans_num']  # Assuming 'trans_num' is unique
            explanation = explain_transaction(avg_probability, input_dict, transaction_id)
            
            # Display the results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            # Display average probability
            st.write(f"The transaction has a {avg_probability:.2%} probability of being fraudulent.")
            
            # Display probabilities from each model
            fig_probs = create_model_probability_chart(probabilities)
            st.plotly_chart(fig_probs, use_container_width=True)
            
            # Display explanation
            st.markdown("---")
            st.subheader("Explanation of Prediction")
            st.markdown(explanation)
