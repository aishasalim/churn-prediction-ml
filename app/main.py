import streamlit as st
import pandas as pd
import pickle
import numpy as np
from openai import OpenAI
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
import joblib

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
    model_path = os.path.join(current_dir, filename)
    
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



# xgboost_model = load_model('../models/xgb_model.pkl')
# random_forest_model = load_model('../models/rf_model.pkl')
# decision_tree_model = load_model('../models/dt_model.pkl')
# svm_model = load_model('../models/svm_model.pkl')
# knn_model = load_model('../models/knn_model.pkl')
# voting_classifier_model = load_model('../models/voting_classifier.pkl')
# xgboost_SMOTE_model = load_model('../models/xgboost-SMOTE.pkl')
# xgboost_featureEngineered_model = load_model('../models/xgboost-featureEngineered.pkl')
best_lr_model = load_model('models/best_lr_model.pkl')
stacking_model = load_model('models/stacking_model.pkl')
gbc_model = load_model('models/gbc_model.pkl')


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
st.title("Customer Churn Prediction")

df = pd.read_csv("../data/churn.csv")

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
