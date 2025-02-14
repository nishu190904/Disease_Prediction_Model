import os
import joblib 
import streamlit as st  
from streamlit_option_menu import option_menu


st.set_page_config(
    page_title="Prediction of Disease Outbreaks",
    layout="wide",
    page_icon="🏥"
)


diabetes_model = joblib.load(r"C:\Users\ASUS\OneDrive\Desktop\Project\models\diabetes_model.pkl")
heart_disease_model = joblib.load(r"C:\Users\ASUS\OneDrive\Desktop\Project\models\heart_model.pkl")
parkinsons_model = joblib.load(r"C:\Users\ASUS\OneDrive\Desktop\Project\models\parkinsons_model.pkl")


with st.sidebar:
    selected = option_menu(
        "Prediction of Disease Outbreak System",
        ["Diabetes Prediction", "Heart Disease Prediction", "Parkinsons Prediction"],
        menu_icon="hospital-fill",
        icons=["activity", "heart", "person"],
        default_index=0
    )


if selected == "Diabetes Prediction":       
    st.title("Diabetes Prediction")  

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input("Number of Pregnancies")
        SkinThickness = st.text_input("Skin Thickness Value")
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")

    with col2:
        Glucose = st.text_input("Glucose Level")
        Insulin = st.text_input("Insulin Level")
        Age = st.text_input("Age of the Person")

    with col3:
        BloodPressure = st.text_input("Blood Pressure Value")
        BMI = st.text_input("BMI Value")

diab_diagnosis = ''

if st.button('Diabetes Test Result'):
    user_input = [
        Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, 
        BMI, DiabetesPedigreeFunction, Age
    ]

    
    user_input = [float(x) if x.strip() else 0.0 for x in user_input]

    diab_prediction = diabetes_model.predict([user_input])

    if diab_prediction[0] == 1:
        diab_diagnosis = "The person is diabetic"
    else:
        diab_diagnosis = "The person is not diabetic"

    st.success(diab_diagnosis)


if selected == "Heart Disease Prediction":       
    st.title("Heart Disease Prediction")  

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input("Age")
        trestbps = st.text_input("Resting Blood Pressure")
        restecg = st.text_input("Resting Electrocardiographic Results")
        slope = st.text_input("Slope of the Peak Exercise ST Segment")

    with col2:
        sex = st.text_input("Sex (1 = Male, 0 = Female)")
        chol = st.text_input("Serum Cholesterol in mg/dl")
        thalach = st.text_input("Maximum Heart Rate Achieved")
        ca = st.text_input("Number of Major Vessels (0-3) Colored by Fluoroscopy")

    with col3:
        cp = st.text_input("Chest Pain Type (0-3)")
        fbs = st.text_input("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)")
        exang = st.text_input("Exercise Induced Angina (1 = Yes, 0 = No)")
        thal = st.text_input("Thalassemia (0-3)")
        oldpeak = st.text_input("ST Depression Induced by Exercise")  

heart_diagnosis = ''

if st.button('Heart Disease Test Result'):
    user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    
    try:
        user_input = [float(x) if x else 0.0 for x in user_input]  
    except ValueError:
        st.error("Please enter valid numerical values")
        st.stop()
    
    heart_prediction = heart_disease_model.predict([user_input])
    
    if heart_prediction[0] == 1:
        heart_diagnosis = 'The person has heart disease'
    else:
        heart_diagnosis = 'The person does not have heart disease'
    
    st.success(heart_diagnosis)


if selected == "Parkinsons Prediction":       
    st.title("Parkinson's Disease Prediction")  

    col1, col2, col3 = st.columns(3)

    with col1:
        Fo = st.text_input("MDVP:Fo(Hz)")
        Jitter_percent = st.text_input("MDVP:Jitter(%)")
        RAP = st.text_input("MDVP:RAP")
        Shimmer = st.text_input("MDVP:Shimmer")
        APQ3 = st.text_input("Shimmer:APQ3")
        DDA = st.text_input("Shimmer:DDA")
        RPDE = st.text_input("RPDE")
        spread1 = st.text_input("Spread1")

    with col2:
        Fhi = st.text_input("MDVP:Fhi(Hz)")
        Jitter_Abs = st.text_input("MDVP:Jitter(Abs)")
        PPQ = st.text_input("MDVP:PPQ")
        Shimmer_dB = st.text_input("MDVP:Shimmer(dB)")
        APQ5 = st.text_input("Shimmer:APQ5")
        NHR = st.text_input("NHR")
        DFA = st.text_input("DFA")
        spread2 = st.text_input("Spread2")

    with col3:
        Flo = st.text_input("MDVP:Flo(Hz)")
        DDP = st.text_input("Jitter:DDP")
        APQ = st.text_input("MDVP:APQ")
        HNR = st.text_input("HNR")
        D2 = st.text_input("D2")
        PPE = st.text_input("PPE")

    parkinsons_diagnosis = ""

if st.button("Parkinson's Test Result"):
    user_input = [
        Fo, Fhi, Flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
        Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR,
        RPDE, DFA, spread1, spread2, D2, PPE  
    ]

    
    user_input = [float(x) if x.strip() else 0.0 for x in user_input]

    parkinsons_prediction = parkinsons_model.predict([user_input])

    if parkinsons_prediction[0] == 1:
        parkinsons_diagnosis = "The person has Parkinson’s Disease"
    else:
        parkinsons_diagnosis = "The person does not have Parkinson’s Disease"

    st.success(parkinsons_diagnosis)
