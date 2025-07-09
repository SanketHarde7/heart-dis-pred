import streamlit as st
import pandas as pd
import joblib

model=joblib.load('heart_dis.pkl')


st.title("❤️ Heart Disease Prediction ")

age=st.number_input("Age",min_value=1,max_value=120)
sex=st.selectbox("Sex",("Male","female"))
cp=st.selectbox("ChestPainType",('ATA','NAP','ASV','TA'))
resting_ecg=st.selectbox("RestingECG",('Normal',"ST",'LVH'))
exercise_agina=st.selectbox("ExerciseAgina",('Yes','No'))
st_slope=st.selectbox("ST_Slope",['Up','Flat','Down'])
resting_bp=st.number_input("Resting Blood Pressure ",min_value=50,max_value=200)
cholesterol=st.number_input("cholesterol",min_value=100,max_value=500)
fasting_bs=st.selectbox("Fasting Blood Pressure>120mg/dl ",("Yes","No"))
max_hr=st.number_input("Maximum heart rate achived",min_value=50,max_value=250)
oldpeak=st.number_input("Oldpeak(ST depression)",min_value=0.0,max_value=10.0,step=0.1)
fasting_bs =1 if fasting_bs=="Yes" else 0
sex=1 if sex=='Male' else 0
cp_map={"ATA":0,"NAP":1,"ASV":2,"TA":3}
ecg_map={'Normal':0,"ST":1,"LVH":2}
agina=1 if exercise_agina =="Yes" else 0
slope_map={"Up":0,"Flat":1,"Down":2}

input_data = pd.DataFrame([[age, sex, cp_map[cp], resting_bp, cholesterol, fasting_bs, 
                            ecg_map[resting_ecg], max_hr, agina, oldpeak, slope_map[st_slope]]],
    columns=["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", 
             "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", 
             "Oldpeak", "ST_Slope"])


if st.button("predict"):
    prediction=model.predict(input_data)[0]
    if prediction==1:
        st.error("high risk of heart disease")
    else:
        st.success("No heart disease dected.")






# Age Sex ChestPainType  RestingBP  Cholesterol  FastingBS RestingECG  MaxHR ExerciseAngina  Oldpeak ST_Slope  HeartDisease