import numpy as np
import pickle
import pandas as pd
import streamlit as st
from PIL import Image



st.set_page_config(
    page_title="Employee detection system",
    page_icon='🏦',
    layout="wide", 
    initial_sidebar_state="expanded",
)
st.subheader('Система определения срока деятельности рабочего')

model_selected = st.radio('Какой анализ вы бы хотели использовать?', ('KNeighborsClassifier', 'LogisticRegression', 'DecisionTreeClassifier',  'RandomForestClassifier', 'AdaBoostClassifier', 'XGBClassifier', 'CatBoostClassifier', 'Default'))


if model_selected == 'KNeighborsClassifier':
    pickle_in = open("employee_KNN.pkl","rb")
    classifier=pickle.load(pickle_in)
elif model_selected in ['LogisticRegression', 'Default']:
    pickle_in = open("employee_LogReg.pkl","rb")
    classifier=pickle.load(pickle_in)
elif model_selected == 'DecisionTreeClassifier':
    pickle_in = open("employee_DecisionTree.pkl","rb")
    classifier=pickle.load(pickle_in)
elif model_selected == 'RandomForestClassifier':
    pickle_in = open("employee_RandomForest.pkl","rb")
elif model_selected == 'AdaBoostClassifier':
    pickle_in = open("employee_AdaBoost.pkl","rb")
elif model_selected == 'XGBClassifier':
    pickle_in = open("employee_xgboos.pkl","rb")
elif model_selected == 'CatBoostClassifier':
    pickle_in = open("employee_catboost.pkl","rb")
    classifier=pickle.load(pickle_in)
    
def predict_note_authentication(JoiningYear, PaymentTier, Age, Gender, EverBenched, ExperienceInCurrentDomain, Bachelors, Masters, PHD, Bangalore, New_Delhi, Pune):
    prediction=classifier.predict([[JoiningYear, PaymentTier, Age, Gender, EverBenched, ExperienceInCurrentDomain, Bachelors, Masters, PHD, Bangalore, New_Delhi, Pune]])
    print(prediction)
    return prediction 



def main():
    st.title("Система определения срока деятельности рабочего")
    JoiningYear = st.radio('Ваш пол?(0 - male, 1 - female)', (0, 1))
    JoiningYear = st.number_input('В каком году начал деятельность(используйте только цифры)?', step=1, value=0)
    PaymentTier = st.number_input('Уровень оплаты?(используйте цифры от 1 до 3)?', step=1, value=0) 
    Age = st.number_input('Сколько вам полных лет?(используйте только цифры)?', step=1, value=0)
    Gender = st.radio('Ваш пол?(1 - male, 0 - female)', (0, 1))
    EverBenched = st.radio('когда-либо сидел на скамейке запасных?(0 - нет, 1 - да)', (0, 1))    
    ExperienceInCurrentDomain =st.number_input('Опыт работы в текущей области?(используйте только цифры)?',step=1, value=0)
    Bachelors = st.radio('Имеете степень бакалавриата?(1 - да, 0 - нет)', (0, 1))
    Masters = st.radio('Имеете степень магистра?(1 - да, 0 - нет)', (0, 1))
    PHD = st.radio('Имеете степень PHD?(1 - да, 0 - нет)', (0, 1))
    Bangalore = st.radio('Вы из города Bangalore?(1 - да, 0 - нет)', (0, 1))
    New_Delhi = st.radio('Вы из города New Delhi?(1 - да, 0 - нет)', (0, 1))
    Pune = st.radio('Вы из города Pune?(1 - да, 0 - нет)', (0, 1))
                     
    result=""
    if st.button("Predict"):
        result=int(predict_note_authentication(JoiningYear, PaymentTier, Age, Gender, EverBenched, ExperienceInCurrentDomain, Bachelors, Masters, PHD, Bangalore, New_Delhi, Pune)) 
     #st.success('The output is {}'.format(result))
    st.success('Результат системы(1 - Will stay, Результат системы(0 - Will not stay) {}'.format(result))
                     
    
    if st.button("Built by Aziz Rasulov"):
        st.text("Эта программа предназначена для предсказания длительности деятельности рабочего.")
        
              
                  
if __name__=='__main__':
    main()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    