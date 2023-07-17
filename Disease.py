import pickle 
import streamlit as st
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from tensorflow import keras

def main():

    #with open('model.sav', 'rb') as f:
    #diabetes_model = pickle.load(f)
    diabetes_model = keras.models.load_model("model.h5")

    
    # page title
    st.title('Heart Disease Detection using ML')
        
        
    # getting the input data from the user
    col1, col2, col3, col4 = st.columns(4)
        
    with col1:
        BMI = st.text_input('BMI')
            
    with col2:
       Smoking = st.text_input('Smoking(0 no/ 1 yes)')
        
    with col3:
        AlcoholDrinking = st.text_input('AlcoholDrinking (0 no/ 1 yes)')
        
    with col4:
        Stroke = st.text_input('Stroke (0 no / 1 yes) ')
        
    with col1:
        PhysicalHealth = st.text_input('PhysicalHealth')
        
    with col2:
        Sex = st.text_input('Sex(0 female/ 1 male)')

    with col3:
        AgeCategory = st.text_input('AgeCategory')

    with col4:
        Diabetic = st.text_input('Diabetic (0 no / 1 yes)')

    
        
        
    # code for Prediction
    diab_diagnosis = ''
        
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result '):
        input_data = [float(BMI),
                      int(Smoking),
                      int(AlcoholDrinking),
                      int(Stroke),
                      float(PhysicalHealth),
                      int(Sex),
                      int(AgeCategory),
                      int(Diabetic),
                      ]
        data = np.asarray(input_data)
        data_reshaped = data.reshape(1,-1)
        '''diab_prediction = diabetes_model.predict(data_reshaped)
        diab_percentage = _model.predict_proba(data_reshaped)
        prob = np.max(diab_percentage, axis=1)
        max_prob = np.round(prob, 3)'''
    
        if (diab_prediction >= 0.17 ):
            diab_diagnosis = 'The person may have heart disease'
            
        else:
            diab_diagnosis = 'The person may not have heart disease'
        
    st.success(diab_diagnosis)

if _name_ == '_main_':
    main()