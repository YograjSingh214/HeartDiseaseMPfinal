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
    st.title('Heart Disease Detection using ML-ANN')
        
        
    # getting the input data from the user
    col1, col2, col3, col4 = st.columns(4)
        
    with col1:
        HighBp = st.text_input('High BP (0 no / 1 yes)')
            
    with col2:
        HighChol = st.text_input('High cholestrol (0 no /1 yes)')
        
    with col3:
        BMI = st.text_input('BMI (12-98)')
        
    with col4:
        Smoker = st.text_input('Do you smoke (0 no / 1 yes) ')
        
    with col1:
        Stroke = st.text_input('Stroke (0 no / 1 yes)')
        
    with col2:
        Diabetes = st.text_input('Diabetes (0, 1, 2)')

    with col3:
        PhysActivity = st.text_input('physical activity (0 no / 1 yes)')

    with col4:
        HvyAlcoholConsump = st.text_input('Alcoholic (0 no / 1 yes)')

    with col1:
        GenHlth = st.text_input('GenHlth (1-5)')

    with col2:
        PhysHlth = st.text_input('PhysHlth (0-30)')

    with col3:
        Diffwalk = st.text_input('Difficulty in walking (0 no / 1 yes)')

    with col4:
        Gender = st.text_input('Gender (0 female / 1 male)')

    with col1:
        Age = st.text_input('Age')
        
        
    # code for Prediction
    diab_diagnosis = ''
        
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        input_data = [int(HighBp),
                      int(HighChol),
                      int(BMI),
                      int(Smoker),
                      int(Stroke),
                      int(Diabetes),
                      int(PhysActivity),
                      int(HvyAlcoholConsump),
                      int(GenHlth),
                      int(PhysHlth),
                      int(Diffwalk),
                      int(Gender),
                      int(Age)]
        data = np.asarray(input_data)
        data_reshaped = data.reshape(1,-1)
        diab_prediction = diabetes_model.predict(data_reshaped)
        #diab_percentage = _model.predict_proba(data_reshaped)
        #prob = np.max(diab_percentage, axis=1)
        #max_prob = np.round(prob, 3)
    
        if (diab_prediction[0] > 0.5):
            diab_diagnosis = 'The person may have heart disease'
            
        else:
            diab_diagnosis = 'The person may not have heart disease'
        
    st.success(diab_diagnosis)

if __name__ == '__main__':
    main()
