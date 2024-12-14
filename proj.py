# -*- coding: utf-8 -*-
"""
Created on 13/12/2024

@author: 

* Furkan Özbek
* Emir Alparslan Dikici
* Berat Yüceldi
* Zeynep Ece Aşkın
         
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# loading the saved model
loaded_model = pickle.load(open('bank_model.pkl.sav', 'rb'))

# Function for prediction
def prediction_function(input_data):
    # Convert input data to numpy array and reshape
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction

def main():
    # Title of the Streamlit app
    st.title('Bank Marketing Prediction Web App')

    # Collecting additional input data from the user
    duration = st.number_input('Duration', min_value=0)
    previous = st.number_input('Previous', min_value=0)
    emp_var_rate = st.number_input('Employment Variation Rate', value=0.0)
    euribor3m = st.number_input('Euribor 3 Month Rate', value=0.0)
    nr_employed = st.number_input('Number of Employed', value=0.0)
    contacted_before = st.number_input('Contacted Before (0 or 1)', min_value=0, max_value=1, step=1)
    contact_cellular = st.number_input('Contact Cellular (0 or 1)', min_value=0, max_value=1, step=1)  # New feature

    # Code for prediction
    diagnosis = ''
    
    # Button for prediction
    if st.button('Predict'):
        # Updated input data to match the model's expected feature count
        input_data = [duration, previous, emp_var_rate, euribor3m, nr_employed, contacted_before, contact_cellular]
        diagnosis = prediction_function(input_data)
    
    st.success(f'The prediction is: {diagnosis}')


if __name__ == '__main__':
    main()

# to run ---- streamlit run proj.py
