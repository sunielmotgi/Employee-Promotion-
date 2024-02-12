# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 13:30:35 2024

@author: sunie
"""

import streamlit as st
import pandas as pd
import pickle
st.sidebar.header('User Input Parameters')

st.markdown("<h1 style='text-align: centre; color: black; font-size: 25px';>SUNIEL </h1>", unsafe_allow_html=True)


def user_input_features():
    DEP = st.sidebar.selectbox("department",('Sales & Marketing', 'Operations', 'Technology', 'Analytics',
       'R&D', 'Procurement', 'Finance', 'HR', 'Legal'))
    EDU = st.sidebar.selectbox("education",("Master's & above", "Bachelor's", 'nan', 'Below Secondary'))
    GEN = st.sidebar.selectbox('gender',('f', 'm') )
    
    
    NoT = st.sidebar.selectbox("no_of_trainings",(1,  2,  3,  4,  7,  5,  6,  8, 10,  9))
    AGE = st.sidebar.selectbox("age",(35, 30, 34, 39, 45, 31, 33, 28, 32, 49, 37, 38, 41, 27, 29, 26, 24,
       57, 40, 42, 23, 59, 44, 50, 56, 20, 25, 47, 36, 46, 60, 43, 22, 54,
       58, 48, 53, 55, 51, 52, 21))
    
    PYR= st.sidebar.selectbox("previous_year_rating",(5.,  3.,  1.,  4., 2.))
    
    LoS = st.sidebar.selectbox("length_of_service",(8,  4,  7, 10,  2,  5,  6,  1,  3, 16,  9, 11, 26, 12, 17, 14, 13,
       19, 15, 23, 18, 20, 22, 25, 28, 24, 31, 21, 29, 30, 34, 27, 33, 32,
       37))
    AW = st.sidebar.selectbox("awards_won",(0, 1))
    ATS = st.sidebar.selectbox("avg_training_score",(49., 60., 50., 73., 85., 59., 63., 83., 54., 80., 84., 77., 51.,
       46., 75., 68., 79., 72., 58., 87., 47., 57., 52., 88., 71., 48.,
       65., 62., 53., 78., 44., 91., 82., 69., 74., 86., 90., 92., 67.,
       89., 56., 76., 81., 70., 55., 39., 94., 93., 64., 66., 95., 42.,
       96., 40., 99., 43., 97., 41., 98.))
    
    user_input = {'department':DEP,
              'education': EDU,
              'gender': GEN,
              'no_of_trainings': NoT,
              'age': AGE,
              'previous_year_rating': PYR,
              'length_of_service': LoS,
              'awards_won': AW,
              'avg_training_score': ATS,}
    
    features = pd.DataFrame(user_input,index = [0])
    return features 

df1= user_input_features()
st.subheader('User Input Dataframe')
st.write(df1)

with open("final_model.sav",mode="rb") as f1:
    model = pickle.load(f1)

prediction = model.predict(df1)
prediction_proba = model.predict_proba(df1)

st.subheader('Predicted Result')

st.write(prediction[0])
if (prediction[0]==0):
    print("is not pramoted")
else:
    print("is pramoted")


















