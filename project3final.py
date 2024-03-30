# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 20:29:14 2023

@author: shiva
"""

import streamlit as st
#from joblib import load
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd

le=LabelEncoder()

st.title('LIVER DISEASE PREDICTION')
st.image('https://www.ddw-online.com/wp-content/uploads/2022/10/Shutterstock_1906753141-1.jpg')
st.sidebar.header('User Input Parameters')


model=pickle.load(open('C:/Users/shiva/Desktop/project-3/deployment/randomforest_model.sav','rb'))
lev=pickle.load(open('C:/Users/shiva/Desktop/project-3/deployment/label_project3.sav','rb'))



def predict_decease(dt):
    input_data=np.asarray(dt)
    reshape_data=input_data.reshape(1,12)
    pred=model.predict(reshape_data)
    result=lev.inverse_transform(pred)
    return result

        
    #print(pred)
def main():
    
    st.title('Predict Liver Disease')
    age=st.number_input('Enter your age',min_value=0)
    option=["Male","Female"]
    sex=st.selectbox('select your sex',options=option)
    if sex=='Male':
        sex1=1
    else:
        sex1=0

    albumin=st.sidebar.number_input('Enter your albumin value',min_value=0)
    alkaline_phosphatase=st.sidebar.number_input('Enter your alkaline_phosphatase value',min_value=0)
    alanine_aminotransferase=st.sidebar.number_input('Enter your alanine_aminotransferase value',min_value=0)
    aspartate_aminotransferase=st.sidebar.number_input('Enter your aspartate_aminotransferase value',min_value=0)
    bilirubin=st.sidebar.number_input('Enter your bilirubin value',min_value=0)
    cholinesterase=st.sidebar.number_input('Enter your cholinesterase value',min_value=0)
    cholesterol=st.sidebar.number_input('Enter your cholesterol value',min_value=0)
    creatinina=st.sidebar.number_input('Enter your creatinina value',min_value=0)
    gamma_glutamyl_transferase=st.sidebar.number_input('Enter your gamma_glutamyl_transferase value',min_value=0)
    protein=st.sidebar.number_input('Enter your protein value',min_value=0)

    decease=''
    
    if st.button('predict'):
        decease=predict_decease([age,sex1,albumin,alkaline_phosphatase,alanine_aminotransferase,aspartate_aminotransferase,bilirubin,cholinesterase,cholesterol,creatinina,gamma_glutamyl_transferase,protein])
    #st.write('jkhd')
    st.success(decease)
    df = {'age': age, 'sex': sex1, 'albumin': albumin,
          'alkaline_phosphatase': alkaline_phosphatase, 
          'alanine_aminotransferase': alanine_aminotransferase,
          'aspartate_aminotransferase': aspartate_aminotransferase,
          'bilirubin': bilirubin, 'cholinesterase': cholinesterase,
          'cholesterol': cholesterol, 'creatinina': creatinina,
          'gamma_glutamyl_transferase': gamma_glutamyl_transferase,
          'protein': protein}   

    
    df1 =pd.DataFrame(df,index=[0]) 

    st.subheader('User Input Parameters')
    st.write(df1)

if __name__=='__main__':
    main()
