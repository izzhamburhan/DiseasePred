# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 22:46:18 2023

@author: Izzham Burhan
"""

import numpy as np
import pickle 
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('C:/Users/Izzham Burhan/Documents/Python Scripts/DiseasePred/trained_model.sav', 'rb'))

# creating a function for Prediction
def disease_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = classifier.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
         print('The person have Acne')
    elif (prediction[0] == 1):
        print('The person have AIDS')
    elif (prediction[0] == 2):
        print('The person have Allergy')
    elif (prediction[0] == 3):
        print('The person have Chicken Pox')
    elif (prediction[0] == 4):
        print('The person have Common Cold ')
    elif (prediction[0] == 5):
        print('The person have Dengue')
    elif (prediction[0] == 6):
        print('The person have Diabetes')
    elif (prediction[0] == 7):
        print('The person have GERD')
    elif (prediction[0] == 8):
        print('The person have Heart Attack')
    elif (prediction[0] == 9):
        print('The person have Hypertension')
    elif (prediction[0] == 10):
        print('The person have Malaria')
    elif (prediction[0] == 11):
        print('The person have Migraine')
    
def __main__():
    # giving a title
    st.title('Disease Prediction Web App')
    
    itching = st.text_input('Do you have Itching ?')
    skin_rash = st.text_input('Do you have Rash Skin ?')
    continuous_sneezing = st.text_input('Do you Continuous Sneezing ?')
    shivering = st.text_input('Do you have Shivering ?')
    chills = st.text_input('Do you have Chills ?')
    joint_pain = st.text_input('Do you have Joint Pain ?')
    stomach_pain = st.text_input('Do you have Stomach Pain ?')
    acidity = st.text_input('Do you feel acidity ?')
    ulcers_on_tongue = st.text_input('Do you have Ulcers on Tongue ?')
    muscle_wasting = st.text_input('Do you Muscle Wasting ?')
    vomiting = st.text_input('Do you have Vomiting earlier ?')
    fatigue = st.text_input('Do you have Fatigue ?')
    weight_loss = st.text_input('Do you Loss Weight ?')
    restlessness = st.text_input('Do you have Restlessness ?')
    lethargy = st.text_input('Do you have Lethargy ?')
    patches_in_throat = st.text_input('Do you have Patches in Throat ?')
    irregular_sugar_level = st.text_input('Do you have Irregular Sugar Level ?')
    cough = st.text_input('Do you have Cough ?')
    high_fever = st.text_input('Do you Fever in Level High ?')
    breathlessness = st.text_input('Do you have Breathlessness ?')
    sweating = st.text_input('Do you have Sweating ?')
    indigestion = st.text_input('Do you have Indigestion ?')
    headache = st.text_input('Do you have Headache ?')
    nausea = st.text_input('Do you have Nausea ?')
    loss_of_appetite = st.text_input('Do you Loss some Appetite ?')
    pain_behind_the_eyes = st.text_input('Do you have Pain behind the Eyes ?')
    back_pain = st.text_input('Do you have Back Pain ?')
    diarrhoea = st.text_input('Do you have Diarrhoea ?')
    mild_fever = st.text_input('Do you have Mild Fever ?')
    swelled_lymph_nodes = st.text_input('Do you have Swelled lymph Nodes ?')
    malaise = st.text_input('Do you have Malaise ?')
    blurred_and_distorted_vision = st.text_input('Do you have Blurred or Distorted Vision ?')
    phlegm = st.text_input('Do you have Phelgm ?')
    throat_irritation = st.text_input('Do you have Throat Irritation ?')
    redness_of_eyes = st.text_input('Do you have Redness of Eyes ?')
    sinus_pressure = st.text_input('Do you have Sines Pressure ?')
    runny_nose = st.text_input('Do you have Runny Nose ?')
    congestion = st.text_input('Do you have Congestion ?')
    chest_pain = st.text_input('Do you have Chest Pain ?')
    dizziness = st.text_input('Do you have Dizziness ?')
    obesity = st.text_input('Do you have Obesity ?')
    excessive_hunger = st.text_input('Do you have Excessive Hunger ?')
    extra_marital_contacts = st.text_input('Do you have Extra Marital Contacts ?')
    loss_of_balance = st.text_input('Do you have Loss of Balance ?')
    loss_of_smell = st.text_input('Do you have Loss of Smell ?')
    depression = st.text_input('Do you have Depression ?')
    irritability = st.text_input('Do you have Irritability ?')
    muscle_pain = st.text_input('Do you have Muscle Pain ?')
    red_spots_over_body = st.text_input('Do you have Red Spots over Body ?')
    watering_from_eyes = st.text_input('Do you have Watering from eyes ?')
    increased_appetite = st.text_input('Do you have Increased Appetite ?')
    polyuria = st.text_input('Do you have Polyuria ?')
    lack_of_concentration = st.text_input('Do you have Lack of Concentration ?')
    visual_disturbances = st.text_input('Do you have Visual Disturbances ?')
    pus_filled_pimples = st.text_input('Do you have Pus filled pimples ?')
    blackheads = st.text_input('Do you have Blackheads ?')
    scurring = st.text_input('Do you have Scurring ?')
    
    # code for Prediction
    prognosis = ''
    
    # creating a button for Prediction
    
    if st.button('Disease Test Result'):
        prognosis = disease_prediction([itching,skin_rash,continuous_sneezing,shivering,chills,joint_pain,stomach_pain,acidity,ulcers_on_tongue,muscle_wasting,vomiting,fatigue,weight_loss,restlessness,lethargy,patches_in_throat,irregular_sugar_level,cough,high_fever,breathlessness,sweating,indigestion,headache,nausea,loss_of_appetite,pain_behind_the_eyes,back_pain,diarrhoea,mild_fever,swelled_lymph_nodes,malaise,blurred_and_distorted_vision,phlegm,throat_irritation,redness_of_eyes,sinus_pressure,runny_nose,congestion,chest_pain,dizziness,obesity,excessive_hunger,extra_marital_contacts,loss_of_balance,loss_of_smell,depression,irritability,muscle_pain,red_spots_over_body,watering_from_eyes,increased_appetite,polyuria,lack_of_concentration,visual_disturbances,pus_filled_pimples,blackheads,scurring])
        
        
    st.success(diagnosis)
    

if __name__ == '__main__' :
        main()