# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 12:19:34 2024

@author: adham
"""
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import pickle
from streamlit_option_menu import option_menu
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def load_model(file_name):
    with open(file_name, 'rb') as f:
        model = pickle.load(f)
    return model

# Load the models
script_dir = os.path.dirname(os.path.realpath(__file__))  # Get directory of the script
models_dir = os.path.join(script_dir, 'models')  # Path to models directory
parkinsons_model_path = os.path.join(models_dir, 'parkinsons_model.sav')
parkinsons_forest_path = os.path.join(models_dir, 'parkinsons_forest.sav')
parkinsons_model = load_model(parkinsons_model_path)
parkinsons_forest = load_model(parkinsons_forest_path)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)
data = pd.read_csv("C:/Users/adham/Downloads/parkinsons.csv")   
data.drop(["name"],axis="columns",inplace=True)

x=data.drop("status",axis=1)
y=data["status"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

parkinsons_model.fit(X_train, y_train)
model = LogisticRegression(max_iter=100000)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
score=score*100

parkinsons_forest.fit(X_train, y_train)
model_for = RandomForestClassifier(n_estimators=10)
model_for.fit(X_train, y_train)
score_for = model_for.score(X_test, y_test)
score_for=score_for*100


# Main content
st.title("Prediction de Parkinsons!")



def result():
    
    parkinsons_diagnosis = ''

    if st.button('Resultat du test :'):
        user_input=[MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitte_ptg, MDVP_Jitter_ABS, MDVP_RAP, MDVP_PPQ, Jitter_DDP,
                      MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR, RPDE,
                      DFA, spread1, spread2, D2,PPE]
            
        user_input = [float(x) for x in user_input]
        
        parkinsons_prediction = parkinsons_model.predict([user_input])
        
        if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = 'La personne a du parkinsons'
        else:
                parkinsons_diagnosis = 'La personne n a pas du parkinsons'
                
    st.success(parkinsons_diagnosis)


def result_forest():
    parkinsons_diagnosis = ''

    if st.button('Resultat du test :'):
        user_input=[MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitte_ptg, MDVP_Jitter_ABS, MDVP_RAP, MDVP_PPQ, Jitter_DDP,
                      MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR, RPDE,
                      DFA, spread1, spread2, D2,PPE]
            
        user_input = [float(x) for x in user_input]
        
        parkinsons_prediction = parkinsons_forest.predict([user_input])
        
        if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = 'La personne a du parkinsons'
        else:
                parkinsons_diagnosis = 'La personne n a pas du parkinsons'
                
    st.success(parkinsons_diagnosis)

 
#sidebar
with st.sidebar:
    
    selected=option_menu('Prediction de la maladie Parkinsons avec ML',
                         ['Home',
                          'Predire avec la regression logistique',
                          'Predire avec Random forest'
                          ],
                         menu_icon='hospital-fill',
                           icons=['home','activity', 'person'],
                         default_index=0
                         )
    
if selected == 'Home':
    selected_option = st.sidebar.selectbox(
        "Select an option",
        ("DataSet", "Graphes", "Information"),
        
        placeholder="Selectionner une option."
    )

    if selected_option == "DataSet":
        st.header("Description des caracteristiques ")
        data = {
        'Caracteristique': [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
        'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
        'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
        'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'status', 'RPDE', 'DFA',
        'spread1', 'spread2', 'D2', 'PPE'
        ],
        'Description': [
        'Fréquence fondamentale moyenne en Hertz (Hz)',
        'Fréquence maximale des harmoniques en Hertz (Hz)',
        'Fréquence minimale des harmoniques en Hertz (Hz)',
        'Variabilité de la période fondamentale en pourcentage (%)',
        'Mesure absolue de la variation de la période fondamentale',
        'Amplitude relative de la période fondamentale',
        'Variation de la période fondamentale, perturbations, et quasipériodicité',
        'Variabilité de la période fondamentale, rapport de délai double',
        'Variabilité de l amplitude de la période fondamentale',
        'Variabilité de l amplitude de la période fondamentale en décibels (dB)',
        'Amplitude du tremblement, moyenne de la variation d amplitude sur trois segments',
        'Amplitude du tremblement, moyenne de la variation d amplitude sur cinq segments',
        'Amplitude du tremblement, moyenne de la variation d amplitude',
        'Amplitude du tremblement, analyse d aires sous la courbe',
        'Rapport harmonique au bruit',
        'Rapport harmonique au bruit',
        'Statut de la maladie de Parkinson (variable cible binaire)',
        'Dimension de l espace de Reichardt-Pfeiffer',
        'Analyse fractale de la fluctuation',
        'Paramètre de dispersion vocal 1',
        'Paramètre de dispersion vocal 2',
        'Mesure de la complexité de la voix',
        'Excursion de la période fondamentale moyenne',
        ],
    }
        df = pd.DataFrame(data)
        st.markdown(df.style.hide(axis="index").to_html(), unsafe_allow_html=True)
    elif selected_option == "Graphes":
        st.header("Some Graphs :")
        st.write("Exemple de graphe en fonction de MDVP:Fo et MDVP:Fhi")
        data = pd.read_csv("C:/Users/adham/Downloads/parkinsons.csv")
        data.drop(["name"],axis="columns",inplace=True)
        sns.pairplot(data=data[['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)']])
        st.pyplot()
        sns.relplot(x="MDVP:Fo(Hz)", y="status", kind="line", data=data);
        st.pyplot()
        
    elif selected_option == "Information":
        data = pd.read_csv("C:/Users/adham/Downloads/parkinsons.csv")
        data.drop(["name"],axis="columns",inplace=True)
        st.header("Description du dataset")
        st.write(data.describe())
        st.header("Les types des caracteristiques")
        st.write(data.dtypes)
        st.header("Correclation")
        st.write(data.corr())
        


elif(selected=='Predire avec la regression logistique'):
    st.title('Regression logistique')
    st.write("Précision du modèle:", "{:.2f}%".format(score))
    col1, col2, col3 = st.columns(3)
    
    with col1:
        MDVP_Fo=st.text_input('MDVP_Fo')
    with col2:
        MDVP_Fhi=st.text_input('MDVP_Fhi')
    with col3:
        MDVP_Flo=st.text_input('MDVP_Flo')
        
    with col1:
         MDVP_Jitte_ptg=st.text_input('MDVP_Jitte_ptg')
    with col2:
         MDVP_Jitter_ABS=st.text_input('MDVP_Jitter_ABS')
    with col3:
         MDVP_RAP=st.text_input('MDVP_RAP')
         
    with col1:
         MDVP_PPQ=st.text_input('MDVP_PPQ')
    with col2:
         Jitter_DDP=st.text_input('Jitter_DDP')
    with col3:
         MDVP_Shimmer=st.text_input('MDVP_Shimmer')
         
    with col1:
         MDVP_Shimmer_dB=st.text_input('MDVP_Shimmer_dB')
    with col2:
         Shimmer_APQ3=st.text_input('Shimmer_APQ3')
    with col3:
         Shimmer_APQ5=st.text_input('Shimmer_APQ5')
         
    with col1:
         MDVP_APQ=st.text_input('MDVP_APQ')
    with col2:
         Shimmer_DDA=st.text_input('Shimmer_DDA')
    with col3:
         NHR=st.text_input('NHR')
         
    with col1:
         HNR=st.text_input('HNR')
    with col2:
         RPDE=st.text_input('RPDE')
    with col3:
         DFA=st.text_input('DFA')
         
    with col1:
         spread1=st.text_input('spread1')
    with col2:
         spread2=st.text_input('spread2')
    with col3:
         D2=st.text_input('D2')
    
    with col1:
         PPE=st.text_input('PPE')
    result()
     

else:
    st.title('Predire avec Random forest')
    st.write("Précision du modèle:", "{:.2f}%".format(score_for))
    col1, col2, col3 = st.columns(3)
    
    with col1:
        MDVP_Fo=st.text_input('MDVP_Fo')
    with col2:
        MDVP_Fhi=st.text_input('MDVP_Fhi')
    with col3:
        MDVP_Flo=st.text_input('MDVP_Flo')
        
    with col1:
         MDVP_Jitte_ptg=st.text_input('MDVP_Jitte_ptg')
    with col2:
         MDVP_Jitter_ABS=st.text_input('MDVP_Jitter_ABS')
    with col3:
         MDVP_RAP=st.text_input('MDVP_RAP')
         
    with col1:
         MDVP_PPQ=st.text_input('MDVP_PPQ')
    with col2:
         Jitter_DDP=st.text_input('Jitter_DDP')
    with col3:
         MDVP_Shimmer=st.text_input('MDVP_Shimmer')
         
    with col1:
         MDVP_Shimmer_dB=st.text_input('MDVP_Shimmer_dB')
    with col2:
         Shimmer_APQ3=st.text_input('Shimmer_APQ3')
    with col3:
         Shimmer_APQ5=st.text_input('Shimmer_APQ5')
         
    with col1:
         MDVP_APQ=st.text_input('MDVP_APQ')
    with col2:
         Shimmer_DDA=st.text_input('Shimmer_DDA')
    with col3:
         NHR=st.text_input('NHR')
         
    with col1:
         HNR=st.text_input('HNR')
    with col2:
         RPDE=st.text_input('RPDE')
    with col3:
         DFA=st.text_input('DFA')
         
    with col1:
         spread1=st.text_input('spread1')
    with col2:
         spread2=st.text_input('spread2')
    with col3:
         D2=st.text_input('D2')
    
    with col1:
         PPE=st.text_input('PPE')
    result_forest()
#result 











