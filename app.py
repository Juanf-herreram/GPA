import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st 
import numpy as np 
import pickle

GPA = pd.read_csv("Datos GPA.csv")

tab1, tab2, tab3 = st.tabs(['tab1', 'tab2', 'tab3'])
with open('model.pickle', 'rb') as f:
    modelo = pickle.load(f)

with tab1:

    fig, ax = plt.subplots(1, 4, figsize=(10,4))
    #male
    conteo = GPA['male'].value_counts()
    ax[0].bar(conteo.index, conteo.values)
    #campus
    conteo = GPA['campus'].value_counts()
    ax[1].bar(conteo.index, conteo.values)
    #gradMI
    conteo = GPA['gradMI'].value_counts()
    ax[2].bar(conteo.index, conteo.values)
    #colGPA
    ax[3].hist(GPA['colGPA'])
    fig.tight_layout()
    st.pyplot(fig)

with tab2: 
    fig, ax = plt.subplots(1, 3, figsize=(10,4))
    #ColGPA contra male
    sns.boxplot(data=GPA, x= 'male', y = 'colGPA', ax=ax[0])
    ax[0].set_title("GPA vs Sexo")

    #ColGPA contra Campus
    sns.boxplot(data=GPA, x= 'campus', y = 'colGPA', ax=ax[1])
    ax[1].set_title("GPA vs Campus")

    #ColGPA contra Michigan High school
    sns.boxplot(data=GPA, x= 'gradMI', y = 'colGPA', ax=ax[2])
    ax[2].set_title("GPA vs High school")
    fig.tight_layout()
    st.pyplot(fig)

with tab3:
    male = st.selectbox("Sexo", ['Hombre', 'Mujer'])
    if male == 'Hombre': 
        male = 1
    else:
        male = 0
   
    campus = st.selectbox("VIvienda", ['Campus', 'fuera'])
    if campus == 'Campus': 
        campus = 1
    else:
        campus = 0

    gradMI = st.selectbox("High School", ['Michigan', 'Otro'])
    if gradMI == 'Michigan': 
        gradMI = 1
    else:
        gradMI = 0

    if st.button("Predecir"):
        pred = modelo.predict(np.array([[male, campus, gradMI]]))
        st.write(f"Su promedio seria {round(pred[0], 1)}")


