import streamlit as st
import joblib
import numpy as np

modelo = joblib.load('modelos/modelo_rf.pkl')

st.title('Aplicación Machine Learning - Online Retail Dataset')

st.write('Nombre: TU NOMBRE')
st.write('Código ISIL: TU CODIGO')

cantidad = st.number_input('Cantidad')
precio = st.number_input('Precio Unitario')

if st.button('Predecir'):
    datos = np.array([[cantidad, precio]])
    prediccion = modelo.predict(datos)
    st.write('Resultado:', prediccion)
