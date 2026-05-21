import streamlit as st
import joblib
import numpy as np
modelo = joblib.load('modelos/modelo_rf.pkl')
st.title('Aplicación Machine Learning - Online Retail Dataset')
st.write('Nombre: TU NOMBRE COMPLETO')
st.write('Código ISIL: TU CÓDIGO ISIL')
st.markdown('[Ver Google Colab](PEGAR_LINK_COLAB_AQUI)')
cantidad = st.number_input('Cantidad')
precio = st.number_input('Precio Unitario')
pais = st.number_input('Código País')
if st.button('Predecir'):
datos = np.array([[cantidad, precio, pais]])
prediccion = modelo.predict(datos)
st.write('Resultado:', prediccion)
