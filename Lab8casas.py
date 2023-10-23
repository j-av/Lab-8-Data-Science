import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from math import sqrt
import streamlit as st
import matplotlib.pyplot as plt

data = pd.read_csv('C:/Users/USUARIO/Desktop/UVG/Data science/lab8VF/houses_to_rent_v2.csv')

data['animal'] = data['animal'].replace({'acept': 1, 'not acept': 0})
data['furniture'] = data['furniture'].replace({'furnished': 1, 'not furnished': 0})
data['floor'] = data['floor'].replace('-', 0)
city_mapping = {
    'São Paulo': 1,
    'Porto Alegre': 2,
    'Rio de Janeiro': 3,
    'Campinas': 4,
    'Belo Horizonte': 5
}
data['city'] = data['city'].replace(city_mapping)

# Crear el modelo de Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Seleccionar las características y el objetivo
features = ['city', 'area', 'rooms', 'bathroom', 'parking spaces', 'fire insurance (R$)']
X = data[features]
y = data['rent amount (R$)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
gb_model.fit(X_train, y_train)

# Función para realizar predicciones
def make_prediction(city, area, rooms, bathroom, parking_spaces, fire_insurance):
    input_data = [[city, area, rooms, bathroom, parking_spaces, fire_insurance]]
    prediction = gb_model.predict(input_data)
    return prediction[0]

st.title('Predicción de Alquiler de Viviendas')

# Crear widgets interactivos para ingresar los valores
unique_cities = list(city_mapping.keys())  # Obtener las claves del diccionario de ciudades
city = st.radio('Selecciona una ciudad:', unique_cities)
area = st.slider('Área (m²):', 1, 500)
rooms = st.slider('Número de Habitaciones:', 1, 10)
bathroom = st.slider('Número de Baños:', 1, 5)
parking_spaces = st.slider('Plazas de Aparcamiento:', 0, 5)
fire_insurance = st.slider('Seguro contra Incendios (R$):', 0, 500)

# Crear un botón para calcular la predicción
if st.button('Calcular Predicción'):
    city_code = city_mapping[city]
    prediction = make_prediction(city_code, area, rooms, bathroom, parking_spaces, fire_insurance)
    st.markdown(f'### El monto de alquiler estimado (R$) es:')
    st.write(f'## {prediction:.2f}')

# Gráfico de barras para la variable 'city'
st.title('Distribución de viviendas por ciudad')
city_names = list(city_mapping.keys())
city_counts = data['city'].replace({v: k for k, v in city_mapping.items()}).value_counts()
plt.figure(figsize=(12, 8))
city_counts.plot(kind='bar', color='green')
plt.xlabel('Ciudad')
plt.ylabel('Cantidad')
plt.xticks(range(len(city_names)), city_names)  # Establecer etiquetas de ciudad
st.pyplot(plt)
