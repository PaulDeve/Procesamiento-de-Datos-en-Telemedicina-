from sklearn.linear_model 
import LinearRegression
import numpy as np

# Datos: Edad (años) y glucosa (mg/dL)
edad = np.array([25, 30, 35, 40, 45, 50, 55, 60, 65, 70]).reshape(-1, 1)
glucosa = np.array([95, 98, 101, 105, 109, 113, 118, 122, 130, 135])

# Entrenar modelo de regresión lineal
modelo = LinearRegression().fit(edad, glucosa)

# Predicciones para nuevas edades
nuevas_edades = np.array([33, 47, 58]).reshape(-1, 1)
predicciones = modelo.predict(nuevas_edades)

# Mostrar predicciones
for e, g in zip(nuevas_edades.ravel(), predicciones):
    print(f"Edad: {e} años -> Glucosa estimada: {g:.2f} mg/dL")
