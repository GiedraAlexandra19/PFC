import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

def segmentar(datos, longitud_ventana):
    ventanas = []
    for i in range(0, len(datos), longitud_ventana):
        ventanas.append(datos[i:i + longitud_ventana])
    return ventanas

def filtrar(datos, tipo_dato):
    if tipo_dato == "BVP":
        return datos.filter(like="BVP")
    elif tipo_dato == "HR":
        return datos.filter(like="HR")
    elif tipo_dato == "IBI":
        return datos.filter(like="IBI")
    elif tipo_dato == "EDA":
        return datos.filter(like="EDA")
    elif tipo_dato == "ST":
        return datos.filter(like="ST")
    elif tipo_dato == "aceleracion_x":
        return datos.filter(like="aceleracion_x")
    elif tipo_dato == "aceleracion_y":
        return datos.filter(like="aceleracion_y")
    elif tipo_dato == "aceleracion_z":
        return datos.filter(like="aceleracion_z")
    else:
        raise ValueError("Tipo de dato no válido")

def extraer_caracteristicas(ventanas, tipo_dato):
    caracteristicas = pd.DataFrame()
    for ventana in ventanas:
        if tipo_dato == "BVP":
            caracteristicas["RMSSD"] = ventana.mean()
            caracteristicas["SDNN"] = ventana.std()
            caracteristicas["pNN50"] = (ventana.diff() > 0).sum()
        elif tipo_dato == "HR":
            caracteristicas["media"] = ventana.mean()
            caracteristicas["desviacion_tipica"] = ventana.std()
            caracteristicas["varianza"] = ventana.var()
        elif tipo_dato == "IBI":
            caracteristicas["media1"] = ventana.mean()
            caracteristicas["desviacion_tipica"] = ventana.std()
            caracteristicas["varianza"] = ventana.var()
        elif tipo_dato == "EDA":
            caracteristicas["RMSSD1"] = ventana.mean()
            caracteristicas["SDNN"] = ventana.std()
            caracteristicas["pNN50"] = (ventana.diff() > 0).sum()
        elif tipo_dato == "ST":
            caracteristicas["media2"] = ventana.mean()
            caracteristicas["desviacion_tipica"] = ventana.std()
            caracteristicas["varianza"] = ventana.var()
        elif tipo_dato == "aceleracion_x":
            caracteristicas["x_media"] = ventana.mean()
            caracteristicas["x_desviacion_tipica"] = ventana.std()
            caracteristicas["x_varianza"] = ventana.var()
        elif tipo_dato == "aceleracion_y":
            caracteristicas["y_media"] = ventana.mean()
            caracteristicas["y_desviacion_tipica"] = ventana.std()
            caracteristicas["y_varianza"] = ventana.var()
        elif tipo_dato == "aceleracion_z":
            caracteristicas["z_media"] = ventana.mean()
            caracteristicas["z_desviacion_tipica"] = ventana.std()
            caracteristicas["z_varianza"] = ventana.var()

    return caracteristicas

def seleccionar_caracteristicas(caracteristicas):
    correlaciones = caracteristicas.corr()
    caracteristicas_sin_correlaciones = caracteristicas.loc[:, ~caracteristicas.columns.isin(
        correlaciones.index[correlaciones > 0.9].tolist())]
    caracteristicas_sin_faltantes = caracteristicas_sin_correlaciones.dropna()
    return caracteristicas_sin_faltantes

def aprender_modelo(caracteristicas, etiquetas):
    knn = KNeighborsClassifier()
    knn.fit(caracteristicas, etiquetas)
    return knn

def evaluar_modelo(modelo, caracteristicas, etiquetas):
    scores = cross_val_score(modelo, caracteristicas, etiquetas, cv=10)
    precision = np.mean(scores)
    return precision

# Cargar el conjunto de datos
datos = pd.DataFrame({
    "BVP": [0.78, 0.82, 0.88, 0.92, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2],
    "HR": [75, 80, 85, 90, 95, 100, 105, 110, 115, 120],
    "IBI": [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35],
    "EDA": [90, 85, 80, 75, 70, 65, 60, 55, 50, 45],
    "ST": [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    "aceleracion_x": [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4],
    "aceleracion_y": [0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7],
    "aceleracion_z": [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0],
    "stress": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
})

# Segmentar los datos en ventanas de 30 segundos.
longitud_ventana = 30
ventanas = segmentar(datos, longitud_ventana)

# Extraer características de las ventanas.
caracteristicas = []
for tipo_dato in ["BVP", "HR", "IBI", "EDA", "ST", "aceleracion_x", "aceleracion_y", "aceleracion_z"]:
    caracteristicas.append(extraer_caracteristicas(ventanas, tipo_dato))
caracteristicas = pd.concat(caracteristicas, axis=1)

# Seleccionar las características más relevantes.
caracteristicas = seleccionar_caracteristicas(caracteristicas)

# Etiquetar los datos.
etiquetas = datos["stress"].tolist()

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba.
caracteristicas_entrenamiento, caracteristicas_prueba, etiquetas_entrenamiento, etiquetas_prueba = train_test_split(
    caracteristicas, etiquetas, test_size=0.2, random_state=42)

# Aprender un modelo KNN.
modelo = aprender_modelo(caracteristicas_entrenamiento, etiquetas_entrenamiento)

# Evaluar el modelo.
precision = evaluar_modelo(modelo, caracteristicas_prueba, etiquetas_prueba)

print("La precisión del modelo en la validación cruzada es:", precision)

# Crear un nuevo conjunto de datos con datos del usuario.
nuevos_datos = pd.DataFrame({
    "BVP": [1.25, 1.3, 1.35],
    "HR": [125, 130, 135],
    "IBI": [0.3, 0.25, 0.2],
    "EDA": [40, 35, 30],
    "ST": [1.6, 1.7, 1.8],
    "aceleracion_x": [0.5, 0.6, 0.7],
    "aceleracion_y": [-0.8, -0.9, -1.0],
    "aceleracion_z": [-0.3, -0.4, -0.5]
})

# Extraer características de los datos del usuario.
nuevas_caracteristicas = []
for tipo_dato in ["BVP", "HR", "IBI", "EDA", "ST", "aceleracion_x", "aceleracion_y", "aceleracion_z"]:
    nuevas_caracteristicas.append(extraer_caracteristicas([nuevos_datos[tipo_dato]], tipo_dato))
nuevas_caracteristicas = pd.concat(nuevas_caracteristicas, axis=1)

# Predecir si el usuario está estresado.
prediccion = modelo.predict(nuevas_caracteristicas)

# Imprimir la predicción.
if prediccion == 0:
    print("El usuario no está estresado.")
else:
    print("El usuario está estresado.")




    