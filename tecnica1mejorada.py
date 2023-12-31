import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 


def segmentar(datos, longitud_ventana):
    """
  Segmenta los datos en ventanas de la longitud especificada.

  Args:
    datos: Los datos a segmentar.
    longitud_ventana: La longitud de las ventanas en segundos.

  Returns:
    Una lista de ventanas de datos.
  """

    ventanas = []
    for i in range(0, len(datos), longitud_ventana):
        ventanas.append(datos[i:i + longitud_ventana])
    return ventanas

def filtrar(datos, tipo_dato):
    """
  Aplica un filtro a los datos de un tipo de dato específico.

  Args:
    datos: Los datos a filtrar.
    tipo_dato: El tipo de dato de los datos a filtrar.

  Returns:
    Los datos filtrados.
  """
    
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

    """
  Extrae las características de las ventanas de datos.

  Args:
    ventanas: Las ventanas de datos de las que extraer las características.
    tipo_dato: El tipo de dato de las ventanas de datos.

  Returns:
    Un DataFrame con las características extraídas.
  """

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
    """
  Selecciona las características más relevantes para la tarea de clasificación.

  Args:
    caracteristicas: Las características a seleccionar.

  Returns:
    Las características seleccionadas.
  """
    # Se calculan las correlaciones entre las características.
    correlaciones = caracteristicas.corr()

    # Se eliminan las características con correlaciones superiores a 0.9.
    caracteristicas_sin_correlaciones = caracteristicas.loc[:, ~caracteristicas.columns.isin(
        correlaciones.index[correlaciones > 0.9].tolist())]
    
    # Se eliminan las características con valores faltantes.
    caracteristicas_sin_faltantes = caracteristicas_sin_correlaciones.dropna()
    return caracteristicas_sin_faltantes

def distancia_euclidiana(x1, x2):
    """
    Calcula la distancia euclidiana entre dos puntos.
    """
    return np.sqrt(np.sum((x1 - x2)**2))

def knn(entrenamiento, etiquetas_entrenamiento, muestra, k):
    """
    Implementación simple del algoritmo k-NN.
    """
    distancias = [distancia_euclidiana(muestra, x) for x in entrenamiento]
    indices_vecinos = np.argsort(distancias)[:k]
    etiquetas_vecinos = etiquetas_entrenamiento[indices_vecinos]
    clase_predominante = np.bincount(etiquetas_vecinos).argmax()
    return clase_predominante

def evaluar_modelo_knn(caracteristicas_entrenamiento, etiquetas_entrenamiento, caracteristicas_prueba, etiquetas_prueba, k):
    """
    Evalúa el modelo k-NN utilizando métricas de evaluación.

    Args:
        caracteristicas_entrenamiento: Características del conjunto de entrenamiento.
        etiquetas_entrenamiento: Etiquetas del conjunto de entrenamiento.
        caracteristicas_prueba: Características del conjunto de prueba.
        etiquetas_prueba: Etiquetas del conjunto de prueba.
        k: Valor de k para el algoritmo k-NN.

    Returns:
        Un diccionario con las métricas de evaluación.
    """
    predicciones = [knn(caracteristicas_entrenamiento.to_numpy(), etiquetas_entrenamiento, muestra.to_numpy(), k) for _, muestra in caracteristicas_prueba.iterrows()]

    # Calcular métricas
    accuracy = accuracy_score(etiquetas_prueba, predicciones)
    precision = precision_score(etiquetas_prueba, predicciones)
    recall = recall_score(etiquetas_prueba, predicciones)
    f1 = f1_score(etiquetas_prueba, predicciones)

    # Retornar las métricas en un diccionario
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    return metrics

# Cargar el conjunto de datos
datos = pd.DataFrame({
    "BVP": [62, 65, 68, 70, 63, 75, 78, 72, 68, 80, 85, 90, 72, 75, 78, 82, 65, 70, 88, 92],
    "HR": [75, 80, 85, 78, 72, 88, 92, 80, 75, 95, 98, 100, 85, 90, 82, 96, 72, 78, 105, 110],
    "IBI": [800, 750, 710, 770, 825, 680, 660, 750, 800, 630, 610, 600, 710, 670, 730, 640, 825, 770, 570, 550],
    "EDA": [2.3, 2.0, 1.8, 2.5, 2.7, 1.5, 1.3, 2.2, 2.5, 1.0, 0.8, 0.7, 1.7, 1.2, 2.0, 1.1, 2.8, 2.3, 1.2, 0.5],
    "ST": [30, 32, 28, 31, 29, 35, 34, 33, 30, 36, 38, 40, 29, 34, 32, 37, 28, 31, 42, 45],
    "aceleracion_x": [0.1, 0.3, -0.2, 0.0, -0.1, 0.4, -0.5, 0.2, 0.1, -0.3, 0.5, -0.2, 0.0, -0.4, 0.2, 0.1, -0.1, 0.4, -0.3, -0.2],
    "aceleracion_y": [-0.2, 0.1, -0.1, 0.2, 0.1, -0.3, 0.4, 0.3, -0.1, 0.2, -0.4, 0.1, 0.1, -0.2, -0.3, 0.3, 0.0, -0.4, -0.2, -0.7],
    "aceleracion_z": [0.5, 0.4, 0.6, 0.3, -0.2, 0.1, -0.1, 0.0, 0.2, -0.3, 0.1, 0.4, -0.1, 0.3, 0.1, -0.2, 0.2, 0.3, -0.9, -1.0],
    "stress": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
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
caracteristicas_entrenamiento, caracteristicas_prueba, etiquetas_entrenamiento, etiquetas_prueba = train_test_split(caracteristicas, etiquetas, test_size=0.2, random_state=42)

# Aprender un modelo k-NN.
k_valor = 3  # Puedes ajustar el valor de k según tus necesidades.
precision = evaluar_modelo_knn(caracteristicas_entrenamiento, etiquetas_entrenamiento, caracteristicas_prueba, etiquetas_prueba, k_valor)

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
prediccion = knn(caracteristicas_entrenamiento.to_numpy(), etiquetas_entrenamiento, nuevas_caracteristicas.to_numpy(), k_valor)

# Imprimir la predicción.
if prediccion == 0:
    print("El usuario no está estresado.")
else:
    print("El usuario está estresado.")


# Evaluar el modelo k-NN
resultados_metricas = evaluar_modelo_knn(caracteristicas_entrenamiento, etiquetas_entrenamiento, caracteristicas_prueba, etiquetas_prueba, k_valor)

# Imprimir los resultados
print("Resultados de métricas:")
print("Accuracy:", resultados_metricas['accuracy'])
print("Precision:", resultados_metricas['precision'])
print("Recall:", resultados_metricas['recall'])
print("F1 Score:", resultados_metricas['f1_score'])