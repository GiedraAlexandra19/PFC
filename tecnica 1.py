import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

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
  elif tipo_dato == "aceleracion":
    return datos.filter(like="aceleracion")
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
  if tipo_dato == "BVP":
    caracteristicas["RMSSD"] = ventanas.RMSSD()
    caracteristicas["SDNN"] = ventanas.SDNN()
    caracteristicas["pNN50"] = ventanas.pNN50()
  elif tipo_dato == "HR":
    caracteristicas["media"] = ventanas.mean()
    caracteristicas["desviacion_tipica"] = ventanas.std()
    caracteristicas["varianza"] = ventanas.var()
  elif tipo_dato == "IBI":
    caracteristicas["media"] = ventanas.mean()
    caracteristicas["desviacion_tipica"] = ventanas.std()
    caracteristicas["varianza"] = ventanas.var()
  elif tipo_dato == "EDA":
    caracteristicas["RMSSD"] = ventanas.RMSSD()
    caracteristicas["SDNN"] = ventanas.SDNN()
    caracteristicas["pNN50"] = ventanas.pNN50()
  elif tipo_dato == "ST":
    caracteristicas["media"] = ventanas.mean()
    caracteristicas["desviacion_tipica"] = ventanas.std()
    caracteristicas["varianza"] = ventanas.var()
  elif tipo_dato == "aceleracion":
    caracteristicas["x_media"] = ventanas.x.mean()
    caracteristicas["x_desviacion_tipica"] = ventanas.x.std()
    caracteristicas["x_varianza"] = ventanas.x.var()
    caracteristicas["y_media"] = ventanas.y.mean()
    caracteristicas["y_desviacion_tipica"] = ventanas.y.std()
    caracteristicas["y_varianza"] = ventanas.y.var()
    caracteristicas["z_media"] = ventanas.z.mean()

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

def aprender_modelo(caracteristicas, etiquetas):
  """
  Aprende un modelo KNN a partir de las características y etiquetas proporcionadas.

  Args:
    caracteristicas: Las características de entrenamiento.
    etiquetas: Las etiquetas de entrenamiento.

  Returns:
    El modelo KNN aprendido.
  """

  # Se crea el modelo KNN.
  knn = KNeighborsClassifier()

  # Se entrena el modelo.
  knn.fit(caracteristicas, etiquetas)

  return knn

def evaluar_modelo(modelo, caracteristicas, etiquetas):
  """
  Evalúa un modelo KNN utilizando validación cruzada.

  Args:
    modelo: El modelo KNN a evaluar.
    caracteristicas: Las características de prueba.
    etiquetas: Las etiquetas de prueba.

  Returns:
    La precisión del modelo en la validación cruzada.
  """

  # Se realiza la validación cruzada.
  scores = cross_val_score(modelo, caracteristicas, etiquetas, cv=10)

  # Se calcula la precisión media.
  precision = np.mean(scores)

  return precision

# Cargamos los datos.
datos = pd.read_csv("datos.csv")

# Segmentamos los datos en ventanas de 30 segundos.
ventanas = segmentar(datos, 30)

# Extraemos las características de las ventanas.
caracteristicas = []
for tipo_dato in ["BVP", "HR", "IBI", "EDA", "ST", "aceleracion"]:
  caracteristicas.append(extraer_caracteristicas(ventanas, tipo_dato))
caracteristicas = pd.concat(caracteristicas, axis=1)

# Seleccionamos las características más relevantes.
caracteristicas = seleccionar_caracteristicas(caracteristicas)

# Etiquetamos los datos.
etiquetas = datos["stress"].tolist()

# Aprendemos un modelo KNN.
modelo = aprender_modelo(caracteristicas, etiquetas)

# Evaluamos el modelo.
precision = evaluar_modelo(modelo, caracteristicas, etiquetas)

print("La precisión del modelo en la validación cruzada es:", precision)

# Creamos un nuevo conjunto de datos con los datos del usuario.
nuevos_datos = pd.DataFrame({
    "BVP": [123, 124, 125, ...],
    "HR": [78, 79, 80, ...],
    "IBI": [1.2, 1.3, 1.4, ...],
    "EDA": [100, 101, 102, ...],
    "ST": [0.7, 0.8, 0.9, ...],
    "aceleracion": [1, 2, 3, ...]
})

# Extraemos las características de los datos del usuario.
nuevas_caracteristicas = extraer_caracteristicas(nuevos_datos, ["BVP", "HR", "IBI", "EDA", "ST", "aceleracion"])

# Predecimos si el usuario está estresado.
prediccion = modelo.predict(nuevas_caracteristicas)

# Imprimimos la predicción.
if prediccion == 0:
  print("El usuario no está estresado.")
else:
  print("El usuario está estresado.")