import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.utils import to_categorical


from keras.models import Sequential
from keras.layers import Dense , Dropout

from sklearn.preprocessing import SMOTE, MinMaxScaler

def load_data(filename):
    with open(filename, "rb") as f:
        data = np.load(f)
    return data

def split_data(data, split_ratio):
    train_size = int(len(data) * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

def preprocess_data(data):
    smote = SMOTE()
    data = smote.fit_resample(data)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data

def define_model():
    model = Sequential()
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation="softmax"))
    return model

class Client:
    def __init__(self, data):
        self.data = data
        self.model = None

    def train(self):
        self.model = define_model()
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.model.fit(self.data, epochs=10)

    def predict(self, data):
        predictions = self.model.predict(data)
        return predictions

class Server:
    def __init__(self):
        self.models = []

    def add_model(self, model):
        self.models.append(model)

    def aggregate_models(self):
        global_model = Sequential()
        for model in self.models:
            global_model.add(model.layers[0])
        for i in range(1, len(model.layers)):
            global_model.add(Dense(model.layers[i].units, activation=model.layers[i].activation))
        global_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return global_model

    def update_model(self, global_model):
        for model in self.models:
            model.layers[0].set_weights(global_model.layers[0].get_weights())
            for i in range(1, len(global_model.layers)):
                model.layers[i].set_weights(global_model.layers[i].get_weights())

def evaluate(predictions, labels):
    # Convertir las etiquetas a formato one-hot
    true_labels_one_hot = to_categorical(np.argmax(labels, axis=1), num_classes=5)
    
    # Convertir las predicciones a formato one-hot
    predicted_labels_one_hot = to_categorical(np.argmax(predictions, axis=1), num_classes=5)

    # Calcular las métricas
    accuracy = accuracy_score(np.argmax(true_labels_one_hot, axis=1), np.argmax(predicted_labels_one_hot, axis=1))
    precision = precision_score(np.argmax(true_labels_one_hot, axis=1), np.argmax(predicted_labels_one_hot, axis=1), average='weighted')
    recall = recall_score(np.argmax(true_labels_one_hot, axis=1), np.argmax(predicted_labels_one_hot, axis=1), average='weighted')
    f1 = f1_score(np.argmax(true_labels_one_hot, axis=1), np.argmax(predicted_labels_one_hot, axis=1), average='weighted')

    return accuracy, precision, recall, f1

# Carga de datos
data = pd.DataFrame({
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

#data = load_data("datos.csv")

# División de datos
train_data, test_data = split_data(data, 0.8)

# Preprocesamiento de datos
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Creación de clientes
client1 = Client(train_data[:int(len(train_data) / 2)])
client2 = Client(train_data[int(len(train_data) / 2):])

# Entrenamiento de los clientes
client1.train()
client2.train()

# Agregación de los modelos
server = Server()
server.add_model(client1.model)
server.add_model(client2.model)
global_model = server.aggregate_models()

# Actualización de los modelos
server.update_model(global_model)

# Predicción
predictions = server.models[0].predict(test_data)

# Evaluación
accuracy, precision, recall, f1 = evaluate(predictions, test_data)
print(f"Accuracy: {accuracy}")
print(f"Precisión: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

""" Carga los datos de la investigación.
Divide los datos en un conjunto de entrenamiento y un conjunto de prueba.
Preprocesa los datos para mejorar el rendimiento del modelo.
Define el modelo, que es una red neuronal profunda con tres capas ocultas.
Entrena el modelo en el conjunto de entrenamiento.
Evalúa el modelo en el conjunto de prueba. """


