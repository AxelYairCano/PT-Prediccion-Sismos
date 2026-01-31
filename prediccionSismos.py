import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# --- 1. CONFIGURACIÓN DE PARÁMETROS ESPECÍFICOS ---
# Configuración de la ANN para predicción de sismos en Chile.
N_ENTRADAS = 7       # Variaciones de valor-b, Magnitud máxima 7 días, Probabilidad M>6.0, etc.
N_OCULTAS = 15       # Neuronas Capa Oculta (Teorema de Kolmogorov)
N_SALIDA = 1         # Magnitud máxima esperada en los próximos 5 días.

# --- 2. DATOS SIMULADOS ---
# X_simulado: 1000 ejemplos de entrenamiento con 7 características de entrada.
# Cada fila representa un día/ventana de tiempo con sus 7 parámetros sísmicos.
X_simulado = np.random.rand(1000, N_ENTRADAS) * 10 

# y_train_simulado: 1000 valores de salida (Magnitud máxima observada en los siguientes 5 días).
# Valores de magnitud simulados entre 4.0 y 8.0.
y_simulado = np.random.uniform(low=4.0, high=8.0, size=(1000, N_SALIDA))

# Inicialización del modelo secuencial
model = Sequential()

# --- Capa Oculta ---
# Neuronas: 15
# Función de Activación: Sigmoide
# input_shape: Define la forma de la entrada (7 variables).
model.add(Dense(
    units=N_OCULTAS,
    activation='sigmoid', 
    input_shape=(N_ENTRADAS,)
))

# --- Capa de Salida ---
# Neuronas: 1 
# Función de Activación: 'linear' para regresión.
model.add(Dense(
    units=N_SALIDA,
    activation='linear' 
))

# 3. Muestra el Modelo
print("--- Resumen del Modelo de la Red Neuronal ---")
model.summary()

## 1. Configuración para el entrenamiento.
# Loss: MSE (Error Cuadrático Medio).
model.compile(
    optimizer='adam',       # adam optimizador
    loss='mse',             # MSE (Error Cuadrático Medio)
    metrics=['mae']         # MAE (Error Absoluto Medio)
)

## 2. ENTRENAMIENTO
print("\n--- Iniciando Entrenamiento Simulado ---")

history = model.fit(
    X_simulado,
    y_simulado,
    epochs=50,              # Número de iteraciones sobre el dataset
    batch_size=32,          # Tamaño de muestras
    verbose=1
)

print("\n--- Entrenamiento Finalizado ---")

# Simular un nuevo conjunto de datos de prueba con 7 entradas
X_test_simulado = np.random.rand(7, N_ENTRADAS) * 10 

# Realizar la predicción
predictions = model.predict(X_test_simulado)

print("\n--- Predicciones Simuladas de Magnitud Máxima (en 5 días) ---")
for i, pred in enumerate(predictions):
    # La salida es la magnitud máxima esperada
    print(f"Entrada {i+1}: Magnitud predicha = {pred[0]:.2f}")
    