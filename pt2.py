import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

# ==========================================
# 1. CARGA DE DATOS REALES
# ==========================================
print("--- Cargando vectores matemáticos... ---")
try:
    X = np.load("X_sismos.npy")
    y = np.load("y_sismos.npy")

    # --- TRUCO TEMPORAL PARA EVITAR EL ERROR DE 0 DATOS ---
    if len(X) < 10:
        print("⚠️ ADVERTENCIA: Pocos datos. Duplicando para prueba técnica...")
        # Repetimos los datos 20 veces para tener "bulto"
        X = np.tile(X, (20, 1))
        y = np.tile(y, (20,))

    print(f"Datos cargados exitosamente.")
    print(f"Total de ejemplos: {len(X)}")
except FileNotFoundError:
    print("ERROR: No se encuentran 'X_sismos.npy' o 'y_sismos.npy'.")
    print("Ejecuta primero el script de carga de datos.")
    exit()

# ==========================================
# 2. PREPARACIÓN (SPLIT CRONOLÓGICO)
# ==========================================
# El paper NO usa división aleatoria (train_test_split aleatorio) porque es una serie temporal.
# Se entrena con el pasado y se prueba con el futuro[cite: 602].

# Usaremos el 80% más antiguo para entrenar y el 20% más reciente para validar
punto_corte = int(len(X) * 0.80)

X_train = X[:punto_corte]
y_train = y[:punto_corte]

X_test = X[punto_corte:]
y_test = y[punto_corte:]

print(f"Entrenamiento (Pasado): {len(X_train)} vectores")
print(f"Prueba (Futuro): {len(X_test)} vectores")

# ==========================================
# 3. ARQUITECTURA DE LA RED (SEGÚN PAPER)
# ==========================================
# Configuración extraída del Paper y tu Presentación:
# - Feedforward (Secuencial)
# - 7 Entradas
# - 15 Neuronas Ocultas (Teorema Kolmogorov: 2n + 1) [cite: 454]
# - Activación Sigmoide
# - 1 Salida (Regresión lineal para magnitud)

model = Sequential()

# Capa Oculta
model.add(Dense(
    units=15, 
    activation='sigmoid', 
    input_shape=(7,) # Las 7 variables matemáticas (x1...x7)
))

# Capa de Salida
model.add(Dense(
    units=1, 
    activation='linear' # Queremos predecir un valor continuo (Magnitud)
))

# ==========================================
# 4. COMPILACIÓN Y ENTRENAMIENTO
# ==========================================
# El paper usa Backpropagation. 'Adam' es una versión optimizada moderna estándar.
# Loss 'mse' (Error Cuadrático Medio) es ideal para predecir magnitudes.

model.compile(
    optimizer='adam', 
    loss='mse', 
    metrics=['mae'] # Mean Absolute Error (Error promedio en grados de magnitud)
)

print("\n--- Iniciando Entrenamiento... ---")
# El paper menciona entrenar hasta converger. 500 épocas es un buen inicio[cite: 528].
history = model.fit(
    X_train, 
    y_train,
    epochs=50, 
    batch_size=10, # Batch pequeño ayuda a generalizar mejor en pocos datos
    verbose=1,
    validation_data=(X_test, y_test)
)

# ==========================================
# 5. EVALUACIÓN Y GUARDADO
# ==========================================
print("\n--- Evaluación Final en Datos de Prueba ---")
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Error Promedio (MAE): {mae:.4f} grados de magnitud")

# Guardar el modelo entrenado ("El Cerebro")
model.save("modelo_sismos_chile.h5")
print("\nModelo guardado como 'modelo_sismos_chile.h5'")

# --- PRUEBA RÁPIDA DE PREDICCIÓN ---
print("\n--- Ejemplo de Predicciones vs Realidad ---")
predicciones = model.predict(X_test[:5]) # Predecir los primeros 5 del test

for i in range(len(predicciones)):
    print(f"Caso {i+1}: Real={y_test[i]:.2f} vs Predicho={predicciones[i][0]:.2f}")

# ==========================================
# 6. EVALUACIÓN CIENTÍFICA (MÉTRICAS DEL PAPER)
# ==========================================

# 1. Definir el Umbral (T) según el paper 
# Se calcula usando los datos de ENTRENAMIENTO (históricos), no los de prueba.
media_mag = np.mean(y_train)
std_mag = np.std(y_train)
UMBRAL = media_mag + (0.6 * std_mag)

print(f"\n--- Parámetros de Alarma ---")
print(f"Magnitud Promedio Histórica: {media_mag:.2f}")
print(f"Umbral de Activación (T): {UMBRAL:.2f}")

# 2. Convertir Predicciones y Realidad a Binario (1=Sismo Fuerte, 0=Ruido)
# Aplanamos los arrays para compararlos
y_real_bin = (y_test >= UMBRAL).astype(int)
y_pred_bin = (predicciones.flatten() >= UMBRAL).astype(int)

# 3. Calcular Matriz de Confusión 
TP = np.sum((y_pred_bin == 1) & (y_real_bin == 1)) # Alarma Correcta
TN = np.sum((y_pred_bin == 0) & (y_real_bin == 0)) # Silencio Correcto
FP = np.sum((y_pred_bin == 1) & (y_real_bin == 0)) # Falsa Alarma
FN = np.sum((y_pred_bin == 0) & (y_real_bin == 1)) # Sismo No Detectado

print(f"\n--- Matriz de Confusión ---")
print(f"Verdaderos Positivos (TP): {TP}")
print(f"Verdaderos Negativos (TN): {TN}")
print(f"Falsos Positivos (FP): {FP}")
print(f"Falsos Negativos (FN): {FN}")

# 4. Calcular Métricas Finales [cite: 494, 513]
# Evitamos división por cero con un pequeño epsilon o chequeo
def calc_prob(num, den):
    return (num / den * 100) if den > 0 else 0

P1 = calc_prob(TP, TP + FP)  # Confiabilidad de la alarma
P0 = calc_prob(TN, TN + FN)  # Confiabilidad de la calma
Sn = calc_prob(TP, TP + FN)  # Sensibilidad (¿Cuántos sismos atrapamos?)
Sp = calc_prob(TN, TN + FP)  # Especificidad (¿Qué tan buenos somos ignorando ruido?)

print(f"\n--- RESULTADOS FINALES DEL PROYECTO ---")
print(f"P1 (Probabilidad Acierto Alarma): {P1:.2f}%")
print(f"P0 (Probabilidad Acierto Calma):  {P0:.2f}%")
print(f"Sensibilidad (Sn): {Sn:.2f}%")
print(f"Especificidad (Sp): {Sp:.2f}%")

# Interpretación automática
if P1 > 50 and P0 > 70:
    print("\nCONCLUSIÓN: El modelo es prometedor (supera el azar).")
else:
    print("\nCONCLUSIÓN: El modelo necesita más datos o ajuste (rendimiento bajo).")