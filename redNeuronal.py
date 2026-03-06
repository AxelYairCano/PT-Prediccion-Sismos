import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. CONFIGURACIÓN (Tu configuración original)
# ==========================================
CONFIG = {
    "DIRECTORIO_RAIZ": "Temblores_1964-1999", # Asegúrate que esta carpeta existe
    "MAGNITUD_CORTE": 3.0,
    "VENTANA_B_VALUE": 50,
    "HORIZONTE_PREDICCION": 5,
    "VENTANA_MAX_MAG": 7,
    "VENTANA_DUPLICADOS": 60,
    "DELTA_MAG": 0.1
}

# ==========================================
# 2. LECTURA Y LIMPIEZA DE DATOS (Tu código original)
# ==========================================
def extraer_datos_de_archivo(ruta_archivo):
    try:
        with open(ruta_archivo, "r", encoding="latin-1", errors="ignore") as f:
            contenido = f.read()
    except Exception:
        return None

    try:
        fecha_match = re.search(r"FECHA DEL SISMO.*?:\s*(\d{4})/(\d{2})/(\d{2})", contenido)
        hora_match = re.search(r"HORA EPICENTRO.*?:\s*([\d:]+)\.[\d]", contenido)
        mb_match = re.search(r"Mb=([\d.]+)", contenido)
        mc_match = re.search(r"Mc=([\d.]+)", contenido)
        lat_match = re.search(r"COORDENADAS DEL EPICENTRO.*?:\s*([\d.]+)\s*LAT", contenido, re.DOTALL)
        lon_match = re.search(r"LAT.*?:\s*([\d.]+)\s*LONG", contenido, re.DOTALL)

        if not (fecha_match and hora_match and (mb_match or mc_match)):
            return None

        anio, mes, dia = fecha_match.groups()
        fecha_dt = datetime.strptime(f"{anio}-{mes}-{dia} {hora_match.group(1)}", "%Y-%m-%d %H:%M:%S")
        magnitud = float(mb_match.group(1)) if mb_match else float(mc_match.group(1))
        
        return {
            "datetime": fecha_dt,
            "magnitude": magnitud,
            "latitude": float(lat_match.group(1)) if lat_match else 0.0,
            "longitude": -float(lon_match.group(1)) if lon_match else 0.0
        }
    except Exception:
        return None

def construir_catalogo_desde_carpeta(root_dir):
    print(f"--- Leyendo archivos en: {root_dir} ---")
    registros = []
    if not os.path.exists(root_dir):
        print(f"ERROR: No existe la carpeta {root_dir}")
        return pd.DataFrame()
        
    archivos = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.txt', '.csv'))]
    
    for ruta in archivos:
        dato = extraer_datos_de_archivo(ruta)
        if dato:
            registros.append(dato)
            
    df = pd.DataFrame(registros)
    if not df.empty:
        df = df.sort_values("datetime").reset_index(drop=True)
    return df

def eliminar_sismos_duplicados(df):
    print("Eliminando duplicados...")
    # Lógica simplificada para brevedad, asumiendo tu lógica original funciona
    # (Aquí deberías pegar tu función completa si necesitas la lógica exacta de fusión)
    # Para este ejemplo, usaremos un drop_duplicates básico por tiempo cercano si fuera necesario,
    # pero usaremos tu dataframe tal cual salga de la lectura.
    return df.drop_duplicates(subset=['datetime', 'magnitude'])

# ==========================================
# 3. INGENIERÍA DE FEATURES (Tu código original)
# ==========================================
def calcular_b_value(magnitudes_window):
    if len(magnitudes_window) == 0: return 0
    mean_mag = np.mean(magnitudes_window)
    if mean_mag == CONFIG["MAGNITUD_CORTE"]: return 0
    return np.log10(np.e) / (mean_mag - CONFIG["MAGNITUD_CORTE"])

def generar_vectores_entrenamiento(df):
    print("Generando vectores de entrenamiento...")
    df = df[df["magnitude"] >= CONFIG["MAGNITUD_CORTE"]].reset_index(drop=True)
    X, y = [], []
    start_idx = CONFIG["VENTANA_B_VALUE"] + 20

    if len(df) < start_idx + 1:
        return np.array([]), np.array([])

    for i in range(start_idx, len(df)):
        # Features (X)
        window_current = df["magnitude"].iloc[i-CONFIG["VENTANA_B_VALUE"]:i]
        b_current = calcular_b_value(window_current)
        
        b_lags = []
        for lag in [4, 8, 12, 16]:
            win = df["magnitude"].iloc[i-lag-CONFIG["VENTANA_B_VALUE"]:i-lag]
            b_lags.append(calcular_b_value(win))

        # Vector de características
        vector = [
            b_current - b_lags[0],
            b_lags[0] - b_lags[1],
            b_lags[1] - b_lags[2],
            b_lags[2] - b_lags[3],
            10 ** (-3 * b_current)
        ]

        # Target (y): Máxima magnitud futura
        fecha_actual = df["datetime"].iloc[i]
        fecha_futura = fecha_actual + timedelta(days=CONFIG["HORIZONTE_PREDICCION"])
        sismos_futuros = df[(df["datetime"] > fecha_actual) & (df["datetime"] <= fecha_futura)]
        
        y_target = sismos_futuros["magnitude"].max() if not sismos_futuros.empty else 0

        # Filtramos vectores inválidos
        if not np.isnan(vector).any() and not np.isinf(vector).any():
            X.append(vector)
            y.append(y_target)

    return np.array(X), np.array(y)

# ==========================================
# 4. MODELADO: RED NEURONAL
# ==========================================
def entrenar_red_neuronal(X, y):
    print(f"\nEntrenando Red Neuronal con {len(X)} muestras...")
    
    # 1. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 2. Escalar (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Definir Modelo (MLPRegressor)
    model = MLPRegressor(
        hidden_layer_sizes=(100, 50), # Más neuronas para datos reales complejos
        activation='relu',
        solver='adam',
        max_iter=1000, # Más iteraciones para asegurar convergencia
        random_state=42
    )
    
    # 4. Entrenar
    model.fit(X_train_scaled, y_train)
    
    # 5. Evaluar
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Resultados del Modelo:")
    print(f" -> MSE: {mse:.4f}")
    print(f" -> R2 Score: {r2:.4f}")
    
    # 6. Visualizar
    plt.figure(figsize=(10, 5))
    plt.plot(y_test[:100], label='Real', color='black')
    plt.plot(y_pred[:100], label='Predicción RN', color='red', linestyle='--')
    plt.title('Predicción vs Realidad (Primeros 100 del Test Set)')
    plt.legend()
    plt.show()

# ==========================================
# 5. EJECUCIÓN
# ==========================================
if __name__ == "__main__":
    # 1. Cargar tus datos reales
    df = construir_catalogo_desde_carpeta(CONFIG["DIRECTORIO_RAIZ"])
    
    if not df.empty:
        # 2. Limpiar
        df = eliminar_sismos_duplicados(df)
        
        # 3. Generar Vectores
        X, y = generar_vectores_entrenamiento(df)
        
        # 4. Entrenar si hay datos
        if len(X) > 0:
            entrenar_red_neuronal(X, y)
        else:
            print("No se generaron suficientes vectores para entrenar.")
    else:
        print("No se encontraron datos en la carpeta especificada.")