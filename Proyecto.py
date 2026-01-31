import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# 1. CONFIGURACIÓN GENERAL
# ==========================================
DIRECTORIO_RAIZ = "2017"   # Tu carpeta principal
MAGNITUD_CORTE = 3.0       # M0 solo sismos >= 3.0 
VENTANA_B_VALUE = 50       # N=50 sismos para calcular el b-value
HORIZONTE_PREDICCION = 5   # Días a predecir (Target)
VENTANA_MAX_MAG = 7        # Días hacia atrás para Input x6 (Bath/Omori)

# Diccionario para convertir meses de español a número
MESES = {
    "ENE": "01", "FEB": "02", "MAR": "03", "ABR": "04", "MAY": "05", "JUN": "06",
    "JUL": "07", "AGO": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DIC": "12"
}

# ==========================================
# 2. MÓDULO DE LECTURA Y EXTRACCIÓN DE DATOS
# ==========================================
def extraer_datos_de_archivo(ruta_archivo):
    """Lee un archivo de texto y extrae fecha, hora, lat, long y magnitud usando Regex."""
    with open(ruta_archivo, "r", encoding="latin-1", errors="ignore") as f:
        contenido = f.read()

    # Patrones Regex para encontrar los datos en el texto desordenado
    try:
        # 1. FECHA (Ej: 12/ENE/17)
        fecha_match = re.search(r"FECHA DEL SISMO .*?:\s*(\d{2})/([A-Z]{3})/(\d{2})", contenido)
        # 2. HORA (Ej: 10:26:58)
        hora_match = re.search(r"HORA EPICENTRO .*?:\s*([\d:.]+)", contenido)
        # 3. MAGNITUD (Ej: Mc=5.0)
        mag_match = re.search(r"Mc=\s*([\d.]+)", contenido)
        # 4. LATITUD y LONGITUD
        lat_match = re.search(r"COORDENADAS .*?:\s*([\d.]+)\s*LAT", contenido)
        lon_match = re.search(r":\s*([\d.]+)\s*LONG", contenido)

        if fecha_match and hora_match and mag_match:
            dia, mes_txt, anio = fecha_match.groups()
            hora = hora_match.group(1).split(".")[0] # Quitamos milisegundos si hay
            
            # Construir fecha datetime
            mes_num = MESES.get(mes_txt, "01")
            # Asumimos siglo 2000 (ajustar si tienes datos de 1999)
            fecha_str = f"20{anio}-{mes_num}-{dia} {hora}"
            fecha_dt = datetime.strptime(fecha_str, "%Y-%m-%d %H:%M:%S")
            
            return {
                "datetime": fecha_dt,
                "magnitude": float(mag_match.group(1)),
                "latitude": float(lat_match.group(1)) if lat_match else 0.0,
                "longitude": -float(lon_match.group(1)) if lon_match else 0.0 # Oeste es negativo
            }
    except Exception as e:
        return None # Si falla el regex en algún archivo, lo saltamos
    return None

def construir_catalogo_desde_carpetas(root_dir):
    print(f"--- Explorando carpeta: {root_dir} ---")
    registros = []
    
    # Recorrer carpetas (ej: 2017 -> 2017-01-12...)
    subcarpetas = sorted([f.path for f in os.scandir(root_dir) if f.is_dir()])
    
    for carpeta in subcarpetas:
        archivos = [f.name for f in os.scandir(carpeta) if f.name.endswith(('.txt', '.csv'))]
        
        if not archivos:
            continue
            
        # IMPORTANTE: Tomamos solo el PRIMER archivo de cada carpeta de evento
        # para evitar duplicar el mismo sismo reportado por diferentes estaciones.
        ruta_archivo = os.path.join(carpeta, archivos[0])
        dato = extraer_datos_de_archivo(ruta_archivo)
        
        if dato:
            registros.append(dato)
            
    df = pd.DataFrame(registros)
    if not df.empty:
        # Ordenar cronológicamente es CRÍTICO para el cálculo de b-values
        df = df.sort_values("datetime").reset_index(drop=True)
    
    return df

# ==========================================
# 3. MÓDULO MATEMÁTICO
# ==========================================
def calcular_b_value(magnitudes_window):
    """ Fórmula de Aki & Utsu (Máxima Verosimilitud) """
    if len(magnitudes_window) == 0: return 0
    mean_mag = np.mean(magnitudes_window)
    if mean_mag == MAGNITUD_CORTE: return 0
    # b = log10(e) / (Promedio - M_corte)
    return np.log10(np.e) / (mean_mag - MAGNITUD_CORTE)

def generar_vectores_entrenamiento(df):
    """ Genera X (7 variables) e y (Target) recorriendo evento por evento. """
    
    # [cite_start]Filtrar sismos pequeños (ruido) según el paper
    df = df[df["magnitude"] >= MAGNITUD_CORTE].reset_index(drop=True)
    
    print(f"Sismos válidos para el modelo (M>={MAGNITUD_CORTE}): {len(df)}")
    
    X, y = [], []
    
    # Necesitamos historia suficiente para el primer cálculo (50 eventos + 20 lags)
    start_idx = VENTANA_B_VALUE + 20
    
    if len(df) < start_idx + 1:
        print(f"⚠ ERROR: Insuficientes datos. Se necesitan al menos {start_idx} sismos.")
        return np.array([]), np.array([])

    for i in range(start_idx, len(df)):
        # --- INPUT 1-5: Variaciones del b-value ---
        # b actual (i-50 a i)
        window_current = df["magnitude"].iloc[i-VENTANA_B_VALUE : i]
        b_current = calcular_b_value(window_current)
        
        # b lags (retrasados 4, 8, 12, 16 pasos)
        b_lags = []
        for lag in [4, 8, 12, 16]:
            win = df["magnitude"].iloc[i - lag - VENTANA_B_VALUE : i - lag]
            b_lags.append(calcular_b_value(win))
            
        x1 = b_current - b_lags[0]
        x2 = b_lags[0] - b_lags[1]
        x3 = b_lags[1] - b_lags[2]
        x4 = b_lags[2] - b_lags[3]
        x5 = b_lags[3] - (calcular_b_value(df["magnitude"].iloc[i - 20 - VENTANA_B_VALUE : i - 20]))
        
        # --- INPUT 6: Magnitud Máxima últimos 7 días (Bath/Omori) ---
        fecha_actual = df["datetime"].iloc[i]
        fecha_inicio_semana = fecha_actual - timedelta(days=VENTANA_MAX_MAG)
        
        sismos_semana = df[
            (df["datetime"] >= fecha_inicio_semana) & 
            (df["datetime"] < fecha_actual)
        ]
        x6 = sismos_semana["magnitude"].max() if not sismos_semana.empty else 0
        
        # --- INPUT 7: Probabilidad sismo >= 6.0 ---
        x7 = 10**(-3 * b_current)
        
        # --- TARGET (y): Magnitud máxima próximos 5 días ---
        fecha_limite_futura = fecha_actual + timedelta(days=HORIZONTE_PREDICCION)
        
        sismos_futuros = df[
            (df["datetime"] > fecha_actual) & 
            (df["datetime"] <= fecha_limite_futura)
        ]
        y_target = sismos_futuros["magnitude"].max() if not sismos_futuros.empty else 0
        
        # Guardar vector
        vector = [x1, x2, x3, x4, x5, x6, x7]
        
        # Solo agregar si no hay NaNs (errores de cálculo)
        if not np.isnan(vector).any():
            X.append(vector)
            y.append(y_target)
            
    return np.array(X), np.array(y)

# ==========================================
# 4. EJECUCIÓN PRINCIPAL
# ==========================================

# 1. Crear catálogo desde las carpetas
print("1. Leyendo archivos...")
df_catalogo = construir_catalogo_desde_carpetas(DIRECTORIO_RAIZ)

if not df_catalogo.empty:
    print("\n--- Vista previa del catálogo ---")
    print(df_catalogo.head())
    
    # 2. Generar Matrices para la Red Neuronal
    print("\n2. Procesando matemática del paper...")
    X_train, y_train = generar_vectores_entrenamiento(df_catalogo)
    
    if len(X_train) > 0:
        print("\n✅ ¡ÉXITO! Datos listos para la Red Neuronal.")
        print(f"Dimensiones Entrada (X): {X_train.shape} (Eventos, 7 Variables)")
        print(f"Dimensiones Salida (y): {y_train.shape} (Eventos, Magnitud Futura)")
        
        # Opcional: Guardar para usar en otro script
        # np.save("X_train.npy", X_train)
        # np.save("y_train.npy", y_train)
    else:
        print("\n⚠ ADVERTENCIA: No se generaron vectores. Probablemente tienes muy pocos sismos en la carpeta '2017'.")
        print("El algoritmo necesita una historia de al menos 70 sismos para empezar a calcular.")
else:
    print("Error: No se encontraron archivos válidos en la estructura de carpetas.")
    