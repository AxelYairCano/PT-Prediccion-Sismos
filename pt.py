import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# =============================================================================
# 1. CONFIGURACIÓN
# =============================================================================
# Definimos las constantes físicas y operativas del modelo.
DIRECTORIO_RAIZ = "Temblores_1964-1999"    # Carpeta donde buscarás los archivos CIRES
MAGNITUD_CORTE = 3.0        # (M0) Magnitud mínima para considerar un sismo
HORIZONTE_PREDICCION = 5    # Días hacia el futuro que la IA intentará predecir
VENTANA_MAX_MAG = 7         # Días hacia el pasado para buscar la magnitud máxima

# Configuración dinámica: El paper exige 50 sismos, pero si tienes pocos datos,
# el código se ajustará automáticamente para funcionar en "Modo Demo".
CONFIG = {
    "VENTANA_B_VALUE": 50,  # N=50 sismos para calcular la estadística b-value
    "LAG_STEP": 4           # Saltos para comparar variaciones (t, t-4, t-8...)
}

# Diccionario auxiliar para traducir las fechas que vienen en texto (Español -> Número)
MESES = {
    "ENE": "01", "FEB": "02", "MAR": "03", "ABR": "04", "MAY": "05", "JUN": "06",
    "JUL": "07", "AGO": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DIC": "12"
}

# =============================================================================
# 2. EXTRACCIÓN
# =============================================================================
def extraer_datos_de_archivo(ruta_archivo):
    """
    Abre un archivo de texto crudo (formato CIRES) y busca patrones específicos
    para extraer la fecha, hora, magnitud y coordenadas.
    """
    with open(ruta_archivo, "r", encoding="latin-1", errors="ignore") as f:
        contenido = f.read()

    try:
        # 1. Busca la fecha del sismo: "FECHA DEL SISMO ... : 25/DIC/17"
        fecha_match = re.search(r"FECHA DEL SISMO .*?:\s*(\d{2})/([A-Z]{3})/(\d{2})", contenido)
        
        # 2. Busca la hora: "HORA EPICENTRO ... : 20:23:11.0"
        hora_match = re.search(r"HORA EPICENTRO .*?:\s*([\d:.]+)", contenido)
        
        # 3. Busca la magnitud. El patrón [\s/]* permite leer tanto "Mc=5.0" como "/Mc=5.0"
        mag_match = re.search(r"Mc=\s*([\d.]+)", contenido)
        
        # 4. Busca coordenadas (Latitud y Longitud)
        lat_match = re.search(r"EPICENTRO\s*:\s*([\d.]+)\s*LAT", contenido)
        # Respaldo por si el formato cambia ligeramente en otros archivos
        if not lat_match:
            lat_match = re.search(r"COORDENADAS.*?:\s*([\d.]+)\s*LAT", contenido, re.DOTALL)
        lon_match = re.search(r":\s*([\d.]+)\s*LONG", contenido)

        # Si encontramos fecha, hora y magnitud, procesamos:
        if fecha_match and hora_match and mag_match:
            dia, mes_txt, anio = fecha_match.groups()
            hora = hora_match.group(1).split(".")[0] # Limpiamos milisegundos
            
            # Traducimos "DIC" a "12"
            mes_num = MESES.get(mes_txt, "01")
            
            # Construimos un objeto fecha real
            fecha_str = f"20{anio}-{mes_num}-{dia} {hora}"
            fecha_dt = datetime.strptime(fecha_str, "%Y-%m-%d %H:%M:%S")
            
            # Retornamos el diccionario limpio
            return {
                "datetime": fecha_dt,
                "magnitude": float(mag_match.group(1)),
                "latitude": float(lat_match.group(1)) if lat_match else 0.0,
                "longitude": -float(lon_match.group(1)) if lon_match else 0.0 # Longitud Oeste es negativa
            }
    except Exception:
        return None # Si el archivo está dañado, lo ignoramos
    return None

def construir_catalogo_desde_carpetas(root_dir):
    """
    Explorador de carpetas: Entra a '2017', busca subcarpetas y extrae el primer
    archivo válido de cada evento sísmico.
    """
    print(f"--- Explorando carpeta: {root_dir} ---")
    registros = []
    
    # Listamos todas las subcarpetas (cada carpeta es un evento sísmico)
    subcarpetas = sorted([f.path for f in os.scandir(root_dir) if f.is_dir()])
    
    for carpeta in subcarpetas:
        # Buscamos archivos .txt o .csv dentro del evento
        archivos = [f.name for f in os.scandir(carpeta) if f.name.endswith(('.txt', '.csv'))]
        if not archivos: continue
        
        # Tomamos el primer archivo para evitar duplicados del mismo sismo
        ruta_archivo = os.path.join(carpeta, archivos[0])
        dato = extraer_datos_de_archivo(ruta_archivo)
        
        if dato: registros.append(dato)
            
    # Creamos un DataFrame (Tabla) con todos los sismos encontrados
    df = pd.DataFrame(registros)
    
    # Ordenamos por fecha para calcular leyes sísmicas.
    if not df.empty:
        df = df.sort_values("datetime").reset_index(drop=True)
    return df

# =============================================================================
# 3. LEYES MATEMÁTICAS
# =============================================================================
def calcular_b_value(magnitudes):
    """
    Calcula el valor-b de Gutenberg-Richter usando el Método de Máxima Verosimilitud (Aki & Utsu).
    Fórmula: b = log10(e) / (Promedio(Magnitudes) - Magnitud_Corte)
    Representa la presión tectónica: b bajo = alto estrés (peligro).
    """
    if len(magnitudes) == 0: return 0
    mean_mag = np.mean(magnitudes)
    
    # Evitar división por cero si el promedio es igual al corte
    if mean_mag <= MAGNITUD_CORTE: return 0.5 
    
    return np.log10(np.e) / (mean_mag - MAGNITUD_CORTE)

def generar_vectores_entrenamiento(df):
    """
    Transforma la lista de sismos en Vectores Matemáticos (X) y Objetivos (y).
    """
    # 1. Filtramos sismos muy pequeños (ruido), según indica el paper
    df = df[df["magnitude"] >= MAGNITUD_CORTE].reset_index(drop=True)
    n_sismos = len(df)
    print(f"Sismos válidos para cálculo (M>={MAGNITUD_CORTE}): {n_sismos}")
    
    # --- LÓGICA DE AUTO-AJUSTE (MODO DEMO) ---
    # Si tienes menos de 70 sismos, el código relaja las reglas matemáticas para que no falle.
    if n_sismos < 70:
        print("\n[MODO DEMO] Pocos datos. Reduciendo ventana histórica para prueba.")
        CONFIG["VENTANA_B_VALUE"] = 2  # Usar solo 2 sismos para promedio (Mínimo posible)
        CONFIG["LAG_STEP"] = 1         # Comparar con el sismo inmediatamente anterior
    
    ventana = CONFIG["VENTANA_B_VALUE"]
    step = CONFIG["LAG_STEP"]
    
    # Calculamos desde qué sismo podemos empezar a crear vectores.
    # Necesitamos historia suficiente para calcular las variaciones pasadas.
    start_idx = ventana + (5 * step)
    
    if n_sismos <= start_idx:
        print(f"⚠ ERROR: Faltan datos. Necesitas al menos {start_idx + 1} sismos.")
        return np.array([]), np.array([])

    X, y = [], []

    # Recorremos el catálogo sismo a sismo
    for i in range(start_idx, n_sismos):
        
        # --- CREACIÓN DE ENTRADAS (X) ---
        
        # INPUT 1: Valor-b actual (calculado con los últimos N sismos)
        window_current = df["magnitude"].iloc[i-ventana : i]
        b_current = calcular_b_value(window_current)
        
        # INPUTS 2,3,4,5: Valores-b pasados (para ver la tendencia o variación)
        # Calculamos el b-value hace 'step' pasos, hace 2*'step' pasos, etc.
        b_lags = []
        for k in range(1, 6): 
            lag_idx = k * step
            # Ventana histórica desplazada hacia atrás
            win = df["magnitude"].iloc[i - lag_idx - ventana : i - lag_idx]
            b_lags.append(calcular_b_value(win))
            
        # x1 a x5 son las DIFERENCIAS (Deltas), es decir, cuánto cambió el b-value
        x1 = b_current - b_lags[0]
        x2 = b_lags[0] - b_lags[1]
        x3 = b_lags[1] - b_lags[2]
        x4 = b_lags[2] - b_lags[3]
        x5 = b_lags[3] - b_lags[4]
        
        # INPUT 6: Ley de Bath/Omori (Magnitud máxima en los últimos 7 días)
        fecha_actual = df["datetime"].iloc[i]
        fecha_inicio_semana = fecha_actual - timedelta(days=VENTANA_MAX_MAG)
        
        # Filtramos sismos ocurridos en esa semana previa
        sismos_semana = df[(df["datetime"] >= fecha_inicio_semana) & (df["datetime"] < fecha_actual)]
        x6 = sismos_semana["magnitude"].max() if not sismos_semana.empty else 0
        
        # INPUT 7: Probabilidad teórica (Fórmula exponencial del paper)
        x7 = 10**(-3 * b_current)
        
        # --- CREACIÓN DEL OBJETIVO (y) ---
        # Miramos hacia el FUTURO (5 días adelante) para ver qué pasó realmente
        fecha_limite_futura = fecha_actual + timedelta(days=HORIZONTE_PREDICCION)
        
        sismos_futuros = df[(df["datetime"] > fecha_actual) & (df["datetime"] <= fecha_limite_futura)]
        
        # El target es la magnitud máxima que ocurrió. Si no hubo sismo, es 0.
        y_target = sismos_futuros["magnitude"].max() if not sismos_futuros.empty else 0
        
        # Guardamos el vector
        vector = [x1, x2, x3, x4, x5, x6, x7]
        X.append(vector)
        y.append(y_target)
            
    # Convertimos listas a matrices Numpy (Formato que entiende TensorFlow/Keras)
    return np.array(X), np.array(y)

# =============================================================================
# 4. EJECUCIÓN DEL PROGRAMA
# =============================================================================
print("1. Iniciando lectura de archivos...")
df_catalogo = construir_catalogo_desde_carpetas(DIRECTORIO_RAIZ)

if not df_catalogo.empty:
    print(f"\nSe encontraron {len(df_catalogo)} archivos procesables.")
    print("2. Generando vectores matemáticos para la Red Neuronal...")
    
    X_train, y_train = generar_vectores_entrenamiento(df_catalogo)
    
    if len(X_train) > 0:
        print("\n ¡VECTORES GENERADOS CON ÉXITO!")
        print(f"-> Tienes {len(X_train)} ejemplos para entrenar.")
        print(f"-> Cada ejemplo tiene 7 variables de entrada (X).")
        print(f"-> Cada ejemplo tiene 1 objetivo de salida (y).")
        
        print("\nEjemplo de vector de entrada (Lo que verá la red):")
        print(np.round(X_train[0], 4))
        
        # Guardamos los archivos csv para usarlos en el entrenamiento de la red
        np.save("X_sismos.npy", X_train)
        np.save("y_sismos.npy", y_train)
        print("\nArchivos guardados: X_sismos.npy, y_sismos.npy")
    else:
        print("\n⚠ AVISO: No se generaron vectores. Necesitas más archivos en la carpeta.")
else:
    print("Error: No se encontraron archivos válidos.")