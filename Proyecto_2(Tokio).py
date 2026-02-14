import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURACIÓN GENERAL (TOKIO)
# ==========================================
CONFIG = {
    "DIRECTORIO_RAIZ": "Temblores_1964-1999",
    "MAGNITUD_CORTE": 3.0,
    "MAGNITUD_OBJETIVO": 5.0,      
    "VENTANA_B_VALUE": 50,
    "HORIZONTE_PREDICCION": 7,     
    "VENTANA_MAX_MAG": 7,
    "VENTANA_DUPLICADOS": 60,
    "DELTA_MAG": 0.1,
    "TIPO_MAGNITUD": "Mc"
}

# ==========================================
# 2. EXTRACCIÓN DE DATOS
# ==========================================
def extraer_datos_de_archivo(ruta_archivo):
    try:
        with open(ruta_archivo, "r", encoding="latin-1", errors="ignore") as f:
            contenido = f.read()
    except:
        return None

    try:
        fecha_match = re.search(r"FECHA DEL SISMO.*?:\s*(\d{4})/(\d{2})/(\d{2})", contenido)
        hora_match = re.search(r"HORA EPICENTRO.*?:\s*([\d:]+)\.[\d]", contenido)
        mb_match = re.search(r"Mb=([\d.]+)", contenido)
        mc_match = re.search(r"Mc=([\d.]+)", contenido)

        if not (fecha_match and hora_match and (mb_match or mc_match)):
            return None

        anio, mes, dia = fecha_match.groups()
        hora = hora_match.group(1)
        fecha_dt = datetime.strptime(f"{anio}-{mes}-{dia} {hora}", "%Y-%m-%d %H:%M:%S")

        if CONFIG["TIPO_MAGNITUD"] == "Mb":
            if not mb_match:
                return None
            magnitud = float(mb_match.group(1))
        else:
            if not mc_match:
                return None
            magnitud = float(mc_match.group(1))

        return {
            "datetime": fecha_dt,
            "magnitude": magnitud
        }
    except:
        return None


def construir_catalogo_desde_carpeta(root_dir):
    registros = []
    archivos = [
        os.path.join(root_dir, f)
        for f in os.listdir(root_dir)
        if f.endswith(('.txt', '.csv'))
    ]

    for ruta_archivo in archivos:
        dato = extraer_datos_de_archivo(ruta_archivo)
        if dato:
            registros.append(dato)

    df = pd.DataFrame(registros)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


# ==========================================
# 3. ELIMINAR DUPLICADOS
# ==========================================
def eliminar_sismos_duplicados(df):
    ventana_segundos = CONFIG["VENTANA_DUPLICADOS"]
    delta_mag = CONFIG["DELTA_MAG"]

    eventos_unicos = []
    usado = np.zeros(len(df), dtype=bool)

    for i in range(len(df)):
        if usado[i]:
            continue

        evento = df.iloc[i]
        indices = [i]

        for j in range(i + 1, len(df)):
            if usado[j]:
                continue

            dt = abs((df.iloc[j]["datetime"] - evento["datetime"]).total_seconds())
            dm = abs(df.iloc[j]["magnitude"] - evento["magnitude"])

            if dt <= ventana_segundos and dm <= delta_mag:
                indices.append(j)

        sub_df = df.iloc[indices]

        evento_fusionado = {
            "datetime": sub_df["datetime"].min(),
            "magnitude": sub_df["magnitude"].mean()
        }

        eventos_unicos.append(evento_fusionado)
        usado[indices] = True

    df_limpio = pd.DataFrame(eventos_unicos).sort_values("datetime").reset_index(drop=True)
    return df_limpio


# ==========================================
# 4. CÁLCULO b-VALUE
# ==========================================
def calcular_b_value(magnitudes_window):
    if len(magnitudes_window) == 0:
        return 0
    mean_mag = np.mean(magnitudes_window)
    if mean_mag == CONFIG["MAGNITUD_CORTE"]:
        return 0
    return np.log10(np.e) / (mean_mag - CONFIG["MAGNITUD_CORTE"])


# ==========================================
# 5. GENERACIÓN DATASET (CLASIFICACIÓN)
# ==========================================
def generar_vectores_entrenamiento(df):
    df_filtrado = df[df["magnitude"] >= CONFIG["MAGNITUD_CORTE"]].reset_index(drop=True)

    X, y, fechas = [], [], []
    start_idx = CONFIG["VENTANA_B_VALUE"] + 20

    for i in range(start_idx, len(df_filtrado)):
        window_current = df_filtrado["magnitude"].iloc[i-CONFIG["VENTANA_B_VALUE"]:i]
        b_current = calcular_b_value(window_current)

        b_lags = []
        for lag in [4, 8, 12, 16]:
            win = df_filtrado["magnitude"].iloc[i-lag-CONFIG["VENTANA_B_VALUE"]:i-lag]
            b_lags.append(calcular_b_value(win))

        x1 = b_current - b_lags[0]
        x2 = b_lags[0] - b_lags[1]
        x3 = b_lags[1] - b_lags[2]
        x4 = b_lags[2] - b_lags[3]
        x5 = b_lags[3] - calcular_b_value(
            df_filtrado["magnitude"].iloc[i-20-CONFIG["VENTANA_B_VALUE"]:i-20]
        )

        fecha_actual = df_filtrado["datetime"].iloc[i]
        fecha_inicio = fecha_actual - timedelta(days=CONFIG["VENTANA_MAX_MAG"])

        sismos_ventana = df_filtrado[
            (df_filtrado["datetime"] >= fecha_inicio) &
            (df_filtrado["datetime"] < fecha_actual)
        ]

        x6 = sismos_ventana["magnitude"].max() if not sismos_ventana.empty else 0
        x7 = 10 ** (-3 * b_current)

        # ===============================
        # TARGET BINARIO (TOKIO)
        # ===============================
        fecha_futura = fecha_actual + timedelta(days=CONFIG["HORIZONTE_PREDICCION"])
        sismos_futuros = df_filtrado[
            (df_filtrado["datetime"] > fecha_actual) &
            (df_filtrado["datetime"] <= fecha_futura)
        ]

        if not sismos_futuros.empty and \
           (sismos_futuros["magnitude"] >= CONFIG["MAGNITUD_OBJETIVO"]).any():
            y_target = 1
        else:
            y_target = 0

        X.append([x1, x2, x3, x4, x5, x6, x7])
        y.append(y_target)
        fechas.append(fecha_actual)

    return np.array(X), np.array(y), fechas


# ==========================================
# 6. EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":

    print("=== PIPELINE DE PROCESAMIENTO SÍSMICO TOKIO ===")

    df_catalogo = construir_catalogo_desde_carpeta(CONFIG["DIRECTORIO_RAIZ"])

    print(f"Eventos leídos: {len(df_catalogo)}")

    df_catalogo = eliminar_sismos_duplicados(df_catalogo)

    print(f"Eventos tras limpieza: {len(df_catalogo)}")

    b_global = calcular_b_value(df_catalogo["magnitude"])
    print(f"b-value global: {b_global:.4f}")

    X_train, y_train, fechas_train = generar_vectores_entrenamiento(df_catalogo)

    print("\n=== DATASET GENERADO ===")
    print(f"Total muestras: {len(y_train)}")
    print(f"Clase 1 (M ≥ {CONFIG['MAGNITUD_OBJETIVO']}): {np.sum(y_train == 1)}")
    print(f"Clase 0: {np.sum(y_train == 0)}")
    print(f"Porcentaje clase 1: {100*np.mean(y_train):.2f}%")
    print(f"Shape X: {X_train.shape}")
