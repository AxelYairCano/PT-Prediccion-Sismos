import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CONFIGURACIÓN GENERAL
# ==========================================
CONFIG = {
    "DIRECTORIO_RAIZ": "Temblores_1964-1999",
    "MAGNITUD_CORTE": 3.0,
    "VENTANA_B_VALUE": 50,
    "HORIZONTE_PREDICCION": 5,
    "VENTANA_MAX_MAG": 7,
    "VENTANA_DUPLICADOS": 60,
    "DELTA_MAG": 0.1
}

# ==========================================
# 2. EXTRACCIÓN DE DATOS
# ==========================================
def extraer_datos_de_archivo(ruta_archivo):
    try:
        with open(ruta_archivo, "r", encoding="latin-1", errors="ignore") as f:
            contenido = f.read()
    except Exception as e:
        print(f"Error leyendo {ruta_archivo}: {e}")
        return None

    try:
        fecha_match = re.search(r"FECHA DEL SISMO.*?:\s*(\d{4})/(\d{2})/(\d{2})", contenido)
        hora_match = re.search(r"HORA EPICENTRO.*?:\s*([\d:]+)\.[\d]", contenido)

        # ------------------------------------------
        # Mb: Magnitud de ondas de cuerpo (ondas P)
        # Mc: Magnitud de coda (energía tardía)
        # Se usa Mb si existe; en caso contrario Mc
        # ------------------------------------------
        mb_match = re.search(r"Mb=([\d.]+)", contenido)
        mc_match = re.search(r"Mc=([\d.]+)", contenido)

        lat_match = re.search(r"COORDENADAS DEL EPICENTRO.*?:\s*([\d.]+)\s*LAT", contenido, re.DOTALL)
        lon_match = re.search(r"LAT.*?:\s*([\d.]+)\s*LONG", contenido, re.DOTALL)

        if not (fecha_match and hora_match and (mb_match or mc_match)):
            return None

        anio, mes, dia = fecha_match.groups()
        hora = hora_match.group(1)
        fecha_dt = datetime.strptime(f"{anio}-{mes}-{dia} {hora}", "%Y-%m-%d %H:%M:%S")

        if mb_match:
            magnitud = float(mb_match.group(1))
            tipo_magnitud = "Mb"
        elif mc_match:
            magnitud = float(mc_match.group(1))
            tipo_magnitud = "Mc"
        else:
            return None

        lat = float(lat_match.group(1)) if lat_match else 0.0
        lon = -float(lon_match.group(1)) if lon_match else 0.0

        return {
            "datetime": fecha_dt,
            "magnitude": magnitud,
            "magnitude_mb": float(mb_match.group(1)) if mb_match else None,
            "magnitude_mc": float(mc_match.group(1)) if mc_match else None,
            "tipo_magnitud": tipo_magnitud,
            "latitude": lat,
            "longitude": lon,
            "archivo_origen": os.path.basename(ruta_archivo)
        }

    except Exception as e:
        print(f"Error procesando {ruta_archivo}: {e}")
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
    return df.sort_values("datetime").reset_index(drop=True)


# ==========================================
# 3. ELIMINAR SISMOS DUPLICADOS
# ==========================================
def eliminar_sismos_duplicados(df, ventana_segundos=None, delta_mag=None):

    ventana_segundos = ventana_segundos or CONFIG["VENTANA_DUPLICADOS"]
    delta_mag = delta_mag or CONFIG["DELTA_MAG"]

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

        tipos = sub_df["tipo_magnitud"].unique()
        tipo_final = tipos[0] if len(tipos) == 1 else "Mixto"

        evento_fusionado = {
            "datetime": sub_df["datetime"].min(),
            "magnitude": sub_df["magnitude"].mean(),
            "magnitude_mb": sub_df["magnitude_mb"].mean(),
            "magnitude_mc": sub_df["magnitude_mc"].mean(),
            "tipo_magnitud": tipo_final,
            "latitude": sub_df["latitude"].mean(),
            "longitude": sub_df["longitude"].mean(),
            "duplicados_encontrados": len(indices) - 1
        }

        eventos_unicos.append(evento_fusionado)
        usado[indices] = True

    return pd.DataFrame(eventos_unicos).sort_values("datetime").reset_index(drop=True)


# ==========================================
# 4. MODELO MATEMÁTICO
# ==========================================
def calcular_b_value(magnitudes_window):
    if len(magnitudes_window) == 0:
        return 0
    mean_mag = np.mean(magnitudes_window)
    if mean_mag == CONFIG["MAGNITUD_CORTE"]:
        return 0
    return np.log10(np.e) / (mean_mag - CONFIG["MAGNITUD_CORTE"])


# ==========================================
# 5. VISUALIZACIONES
# ==========================================
def crear_visualizaciones(df_catalogo):

    fig = plt.figure(figsize=(16, 10))

    # Mb vs Mc mejorada
    ax = plt.subplot(1, 1, 1)
    valid_data = df_catalogo.dropna(subset=['magnitude_mb', 'magnitude_mc'])

    ax.scatter(valid_data['magnitude_mb'],
               valid_data['magnitude_mc'],
               alpha=0.6,
               s=50)

    min_val = min(valid_data['magnitude_mb'].min(),
                  valid_data['magnitude_mc'].min())
    max_val = max(valid_data['magnitude_mb'].max(),
                  valid_data['magnitude_mc'].max())

    ax.plot([min_val, max_val], [min_val, max_val], 'r--')

    ax.set_xlabel('Magnitud Mb')
    ax.set_ylabel('Magnitud Mc')
    ax.set_title('Comparación Mb vs Mc')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ==========================================
# 6. MAIN
# ==========================================
if __name__ == "__main__":

    df_catalogo = construir_catalogo_desde_carpeta(CONFIG["DIRECTORIO_RAIZ"])

    print("\nEventos leídos:", len(df_catalogo))

    df_catalogo = eliminar_sismos_duplicados(df_catalogo)

    print("\nDistribución por tipo:")
    print(df_catalogo["tipo_magnitud"].value_counts())

    df_mb = df_catalogo[df_catalogo["tipo_magnitud"] == "Mb"]
    df_mc = df_catalogo[df_catalogo["tipo_magnitud"] == "Mc"]

    b_total = calcular_b_value(df_catalogo["magnitude"])
    b_mb = calcular_b_value(df_mb["magnitude"])
    b_mc = calcular_b_value(df_mc["magnitude"])

    print("\nComparación de b-values:")
    print(f"Total: {b_total:.4f}")
    print(f"Mb:    {b_mb:.4f}")
    print(f"Mc:    {b_mc:.4f}")

    df_comparacion = df_catalogo.dropna(subset=['magnitude_mb', 'magnitude_mc'])
    df_comparacion["dif_mb_mc"] = (
        df_comparacion["magnitude_mb"] -
        df_comparacion["magnitude_mc"]
    )

    print("\nDiferencia Mb - Mc")
    print("Media:", df_comparacion["dif_mb_mc"].mean())
    print("Std:", df_comparacion["dif_mb_mc"].std())

    crear_visualizaciones(df_catalogo)
