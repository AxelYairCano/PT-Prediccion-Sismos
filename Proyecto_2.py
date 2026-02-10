import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
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
    """ Extrae datos sísmicos """
    try:
        with open(ruta_archivo, "r", encoding="latin-1", errors="ignore") as f:
            contenido = f.read()
    except Exception as e:
        print(f" Error leyendo {ruta_archivo}: {e}")
        return None

    try:
        # FECHA: Formato YYYY/MM/DD (nuevo)
        fecha_match = re.search(
            r"FECHA DEL SISMO.*?:\s*(\d{4})/(\d{2})/(\d{2})", 
            contenido
        )
        
        # HORA: Formato HH:MM:SS.D (con punto decimal)
        hora_match = re.search(
            r"HORA EPICENTRO.*?:\s*([\d:]+)\.[\d]",
            contenido
        )
        
        # MAGNITUDES: Extraer Mb Y Mc
        mb_match = re.search(r"Mb=([\d.]+)", contenido)
        mc_match = re.search(r"Mc=([\d.]+)", contenido)
        
        # COORDENADAS: LAT y LONG en líneas separadas
        lat_match = re.search(
            r"COORDENADAS DEL EPICENTRO.*?:\s*([\d.]+)\s*LAT",
            contenido,
            re.DOTALL
        )
        lon_match = re.search(
            r"LAT.*?:\s*([\d.]+)\s*LONG",
            contenido,
            re.DOTALL
        )

        if not (fecha_match and hora_match and (mb_match or mc_match)):
            return None

        anio, mes, dia = fecha_match.groups()
        hora = hora_match.group(1)

        # Construir datetime
        fecha_str = f"{anio}-{mes}-{dia} {hora}"
        fecha_dt = datetime.strptime(fecha_str, "%Y-%m-%d %H:%M:%S")

        # Usar Mb si existe, sino Mc
        magnitud = float(mb_match.group(1)) if mb_match else float(mc_match.group(1))
        
        # Extraer coordenadas (pueden estar en múltiples líneas)
        lat = float(lat_match.group(1)) if lat_match else 0.0
        lon = -float(lon_match.group(1)) if lon_match else 0.0

        return {
            "datetime": fecha_dt,
            "magnitude": magnitud,
            "magnitude_mb": float(mb_match.group(1)) if mb_match else None,
            "magnitude_mc": float(mc_match.group(1)) if mc_match else None,
            "latitude": lat,
            "longitude": lon,
            "archivo_origen": os.path.basename(ruta_archivo)
        }
    except Exception as e:
        print(f" Error procesando {ruta_archivo}: {e}")
        return None


def construir_catalogo_desde_carpeta(root_dir):
    """ Construccion de catálogo """
    print(f"\n--- Leyendo archivos en: {root_dir} ---")
    
    if not os.path.exists(root_dir):
        print(f" Directorio no existe: {root_dir}")
        return pd.DataFrame()
    
    registros = []
    archivos = [
        os.path.join(root_dir, f)
        for f in os.listdir(root_dir)
        if f.endswith(('.txt', '.csv'))
    ]

    print(f"Archivos encontrados: {len(archivos)}")

    for idx, ruta_archivo in enumerate(archivos, 1):
        if idx % 100 == 0:
            print(f"  Procesados {idx}/{len(archivos)}...")
        
        dato = extraer_datos_de_archivo(ruta_archivo)
        if dato:
            registros.append(dato)

    if not registros:
        print(" No se extrajeron datos de ningún archivo")
        return pd.DataFrame()

    df = pd.DataFrame(registros)
    df = df.sort_values("datetime").reset_index(drop=True)
    
    print(f"{len(df)} registros extraídos exitosamente")
    return df


# ==========================================
# 3. ELIMINAR SISMOS DUPLICADOS
# ==========================================
def eliminar_sismos_duplicados(df, ventana_segundos=None, delta_mag=None):
    """ Eliminamos duplicados con parámetros configurables """
    ventana_segundos = ventana_segundos or CONFIG["VENTANA_DUPLICADOS"]
    delta_mag = delta_mag or CONFIG["DELTA_MAG"]
    
    print(f"\nEliminando duplicados (ventana: {ventana_segundos}s, ΔM: {delta_mag})...")
    
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
            "magnitude": sub_df["magnitude"].mean(),
            "magnitude_mb": sub_df["magnitude_mb"].mean(),
            "magnitude_mc": sub_df["magnitude_mc"].mean(),
            "latitude": sub_df["latitude"].mean(),
            "longitude": sub_df["longitude"].mean(),
            "duplicados_encontrados": len(indices) - 1
        }

        eventos_unicos.append(evento_fusionado)
        usado[indices] = True

    df_limpio = pd.DataFrame(eventos_unicos).sort_values("datetime").reset_index(drop=True)
    print(f" {len(df_limpio)} eventos únicos ({len(df) - len(df_limpio)} duplicados eliminados)")
    
    return df_limpio


# ==========================================
# 4. MÓDELOS MATEMÁTICOS
# ==========================================
def calcular_b_value(magnitudes_window):
    """Calcula b-value de Gutenberg-Richter"""
    if len(magnitudes_window) == 0:
        return 0
    mean_mag = np.mean(magnitudes_window)
    if mean_mag == CONFIG["MAGNITUD_CORTE"]:
        return 0
    return np.log10(np.e) / (mean_mag - CONFIG["MAGNITUD_CORTE"])


def generar_vectores_entrenamiento(df):
    """Genera features para entrenamiento de modelo"""
    df_filtrado = df[df["magnitude"] >= CONFIG["MAGNITUD_CORTE"]].reset_index(drop=True)
    print(f"\nSismos válidos (M ≥ {CONFIG['MAGNITUD_CORTE']}): {len(df_filtrado)}")

    X, y, fechas = [], [], []
    start_idx = CONFIG["VENTANA_B_VALUE"] + 20

    if len(df_filtrado) < start_idx + 1:
        print(" Insuficientes sismos para entrenamiento")
        return np.array([]), np.array([]), []

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

        fecha_futura = fecha_actual + timedelta(days=CONFIG["HORIZONTE_PREDICCION"])
        sismos_futuros = df_filtrado[
            (df_filtrado["datetime"] > fecha_actual) &
            (df_filtrado["datetime"] <= fecha_futura)
        ]
        y_target = sismos_futuros["magnitude"].max() if not sismos_futuros.empty else 0

        vector = [x1, x2, x3, x4, x5, x6, x7]

        if not np.isnan(vector).any() and not np.isinf(vector).any():
            X.append(vector)
            y.append(y_target)
            fechas.append(fecha_actual)

    return np.array(X), np.array(y), fechas

# ==========================================
# 5. VISUALIZACIONES
# ==========================================
def crear_visualizaciones(df_catalogo, X_train, y_train, fechas_train):
    """Crea gráficas de análisis sísmico"""
    print("\n[5/5] Generando visualizaciones...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. TIMELINE DE SISMOS
    ax1 = plt.subplot(3, 3, 1)
    ax1.scatter(df_catalogo['datetime'], df_catalogo['magnitude'], 
                alpha=0.6, s=50, c=df_catalogo['magnitude'], cmap='YlOrRd')
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Magnitud')
    ax1.set_title('Timeline de Sismos (1964-1999)')
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. DISTRIBUCIÓN DE MAGNITUDES
    ax2 = plt.subplot(3, 3, 2)
    ax2.hist(df_catalogo['magnitude'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Magnitud')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Distribución de Magnitudes')
    ax2.axvline(df_catalogo['magnitude'].mean(), color='red', linestyle='--', 
                label=f'Media: {df_catalogo["magnitude"].mean():.2f}')
    ax2.legend()
    
    # 3. SISMOS POR AÑO
    ax3 = plt.subplot(3, 3, 3)
    df_catalogo['año'] = df_catalogo['datetime'].dt.year
    sismos_por_año = df_catalogo.groupby('año').size()
    ax3.bar(sismos_por_año.index, sismos_por_año.values, color='teal', alpha=0.7)
    ax3.set_xlabel('Año')
    ax3.set_ylabel('Cantidad de Sismos')
    ax3.set_title('Sismos por Año')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. MAPA DE COORDENADAS
    ax4 = plt.subplot(3, 3, 4)
    scatter = ax4.scatter(df_catalogo['longitude'], df_catalogo['latitude'],
                          c=df_catalogo['magnitude'], s=100, cmap='RdYlBu_r', alpha=0.6)
    ax4.set_xlabel('Longitud')
    ax4.set_ylabel('Latitud')
    ax4.set_title('Mapa de Epicentros (México)')
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Magnitud')
    
    # 5. BOX PLOT DE MAGNITUDES
    ax5 = plt.subplot(3, 3, 5)
    df_catalogo['mes'] = df_catalogo['datetime'].dt.month
    df_catalogo.boxplot(column='magnitude', by='mes', ax=ax5)
    ax5.set_xlabel('Mes')
    ax5.set_ylabel('Magnitud')
    ax5.set_title('Magnitudes por Mes')
    plt.sca(ax5)
    plt.xticks(range(1, 13), ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                               'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
    
    # 6. MAGNITUDES Mb vs Mc
    ax6 = plt.subplot(3, 3, 6)
    valid_data = df_catalogo.dropna(subset=['magnitude_mb', 'magnitude_mc'])
    ax6.scatter(valid_data['magnitude_mb'], valid_data['magnitude_mc'], 
                alpha=0.6, s=50, color='purple')
    ax6.set_xlabel('Magnitud Mb')
    ax6.set_ylabel('Magnitud Mc')
    ax6.set_title('Comparación Mb vs Mc')
    ax6.grid(True, alpha=0.3)
    
    # 7. VARIABLE TARGET (y_train)
    if len(y_train) > 0:
        ax7 = plt.subplot(3, 3, 7)
        ax7.hist(y_train, bins=30, color='coral', edgecolor='black', alpha=0.7)
        ax7.set_xlabel('Magnitud Futura (5 días)')
        ax7.set_ylabel('Frecuencia')
        ax7.set_title('Distribución de Target (y_train)')
        ax7.axvline(y_train.mean(), color='darkred', linestyle='--',
                   label=f'Media: {y_train.mean():.2f}')
        ax7.legend()
    
    # 8. FEATURES (X_train)
    if len(X_train) > 0:
        ax8 = plt.subplot(3, 3, 8)
        feature_names = ['ΔB₁', 'ΔB₂', 'ΔB₃', 'ΔB₄', 'ΔB₅', 'Mag_Max', 'Exp']
        mean_features = X_train.mean(axis=0)
        ax8.barh(feature_names, mean_features, color='lightgreen', edgecolor='black')
        ax8.set_xlabel('Valor Promedio')
        ax8.set_title('Promedio de Features')
        ax8.grid(True, alpha=0.3, axis='x')
    
    # 9. CORRELACIÓN FEATURES vs TARGET
    if len(X_train) > 0:
        ax9 = plt.subplot(3, 3, 9)
        correlaciones = []
        feature_names = ['ΔB₁', 'ΔB₂', 'ΔB₃', 'ΔB₄', 'ΔB₅', 'Mag_Max', 'Exp']
        for i in range(X_train.shape[1]):
            corr = np.corrcoef(X_train[:, i], y_train)[0, 1]
            correlaciones.append(corr)
        
        colors = ['green' if c > 0 else 'red' for c in correlaciones]
        ax9.barh(feature_names, correlaciones, color=colors, alpha=0.7, edgecolor='black')
        ax9.set_xlabel('Correlación con Target')
        ax9.set_title('Correlación Features vs Target')
        ax9.axvline(0, color='black', linestyle='-', linewidth=0.8)
        ax9.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('analisis_sismos.png', dpi=300, bbox_inches='tight')
    #print(" Gráficas guardadas en 'analisis_sismos.png'")

    output_filename = f"analisis_sismos_{CONFIG['DIRECTORIO_RAIZ']}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f" Gráficas guardadas en '{output_filename}' (datos de {CONFIG['DIRECTORIO_RAIZ']})")
    plt.show()
    print(" Visualizaciones generadas exitosamente")


# ==========================================
# 6. EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("----- PIPELINE DE PROCESAMIENTO SÍSMICO -----")
    print("=" * 60)

    print("\n[1/5] Construyendo catálogo completo...")
    df_catalogo = construir_catalogo_desde_carpeta(CONFIG["DIRECTORIO_RAIZ"])

    if df_catalogo.empty:
        print(" No se encontraron datos válidos")
        exit(1)

    print(f"\nEventos leídos (con duplicados): {len(df_catalogo)}")
    print(f"Rango de fechas: {df_catalogo['datetime'].min()} a {df_catalogo['datetime'].max()}")

    print("\n[2/5] Eliminando sismos duplicados...")
    df_catalogo = eliminar_sismos_duplicados(df_catalogo)

    print("\n[3/5] Generando dataset de entrenamiento...")
    X_train, y_train, fechas_train = generar_vectores_entrenamiento(df_catalogo)

    print("\n[4/5] Validando dataset...")
    if len(X_train) > 0:
        print("Dataset listo para usar")
        print(f"   X shape: {X_train.shape}")
        print(f"   y shape: {y_train.shape}")
        print(f"   Rango de magnitudes: [{y_train.min():.2f}, {y_train.max():.2f}]")
        print(f"   Promedio y: {y_train.mean():.2f}")
        print(f"   Desviación estándar: {y_train.std():.2f}")
        
        # Crear visualizaciones
        crear_visualizaciones(df_catalogo, X_train, y_train, fechas_train)
    else:
        print(" No se generaron vectores de entrenamiento")

    print("\n" + "=" * 60)

