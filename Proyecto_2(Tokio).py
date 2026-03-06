import os
import re
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score
from sklearn.model_selection import StratifiedKFold

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
    "TIPO_MAGNITUD": "Mb" # MAGNITUDES: Mb o Mc
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
# 6. ENTRENAMIENTO Y EVALUACIÓN MODELO
# ==========================================
def entrenar_y_evaluar_modelo(X, y):
    print("\n=== ENTRENANDO RED NEURONAL ===")

    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y  # importante por desbalance
    )

    # Estandarización
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Modelo MLP
    modelo = MLPClassifier(
        hidden_layer_sizes=(10,),
        activation='relu',
        solver='adam',
        max_iter=2000,
        random_state=42
    )

    modelo.fit(X_train, y_train)

    # Predicciones
    y_pred = modelo.predict(X_test)

    # Matriz de confusión
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Métricas derivadas
    sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    mcc = matthews_corrcoef(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n=== RESULTADOS ===")
    print("Matriz de Confusión:")
    print(f"TN: {tn}  FP: {fp}")
    print(f"FN: {fn}  TP: {tp}")

    print("\nMétricas:")
    print(f"Sensibilidad (Recall+): {sensibilidad:.4f}")
    print(f"Especificidad:          {especificidad:.4f}")
    print(f"PPV (Precision):        {ppv:.4f}")
    print(f"NPV:                    {npv:.4f}")
    print(f"MCC:                    {mcc:.4f}")
    print(f"Accuracy:               {accuracy:.4f}")

    return modelo, scaler


# ==========================================
# 7. VALIDACIÓN CRUZADA
# ==========================================
def validacion_cruzada_mlp(X, y, n_splits=5):
    print("\n=== VALIDACIÓN CRUZADA (K-FOLD) ===")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    mcc_scores = []
    sensibilidad_scores = []
    especificidad_scores = []
    accuracy_scores = []

    fold = 1

    for train_index, test_index in skf.split(X, y):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        modelo = MLPClassifier(
            hidden_layer_sizes=(10,),
            activation='relu',
            solver='adam',
            max_iter=2000,
            random_state=42
        )

        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0
        especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        mcc_scores.append(mcc)
        sensibilidad_scores.append(sensibilidad)
        especificidad_scores.append(especificidad)
        accuracy_scores.append(accuracy)

        print(f"\nFold {fold}:")
        print(f"  MCC: {mcc:.4f}")
        print(f"  Sensibilidad: {sensibilidad:.4f}")
        print(f"  Especificidad: {especificidad:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")

        fold += 1

    print("\n=== RESULTADOS PROMEDIO ===")
    print(f"MCC promedio: {np.mean(mcc_scores):.4f} ± {np.std(mcc_scores):.4f}")
    print(f"Sensibilidad promedio: {np.mean(sensibilidad_scores):.4f}")
    print(f"Especificidad promedio: {np.mean(especificidad_scores):.4f}")
    print(f"Accuracy promedio: {np.mean(accuracy_scores):.4f}")


def comparar_arquitecturas(X, y):

    print("\n=== COMPARACIÓN DE ARQUITECTURAS ===")

    arquitecturas = {
        "M1_(5,)": (5,),
        "M2_(10,)": (10,),
        "M3_(15,)": (15,),
        "M4_(10,5)": (10, 5)
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    resultados = {}

    for nombre, arquitectura in arquitecturas.items():

        mcc_scores = []

        for train_index, test_index in skf.split(X, y):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            modelo = MLPClassifier(
                hidden_layer_sizes=arquitectura,
                activation='relu',
                solver='adam',
                max_iter=2000,
                random_state=42
            )

            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)

            mcc = matthews_corrcoef(y_test, y_pred)
            mcc_scores.append(mcc)

        resultados[nombre] = (np.mean(mcc_scores), np.std(mcc_scores))

    print("\n=== RESULTADOS ===")
    for nombre, (mean_mcc, std_mcc) in resultados.items():
        print(f"{nombre} → MCC: {mean_mcc:.4f} ± {std_mcc:.4f}")

    return resultados

def graficos_sismicos(df, X=None, y=None, 
                      guardar=False, 
                      nombre_archivo="analisis_sismico.png"):

    plt.style.use("ggplot")
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle("Análisis Sísmico - Tokio", fontsize=16)

    # ==========================================
    # 1️⃣ Timeline
    # ==========================================
    axes[0,0].scatter(df["datetime"], df["magnitude"],
                      c=df["magnitude"], cmap="YlOrRd", alpha=0.7)
    axes[0,0].set_title("Timeline de Sismos")
    axes[0,0].set_xlabel("Fecha")
    axes[0,0].set_ylabel("Magnitud")

    # ==========================================
    # 2️⃣ Histograma
    # ==========================================
    axes[0,1].hist(df["magnitude"], bins=25, alpha=0.8)
    axes[0,1].axvline(df["magnitude"].mean(),
                      color='red', linestyle='--',
                      label=f"Media: {df['magnitude'].mean():.2f}")
    axes[0,1].legend()
    axes[0,1].set_title("Distribución de Magnitudes")

    # ==========================================
    # 3️⃣ Sismos por Año
    # ==========================================
    sismos_por_anio = df.groupby(df["datetime"].dt.year).size()
    axes[1,0].bar(sismos_por_anio.index, sismos_por_anio.values)
    axes[1,0].set_title("Sismos por Año")
    axes[1,0].set_xlabel("Año")
    axes[1,0].set_ylabel("Cantidad")

    # ==========================================
    # 4️⃣ Magnitudes por Mes
    # ==========================================
    df_temp = df.copy()
    df_temp["Mes"] = df_temp["datetime"].dt.month
    sns.boxplot(x="Mes", y="magnitude", data=df_temp, ax=axes[1,1])
    axes[1,1].set_title("Magnitudes por Mes")

    # ==========================================
    # 5️⃣ Distribución Target
    # ==========================================
    if y is not None:
        axes[2,0].hist(y, bins=20)
        axes[2,0].set_title("Distribución Target")

    # ==========================================
    # 6️⃣ Correlación Features vs Target
    # ==========================================
    if X is not None and y is not None:
        corrs = [np.corrcoef(X[:,i], y)[0,1] for i in range(X.shape[1])]
        axes[2,1].barh(range(len(corrs)), corrs)
        axes[2,1].set_title("Correlación Features vs Target")

    plt.tight_layout()

    # ==========================================
    # 💾 GUARDAR IMAGEN
    # ==========================================
    if guardar:
        plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
        print(f"\nImagen guardada como: {nombre_archivo}")

    plt.show()
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

    if len(X_train) > 0:
        modelo, scaler = entrenar_y_evaluar_modelo(X_train, y_train)
    else:
        print("No hay suficientes datos para entrenar.")

    validacion_cruzada_mlp(X_train, y_train)

    comparar_arquitecturas(X_train, y_train)

    graficos_sismicos(df_catalogo, X_train, y_train, guardar=True, nombre_archivo=f"analisis_sismico_tokio_{CONFIG['TIPO_MAGNITUD']}.png")