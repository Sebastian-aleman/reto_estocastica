"""Datos del reto CONAFOR: especies, costos, monitoreo y plantas por polígono."""
import numpy as np
import pandas as pd

ABREVIATURAS = ["LE", "SA", "SC", "ST", "OC", "OE", "OR", "OS", "PL", "YF"]

NOMBRES = [
    "Agave lechuguilla", "Agave salmiana", "Agave scabra", "Agave striata",
    "Opuntia cantabrigiensis", "Opuntia engelmannii", "Opuntia robusta",
    "Opuntia streptacantha", "Prosopis laevigata", "Yucca filifera",
]

COSTOS = [26, 26, 26, 26, 18, 21, 18, 18, 26.5, 26]
PRESUPUESTO_HA = 12500

VIVOS = {
    2021: [2.1, 9.8, 2.1, 2.1, 2.45, 1.9, 3.65, 3.2, 4.3, 1.3],
    2022: [1.98, 9.40, 1.73, 2.00, 2.27, 1.71, 3.50, 2.84, 3.53, 1.15],
    2023: [1.84, 8.53, 1.29, 1.83, 1.88, 1.40, 3.21, 2.42, 2.89, 0.83],
}

# Cantidad de plantas establecidas en cada polígono por especie (P1..P30)
PLANTAS = {
    "LE": [8, 58, 66, 10, 65, 67, 41, 32, 58, 47, 48, 36, 50, 35, 40,
           70, 56, 45, 56, 69, 54, 75, 44, 10, 44, 72, 74, 67, 60, 34],
    "SA": [46, 263, 236, 52, 280, 306, 209, 252, 269, 209, 233, 182, 189, 186, 311,
           290, 233, 204, 223, 285, 287, 292, 288, 59, 155, 291, 342, 309, 280, 195],
    "SC": [16, 47, 51, 15, 66, 63, 43, 46, 58, 47, 41, 43, 49, 38, 52,
           55, 39, 34, 51, 57, 70, 60, 69, 15, 28, 73, 52, 67, 61, 56],
    "ST": [16, 49, 50, 9, 61, 61, 49, 40, 53, 41, 37, 44, 28, 43, 62,
           54, 46, 33, 48, 75, 62, 68, 51, 11, 31, 50, 55, 66, 53, 58],
    "OC": [11, 60, 71, 15, 92, 81, 48, 73, 57, 66, 48, 51, 43, 39, 68,
           84, 71, 56, 50, 77, 89, 104, 69, 17, 39, 66, 70, 86, 76, 55],
    "OE": [14, 44, 56, 10, 54, 62, 52, 39, 58, 43, 40, 41, 33, 44, 63,
           50, 45, 46, 48, 74, 58, 62, 60, 18, 39, 62, 76, 54, 68, 46],
    "OR": [18, 111, 100, 18, 124, 118, 87, 86, 94, 96, 94, 68, 73, 73, 132,
           105, 83, 82, 93, 107, 126, 109, 122, 17, 65, 125, 100, 120, 109, 91],
    "OS": [12, 95, 78, 20, 114, 91, 60, 79, 91, 71, 65, 72, 76, 61, 89,
           77, 81, 48, 90, 103, 109, 95, 100, 10, 50, 101, 100, 103, 85, 63],
    "PL": [15, 98, 123, 23, 106, 133, 97, 91, 117, 108, 94, 67, 97, 74, 149,
           121, 97, 95, 120, 104, 139, 125, 106, 39, 67, 122, 129, 141, 135, 86],
    "YF": [9, 39, 28, 11, 41, 45, 27, 26, 37, 29, 27, 19, 25, 23, 53,
           40, 32, 17, 33, 25, 44, 43, 39, 8, 25, 31, 44, 33, 36, 23],
}

# Área del polígono (ha); P13 aparece como "05.05" en el documento original
AREAS = [1.28, 6.64, 6.76, 1.38, 8, 7.82, 5.53, 5.64, 7.11, 6.11,
         5.64, 4.92, 5.05, 4.75, 7.97, 7.34, 5.98, 5.4, 6.28, 7.6,
         8, 8, 7.67, 1.47, 4.19, 7.52, 8, 8, 7.56, 5.4]

POLIGONOS = [f"P{k}" for k in range(1, 31)]


def cargar_especies():
    especies = pd.DataFrame({"abrev": ABREVIATURAS, "nombre": NOMBRES})
    especies["genero"] = especies["nombre"].str.split().str[0]
    especies["costo"] = COSTOS
    return especies.set_index("abrev")


def cargar_monitoreo():
    monitoreo = pd.DataFrame(VIVOS, index=ABREVIATURAS)
    monitoreo.index.name = "abrev"
    return monitoreo


def cargar_plantas():
    plantas = pd.DataFrame(PLANTAS, index=POLIGONOS).T
    plantas.index.name = "abrev"
    return plantas


def cargar_areas():
    return pd.Series(AREAS, index=POLIGONOS, name="area_ha")


def calcular_prop_base(monitoreo):
    vivos_2021 = monitoreo[2021]
    return (vivos_2021 / vivos_2021.sum()).rename("prop_base")


def calcular_supervivencia(monitoreo):
    supervivencia = pd.DataFrame({
        "sup_2022_2021": monitoreo[2022] / monitoreo[2021],
        "sup_2023_2022": monitoreo[2023] / monitoreo[2022],
        "sup_acum": monitoreo[2023] / monitoreo[2021],
    })
    supervivencia["mortalidad_acum"] = 1 - supervivencia["sup_acum"]
    return supervivencia


def calcular_densidad(plantas, areas):
    """Densidad (plantas/ha) por especie y polígono, más una fila 'Total'.

    Regresa (densidad, resumen); resumen tiene media, desviación estándar
    muestral, CV y la razón var/media (índice de dispersión) de cada fila
    a través de los 30 polígonos.
    """
    densidad = plantas.div(areas, axis=1)
    densidad.loc["Total"] = densidad.sum(axis=0)
    media = densidad.mean(axis=1)
    varianza = densidad.var(axis=1, ddof=1)
    resumen = pd.DataFrame({
        "media": media,
        "desv_est": np.sqrt(varianza),
        "cv": np.sqrt(varianza) / media,
        "var_media": varianza / media,
    })
    return densidad, resumen
