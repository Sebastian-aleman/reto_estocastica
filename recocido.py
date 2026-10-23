"""Recocido simulado para acomodar especies en la malla tresbolillo toroidal.

Representación: `asig` es un arreglo con la especie de cada nodo (0..9). No hay
nodos vacíos: la malla tiene exactamente tantos nodos como plantas
(existentes + nuevas) y todos tienen grado 6.
"""
import math
import random

import numpy as np
import pandas as pd

try:
    from numba import njit
    HAY_NUMBA = True
except ImportError:
    HAY_NUMBA = False

    def njit(funcion=None, **opciones):
        return funcion if funcion is not None else (lambda f: f)

N_ESPECIES = 10
SIN_ASIGNAR = -1          # solo mientras se construye una asignación
TOTAL_EXISTENTES = 129    # plantas existentes por ha
SECUENCIA_ACTUAL = ["SA", "PL", "OS", "OR", "SC", "OC", "SA", "OE", "ST", "SA", "OE",
                    "PL", "SA", "OC", "LE", "SA", "OS", "YF", "SA", "OR", "PL"]


# ---------------------------------------------------------------- Datos de entrada

def redondear_a_total(valores, total):
    """Redondeo por mayor residuo: enteros que suman exactamente `total`."""
    escalados = valores / valores.sum() * total
    enteros = np.floor(escalados).astype(int)
    faltan = total - enteros.sum()
    enteros.iloc[np.argsort(-(escalados - enteros).to_numpy())[:faltan]] += 1
    return enteros


def cargar_cantidades(carpeta, n_nodos, densidad_ha):
    """Plantas existentes y nuevas por especie para una malla de n_nodos.

    Las existentes son 129 por ha, o sea 129/densidad_ha de los nodos; las nuevas
    llenan el resto, así que existentes + nuevas = n_nodos (sin vacíos). El reparto
    por especie sigue X_esperado (existentes) y ha_media de esc1 (nuevas).
    """
    x_esperado = pd.read_csv(carpeta / "X_esperado.inc", sep=r"\s+", header=None,
                             index_col=0).iloc[:, 0]
    compras = pd.read_csv(carpeta / "compras_esperadas.csv")
    ha_media = compras[compras["escenario"] == "esc1"].set_index("especie")["ha_media"]
    total_existentes = int(round(TOTAL_EXISTENTES * n_nodos / densidad_ha))
    return (redondear_a_total(x_esperado, total_existentes),
            redondear_a_total(ha_media, n_nodos - total_existentes))


def preparar_malla(nodos, aristas):
    """Aristas con índices desde 0 y matriz de vecinos (n, 6); todos los nodos deben tener grado 6."""
    n = len(nodos)
    aristas_idx = np.array(aristas) - 1
    vecinos = [[] for _ in range(n)]
    for a, b in aristas_idx:
        vecinos[a].append(b)
        vecinos[b].append(a)
    assert all(len(lista) == 6 for lista in vecinos), "la malla debe ser toroidal (grado 6)"
    return aristas_idx, np.array(vecinos, dtype=np.int64)


def competencia_total(asig, aristas_idx, W_mat):
    return W_mat[asig[aristas_idx[:, 0]], asig[aristas_idx[:, 1]]].sum()


def contenido(cantidades):
    """Lista de contenidos: cada especie repetida según su cantidad."""
    return np.repeat(np.arange(N_ESPECIES), cantidades.to_numpy())


# ---------------------------------------------------------------- Escenarios y líneas base

def generar_existentes(existentes, n_nodos, semilla):
    """Coloca las plantas existentes en nodos al azar. Regresa (asig, fijo).

    Los nodos libres quedan en SIN_ASIGNAR; las líneas base los llenan todos.
    """
    rng = np.random.default_rng(semilla)
    asig = np.full(n_nodos, SIN_ASIGNAR, dtype=np.int64)
    nodos_exist = rng.choice(n_nodos, existentes.sum(), replace=False)
    asig[nodos_exist] = contenido(existentes)
    fijo = np.zeros(n_nodos, dtype=bool)
    fijo[nodos_exist] = True
    return asig, fijo


def linea_base_actual(asig_exist, fijo, especies):
    """Secuencia de 21 plantas repetida cíclicamente en todos los nodos libres, fila por fila.

    Los nodos ya están en orden de fila (malla.py los genera así). Los nodos con
    planta existente se excluyen, así que la secuencia se los salta y sigue.
    """
    asig = asig_exist.copy()
    libres = np.flatnonzero(~fijo)
    codigos = [especies.get_loc(s) for s in SECUENCIA_ACTUAL]
    asig[libres] = [codigos[k % len(codigos)] for k in range(len(libres))]
    return asig


def cantidades_linea_base(especies, n_nuevas):
    """Plantas nuevas por especie que pone la secuencia actual repetida en n_nuevas nodos."""
    plantas = [SECUENCIA_ACTUAL[k % len(SECUENCIA_ACTUAL)] for k in range(n_nuevas)]
    return pd.Series(plantas).value_counts().reindex(especies, fill_value=0)


def linea_base_aleatoria(asig_exist, fijo, nuevas, semilla):
    """Mismas cantidades que el recocido, repartidas al azar en los nodos libres."""
    rng = np.random.default_rng(semilla)
    asig = asig_exist.copy()
    libres = np.flatnonzero(~fijo)
    assert nuevas.sum() == len(libres), "las nuevas deben llenar todos los nodos libres"
    asig[libres] = rng.permutation(contenido(nuevas))
    return asig


# ---------------------------------------------------------------- Recocido simulado

@njit(cache=True)
def _delta(asig, vecinos, W, u, v):
    """Cambio del objetivo al intercambiar el contenido de u y v (solo sus vecinos)."""
    a = asig[u]
    b = asig[v]
    d = 0.0
    for k in vecinos[u]:
        if k != v:
            d += W[b][asig[k]] - W[a][asig[k]]
    for k in vecinos[v]:
        if k != u:
            d += W[a][asig[k]] - W[b][asig[k]]
    return d


@njit(cache=True)
def _muestrear_empeoramientos(asig, movibles, vecinos, W, n_muestras, semilla):
    random.seed(semilla)
    deltas = np.zeros(n_muestras)
    m = len(movibles)
    k = 0
    intentos = 0
    while k < n_muestras and intentos < 100 * n_muestras:
        intentos += 1
        u = movibles[int(random.random() * m)]
        v = movibles[int(random.random() * m)]
        if asig[u] != asig[v]:
            d = _delta(asig, vecinos, W, u, v)
            if d > 1e-12:
                deltas[k] = d
                k += 1
    return deltas[:k]


@njit(cache=True)
def _nucleo_recocido(asig, movibles, vecinos, W, valor, T0, alfa, L, T_min,
                     max_sin_mejora, fraccion_frio, max_niveles, semilla):
    random.seed(semilla)
    m = len(movibles)
    mejor = asig.copy()
    valor_mejor = valor
    hist_T = np.zeros(max_niveles)
    hist_actual = np.zeros(max_niveles)
    hist_mejor = np.zeros(max_niveles)
    T = T0
    nivel = 0
    sin_mejora = 0
    while T >= T_min and sin_mejora < max_sin_mejora and nivel < max_niveles:
        mejoro = False
        movimientos = 0
        while movimientos < L:
            u = movibles[int(random.random() * m)]
            v = movibles[int(random.random() * m)]
            a = asig[u]
            b = asig[v]
            if a == b:
                continue
            movimientos += 1
            d = _delta(asig, vecinos, W, u, v)
            if d <= 0 or random.random() < math.exp(-d / T):
                asig[u] = b
                asig[v] = a
                valor += d
                if valor < valor_mejor - 1e-9:
                    valor_mejor = valor
                    mejor = asig.copy()
                    mejoro = True
        hist_T[nivel] = T
        hist_actual[nivel] = valor
        hist_mejor[nivel] = valor_mejor
        nivel += 1
        # Los niveles sin mejora solo cuentan con el sistema frío: en caliente el mejor
        # valor puede estancarse por azar mientras la solución actual sigue lejos de él
        if mejoro or T > fraccion_frio * T0:
            sin_mejora = 0
        else:
            sin_mejora += 1
        T *= alfa
    return mejor, valor_mejor, hist_T[:nivel], hist_actual[:nivel], hist_mejor[:nivel]


def recocido_simulado(asig_inicial, movibles, vecinos, aristas_idx, W_mat, semilla,
                      alfa=0.995, L=2000, T_min=1e-3, max_sin_mejora=100, fraccion_frio=0.01,
                      n_muestras_T0=1000, aceptacion_T0=0.8):
    """Minimiza la competencia total intercambiando las especies de dos nodos movibles.

    El único movimiento es el intercambio especie-especie entre nodos nuevos
    (los existentes no se mueven y no hay vacíos), así que las cantidades se conservan.

    T0 se elige para que un empeoramiento de tamaño medio (entre 1000 muestreados)
    se acepte con probabilidad `aceptacion_T0`: T0 = -media(delta) / ln(aceptacion_T0).
    Termina cuando T < T_min o tras `max_sin_mejora` niveles sin mejora con T < fraccion_frio * T0.
    """
    assert (asig_inicial >= 0).all(), "todos los nodos deben tener especie"
    if HAY_NUMBA:
        datos = (asig_inicial.astype(np.int64), movibles.astype(np.int64), vecinos, W_mat)
    else:
        datos = (asig_inicial.tolist(), movibles.tolist(), vecinos.tolist(), W_mat.tolist())
    asig, movibles_, vecinos_, W_ = datos

    deltas = _muestrear_empeoramientos(asig, movibles_, vecinos_, W_, n_muestras_T0, semilla)
    T0 = -deltas.mean() / math.log(aceptacion_T0)
    max_niveles = int(math.ceil(math.log(T_min / T0) / math.log(alfa))) + 1

    valor_inicial = competencia_total(asig_inicial, aristas_idx, W_mat)
    mejor, valor_mejor, hist_T, hist_actual, hist_mejor = _nucleo_recocido(
        asig, movibles_, vecinos_, W_, valor_inicial, T0, alfa, L, T_min,
        max_sin_mejora, fraccion_frio, max_niveles, semilla + 1)

    mejor = np.asarray(mejor, dtype=np.int64)
    historial = pd.DataFrame({
        "iteracion": np.arange(1, len(hist_T) + 1) * L,
        "temperatura": hist_T,
        "actual": hist_actual,
        "mejor": hist_mejor,
    })
    return {
        "asig": mejor,
        "valor": competencia_total(mejor, aristas_idx, W_mat),   # recalculado sin errores acumulados
        "valor_inicial": valor_inicial,
        "T0": T0,
        "historial": historial,
    }
