"""Simulación de la siembra de 1 ha: existentes aleatorias, compra y acomodo con recocido.

Malla: toroide exacto de 47 x 14 = 658 nodos (1 ha), todos ocupados y de grado 6.
Reutiliza datos, competencia (w_st), malla y recocido del Entregable 2.
"""
import sys
import time
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

sys.path.insert(0, str(Path(__file__).parent.parent))

import competencia  # noqa: E402
import datos  # noqa: E402
import malla  # noqa: E402
import recocido as rc  # noqa: E402
from comparacion import Malla  # noqa: E402

DENSIDAD_HA = 658                 # único parámetro de densidad: de él salen sep, el toroide y d_s
MEDIA_EXISTENTES = 129.18
DESV_EXISTENTES = 3.35
SEMILLA_MAESTRA = 20261022
BANDA_META = 0.10                 # holgura permitida en la meta por especie (±10%)
PARAMETROS_RECOCIDO = {"alfa": 0.995, "L": 2000, "T_min": 1e-3, "max_sin_mejora": 100}


def generar_semillas(n):
    """Semillas independientes; la corrida k usa la misma semilla para cualquier n > k."""
    return np.random.SeedSequence(SEMILLA_MAESTRA).spawn(n)


@lru_cache(maxsize=1)
def cargar_modelo():
    """Datos, w_st y malla; se construyen una sola vez por proceso."""
    especies = datos.cargar_especies()
    monitoreo = datos.cargar_monitoreo()
    supervivencia = datos.calcular_supervivencia(monitoreo)
    prop_base = datos.calcular_prop_base(monitoreo)
    p_s = prop_base / prop_base.sum()
    W = competencia.calcular_W(competencia.matriz_base_C(especies),
                               competencia.calcular_vulnerabilidad(supervivencia["mortalidad_acum"]))

    sep = malla.separacion_tresbolillo(DENSIDAD_HA)
    n_col, n_fil = malla.dimensiones_toroide_exacto(DENSIDAD_HA, sep)
    nodos, aristas = malla.generar_malla_toroide_exacto(DENSIDAD_HA)
    assert len(nodos) == DENSIDAD_HA == 658, len(nodos)
    assert len(aristas) == 3 * DENSIDAD_HA == 1974, len(aristas)
    assert set(malla.grados(nodos, aristas).values()) == {6}

    # Recorrido en serpentina: filas pares izq->der, impares der->izq (los ids van fila por fila)
    orden_serpentina = np.arange(len(nodos)).reshape(n_fil, n_col)
    orden_serpentina[1::2] = orden_serpentina[1::2, ::-1]

    return {
        "especies": especies.index,
        "p_s": p_s,
        "c_s": especies["costo"],
        "d_s": rc.redondear_a_total(p_s, DENSIDAD_HA),
        "malla": Malla(nodos, aristas, W),
        "n_col": n_col,
        "n_fil": n_fil,
        "orden_serpentina": orden_serpentina.ravel(),
    }


def calentar():
    """Compila (o carga de caché) el núcleo de numba; sirve de initializer de Pool."""
    simular_hectarea(np.random.SeedSequence(0), L=10)


def fila_resultado(resultado):
    """Resultado de una corrida sin conteos ni asignación, con x_s y q_s en columnas."""
    fila = {k: v for k, v in resultado.items() if not isinstance(v, (pd.Series, np.ndarray))}
    fila.update({f"x_{s}": v for s, v in resultado["x_s"].items()})
    fila.update({f"q_{s}": v for s, v in resultado["q_s"].items()})
    return fila


# ---------------------------------------------------------------- Pasos de la simulación

def calcular_q(x_s, d_s, banda=BANDA_META):
    """Plantas a comprar q_s = max(0, d_s - x_s), ajustadas para que sum(q_s) = 658 - x_tot.

    Si alguna especie ya tiene más existentes que su meta, sum(q_s) se pasa; el exceso
    se quita de una en una de la especie con mayor holgura sobre (1 - banda)·d_s.
    Regresa (q_s, plantas quitadas).
    """
    q_s = (d_s - x_s).clip(lower=0)
    exceso = int(q_s.sum() - (d_s.sum() - x_s.sum()))
    if exceso == 0:
        return q_s, 0
    warnings.warn(f"sum(q_s) excede 658 - x_tot por {exceso}; se ajusta dentro de ±{banda:.0%}")
    for _ in range(exceso):
        holgura = (x_s + q_s - (1 - banda) * d_s).where(q_s > 0, -np.inf)
        q_s[holgura.idxmax()] -= 1
    assert q_s.sum() == d_s.sum() - x_s.sum()
    return q_s, exceso


def calcular_q_costo_minimo(x_s, p_s, c_s, densidad_ha=DENSIDAD_HA, banda=BANDA_META):
    """q_s de costo mínimo con un problema lineal entero (PuLP/CBC).

    min sum c_s q_s  s.a.  sum q_s = densidad_ha - x_tot,
    ceil((1-banda) p_s densidad_ha) <= x_s + q_s <= floor((1+banda) p_s densidad_ha),  q_s >= 0 entero.
    Regresa None si es infactible.
    """
    meta = p_s * densidad_ha
    minimo = np.ceil((1 - banda) * meta - 1e-9).astype(int)
    maximo = np.floor((1 + banda) * meta + 1e-9).astype(int)
    especies = list(x_s.index)

    prob = pulp.LpProblem("q_costo_minimo", pulp.LpMinimize)
    q = prob.add_variable_dict("q", especies, lowBound=0, cat="Integer")
    prob += pulp.lpSum(float(c_s[s]) * q[s] for s in especies)
    prob += pulp.lpSum(q[s] for s in especies) == densidad_ha - int(x_s.sum())
    for s in especies:
        prob += int(x_s[s]) + q[s] >= int(minimo[s])
        prob += int(x_s[s]) + q[s] <= int(maximo[s])
    estado = prob.solve(pulp.COIN_CMD(msg=False))
    if estado.status_str != "Optimal":
        return None
    return pd.Series([int(round(q[s].value())) for s in especies], index=x_s.index)


def competencia_existentes(asig, fijo, malla_ha):
    """Suma de w_st solo sobre aristas con ambos extremos existentes."""
    a, b = malla_ha.aristas_idx[:, 0], malla_ha.aristas_idx[:, 1]
    ambos = fijo[a] & fijo[b]
    return malla_ha.W_mat[asig[a[ambos]], asig[b[ambos]]].sum()


def contar_vecindad(asig, aristas_idx):
    """conteo[s,t] = aristas que unen s y t, en ambos sentidos (la diagonal cuenta 2 por arista)."""
    conteo = np.zeros((rc.N_ESPECIES, rc.N_ESPECIES), dtype=np.int64)
    s, t = asig[aristas_idx[:, 0]], asig[aristas_idx[:, 1]]
    np.add.at(conteo, (s, t), 1)
    np.add.at(conteo, (t, s), 1)
    return conteo


def contar_recorrido(asig, orden):
    """conteo[s,t] = transiciones s -> t al recorrer los nodos en el orden dado."""
    secuencia = asig[orden]
    conteo = np.zeros((rc.N_ESPECIES, rc.N_ESPECIES), dtype=np.int64)
    np.add.at(conteo, (secuencia[:-1], secuencia[1:]), 1)
    return conteo


# ---------------------------------------------------------------- Simulación de 1 ha

def simular_hectarea(semilla, escenario="base", **parametros_recocido):
    """Una hectárea: existentes, compra, orden actual y recocido. `semilla` es un SeedSequence o entero.

    escenario "base": q_s = d_s - x_s. "costo_minimo": q_s del problema entero de costo mínimo
    (si es infactible se usa la base y se marca `infactible`). La elección de q_s no consume
    números aleatorios, así que con la misma semilla ambos escenarios tienen las mismas existentes.
    """
    modelo = cargar_modelo()
    malla_ha = modelo["malla"]
    especies, p_s = modelo["especies"], modelo["p_s"]
    rng = np.random.default_rng(semilla)

    # 1-2. Existentes y su reparto por especie
    x_tot = int(np.clip(round(rng.normal(MEDIA_EXISTENTES, DESV_EXISTENTES)), 0, DENSIDAD_HA))
    x_s = pd.Series(rng.multinomial(x_tot, p_s.to_numpy()), index=especies)

    # 3-4. Existentes en nodos al azar y su competencia entre sí
    asig_exist, fijo = rc.generar_existentes(x_s, DENSIDAD_HA, int(rng.integers(2**32)))
    z_existentes = competencia_existentes(asig_exist, fijo, malla_ha)

    # 5-6. Compra y costo
    q_s, ajuste_q = calcular_q(x_s, modelo["d_s"])
    infactible = False
    if escenario == "costo_minimo":
        q_minimo = calcular_q_costo_minimo(x_s, p_s, modelo["c_s"])
        infactible = q_minimo is None
        if not infactible:
            q_s, ajuste_q = q_minimo, 0
    costo = float((modelo["c_s"] * q_s).sum())

    # 7. Orden actual: secuencia de 21 plantas en los nodos libres
    asig_actual = rc.linea_base_actual(asig_exist, fijo, especies)
    z_orden_actual = rc.competencia_total(asig_actual, malla_ha.aristas_idx, malla_ha.W_mat)

    # 8. Recocido sobre los nodos libres con las cantidades q_s
    semilla_sa = int(rng.integers(2**31 - 1))
    inicial = rc.linea_base_aleatoria(asig_exist, fijo, q_s, semilla_sa)
    inicio = time.perf_counter()
    sa = rc.recocido_simulado(inicial, np.flatnonzero(~fijo), malla_ha.vecinos, malla_ha.aristas_idx,
                              malla_ha.W_mat, semilla_sa, **{**PARAMETROS_RECOCIDO, **parametros_recocido})
    tiempo_recocido = time.perf_counter() - inicio
    asig_final = sa["asig"]

    # 9. Conteos de vecindad y de recorrido en serpentina
    aristas_idx, orden = malla_ha.aristas_idx, modelo["orden_serpentina"]
    return {
        "escenario": escenario,
        "infactible": infactible,
        "x_tot": x_tot,
        "x_s": x_s,
        "q_s": q_s,
        "ajuste_q": ajuste_q,
        "costo": costo,
        "z_existentes": z_existentes,
        "z_orden_actual": z_orden_actual,
        "z_inicial_recocido": sa["valor_inicial"],
        "z_final": sa["valor"],
        "T0": sa["T0"],
        "niveles": len(sa["historial"]),
        "tiempo_recocido": tiempo_recocido,
        "conteo_vecindad_final": contar_vecindad(asig_final, aristas_idx),
        "conteo_vecindad_actual": contar_vecindad(asig_actual, aristas_idx),
        "conteo_recorrido_final": contar_recorrido(asig_final, orden),
        "conteo_recorrido_actual": contar_recorrido(asig_actual, orden),
        "asignacion_final": asig_final,
        "fijo": fijo,
    }
