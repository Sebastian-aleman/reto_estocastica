"""Pruebas simples con assert. Se corren con `python test_reto.py`."""
from pathlib import Path

import numpy as np
import pandas as pd

import competencia
import datos
import distribuciones
import malla
import recocido
import simulacion

DENSIDAD_HA = 658
SEP = malla.separacion_tresbolillo(DENSIDAD_HA)


def test_simetria_W():
    especies = datos.cargar_especies()
    mortalidad = datos.calcular_supervivencia(datos.cargar_monitoreo())["mortalidad_acum"]
    C = competencia.matriz_base_C(especies)
    W = competencia.calcular_W(C, competencia.calcular_vulnerabilidad(mortalidad))
    assert np.allclose(W, W.T)


def test_separacion_densidad():
    assert abs(SEP - 4.19) < 0.005, SEP
    for densidad in [658, 568]:
        sep = malla.separacion_tresbolillo(densidad)
        n_col, n_fil = malla.dimensiones_toroide(100, 100, sep)
        nodos, _ = malla.generar_malla_tresbolillo(100, 100, sep, toroidal=True)
        area_ha = n_col * sep * n_fil * sep * np.sqrt(3) / 2 / 10000
        assert n_fil % 2 == 0 and len(nodos) == n_col * n_fil
        assert abs(len(nodos) / area_ha - densidad) < 1e-6


def test_grados_toroidal():
    dy = SEP * np.sqrt(3) / 2
    nodos, aristas = malla.generar_malla_tresbolillo(10 * SEP, 10 * dy, SEP, toroidal=True)
    assert len(nodos) == 100
    assert all(g == 6 for g in malla.grados(nodos, aristas).values())
    nodos_ha, aristas_ha = malla.generar_malla_tresbolillo(100, 100, SEP, toroidal=True)
    assert all(g == 6 for g in malla.grados(nodos_ha, aristas_ha).values())


def test_sin_vacios():
    """Existentes + nuevas llenan la malla y las líneas base no dejan nodos sin especie."""
    nodos, _ = malla.generar_malla_tresbolillo(100, 100, SEP, toroidal=True)
    existentes, nuevas = recocido.cargar_cantidades(Path(__file__).parent / "salidas",
                                                    len(nodos), DENSIDAD_HA)
    assert existentes.sum() + nuevas.sum() == len(nodos)
    asig_exist, fijo = recocido.generar_existentes(existentes, len(nodos), 0)
    especies = pd.Index(datos.ABREVIATURAS)
    for asig in [recocido.linea_base_actual(asig_exist, fijo, especies),
                 recocido.linea_base_aleatoria(asig_exist, fijo, nuevas, 0)]:
        assert (asig >= 0).all()
        assert (np.bincount(asig[fijo], minlength=10) == existentes.to_numpy()).all()


def test_beta_media():
    supervivencia = datos.calcular_supervivencia(datos.cargar_monitoreo())
    beta = distribuciones.ajustar_beta_supervivencia(supervivencia)
    assert np.allclose(beta["a"] / (beta["a"] + beta["b"]), beta["media"])


def test_compras_con_ajuste_mayores():
    especies = datos.cargar_especies()
    monitoreo = datos.cargar_monitoreo()
    plantas = datos.cargar_plantas()
    areas = datos.cargar_areas()
    _, ajuste, _ = distribuciones.ajustar_densidad(distribuciones.calcular_plantas_por_ha(plantas, areas))
    beta = distribuciones.ajustar_beta_supervivencia(datos.calcular_supervivencia(monitoreo))
    resultado = simulacion.simular_compras(
        areas, ajuste, distribuciones.ajustar_dirichlet_multinomial(plantas),
        datos.calcular_prop_base(monitoreo), beta, especies["costo"], n_iter=50)
    assert (resultado["compras"]["esc2"] >= resultado["compras"]["esc1"]).all()


def test_delta_recocido():
    """El cambio calculado con los vecinos coincide con recalcular todo el objetivo."""
    especies = datos.cargar_especies()
    mortalidad = datos.calcular_supervivencia(datos.cargar_monitoreo())["mortalidad_acum"]
    W = competencia.calcular_W(competencia.matriz_base_C(especies),
                               competencia.calcular_vulnerabilidad(mortalidad))
    W_mat = W.to_numpy()
    dy = SEP * np.sqrt(3) / 2
    nodos, aristas = malla.generar_malla_tresbolillo(10 * SEP, 10 * dy, SEP, toroidal=True)
    aristas_idx, vecinos = recocido.preparar_malla(nodos, aristas)
    rng = np.random.default_rng(0)
    asig = rng.integers(0, recocido.N_ESPECIES, len(nodos))
    for _ in range(200):
        u, v = rng.integers(0, len(nodos), 2)
        antes = recocido.competencia_total(asig, aristas_idx, W_mat)
        delta = recocido._delta(asig, vecinos, W_mat, u, v)
        asig[u], asig[v] = asig[v], asig[u]
        assert abs(recocido.competencia_total(asig, aristas_idx, W_mat) - antes - delta) < 1e-9


def ejecutar_tests():
    test_simetria_W()
    test_separacion_densidad()
    test_grados_toroidal()
    test_sin_vacios()
    test_beta_media()
    test_compras_con_ajuste_mayores()
    test_delta_recocido()
    print("Todas las pruebas pasaron")


if __name__ == "__main__":
    ejecutar_tests()
