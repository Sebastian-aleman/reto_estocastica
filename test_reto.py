"""Pruebas simples con assert. Se corren con `python test_reto.py`."""
import numpy as np

import competencia
import datos
import distribuciones
import malla
import recocido
import simulacion

SEP = 3.4


def test_simetria_W():
    especies = datos.cargar_especies()
    mortalidad = datos.calcular_supervivencia(datos.cargar_monitoreo())["mortalidad_acum"]
    C = competencia.matriz_base_C(especies)
    W = competencia.calcular_W(C, competencia.calcular_vulnerabilidad(mortalidad))
    assert np.allclose(W, W.T)


def test_nodos_1ha():
    nodos, _ = malla.generar_malla_tresbolillo(100, 100, SEP)
    assert abs(len(nodos) - 999) <= 10, len(nodos)


def test_grados_toroidal():
    dy = SEP * np.sqrt(3) / 2
    nodos, aristas = malla.generar_malla_tresbolillo(10 * SEP, 10 * dy, SEP, toroidal=True)
    assert len(nodos) == 100
    assert all(g == 6 for g in malla.grados(nodos, aristas).values())
    nodos_ha, aristas_ha = malla.generar_malla_tresbolillo(100, 100, SEP, toroidal=True)
    assert all(g == 6 for g in malla.grados(nodos_ha, aristas_ha).values())


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
    W_ext = recocido.W_extendida(W)
    nodos, aristas = malla.generar_malla_tresbolillo(30, 30, SEP)
    aristas_idx, vecinos = recocido.preparar_malla(nodos, aristas)
    rng = np.random.default_rng(0)
    asig = np.append(rng.integers(0, recocido.N_ESPECIES + 1, len(nodos)), recocido.VACIO)
    for _ in range(200):
        u, v = rng.integers(0, len(nodos), 2)
        antes = recocido.competencia_total(asig, aristas_idx, W_ext)
        delta = recocido._delta(asig, vecinos, W_ext, u, v)
        asig[u], asig[v] = asig[v], asig[u]
        assert abs(recocido.competencia_total(asig, aristas_idx, W_ext) - antes - delta) < 1e-9


def ejecutar_tests():
    test_simetria_W()
    test_nodos_1ha()
    test_grados_toroidal()
    test_beta_media()
    test_compras_con_ajuste_mayores()
    test_delta_recocido()
    print("Todas las pruebas pasaron")


if __name__ == "__main__":
    ejecutar_tests()
