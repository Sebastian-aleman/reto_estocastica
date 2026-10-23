"""Corre todo el flujo del reto CONAFOR y guarda resultados en salidas/."""
from pathlib import Path

import numpy as np
import pandas as pd

import comparacion
import competencia
import datos
import distribuciones
import malla
import recocido as rc
import simulacion
from test_reto import ejecutar_tests

DENSIDAD_HA = 658          # plantas (= nodos) por ha; se puede cambiar a 568
TIEMPO_MILP = 900
SALIDAS = Path(__file__).parent / "salidas"


def titulo(texto):
    print(f"\n{'=' * 78}\n{texto}\n{'=' * 78}")


def main():
    SALIDAS.mkdir(exist_ok=True)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    # 1. Datos
    especies = datos.cargar_especies()
    monitoreo = datos.cargar_monitoreo()
    plantas = datos.cargar_plantas()
    areas = datos.cargar_areas()
    prop_base = datos.calcular_prop_base(monitoreo)
    supervivencia = datos.calcular_supervivencia(monitoreo)
    _, resumen_densidad = datos.calcular_densidad(plantas, areas)

    titulo("1. Datos")
    print("Especies:\n", especies.join(prop_base).round(3))
    print("\nSupervivencia:\n", supervivencia.round(3))
    print("\nDensidad por ha (30 polígonos):\n", resumen_densidad.round(3))

    # 2. Competencia
    C = competencia.matriz_base_C(especies)
    vul = competencia.calcular_vulnerabilidad(supervivencia["mortalidad_acum"])
    W = competencia.calcular_W(C, vul)
    competencia.guardar_W(W, SALIDAS)
    competencia.graficar_heatmap_W(W, SALIDAS / "heatmap_W.png")
    G = competencia.graficar_grafo_competencia(W, prop_base, especies, SALIDAS / "grafo_competencia.png")

    titulo("2. Competencia")
    print("Matriz W:\n", W.round(3))
    print(f"\nAristas del grafo con W > 0.5: {G.number_of_edges()} de 45")

    # 3. Malla
    sep = malla.separacion_tresbolillo(DENSIDAD_HA)
    dy = sep * np.sqrt(3) / 2
    nodos_ha, aristas_ha = malla.generar_malla_tresbolillo(100, 100, sep)
    malla.graficar_malla(nodos_ha, aristas_ha, SALIDAS / "malla_1ha.png")
    nodos_toro, aristas_toro = malla.generar_malla_tresbolillo(100, 100, sep, toroidal=True)
    n_col, n_fil = malla.dimensiones_toroide(100, 100, sep)
    nodos_tile, aristas_tile = malla.generar_malla_tresbolillo(10 * sep, 10 * dy, sep, toroidal=True)
    malla.exportar_malla_gams(nodos_tile, aristas_tile, SALIDAS)

    titulo("3. Malla")
    print(f"Densidad: {DENSIDAD_HA} plantas/ha -> separación sep = sqrt(2*10000 / (sqrt(3)*{DENSIDAD_HA})) "
          f"= {sep:.4f} m (entre filas {dy:.4f} m)")
    print(f"Malla plana 1 ha: {len(nodos_ha)} nodos")
    print(f"Malla toroidal 1 ha: {n_col} x {n_fil} = {len(nodos_toro)} nodos, "
          f"{len(nodos_toro) / (n_col * sep * n_fil * dy / 10000):.2f} nodos/ha, todos de grado "
          f"{set(malla.grados(nodos_toro, aristas_toro).values())}")
    print(f"Malla toroidal 10x10: {len(nodos_tile)} nodos, {len(aristas_tile)} aristas")

    # 4A. Plantas por ha
    plantas_por_ha = distribuciones.calcular_plantas_por_ha(plantas, areas)
    ajustes, ajuste_densidad, dists = distribuciones.ajustar_densidad(plantas_por_ha)
    distribuciones.guardar_ajuste(ajuste_densidad, SALIDAS / "ajuste_densidad.json")
    distribuciones.graficar_densidad(plantas_por_ha, dists, SALIDAS / "ajuste_densidad.png")
    razon_var_media = plantas_por_ha.var() / plantas_por_ha.mean()

    titulo("4A. Ajuste de plantas existentes por ha")
    print(f"Media = {plantas_por_ha.mean():.2f}, varianza = {plantas_por_ha.var():.2f}, "
          f"var/media = {razon_var_media:.3f}")
    print(ajustes.round(4).to_string())
    print(f"\nvar/media = {razon_var_media:.3f} < 1: hay SUBDISPERSIÓN. La Poisson obliga var = media "
          f"(var = {plantas_por_ha.mean():.1f}), mucho más que la observada, por eso se rechaza.\n"
          f"La Binomial(658, p) tiene var/media = 1 - p = {1 - dists['Binomial'].args[1]:.3f}, "
          "también demasiado grande.")
    parametros = ", ".join(f"{k} = {v:.4g}" for k, v in ajuste_densidad["parametros"].items())
    print(f"Mejor por AIC: {ajuste_densidad['distribucion']} ({parametros})")

    # 4B. Ocurrencia por especie
    tabla_prop = distribuciones.comparar_proporciones(plantas, prop_base)
    distribuciones.graficar_proporciones(tabla_prop, SALIDAS / "proporciones_especie.png")
    chi2, gl, p_valor = distribuciones.prueba_homogeneidad(plantas)
    ajuste_especies = distribuciones.ajustar_dirichlet_multinomial(plantas)

    titulo("4B. Ocurrencia por especie")
    print(tabla_prop.round(4).to_string())
    print(f"\nChi-cuadrada de homogeneidad (30x10): X2 = {chi2:.2f}, gl = {gl}, p = {p_valor:.4f}")
    print("  -> " + ("No se rechaza que la mezcla de especies sea igual en todos los polígonos."
                     if p_valor > 0.05 else "La mezcla de especies difiere entre polígonos."))
    print(f"Dirichlet-multinomial: rho = {ajuste_especies['rho']:.2e}, "
          f"alpha0 = {ajuste_especies['alpha0']:,.0f}")
    print(f"  Inflación de varianza vs multinomial con N = {ajuste_especies['N_medio']:.0f} "
          f"plantas: x{ajuste_especies['factor_sobredispersion']:.3f}")
    print(f"  logL multinomial = {ajuste_especies['log_ver_mult']:.2f} (AIC {ajuste_especies['aic_mult']:.2f}); "
          f"logL DM = {ajuste_especies['log_ver_dm']:.2f} (AIC {ajuste_especies['aic_dm']:.2f})")
    print(f"  -> Se usa: {ajuste_especies['distribucion']} "
          "(la DM solo si infla la varianza >= 20% y mejora el AIC en más de 2)")

    # 4C. Supervivencia
    beta = distribuciones.ajustar_beta_supervivencia(supervivencia)
    markov = distribuciones.supervivencia_markov(beta, supervivencia)

    titulo("4C. Supervivencia (Beta + cadena de Markov vivo/muerto)")
    print(beta.round(4).to_string())
    print()
    print(markov.round(4).to_string())
    print(f"\nsup_acum observada (2023/2021) abarca 2 años y se compara con sup_2: "
          f"error absoluto medio = {markov['error_2_anios'].abs().mean():.4f}, "
          f"máximo = {markov['error_2_anios'].abs().max():.4f}")

    # 5. Monte Carlo de compras
    costos = especies["costo"]
    resultado = simulacion.simular_compras(areas, ajuste_densidad, ajuste_especies, prop_base, beta, costos)
    resumen_especies = simulacion.resumen_por_especie(resultado, areas, especies.index)
    resumen_costos = simulacion.resumen_costos(resultado, areas, datos.PRESUPUESTO_HA)
    planes, ahorros = simulacion.comparar_planes(resultado, areas, prop_base, costos)
    x_esperado = simulacion.guardar_resultados(resumen_especies, resultado, areas, SALIDAS)
    simulacion.graficar_simulacion(resultado, areas, especies.index, datos.PRESUPUESTO_HA, SALIDAS)

    titulo(f"5. Monte Carlo de compras ({simulacion.N_ITER:,} iteraciones, semilla {simulacion.SEMILLA})")
    for esc, nombre in simulacion.ESCENARIOS.items():
        print(f"\nCompra por especie, {esc} ({nombre}):")
        print(resumen_especies[resumen_especies["escenario"] == esc]
              .drop(columns="escenario").set_index("especie").round(1).to_string())
    print("\nCosto por ha (prob_excede: proyecto completo; prob_excede_poligono: cada polígono):")
    print(resumen_costos.round(3).to_string())
    print("\nPlanes (por ha y total de los 30 polígonos):")
    print(planes.round(1).to_string())
    print("\nAhorro esperado (positivo = el Monte Carlo compra menos que la referencia):")
    print(ahorros.round(1).to_string())
    print("\nExistentes esperados por ha (X_esperado.inc):")
    print(x_esperado.round(2).to_string())

    # 6. Acomodo espacial: recocido simulado vs líneas base
    titulo("6. Acomodo espacial en 1 ha: recocido simulado vs líneas base")
    print(comparacion.ejecutar_paso4(W, prop_base, nodos_tile, aristas_tile, SALIDAS,
                                     densidad_ha=DENSIDAD_HA, tiempo_milp=TIEMPO_MILP))

    # 7. Pruebas
    titulo("7. Pruebas")
    ejecutar_tests()

    titulo("Supuestos usados")
    supuestos = [
        f"Densidad existente: {ajuste_densidad['distribucion']} ajustada por máxima verosimilitud a "
        "total/área de los 30 polígonos; Poisson y Binomial se evalúan con la densidad redondeada.",
        "Binomial: n = 658 nodos por ha; KS con distribuciones discretas es aproximada.",
        "Chi-cuadrada de Poisson/Binomial con 5 clases de probabilidad similar.",
        f"Reparto por especie: {ajuste_especies['distribucion']} con p = p_poligonos "
        "(proporciones observadas en los polígonos).",
        "alpha0 por momentos: suma de cuadrados de las proporciones entre polígonos vs su valor esperado.",
        "Meta por especie: p_s = prop_base (vivos 2021) x 658 x área.",
        "Supervivencia anual ~ Beta por momentos: media de las 2 tasas anuales, varianza muestral "
        f"de esas 2 tasas con mínimo {distribuciones.VAR_MIN_SUPERVIVENCIA}.",
        "Cadena de Markov con s = media de la Beta; sup_acum (2023/2021) se valida contra 2 años.",
        "Monte Carlo: densidad, reparto y sup_1 se sortean de forma independiente por polígono e iteración.",
        "Escenario 2: compra = ceil(necesidad / sup_1), sup_1 sorteada por especie y polígono.",
        f"Costo por ha = costo total / área total ({areas.sum():.2f} ha); presupuesto {datos.PRESUPUESTO_HA:,}/ha.",
        "Plan determinista: 524 plantas/ha x p_s; referencia sin descuento: 658/ha x p_s.",
        f"Malla sin vacíos: sep = sqrt(2*10000 / (sqrt(3)*{DENSIDAD_HA})) = {sep:.4f} m, así cada ha "
        f"tiene exactamente {DENSIDAD_HA} posiciones.",
        f"Acomodo en 1 ha: malla toroidal de {len(nodos_toro)} nodos ({n_col} x {n_fil}, filas par, "
        f"todos de grado 6) para evitar el efecto de borde; existentes = {rc.TOTAL_EXISTENTES}/{DENSIDAD_HA} "
        "de los nodos (reparto de X_esperado) fijas al azar y nuevas = el resto (reparto de ha_media de "
        "esc1); todos los nodos ocupados.",
        "Competencia total = suma de W sobre todas las aristas.",
        "Línea base actual: secuencia de 21 plantas cíclica en todos los nodos libres, en orden de fila.",
        "Recocido: arranca de una asignación aleatoria; el único movimiento es intercambiar las especies "
        "de dos nodos nuevos; T0 tal que un empeoramiento medio se acepta con prob. 0.8; para tras 100 "
        "niveles sin mejora contados con T < 1% de T0.",
        "Comparación justa: recocido con las cantidades de la línea base actual vs con cantidades p_s.",
        "Facilitación: W[PL,SC] = W[PL,YF] = -0.3 solo para optimizar; la competencia se mide con W.",
        f"Validación: malla toroidal 10x10 sin vacíos con 100 plantas proporcionales a prop_base; MILP "
        f"con formulación fuerte (enlace por arista, a[n1,SA] = 1), CBC, límite {TIEMPO_MILP} s.",
    ]
    for i, supuesto in enumerate(supuestos, 1):
        print(f"{i:>2}. {supuesto}")


if __name__ == "__main__":
    main()
