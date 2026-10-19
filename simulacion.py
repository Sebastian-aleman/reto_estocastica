"""Monte Carlo de compras de planta por polígono."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N_ITER = 10_000
SEMILLA = 2026
N_NODOS_HA = 658
PLAN_DETERMINISTA_HA = 524
ESCENARIOS = {"esc1": "Sin ajuste", "esc2": "Con ajuste por supervivencia"}


def muestrear_densidad(ajuste, rng, tamano):
    par = ajuste["parametros"]
    if ajuste["distribucion"] == "Normal":
        return rng.normal(par["media"], par["desv"], tamano)
    if ajuste["distribucion"] == "Poisson":
        return rng.poisson(par["lambda"], tamano)
    return rng.binomial(par["n"], par["p"], tamano)


def simular_compras(areas, ajuste_densidad, ajuste_especies, p_meta, beta, costos,
                    n_iter=N_ITER, semilla=SEMILLA):
    """Simula existentes, necesidad y compras. Arreglos de forma (iteración, polígono, especie)."""
    rng = np.random.default_rng(semilla)
    area = areas.to_numpy()
    n_pol = len(area)

    # 1. Plantas existentes por polígono
    densidad = muestrear_densidad(ajuste_densidad, rng, (n_iter, n_pol))
    existentes = np.clip(np.round(densidad * area), 0, None).astype(np.int64)

    # 2. Reparto por especie
    p_existentes = ajuste_especies["p"].to_numpy()
    if ajuste_especies["distribucion"] == "Dirichlet-multinomial":
        p_existentes = rng.dirichlet(ajuste_especies["alpha0"] * p_existentes, (n_iter, n_pol))
    X = rng.multinomial(existentes, p_existentes)

    # 3. Meta y necesidad
    meta = p_meta.to_numpy() * N_NODOS_HA * area[:, None]
    necesidad = np.maximum(0, np.ceil(meta - X))

    # 4-5. Compras sin y con ajuste por supervivencia del primer año
    sup_1 = rng.beta(beta["a"].to_numpy(), beta["b"].to_numpy(), X.shape)
    compras = {"esc1": necesidad, "esc2": np.ceil(necesidad / sup_1)}

    # 6. Costo por polígono
    c = costos.to_numpy()
    costos_pol = {esc: (compra * c).sum(axis=2) for esc, compra in compras.items()}
    return {"X": X, "compras": compras, "costos": costos_pol}


def _estadisticos(valores):
    return {
        "media": valores.mean(axis=0),
        "desv": valores.std(axis=0),
        "p5": np.percentile(valores, 5, axis=0),
        "p50": np.percentile(valores, 50, axis=0),
        "p95": np.percentile(valores, 95, axis=0),
    }


def resumen_por_especie(resultado, areas, especies):
    """Compra por especie: total de los 30 polígonos y por ha, en ambos escenarios."""
    area_total = areas.sum()
    tablas = []
    for esc, compra in resultado["compras"].items():
        total = compra.sum(axis=1)                    # (iteración, especie)
        tabla = pd.DataFrame({f"total_{k}": v for k, v in _estadisticos(total).items()}, index=especies)
        for k, v in _estadisticos(total / area_total).items():
            tabla[f"ha_{k}"] = v
        tabla.insert(0, "escenario", esc)
        tablas.append(tabla)
    return pd.concat(tablas).rename_axis("especie").reset_index()


def resumen_costos(resultado, areas, presupuesto_ha):
    """Costo por ha del proyecto (30 polígonos) y riesgo de exceder el presupuesto."""
    area = areas.to_numpy()
    filas = {}
    for esc, costo in resultado["costos"].items():
        costo_ha = costo.sum(axis=1) / area.sum()
        costo_ha_pol = costo / area
        filas[esc] = {
            "media": costo_ha.mean(),
            "p5": np.percentile(costo_ha, 5),
            "p95": np.percentile(costo_ha, 95),
            "prob_excede": (costo_ha > presupuesto_ha).mean(),
            "prob_excede_poligono": (costo_ha_pol > presupuesto_ha).mean(),
        }
    return pd.DataFrame(filas).T


def comparar_planes(resultado, areas, p_meta, costos):
    """Plantas y costo por ha y totales: planes fijos vs Monte Carlo."""
    area_total = areas.sum()
    costo_unitario_medio = (p_meta * costos).sum()
    planes = {
        "Comprar 658/ha": (N_NODOS_HA, N_NODOS_HA * costo_unitario_medio),
        "Determinista 524/ha": (PLAN_DETERMINISTA_HA, PLAN_DETERMINISTA_HA * costo_unitario_medio),
    }
    for esc, compra in resultado["compras"].items():
        planes[f"MC {esc}"] = (compra.sum(axis=(1, 2)).mean() / area_total,
                               resultado["costos"][esc].sum(axis=1).mean() / area_total)
    tabla = pd.DataFrame(planes, index=["plantas_ha", "costo_ha"]).T
    tabla["plantas_total"] = tabla["plantas_ha"] * area_total
    tabla["costo_total"] = tabla["costo_ha"] * area_total

    ahorros = []
    for esc in resultado["compras"]:
        for referencia in ["Comprar 658/ha", "Determinista 524/ha"]:
            diferencia = tabla.loc[referencia] - tabla.loc[f"MC {esc}"]
            ahorros.append(pd.Series(diferencia, name=f"MC {esc} vs {referencia}"))
    return tabla, pd.DataFrame(ahorros)


def guardar_resultados(resumen_especies, resultado, areas, carpeta):
    resumen_especies.to_csv(carpeta / "compras_esperadas.csv", index=False)
    x_esperado_ha = resultado["X"].sum(axis=1).mean(axis=0) / areas.sum()
    especies = resumen_especies["especie"].unique()
    lineas = [f"{s:<4}{x:12.6f}" for s, x in zip(especies, x_esperado_ha)]
    (carpeta / "X_esperado.inc").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return pd.Series(x_esperado_ha, index=especies, name="X_esperado_ha")


def graficar_simulacion(resultado, areas, especies, presupuesto_ha, carpeta):
    area_total = areas.sum()

    # Histograma del costo por ha
    fig, ax = plt.subplots(figsize=(9, 5))
    for esc, costo in resultado["costos"].items():
        ax.hist(costo.sum(axis=1) / area_total, bins=50, alpha=0.7, label=ESCENARIOS[esc])
    ax.axvline(presupuesto_ha, color="black", linestyle="--", label=f"Presupuesto {presupuesto_ha:,}")
    ax.set_xlabel("Costo por ha del proyecto ($)")
    ax.set_ylabel("Iteraciones")
    ax.set_title("Distribución del costo por ha (10,000 iteraciones)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(carpeta / "costo_por_ha.png", dpi=150)
    plt.close(fig)

    # Boxplot de compra por ha por especie (cada polígono en cada iteración)
    x = np.arange(len(especies))
    fig, ax = plt.subplots(figsize=(11, 5))
    for desplazamiento, (esc, compra) in zip([-0.2, 0.2], resultado["compras"].items()):
        por_ha = (compra / areas.to_numpy()[:, None]).reshape(-1, len(especies))
        caja = ax.boxplot(por_ha, positions=x + desplazamiento, widths=0.35, patch_artist=True,
                          showfliers=False, medianprops={"color": "black"})
        color = "C0" if esc == "esc1" else "C1"
        for parte in caja["boxes"]:
            parte.set_facecolor(color)
        ax.plot([], [], color=color, linewidth=8, label=ESCENARIOS[esc])
    ax.set_xticks(x, especies)
    ax.set_ylabel("Plantas a comprar por ha")
    ax.set_title("Compra por ha en cada polígono (sin valores atípicos)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(carpeta / "compra_por_especie.png", dpi=150)
    plt.close(fig)

    # Convergencia de la media de compra total
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, (esc, compra) in zip(axes, resultado["compras"].items()):
        total = compra.sum(axis=(1, 2))
        media_acum = np.cumsum(total) / np.arange(1, len(total) + 1)
        ax.plot(media_acum)
        ax.axhline(total.mean(), color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel("Media de compra total")
        ax.set_title(ESCENARIOS[esc])
    axes[-1].set_xlabel("Iteraciones")
    fig.tight_layout()
    fig.savefig(carpeta / "convergencia.png", dpi=150)
    plt.close(fig)
