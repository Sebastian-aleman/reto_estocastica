"""Escenario de costo mínimo: q_s por programación entera (PuLP/CBC) en las mismas 1,000 semillas.

Compara contra el escenario base (q_s = d_s - x_s, salidas_e3/corridas.csv) y guarda en salidas_e3/:
  corridas_costo_minimo.csv, comparacion_escenarios.{csv,md}, hist_costo_escenarios.png
"""
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import sim_siembra as ss  # noqa: E402

SALIDAS = Path(__file__).parent / "salidas_e3"
N_CORRIDAS = 1000
N_PROCESOS = 5
Z_95 = 1.96
PRESUPUESTO_HA = 12500
COLUMNAS_COMUNES = ["x_tot", "z_existentes", "z_orden_actual"]

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
REJILLA = "#e4e3df"
COLOR_BASE = "#2a78d6"
COLOR_MINIMO = "#1baf7a"


def media_ic(valores):
    media = valores.mean()
    margen = Z_95 * valores.std(ddof=1) / np.sqrt(len(valores))
    return media, media - margen, media + margen


def formato_ic(valores, decimales=2):
    media, inf, sup = media_ic(valores)
    return f"{media:,.{decimales}f} [{inf:,.{decimales}f}, {sup:,.{decimales}f}]"


def tabla_comparativa(base, minimo, especies, p_s):
    renglones = []
    for nombre, columna, decimales in [("Costo por ha ($)", "costo", 2), ("z_final", "z_final", 2)]:
        renglones.append((nombre, formato_ic(base[columna], decimales), formato_ic(minimo[columna], decimales),
                          f"{minimo[columna].mean() - base[columna].mean():+,.2f}"))
    excede_base = (base["costo"] > PRESUPUESTO_HA).astype(float)
    excede_minimo = (minimo["costo"] > PRESUPUESTO_HA).astype(float)
    renglones.append((f"P(costo > {PRESUPUESTO_HA:,})", formato_ic(excede_base, 3), formato_ic(excede_minimo, 3),
                      f"{excede_minimo.mean() - excede_base.mean():+.3f}"))
    reduccion = {nombre: (t["z_orden_actual"] - t["z_final"]) / t["z_orden_actual"] * 100
                 for nombre, t in [("base", base), ("minimo", minimo)]}
    renglones.append(("Reducción de z vs orden actual (%)", formato_ic(reduccion["base"]),
                      formato_ic(reduccion["minimo"]), f"{reduccion['minimo'].mean() - reduccion['base'].mean():+.2f}"))
    for s in especies:
        q_base, q_min = base[f"q_{s}"].mean(), minimo[f"q_{s}"].mean()
        final_min = (minimo[f"x_{s}"] + minimo[f"q_{s}"]).mean() / ss.DENSIDAD_HA
        renglones.append((f"q_s {s} (proporción final costo mín. {final_min:.4f} vs p_s {p_s[s]:.4f})",
                          f"{q_base:.2f}", f"{q_min:.2f}", f"{q_min - q_base:+.2f}"))
    q_cols = [f"q_{s}" for s in especies]
    renglones.append(("q total", f"{base[q_cols].sum(axis=1).mean():.2f}",
                      f"{minimo[q_cols].sum(axis=1).mean():.2f}", "+0.00"))
    return pd.DataFrame(renglones, columns=["concepto", "base (q_s = d_s − x_s)", "costo mínimo (ILP)",
                                            "diferencia"])


def graficar_histogramas(base, minimo, ruta):
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
        "axes.edgecolor": REJILLA, "axes.labelcolor": TINTA_SECUNDARIA, "xtick.color": TINTA_SECUNDARIA,
        "ytick.color": TINTA_SECUNDARIA, "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    })
    costos = pd.concat([base["costo"], minimo["costo"]])
    bins = np.arange(np.floor(costos.min() / 10) * 10, costos.max() + 20, 20)
    # "\$" evita que matplotlib interprete el texto entre dos "$" como fórmula
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for tabla, color, etiqueta in [(base, COLOR_BASE, "Base (q_s = d_s − x_s)"),
                                   (minimo, COLOR_MINIMO, "Costo mínimo (ILP)")]:
        prob = (tabla["costo"] > PRESUPUESTO_HA).mean()
        ax.hist(tabla["costo"], bins=bins, histtype="stepfilled", color=color, alpha=0.35)
        ax.hist(tabla["costo"], bins=bins, histtype="step", color=color, linewidth=1.8,
                label=f"{etiqueta}: media \\${tabla['costo'].mean():,.0f}, P(costo > presupuesto) = {prob:.3f}")
    ax.axvline(PRESUPUESTO_HA, color=TINTA, linestyle="--", linewidth=1.5)
    ax.annotate(f"Presupuesto \\${PRESUPUESTO_HA:,}", (PRESUPUESTO_HA, ax.get_ylim()[1] * 0.97),
                xytext=(6, 0), textcoords="offset points", va="top", fontsize=9, color=TINTA_SECUNDARIA)
    ax.set_xlabel("Costo por ha (\\$), bins de \\$20")
    ax.set_ylabel("Corridas")
    ax.set_title(f"Costo por ha: escenario base vs costo mínimo ({len(base):,} corridas, mismas semillas)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def main():
    semillas = ss.generar_semillas(N_CORRIDAS)
    inicio = time.perf_counter()
    with Pool(N_PROCESOS, initializer=ss.calentar) as pool:
        resultados = pool.map(partial(ss.simular_hectarea, escenario="costo_minimo"), semillas, chunksize=10)
    tiempo_total = time.perf_counter() - inicio

    minimo = pd.DataFrame([{"corrida": k, **ss.fila_resultado(r)} for k, r in enumerate(resultados)])
    minimo = minimo.set_index("corrida")
    minimo.to_csv(SALIDAS / "corridas_costo_minimo.csv")
    base = pd.read_csv(SALIDAS / "corridas.csv", index_col="corrida")

    # Mismas semillas -> mismas existentes y mismo orden actual que el escenario base
    assert np.allclose(base[COLUMNAS_COMUNES], minimo[COLUMNAS_COMUNES], rtol=0, atol=1e-9)
    n_infactibles = int(minimo["infactible"].sum())
    factibles = ~minimo["infactible"]

    modelo = ss.cargar_modelo()
    especies = list(modelo["especies"])
    comparacion = tabla_comparativa(base[factibles], minimo[factibles], especies, modelo["p_s"])
    comparacion.to_csv(SALIDAS / "comparacion_escenarios.csv", index=False)

    encabezados = list(comparacion.columns)
    lineas = ["| " + " | ".join(encabezados) + " |", "|" + "---|" * len(encabezados)]
    lineas += ["| " + " | ".join(map(str, fila)) + " |" for fila in comparacion.itertuples(index=False)]
    texto = [f"# Escenario base vs costo mínimo ({factibles.sum():,} corridas factibles de {N_CORRIDAS:,})", "",
             "Media [IC 95%] con IC = media ± 1.96·s/√n (Wald para la probabilidad). Mismas semillas: "
             "mismas existentes en ambos escenarios. Costo mínimo: min Σ c_s q_s s.a. Σ q_s = 658 − x_tot, "
             "ceil(0.9·p_s·658) ≤ x_s + q_s ≤ floor(1.1·p_s·658), q_s ≥ 0 entero (PuLP/CBC); "
             "después recocido con L = 2000.", "",
             *lineas, "",
             f"Corridas infactibles: {n_infactibles}.",
             f"Tiempo del escenario de costo mínimo: {tiempo_total:.1f} s ({N_PROCESOS} procesos).",
             "Las especies de $26 (LE, SA, SC, ST, YF) empatan en costo: el ILP puede repartir entre ellas "
             "de varias formas óptimas y CBC elige una.", ""]
    (SALIDAS / "comparacion_escenarios.md").write_text("\n".join(texto), encoding="utf-8")
    graficar_histogramas(base, minimo, SALIDAS / "hist_costo_escenarios.png")
    print("\n".join(texto))


if __name__ == "__main__":
    main()
