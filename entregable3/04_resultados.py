"""Tablas y figuras del experimento de 1,000 hectáreas (lee las salidas de 03_experimento.py).

a) valores_esperados.{csv,md}: medias por ha con IC 95%
b) matrices de transición (vecindad y recorrido, recocido y orden actual), heatmaps y π vs p_s
c) convergencia.png: media acumulada con banda IC 95%
d) hist_costo.png, hist_competencia.png
e) orden_plantacion_ejemplo.{csv,md} y mapa_ejemplo.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import sim_siembra as ss  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))
import datos  # noqa: E402
from comparacion import COLOR_ESPECIE  # noqa: E402

SALIDAS = Path(__file__).parent / "salidas_e3"
Z_95 = 1.96
PRESUPUESTO_HA = datos.PRESUPUESTO_HA
N_FILAS_EJEMPLO = 3

# Estilo: superficie clara, tinta neutra, azul = recocido, naranja = orden actual
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
REJILLA = "#e4e3df"
COLOR_RECOCIDO = "#2a78d6"
COLOR_ACTUAL = "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
    "axes.edgecolor": REJILLA, "axes.labelcolor": TINTA_SECUNDARIA, "axes.titlecolor": TINTA,
    "xtick.color": TINTA_SECUNDARIA, "ytick.color": TINTA_SECUNDARIA, "text.color": TINTA,
    "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


# ---------------------------------------------------------------- Utilidades

def df_a_md(df, decimales=2, indice=True):
    """Tabla markdown sin depender de tabulate."""
    df = df.reset_index() if indice else df

    def formato(v):
        if isinstance(v, (float, np.floating)):
            return f"{v:,.{decimales}f}"
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)

    lineas = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lineas += ["| " + " | ".join(formato(v) for v in fila) + " |" for fila in df.itertuples(index=False)]
    return "\n".join(lineas)


def guardar_tabla(df, nombre, titulo, decimales=2, indice=True, notas=()):
    df.to_csv(SALIDAS / f"{nombre}.csv", index=indice)
    texto = [f"# {titulo}", "", df_a_md(df, decimales, indice), "", *notas]
    (SALIDAS / f"{nombre}.md").write_text("\n".join(texto) + "\n", encoding="utf-8")


def estadistico(valores):
    n = len(valores)
    media, desv = valores.mean(), valores.std(ddof=1)
    margen = Z_95 * desv / np.sqrt(n)
    return {"media": media, "desv_est": desv, "ic95_inf": media - margen, "ic95_sup": media + margen}


# ---------------------------------------------------------------- a) Valores esperados

def tabla_valores_esperados(corridas, especies, nombres):
    q_cols = [f"q_{s}" for s in especies]
    reduccion = (corridas["z_orden_actual"] - corridas["z_final"]) / corridas["z_orden_actual"] * 100
    renglones = [("Existentes", "x_tot (total)", corridas["x_tot"])]
    renglones += [("Existentes", f"x_s {s} ({nombres[s]})", corridas[f"x_{s}"]) for s in especies]
    renglones += [("A suministrar", f"q_s {s} ({nombres[s]})", corridas[f"q_{s}"]) for s in especies]
    renglones += [
        ("A suministrar", "q total", corridas[q_cols].sum(axis=1)),
        ("Costo", "Costo por ha ($)", corridas["costo"]),
        ("Costo", f"P(costo > {PRESUPUESTO_HA:,})", (corridas["costo"] > PRESUPUESTO_HA).astype(float)),
        ("Competencia", "z_existentes", corridas["z_existentes"]),
        ("Competencia", "z_orden_actual", corridas["z_orden_actual"]),
        ("Competencia", "z_final (recocido)", corridas["z_final"]),
        ("Competencia", "Reducción recocido vs orden actual (%)", reduccion),
    ]
    tabla = pd.DataFrame([{"grupo": g, "concepto": c, **estadistico(v)} for g, c, v in renglones])
    return tabla.set_index(["grupo", "concepto"])


# ---------------------------------------------------------------- b) Matrices de transición

def normalizar_filas(conteo):
    return conteo / conteo.sum(axis=1, keepdims=True)


def distribucion_estacionaria(P):
    """Eigenvector izquierdo de P con valor propio 1, normalizado a suma 1."""
    valores, vectores = np.linalg.eig(P.T)
    pi = np.real(vectores[:, np.argmin(np.abs(valores - 1))])
    return pi / pi.sum()


def dibujar_heatmap(ax, P, especies, titulo, vmax):
    imagen = ax.imshow(P, cmap="Blues", vmin=0, vmax=vmax)
    ax.grid(False)
    ax.set_xticks(range(len(especies)), especies)
    ax.set_yticks(range(len(especies)), especies)
    ax.set_xlabel("t (especie siguiente / vecina)")
    ax.set_ylabel("s (especie actual)")
    for i in range(len(especies)):
        for j in range(len(especies)):
            ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if P[i, j] > 0.55 * vmax else TINTA)
    ax.set_title(titulo)
    return imagen


def graficar_heatmap(P, especies, titulo, ruta, vmax):
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    imagen = dibujar_heatmap(ax, P, especies, titulo, vmax)
    fig.colorbar(imagen, ax=ax, label="P(t | s)", shrink=0.85)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def graficar_heatmap_comparacion(matrices, especies, ruta, vmax=0.5):
    """Recorrido: orden actual y recocido lado a lado con la misma escala de color."""
    fig, ejes = plt.subplots(1, 2, figsize=(14, 6.2), constrained_layout=True)
    for ax, (nombre, titulo) in zip(ejes, [("orden_actual", "Orden actual (secuencia de 21)"),
                                           ("recocido", "Recocido simulado")]):
        imagen = dibujar_heatmap(ax, matrices[("recorrido", nombre)], especies, titulo, vmax)
    fig.colorbar(imagen, ax=ejes, label="P(t | s)", shrink=0.85)
    fig.suptitle("P(t|s) al recorrer la malla en serpentina (1,000 corridas; filas suman 1)")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def matrices_transicion(conteos, especies, p_s):
    matrices = {}
    for tipo in ["vecindad", "recorrido"]:
        for metodo, nombre in [("final", "recocido"), ("actual", "orden_actual")]:
            matrices[(tipo, nombre)] = normalizar_filas(conteos[f"{tipo}_{metodo}"].astype(float))

    vmax = max(P.max() for P in matrices.values())
    texto = ["# Matrices de transición P(t|s) (1,000 corridas; filas suman 1)", "",
             "- Vecindad: aristas (s,t) de la malla contadas en ambos sentidos.",
             "- Recorrido: transiciones s→t al recorrer la malla en serpentina "
             "(fila 1 izq→der, fila 2 der→izq, …).", ""]
    for (tipo, nombre), P in matrices.items():
        tabla = pd.DataFrame(P, index=especies, columns=especies).rename_axis("s \\ t")
        tabla.to_csv(SALIDAS / f"transicion_{tipo}_{nombre}.csv")
        texto += [f"## {tipo.capitalize()}, {nombre.replace('_', ' ')}", "", df_a_md(tabla, 2), ""]
        graficar_heatmap(P, especies, f"P(t|s) de {tipo}, {nombre.replace('_', ' ')}",
                         SALIDAS / f"heatmap_{tipo}_{nombre}.png", vmax)
    (SALIDAS / "matrices_transicion.md").write_text("\n".join(texto), encoding="utf-8")

    pi_recocido = distribucion_estacionaria(matrices[("recorrido", "recocido")])
    pi_actual = distribucion_estacionaria(matrices[("recorrido", "orden_actual")])
    comparacion = pd.DataFrame({
        "p_s": p_s.to_numpy(),
        "pi_recocido": pi_recocido,
        "dif_recocido": pi_recocido - p_s.to_numpy(),
        "pi_orden_actual": pi_actual,
        "dif_orden_actual": pi_actual - p_s.to_numpy(),
    }, index=especies).rename_axis("especie")
    guardar_tabla(comparacion, "estacionaria_vs_ps", "Distribución estacionaria π del recorrido vs p_s",
                  decimales=4, notas=[
                      "π = eigenvector izquierdo de valor 1 de la matriz de recorrido.",
                      "Con el recocido cada ha tiene exactamente d_s = p_s·658 plantas de cada especie, "
                      "así que π ≈ p_s salvo el efecto de los extremos del recorrido. El orden actual "
                      "usa la secuencia de 21 plantas, que no respeta p_s."])
    return matrices, comparacion


# ---------------------------------------------------------------- c) y d) Gráficas del Monte Carlo

def graficar_convergencia(corridas, ruta):
    variables = [("x_tot", "x_tot (existentes/ha)"), ("costo", "Costo por ha ($)"), ("z_final", "z_final")]
    n = np.arange(1, len(corridas) + 1)
    fig, ejes = plt.subplots(len(variables), 1, figsize=(9, 8), sharex=True)
    for ax, (columna, etiqueta) in zip(ejes, variables):
        valores = corridas[columna].to_numpy()
        media = np.cumsum(valores) / n
        desv = pd.Series(valores).expanding().std(ddof=1).to_numpy()
        margen = Z_95 * desv / np.sqrt(n)
        ax.fill_between(n, media - margen, media + margen, color=COLOR_RECOCIDO, alpha=0.18,
                        linewidth=0, label="IC 95%")
        ax.plot(n, media, color=COLOR_RECOCIDO, linewidth=2, label="Media acumulada")
        ax.axhline(media[-1], color=TINTA_SECUNDARIA, linestyle="--", linewidth=1,
                   label=f"Media final = {media[-1]:,.2f}")
        ax.set_ylabel(etiqueta)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    ejes[-1].set_xlabel("Número de corrida")
    ejes[0].set_title("Convergencia del Monte Carlo (banda: media ± 1.96·s/√n)")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def graficar_histogramas(corridas):
    costo = corridas["costo"]
    prob = (costo > PRESUPUESTO_HA).mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(costo, bins=40, color=COLOR_RECOCIDO, edgecolor=SUPERFICIE, linewidth=1)
    ax.axvline(PRESUPUESTO_HA, color=TINTA, linestyle="--", linewidth=1.5)
    ax.annotate(f"Presupuesto ${PRESUPUESTO_HA:,}\nP(costo > presupuesto) = {prob:.3f}",
                (PRESUPUESTO_HA, ax.get_ylim()[1] * 0.95), xytext=(8, 0), textcoords="offset points",
                va="top", fontsize=9, color=TINTA_SECUNDARIA)
    ax.set_xlabel("Costo por ha ($)")
    ax.set_ylabel("Corridas")
    ax.set_title(f"Costo de la planta a suministrar por ha ({len(costo):,} corridas)")
    fig.tight_layout()
    fig.savefig(SALIDAS / "hist_costo.png", dpi=150)
    plt.close(fig)

    # Bins comunes: z_final está mucho más concentrada que z_orden_actual
    z_min = np.floor(corridas[["z_final", "z_orden_actual"]].min().min())
    z_max = np.ceil(corridas[["z_final", "z_orden_actual"]].max().max())
    bins = np.arange(z_min, z_max + 2, 2)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for columna, color, etiqueta in [("z_orden_actual", COLOR_ACTUAL, "Orden actual (secuencia de 21)"),
                                     ("z_final", COLOR_RECOCIDO, "Recocido simulado")]:
        ax.hist(corridas[columna], bins=bins, color=color, edgecolor=SUPERFICIE, linewidth=0.5,
                label=f"{etiqueta}: media {corridas[columna].mean():.1f}")
    ax.set_xlabel("Competencia total z (suma de w_st sobre las 1,974 aristas; bins de 2)")
    ax.set_ylabel("Corridas")
    ax.set_title("Competencia por ha: recocido vs orden actual")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(SALIDAS / "hist_competencia.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- e) Orden de plantación

def orden_plantacion(ejemplo, modelo):
    nodos = modelo["malla"].nodos
    n_col = modelo["n_col"]
    idx = modelo["orden_serpentina"]
    existente = ejemplo["existente"].to_numpy()[idx]
    tabla = pd.DataFrame({
        "orden": np.arange(1, len(idx) + 1),
        "fila": idx // n_col + 1,
        "columna": idx % n_col + 1,
        "x_m": nodos["x"].to_numpy()[idx].round(2),
        "y_m": nodos["y"].to_numpy()[idx].round(2),
        "especie": ejemplo["especie"].to_numpy()[idx],
        "existente": np.where(existente, "sí", "no"),
        "accion": np.where(existente, "existente – no se planta", "plantar"),
    })
    # Número consecutivo de planta nueva, para la cuadrilla
    tabla["planta_nueva"] = np.where(existente, "", (~existente).cumsum().astype(str))
    return tabla


def tabla_primeras_filas(tabla, n_filas, meta):
    filas = []
    for fila, grupo in tabla[tabla["fila"] <= n_filas].groupby("fila", sort=True):
        codigos = [f"[{s}]" if e == "sí" else s for s, e in zip(grupo["especie"], grupo["existente"])]
        filas.append({
            "fila": fila,
            "sentido": "izq → der" if fila % 2 == 1 else "der → izq",
            "plantar": int((grupo["existente"] == "no").sum()),
            "existentes": int((grupo["existente"] == "sí").sum()),
            "secuencia": " ".join(codigos),
        })
    ejemplo = meta["ejemplo"]
    texto = [f"# Orden de plantación: primeras {n_filas} filas (ejemplo)", "",
             f"Corrida {ejemplo['corrida']} (z_final = {ejemplo['z_final']:.2f}, la más cercana a la mediana "
             f"{ejemplo['mediana_z_final']:.2f}; x_tot = {ejemplo['x_tot']}). Recorrido en serpentina; "
             "los códigos entre corchetes son plantas existentes que no se plantan.", "",
             df_a_md(pd.DataFrame(filas), indice=False), ""]
    (SALIDAS / "orden_plantacion_ejemplo.md").write_text("\n".join(texto), encoding="utf-8")
    return filas


def dibujar_hectarea(ax, tabla, especies, colores):
    """Hectárea completa: puntos por especie, existentes con borde negro y la serpentina."""
    ax.grid(False)
    x, y = tabla["x_m"].to_numpy(), tabla["y_m"].to_numpy()
    ax.quiver(x[:-1], y[:-1], np.diff(x), np.diff(y), angles="xy", scale_units="xy", scale=1,
              color=TINTA_SECUNDARIA, alpha=0.3, width=0.0012, headwidth=5, headlength=6, zorder=1)
    for s in especies:
        for es_existente, borde, grosor in [(False, SUPERFICIE, 1.0), (True, TINTA, 1.6)]:
            sel = (tabla["especie"] == s) & ((tabla["existente"] == "sí") == es_existente)
            ax.scatter(x[sel], y[sel], s=55, color=colores[s], edgecolors=borde,
                       linewidths=grosor, zorder=2, label=s if not es_existente else None)
    ax.scatter([], [], s=55, facecolors="none", edgecolors=TINTA, linewidths=1.6, label="Existente")
    ax.annotate("Inicio", (x[0], y[0]), xytext=(-6, -14), textcoords="offset points", fontsize=8,
                color=TINTA_SECUNDARIA)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def graficar_mapa(tabla, especies, meta, ruta):
    fig, ax = plt.subplots(figsize=(16, 5.6))
    dibujar_hectarea(ax, tabla, especies, COLOR_ESPECIE)
    ejemplo = meta["ejemplo"]
    ax.set_title(f"Hectárea ejemplo (47 × 14 = 658 nodos), corrida {ejemplo['corrida']}, "
                 f"z_final = {ejemplo['z_final']:.2f}; flechas: recorrido en serpentina")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=11, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta, dpi=170)
    plt.close(fig)


def color_texto(color):
    """Tinta clara u oscura según la luminancia del relleno."""
    r, g, b = matplotlib.colors.to_rgb(color)
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else TINTA


def graficar_mapa_v2(tabla, especies, meta, ruta, filas_zoom=3, columnas_zoom=15):
    """Arriba la hectárea completa; abajo un zoom con el código de especie y el orden de cada nodo."""
    colores = dict(zip(especies, plt.cm.tab10.colors))
    fig, (ax_ha, ax_zoom) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[4.2, 2.4])

    dibujar_hectarea(ax_ha, tabla, especies, colores)
    zoom = tabla[(tabla["fila"] <= filas_zoom) & (tabla["columna"] <= columnas_zoom)]
    margen = 2.0
    x0, x1 = zoom["x_m"].min() - margen, zoom["x_m"].max() + margen
    y0, y1 = zoom["y_m"].min() - margen, zoom["y_m"].max() + margen
    ax_ha.add_patch(matplotlib.patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                                 edgecolor=TINTA, linestyle="--", linewidth=1.2, zorder=3))
    ejemplo = meta["ejemplo"]
    ax_ha.set_title(f"Hectárea ejemplo (47 × 14 = 658 nodos), corrida {ejemplo['corrida']}, "
                    f"z_final = {ejemplo['z_final']:.2f}; flechas: recorrido en serpentina; "
                    f"recuadro: zoom de abajo")
    ax_ha.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=11, frameon=False, fontsize=10)

    # Zoom: flechas solo entre posiciones consecutivas del recorrido que caen dentro del recuadro
    ax_zoom.grid(False)
    x, y, orden = zoom["x_m"].to_numpy(), zoom["y_m"].to_numpy(), zoom["orden"].to_numpy()
    seguidos = np.flatnonzero(np.diff(orden) == 1)
    dx, dy = x[seguidos + 1] - x[seguidos], y[seguidos + 1] - y[seguidos]
    # Solo el tramo central, para que la punta no quede debajo del círculo siguiente
    ax_zoom.quiver(x[seguidos] + 0.3 * dx, y[seguidos] + 0.3 * dy, 0.4 * dx, 0.4 * dy,
                   angles="xy", scale_units="xy", scale=1, color=TINTA_SECUNDARIA, alpha=0.6,
                   width=0.0016, headwidth=5, headlength=6, zorder=1)
    for f in zoom.itertuples(index=False):
        existente = f.existente == "sí"
        ax_zoom.scatter(f.x_m, f.y_m, s=620, color=colores[f.especie], zorder=2,
                        edgecolors=TINTA if existente else SUPERFICIE, linewidths=2.4 if existente else 1.0)
        ax_zoom.text(f.x_m, f.y_m, f.especie, ha="center", va="center", fontsize=8, fontweight="bold",
                     color=color_texto(colores[f.especie]), zorder=3)
        ax_zoom.annotate(str(f.orden), (f.x_m, f.y_m), xytext=(0, 16), textcoords="offset points",
                         ha="center", fontsize=7, color=TINTA_SECUNDARIA)
    # A dónde sigue el recorrido fuera del recuadro
    for fila, grupo in zoom.groupby("fila"):
        texto = "→ sigue hasta la columna 47" if fila % 2 == 1 else "← llega desde la columna 47"
        ax_zoom.annotate(texto, (grupo["x_m"].max(), grupo["y_m"].iloc[0]), xytext=(18, 0),
                         textcoords="offset points", va="center", fontsize=8, color=TINTA_SECUNDARIA)
    ax_zoom.set_aspect("equal")
    ax_zoom.set_xlim(x0, x1 + 16)
    ax_zoom.set_ylim(y0, y1 + 1)
    ax_zoom.set_yticks(sorted(zoom["y_m"].unique()), [f"Fila {k}" for k in range(1, filas_zoom + 1)])
    ax_zoom.set_xlabel("x (m)")
    ax_zoom.set_title(f"Zoom: filas 1–{filas_zoom}, columnas 1–{columnas_zoom} "
                      "(número = orden de plantación; borde negro = existente, no se planta)")
    fig.tight_layout()
    fig.savefig(ruta, dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------- Principal

def main():
    corridas = pd.read_csv(SALIDAS / "corridas.csv", index_col="corrida")
    conteos = np.load(SALIDAS / "conteos.npz")
    meta = json.loads((SALIDAS / "experimento.json").read_text(encoding="utf-8"))
    ejemplo = pd.read_csv(SALIDAS / "ejemplo_asignacion.csv")
    modelo = ss.cargar_modelo()
    especies = list(modelo["especies"])
    nombres = datos.cargar_especies()["nombre"]

    # a)
    valores = tabla_valores_esperados(corridas, especies, nombres)
    guardar_tabla(valores, "valores_esperados", f"Valores esperados por hectárea ({len(corridas):,} corridas)",
                  notas=["IC 95% = media ± 1.96·s/√n. Para P(costo > presupuesto) s es la desviación de "
                         "la indicadora (IC de Wald)."])
    print(df_a_md(valores))

    # b)
    matrices, comparacion = matrices_transicion(conteos, especies, modelo["p_s"])
    graficar_heatmap_comparacion(matrices, especies, SALIDAS / "heatmap_recorrido_comparacion.png")
    for clave in [("recorrido", "recocido"), ("vecindad", "recocido")]:
        print(f"\nP(t|s) {clave[0]}, {clave[1]}")
        print(df_a_md(pd.DataFrame(matrices[clave], index=especies, columns=especies).rename_axis("s \\ t")))
    print("\nπ vs p_s")
    print(df_a_md(comparacion, 4))

    # c) y d)
    graficar_convergencia(corridas, SALIDAS / "convergencia.png")
    graficar_histogramas(corridas)

    # e)
    tabla = orden_plantacion(ejemplo, modelo)
    tabla.to_csv(SALIDAS / "orden_plantacion_ejemplo.csv", index=False, encoding="utf-8-sig")
    filas = tabla_primeras_filas(tabla, N_FILAS_EJEMPLO, meta)
    graficar_mapa(tabla, especies, meta, SALIDAS / "mapa_ejemplo.png")
    graficar_mapa_v2(tabla, especies, meta, SALIDAS / "mapa_ejemplo_v2.png")
    print("\nPrimeras filas del orden de plantación")
    print(df_a_md(pd.DataFrame(filas), indice=False))


if __name__ == "__main__":
    main()
