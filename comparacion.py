"""Experimentos del recocido simulado, validación con MILP y gráficas del acomodo."""
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp

import malla as mll
import recocido as rc

SEMILLA_ESCENARIO = 2026
N_CORRIDAS = 10
N_ESCENARIOS = 30
N_CORRIDAS_TILE = 5
PARAMETROS_CORTOS = {"alfa": 0.98, "L": 1000}
W_FACILITACION = -0.3
ZOOM = (40, 60)

# Tonos por género: Agave azules, Opuntia naranjas, Prosopis verde, Yucca amarillo
COLOR_ESPECIE = {
    "LE": "#b7d3f6", "SA": "#2a78d6", "SC": "#0d366b", "ST": "#6da7ec",
    "OC": "#f7c4a8", "OE": "#f08a5d", "OR": "#d2501f", "OS": "#8f3410",
    "PL": "#1baf7a", "YF": "#eda100",
}

# Signo para que "mejora" sea positiva: -1 si conviene que baje, +1 si conviene que suba
SENTIDO_METRICAS = {
    "competencia_total": -1,
    "competencia_por_planta": -1,
    "pares_misma_especie": -1,
    "pares_misma_SC": -1,
    "pares_misma_YF": -1,
    "pares_misma_PL": -1,
    "pct_SC_YF_con_PL": 1,
}
METODOS = ["Base actual", "Base aleatoria", "Recocido cantidades base", "Recocido"]


# ---------------------------------------------------------------- Métricas

def calcular_metricas(asig, aristas_idx, vecinos, W_mat, especies):
    cod = {s: especies.get_loc(s) for s in ["SC", "YF", "PL"]}
    a, b = asig[aristas_idx[:, 0]], asig[aristas_idx[:, 1]]
    misma = a == b
    total = W_mat[a, b].sum()

    sc_yf = np.isin(asig, [cod["SC"], cod["YF"]])
    con_pl = (asig[vecinos] == cod["PL"]).any(axis=1)
    return {
        "competencia_total": total,
        "competencia_por_planta": total / len(asig),
        "pares_misma_especie": misma.sum(),
        **{f"pares_misma_{s}": (misma & (a == cod[s])).sum() for s in ["SC", "YF", "PL"]},
        "pct_SC_YF_con_PL": 100 * con_pl[sc_yf].mean(),
    }


def resumir_metricas(tabla):
    """Media ± desviación por método y % de mejora del recocido sobre la base actual."""
    metodos = [m for m in METODOS if m in set(tabla["metodo"])]
    medias = tabla.groupby("metodo")[list(SENTIDO_METRICAS)].mean().T[metodos]
    desv = tabla.groupby("metodo")[list(SENTIDO_METRICAS)].std().T[metodos]
    resumen = pd.DataFrame(index=medias.index)
    for metodo in metodos:
        resumen[metodo] = [f"{m:.2f} ± {d:.2f}" for m, d in zip(medias[metodo], desv[metodo])]
    base = medias["Base actual"].replace(0, np.nan)
    sentido = pd.Series(SENTIDO_METRICAS)
    resumen["% mejora SA vs actual"] = (sentido * (medias["Recocido"] - base) / base * 100).round(1)
    return resumen


def comparacion_justa(tabla):
    """Separa la mejora por acomodo (a - b) del efecto de las cantidades (b - c)."""
    media = tabla.groupby("metodo")["competencia_total"].mean()
    desv = tabla.groupby("metodo")["competencia_total"].std()
    a, b, c = media["Base actual"], media["Recocido cantidades base"], media["Recocido"]
    filas = [
        ("(a) Base actual", a, desv["Base actual"]),
        ("(b) Recocido con cantidades de la base", b, desv["Recocido cantidades base"]),
        ("(c) Recocido con cantidades p_s", c, desv["Recocido"]),
        ("(a) - (b): mejora por acomodo", a - b, np.nan),
        ("(b) - (c): efecto de las cantidades", b - c, np.nan),
        ("(a) - (c): mejora total", a - c, np.nan),
    ]
    resumen = pd.DataFrame(filas, columns=["concepto", "competencia_total", "desv"]).set_index("concepto")
    resumen["% de (a)"] = resumen["competencia_total"] / a * 100
    return resumen.round(2)


# ---------------------------------------------------------------- Experimentos

class Malla:
    """Datos de una malla toroidal listos para el recocido."""

    def __init__(self, nodos, aristas, W):
        self.nodos = nodos
        self.aristas_idx, self.vecinos = rc.preparar_malla(nodos, aristas)
        self.W_mat = W.to_numpy()
        self.especies = W.index

    def metricas(self, asig):
        return calcular_metricas(asig, self.aristas_idx, self.vecinos, self.W_mat, self.especies)


def correr_escenario(malla, existentes, nuevas, semilla_escenario, semilla_sa, nuevas_base=None,
                     **parametros_sa):
    """Líneas base y recocido para un escenario de existentes.

    Si se da `nuevas_base`, también corre el recocido con las cantidades de la base actual.
    """
    asig_exist, fijo = rc.generar_existentes(existentes, len(malla.nodos), semilla_escenario)
    movibles = np.flatnonzero(~fijo)

    def recocer(cantidades):
        # El recocido arranca de una asignación aleatoria con esas cantidades
        inicial = rc.linea_base_aleatoria(asig_exist, fijo, cantidades, semilla_sa)
        return rc.recocido_simulado(inicial, movibles, malla.vecinos, malla.aristas_idx,
                                    malla.W_mat, semilla_sa, **parametros_sa)

    sa = recocer(nuevas)
    soluciones = {
        "Base actual": rc.linea_base_actual(asig_exist, fijo, malla.especies),
        "Base aleatoria": rc.linea_base_aleatoria(asig_exist, fijo, nuevas, semilla_sa),
        "Recocido": sa["asig"],
    }
    if nuevas_base is not None:
        soluciones["Recocido cantidades base"] = recocer(nuevas_base)["asig"]
    filas = [{"metodo": m, **malla.metricas(asig)} for m, asig in soluciones.items()]
    return filas, soluciones, fijo, sa


def experimento_semillas(malla, existentes, nuevas, nuevas_base):
    """10 corridas del recocido con semillas distintas sobre un mismo escenario."""
    filas, mejor = [], None
    for corrida in range(1, N_CORRIDAS + 1):
        filas_corrida, soluciones, fijo, sa = correr_escenario(
            malla, existentes, nuevas, SEMILLA_ESCENARIO, 100 * corrida, nuevas_base)
        filas += [{"corrida": corrida, **f} for f in filas_corrida]
        if mejor is None or sa["valor"] < mejor["sa"]["valor"]:
            mejor = {"soluciones": soluciones, "fijo": fijo, "sa": sa, "corrida": corrida}
    return pd.DataFrame(filas), mejor


def experimento_escenarios(malla, existentes, nuevas, nuevas_base):
    """30 escenarios de existentes con una corrida corta del recocido en cada uno."""
    filas = []
    for escenario in range(1, N_ESCENARIOS + 1):
        filas_esc, _, _, _ = correr_escenario(malla, existentes, nuevas, 1000 + escenario,
                                              escenario, nuevas_base, **PARAMETROS_CORTOS)
        filas += [{"escenario": escenario, **f} for f in filas_esc]
    return pd.DataFrame(filas)


def matriz_facilitacion(W, valor=W_FACILITACION):
    W_fac = W.copy()
    for s in ["SC", "YF"]:
        W_fac.loc["PL", s] = W_fac.loc[s, "PL"] = valor
    return W_fac


def experimento_facilitacion(malla, existentes, nuevas, W_fac):
    """Mismas 10 corridas del recocido pero optimizando con W_fac; métricas con la W original."""
    W_fac_mat = W_fac.to_numpy()
    asig_exist, fijo = rc.generar_existentes(existentes, len(malla.nodos), SEMILLA_ESCENARIO)
    filas = []
    for corrida in range(1, N_CORRIDAS + 1):
        inicial = rc.linea_base_aleatoria(asig_exist, fijo, nuevas, 100 * corrida)
        sa = rc.recocido_simulado(inicial, np.flatnonzero(~fijo), malla.vecinos, malla.aristas_idx,
                                  W_fac_mat, 100 * corrida)
        filas.append({"corrida": corrida, "metodo": "Recocido W_fac", **malla.metricas(sa["asig"])})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------- Validación con MILP

def construir_milp_fuerte(malla, n_s, relajado=False):
    """MILP sin vacíos con restricciones de enlace por arista.

    a[i,s] = 1 si el nodo i lleva la especie s; sum_s a[i,s] = 1 y sum_i a[i,s] = n_s.
    Para cada arista (i,j): sum_t y[i,j,s,t] = a[i,s] y sum_s y[i,j,s,t] = a[j,t], y >= 0.
    Se fija a[n1, SA] = 1 para romper la simetría por traslación del toroide.
    """
    W = malla.W_mat
    nodos = range(len(malla.nodos))
    especies = range(rc.N_ESPECIES)
    aristas = range(len(malla.aristas_idx))

    prob = pulp.LpProblem("tile_fuerte", pulp.LpMinimize)
    a = prob.add_variable_dict("a", (nodos, especies), lowBound=0, upBound=1,
                               cat="Continuous" if relajado else "Binary")
    y = prob.add_variable_dict("y", (aristas, especies, especies), lowBound=0)
    prob += pulp.lpSum(W[s, t] * y[e, s, t] for e in aristas for s in especies for t in especies)
    for i in nodos:
        prob += pulp.lpSum(a[i, s] for s in especies) == 1
    for s in especies:
        prob += pulp.lpSum(a[i, s] for i in nodos) == int(n_s.iloc[s])
    for e, (i, j) in enumerate(malla.aristas_idx):
        for s in especies:
            prob += pulp.lpSum(y[e, s, t] for t in especies) == a[i, s]
            prob += pulp.lpSum(y[e, t, s] for t in especies) == a[j, s]
    prob += a[0, malla.especies.get_loc("SA")] == 1
    return prob, a


def resolver_milp_fuerte(malla, n_s, ruta_log, tiempo_limite=900):
    prob_lp, _ = construir_milp_fuerte(malla, n_s, relajado=True)
    valor_lp = prob_lp.solve(pulp.COIN_CMD(msg=False)).objective

    prob, a = construir_milp_fuerte(malla, n_s)
    estadisticas = prob.solve(pulp.COIN_CMD(msg=False, timeLimit=tiempo_limite, logPath=str(ruta_log)))
    asig = np.full(len(malla.nodos), rc.SIN_ASIGNAR, dtype=np.int64)
    for (i, s), var in a.items():
        if var.value() is not None and var.value() > 0.5:
            asig[i] = s

    cota = estadisticas.best_bound
    if estadisticas.status_str == "Optimal":
        cota = estadisticas.objective
    elif cota is None:
        # PuLP no devuelve la cota cuando CBC para por tiempo; se lee del log
        encontrado = re.search(r"Lower bound:\s*([-\d.eE+]+)", ruta_log.read_text(errors="ignore"))
        cota = float(encontrado.group(1)) if encontrado else valor_lp
    return {
        "lp": valor_lp,
        "valor": estadisticas.objective,
        "cota": cota,
        "estado": estadisticas.status_str,
        "tiempo": estadisticas.time,
        "valor_verificado": rc.competencia_total(asig, malla.aristas_idx, malla.W_mat),
    }


def recocido_tile(malla, n_s):
    """Recocido en la malla toroidal: todos los nodos son movibles, sin existentes ni vacíos."""
    n = len(malla.nodos)
    valores = []
    for semilla in range(1, N_CORRIDAS_TILE + 1):
        rng = np.random.default_rng(semilla)
        inicial = rng.permutation(rc.contenido(n_s))
        sa = rc.recocido_simulado(inicial, np.arange(n), malla.vecinos, malla.aristas_idx,
                                  malla.W_mat, semilla)
        valores.append(sa["valor"])
    return np.array(valores)


def escribir_gams_tile(n_s, ruta, tiempo_limite=900):
    """Formulación fuerte sin vacíos (igual al MILP de PuLP)."""
    cantidades = ", ".join(f"{s} {int(v)}" for s, v in n_s.items())
    ruta.write_text(f"""* Modelo exacto del patrón toroidal 10x10 sin vacíos, formulación fuerte (igual al MILP de PuLP)
* Correr desde la carpeta salidas/ (usa nodos_tile.inc, aristas_tile.inc y W.inc)
Sets s 'especies' /{", ".join(n_s.index)}/
     i 'nodos' /
$include nodos_tile.inc
/
     E(i,i) 'aristas' /
$include aristas_tile.inc
/;
Alias (s,t), (i,j);

Table w(s,t)
$include W.inc
;

Parameter n(s) 'plantas por especie (total {int(n_s.sum())} = todos los nodos)' /{cantidades}/;

Binary Variable a(i,s);
Positive Variable y(i,j,s,t);
Variable z;

* Romper simetría por traslación del toroide
a.fx('n1','SA') = 1;

Equations fo, una_especie(i), cantidad(s), enlace_i(i,j,s), enlace_j(i,j,t);
fo..                    z =e= sum((i,j,s,t)$E(i,j), w(s,t)*y(i,j,s,t));
una_especie(i)..        sum(s, a(i,s)) =e= 1;
cantidad(s)..           sum(i, a(i,s)) =e= n(s);
enlace_i(i,j,s)$E(i,j).. sum(t, y(i,j,s,t)) =e= a(i,s);
enlace_j(i,j,t)$E(i,j).. sum(s, y(i,j,s,t)) =e= a(j,t);

Model tile /all/;
option reslim = {tiempo_limite}, optcr = 0;
Solve tile using MIP minimizing z;
display z.l, a.l;
""", encoding="utf-8")


# ---------------------------------------------------------------- Gráficas y archivos

def graficar_convergencia(historial, ruta):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(historial["iteracion"], historial["actual"], linewidth=1, alpha=0.7, label="Actual")
    ax.plot(historial["iteracion"], historial["mejor"], linewidth=2, label="Mejor")
    ax.set_xlabel("Iteración (movimientos)")
    ax.set_ylabel("Competencia total")
    ax2 = ax.twinx()
    ax2.plot(historial["iteracion"], historial["temperatura"], color="gray", linestyle="--",
             linewidth=1, label="Temperatura")
    ax2.set_yscale("log")
    ax2.set_ylabel("Temperatura (escala log)")
    lineas = ax.get_legend_handles_labels()
    lineas2 = ax2.get_legend_handles_labels()
    ax.legend(lineas[0] + lineas2[0], lineas[1] + lineas2[1], loc="upper right")
    ax.set_title("Convergencia del recocido simulado (1 ha toroidal, sin vacíos)")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def _dibujar_mapa(ax, nodos, asig, fijo, especies, tam, aristas_idx=None, etiquetas=False):
    x, y = nodos["x"].to_numpy(), nodos["y"].to_numpy()
    if aristas_idx is not None:
        largo = np.hypot(x[aristas_idx[:, 0]] - x[aristas_idx[:, 1]],
                         y[aristas_idx[:, 0]] - y[aristas_idx[:, 1]])
        for (a, b), l in zip(aristas_idx, largo):
            # En el toroide las aristas que dan la vuelta no se dibujan
            if l < 2 * largo.min():
                ax.plot(x[[a, b]], y[[a, b]], color="#cccccc", linewidth=0.6, zorder=0)
    for k, s in enumerate(especies):
        es = asig == k
        ax.scatter(x[es], y[es], s=tam, color=COLOR_ESPECIE[s],
                   edgecolors=np.where(fijo[es], "black", "none"), linewidths=1.2, zorder=2)
        if etiquetas:
            for xi, yi in zip(x[es], y[es]):
                ax.text(xi, yi, s, ha="center", va="center", fontsize=8, zorder=3, clip_on=True,
                        color="white" if s in ["SA", "SC", "OR", "OS", "PL"] else "black")
    ax.set_aspect("equal")


def graficar_mapas(nodos, soluciones, fijo, especies, aristas_idx, carpeta):
    for nombre, zoom in [("mapa_1ha.png", None), ("mapa_zoom.png", ZOOM)]:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8.5))
        for ax, metodo in zip(axes, ["Base actual", "Recocido"]):
            if zoom is None:
                _dibujar_mapa(ax, nodos, soluciones[metodo], fijo, especies, tam=28)
            else:
                _dibujar_mapa(ax, nodos, soluciones[metodo], fijo, especies, tam=700,
                              aristas_idx=aristas_idx, etiquetas=True)
                ax.set_xlim(zoom[0] - 1.5, zoom[1] + 1.5)
                ax.set_ylim(zoom[0] - 1.5, zoom[1] + 1.5)
            ax.set_title(metodo)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
        for s, color in COLOR_ESPECIE.items():
            axes[0].scatter([], [], color=color, label=s)
        axes[0].scatter([], [], color="white", edgecolors="black", label="Existente (borde negro)")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=11, frameon=False)
        region = "1 ha toroidal" if zoom is None else f"zoom {zoom[1] - zoom[0]}x{zoom[1] - zoom[0]} m"
        fig.suptitle(f"Acomodo por especie ({region}, sin vacíos): línea base actual vs recocido simulado")
        fig.tight_layout(rect=(0, 0.07, 1, 0.96))
        fig.savefig(carpeta / nombre, dpi=150)
        plt.close(fig)


def graficar_boxplot_escenarios(tabla, ruta):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot([tabla.loc[tabla["metodo"] == m, "competencia_total"] for m in METODOS],
               tick_labels=[m.replace(" cantidades", "\ncantidades") for m in METODOS],
               medianprops={"color": "black"})
    ax.set_ylabel("Competencia total")
    ax.set_title(f"Competencia total en {N_ESCENARIOS} escenarios de existentes (1 ha toroidal, sin vacíos)")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def guardar_solucion(nodos, asig, fijo, especies, ruta):
    nombres = np.array(list(especies))
    pd.DataFrame({
        "id_nodo": nodos["id"],
        "x": nodos["x"].round(3),
        "y": nodos["y"].round(3),
        "especie": nombres[asig],
        "existente": fijo,
    }).to_csv(ruta, index=False)


# ---------------------------------------------------------------- Todo el paso 4

def ejecutar_paso4(W, prop_base, nodos_tile, aristas_tile, carpeta, densidad_ha=658,
                   tiempo_milp=900):
    sep = mll.separacion_tresbolillo(densidad_ha)
    dy = sep * np.sqrt(3) / 2
    nodos_ha, aristas_ha = mll.generar_malla_tresbolillo(100, 100, sep, toroidal=True)
    malla_ha = Malla(nodos_ha, aristas_ha, W)
    n_col, n_fil = mll.dimensiones_toroide(100, 100, sep)
    area_ha = n_col * sep * n_fil * dy / 10000
    existentes, nuevas = rc.cargar_cantidades(carpeta, len(nodos_ha), densidad_ha)
    nuevas_base = rc.cantidades_linea_base(W.index, nuevas.sum())
    texto = []

    cantidades = pd.DataFrame({"existentes": existentes, "nuevas": nuevas,
                               "nuevas_base_actual": nuevas_base,
                               "diferencia_base": nuevas_base - nuevas})
    texto += [f"MALLA SIN VACÍOS: densidad_ha = {densidad_ha}",
              f"Separación del tresbolillo: sep = sqrt(2*10000 / (sqrt(3)*{densidad_ha})) = {sep:.4f} m "
              f"(entre filas {dy:.4f} m)",
              f"Malla toroidal: {n_col} columnas x {n_fil} filas = {len(nodos_ha)} nodos, todos de grado 6 "
              f"(periodo {n_col * sep:.2f} x {n_fil * dy:.2f} m = {area_ha:.4f} ha, "
              f"{len(nodos_ha) / area_ha:.2f} nodos/ha).",
              f"Total: {existentes.sum()} existentes ({rc.TOTAL_EXISTENTES}/{densidad_ha} de los nodos) + "
              f"{nuevas.sum()} nuevas = {len(nodos_ha)} nodos, 0 vacíos",
              "",
              "CANTIDADES POR ESPECIE (malla de 1 ha)", cantidades.to_string(),
              "La línea base actual repite la secuencia de 21 plantas en todos los nodos libres y NO "
              "respeta las proporciones p_s (ver diferencia_base).", ""]

    # 10 corridas con un mismo escenario
    tabla_semillas, mejor = experimento_semillas(malla_ha, existentes, nuevas, nuevas_base)
    sa = mejor["sa"]
    texto += [f"RECOCIDO: numba {'sí' if rc.HAY_NUMBA else 'no'} disponible; alfa = 0.995, L = 2000, "
              "T_min = 1e-3, paro tras 100 niveles sin mejora contados con T < 1% de T0; "
              "movimiento: intercambio especie-especie entre nodos nuevos",
              f"Mejor corrida: #{mejor['corrida']}, T0 = {sa['T0']:.3f}, "
              f"{len(sa['historial'])} niveles, competencia {sa['valor_inicial']:.2f} -> {sa['valor']:.2f}",
              "",
              f"{N_CORRIDAS} CORRIDAS DEL RECOCIDO (mismo escenario de existentes, semilla {SEMILLA_ESCENARIO})",
              resumir_metricas(tabla_semillas).to_string(), "",
              f"COMPARACIÓN JUSTA (competencia total, {N_CORRIDAS} corridas)",
              comparacion_justa(tabla_semillas).to_string(), ""]

    # 30 escenarios de existentes
    tabla_escenarios = experimento_escenarios(malla_ha, existentes, nuevas, nuevas_base)
    texto += [f"{N_ESCENARIOS} ESCENARIOS DE EXISTENTES (recocido corto: alfa = 0.98, L = 1000)",
              resumir_metricas(tabla_escenarios).to_string(), "",
              f"COMPARACIÓN JUSTA (competencia total, {N_ESCENARIOS} escenarios)",
              comparacion_justa(tabla_escenarios).to_string(), ""]

    # Escenario de facilitación
    tabla_fac = experimento_facilitacion(malla_ha, existentes, nuevas, matriz_facilitacion(W))
    recocido_w = tabla_semillas[tabla_semillas["metodo"] == "Recocido"]
    facilitacion = pd.DataFrame({
        "Recocido con W": recocido_w[list(SENTIDO_METRICAS)].mean(),
        "Recocido con W_fac": tabla_fac[list(SENTIDO_METRICAS)].mean(),
    })
    facilitacion["cambio"] = facilitacion["Recocido con W_fac"] - facilitacion["Recocido con W"]
    facilitacion["cambio %"] = (facilitacion["cambio"]
                                / facilitacion["Recocido con W"].replace(0, np.nan) * 100)
    texto += [f"ESCENARIO DE FACILITACIÓN: W[PL,SC] = W[PL,YF] = {W_FACILITACION} (y simétricos); "
              f"{N_CORRIDAS} corridas con las mismas semillas",
              "La competencia se evalúa con la W original.",
              facilitacion.round(2).to_string(), ""]

    # Validación contra el MILP fuerte en la malla toroidal 10x10
    malla_tile = Malla(nodos_tile, aristas_tile, W)
    n_tile = rc.redondear_a_total(prop_base, len(nodos_tile))
    escribir_gams_tile(n_tile, carpeta / "modelo_tile.gms", tiempo_milp)
    valores_sa = recocido_tile(malla_tile, n_tile)
    milp = resolver_milp_fuerte(malla_tile, n_tile, carpeta / "milp_tile.log", tiempo_milp)
    mejor_sa = valores_sa.min()
    texto += [f"VALIDACIÓN: MALLA TOROIDAL 10x10 SIN VACÍOS, FORMULACIÓN FUERTE "
              f"({len(nodos_tile)} plantas proporcionales a p_s, a[n1,SA] = 1)",
              "Plantas por especie: " + ", ".join(f"{s} {v}" for s, v in n_tile.items()),
              f"Relajación LP: {milp['lp']:.4f}",
              f"MILP (CBC, límite {tiempo_milp} s): estado {milp['estado']}, {milp['tiempo']:.0f} s, "
              f"mejor valor {milp['valor']:.4f} (verificado {milp['valor_verificado']:.4f}), "
              f"cota final {milp['cota']:.4f}",
              f"Recocido ({N_CORRIDAS_TILE} corridas): mejor {mejor_sa:.4f}, "
              f"media {valores_sa.mean():.4f} ± {valores_sa.std():.4f}",
              f"Gap recocido vs cota: {(mejor_sa - milp['cota']) / mejor_sa * 100:.2f}%",
              f"Gap recocido vs mejor MILP: {(mejor_sa - milp['valor']) / milp['valor'] * 100:+.2f}% "
              "(negativo = el recocido encontró algo mejor)",
              ""]

    # Gráficas y archivos
    graficar_convergencia(sa["historial"], carpeta / "convergencia_recocido.png")
    graficar_mapas(nodos_ha, mejor["soluciones"], mejor["fijo"], W.index, malla_ha.aristas_idx, carpeta)
    graficar_boxplot_escenarios(tabla_escenarios, carpeta / "boxplot_escenarios.png")
    guardar_solucion(nodos_ha, sa["asig"], mejor["fijo"], W.index, carpeta / "solucion_1ha.csv")
    texto += ["Archivos: solucion_1ha.csv (malla toroidal), modelo_tile.gms, milp_tile.log, "
              "convergencia_recocido.png, mapa_1ha.png, mapa_zoom.png, boxplot_escenarios.png, "
              "resultados_paso4_v3.txt"]

    resumen = "\n".join(texto)
    (carpeta / "resultados_paso4_v3.txt").write_text(resumen + "\n", encoding="utf-8")
    return resumen
