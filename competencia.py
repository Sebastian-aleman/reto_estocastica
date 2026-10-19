"""Matriz de competencia entre especies ponderada por vulnerabilidad."""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

COLOR_GENERO = {"Agave": "#2a78d6", "Opuntia": "#eb6834", "Prosopis": "#1baf7a", "Yucca": "#eda100"}

# Competencia base entre géneros distintos (o mismo género, especies distintas)
COMPETENCIA_GENEROS = {
    ("Agave", "Agave"): 0.7,
    ("Opuntia", "Opuntia"): 0.7,
    ("Agave", "Opuntia"): 0.5,
    ("Agave", "Yucca"): 0.5,
    ("Opuntia", "Yucca"): 0.4,
    ("Agave", "Prosopis"): 0.2,
    ("Opuntia", "Prosopis"): 0.2,
    ("Prosopis", "Yucca"): 0.3,
}


def matriz_base_C(especies):
    abrevs = especies.index
    C = pd.DataFrame(1.0, index=abrevs, columns=abrevs)
    for s in abrevs:
        for t in abrevs:
            if s != t:
                par = tuple(sorted([especies.loc[s, "genero"], especies.loc[t, "genero"]]))
                C.loc[s, t] = COMPETENCIA_GENEROS[par]
    return C


def calcular_vulnerabilidad(mortalidad_acum):
    return mortalidad_acum / mortalidad_acum.mean()


def calcular_W(C, vul):
    promedio = (vul.to_numpy()[:, None] + vul.to_numpy()[None, :]) / 2
    return C * promedio


def guardar_W(W, carpeta):
    """Guarda W.csv y W.inc (cuerpo de una Table de GAMS)."""
    W.to_csv(carpeta / "W.csv")
    tabla = W.rename_axis(index=None, columns=None).round(6).to_string()
    (carpeta / "W.inc").write_text(tabla + "\n", encoding="utf-8")


def graficar_heatmap_W(W, ruta):
    fig, ax = plt.subplots(figsize=(7, 6))
    imagen = ax.imshow(W, cmap="Blues")
    ax.set_xticks(range(len(W)), W.columns)
    ax.set_yticks(range(len(W)), W.index)
    for i in range(len(W)):
        for j in range(len(W)):
            valor = W.iloc[i, j]
            ax.text(j, i, f"{valor:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if valor > 1 else "black")
    fig.colorbar(imagen, label="W[s,t]")
    ax.set_title("Competencia ponderada W[s,t]")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# Agave arriba, Opuntia abajo, Prosopis a la izquierda y Yucca a la derecha.
# Cada fila va en arco y SA (el nodo más grande) en una punta, para que las
# aristas no pasen encima de otros nodos.
POSICIONES_GRAFO = {
    "SA": (-1.5, 1.2), "SC": (-0.5, 1.8), "ST": (0.5, 1.8), "LE": (1.5, 1.2),
    "OC": (-1.5, -1.2), "OE": (-0.5, -1.8), "OR": (0.5, -1.8), "OS": (1.5, -1.2),
    "PL": (-2.4, 0.0), "YF": (2.4, 0.0),
}


def graficar_grafo_competencia(W, prop_base, especies, ruta, umbral=0.5):
    G = nx.Graph()
    G.add_nodes_from(W.index)
    for s in W.index:
        for t in W.columns:
            if s < t and W.loc[s, t] > umbral:
                G.add_edge(s, t, peso=W.loc[s, t])

    pos = {k: np.array(v) for k, v in POSICIONES_GRAFO.items()}
    pesos = [G[s][t]["peso"] for s, t in G.edges]
    tam_nodo = [6000 * prop_base[s] for s in G.nodes]
    colores = [COLOR_GENERO[especies.loc[s, "genero"]] for s in G.nodes]
    autocomp = [W.loc[s, s] for s in G.nodes]

    fig, ax = plt.subplots(figsize=(11, 7))
    bordes = nx.draw_networkx_edges(G, pos, ax=ax, width=[4 * w for w in pesos],
                                    edge_color=pesos, edge_cmap=plt.cm.Greys,
                                    edge_vmin=0, edge_vmax=W.to_numpy().max())
    # Autoaristas W[s,s]: anillo alrededor del nodo
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[2 * n + 600 for n in tam_nodo],
                           node_color="none", edgecolors=plt.cm.Greys(np.array(autocomp) / max(autocomp)),
                           linewidths=[4 * w for w in autocomp])
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=tam_nodo, node_color=colores)
    # Etiqueta hacia afuera, separada del anillo según su tamaño
    for s, n in zip(G.nodes, tam_nodo):
        direccion = pos[s] / np.linalg.norm(pos[s])
        separacion = np.sqrt(2 * n + 600) / 2 + 14
        ax.annotate(s, pos[s], xytext=direccion * separacion, textcoords="offset points",
                    ha="center", va="center", fontsize=11)

    for genero, color in COLOR_GENERO.items():
        ax.scatter([], [], color=color, label=genero)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0), ncol=4, title="Género")
    fig.colorbar(bordes, ax=ax, label="W", shrink=0.6)
    ax.set_title(f"Grafo de competencia (aristas con W > {umbral}; anillo = W[s,s])")
    ax.axis("off")
    ax.margins(0.12, 0.2)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return G
