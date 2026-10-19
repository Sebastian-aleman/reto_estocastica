"""Malla de plantación a tresbolillo (triangular) y su exportación a GAMS."""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def generar_malla_tresbolillo(ancho, alto, sep=3.4, toroidal=False):
    """Nodos (id, x, y) a tresbolillo y aristas entre vecinos a distancia sep.

    Filas separadas sep*sqrt(3)/2; las impares se desplazan sep/2.
    Con toroidal=True los bordes se pegan (necesita un número par de filas).
    """
    dy = sep * np.sqrt(3) / 2
    n_col = int(round(ancho / sep)) if toroidal else int(ancho // sep) + 1
    n_fil = int(round(alto / dy)) if toroidal else int(alto // dy) + 1

    puntos = []
    for j in range(n_fil):
        for k in range(n_col):
            x = k * sep + (j % 2) * sep / 2
            if toroidal or x <= ancho:
                puntos.append((x, j * dy))
    nodos = pd.DataFrame(puntos, columns=["x", "y"])
    nodos.insert(0, "id", range(1, len(nodos) + 1))

    # Distancias entre todos los pares de nodos
    xy = nodos[["x", "y"]].to_numpy()
    dif = xy[:, None, :] - xy[None, :, :]
    if toroidal:
        periodo = np.array([n_col * sep, n_fil * dy])
        dif -= periodo * np.round(dif / periodo)
    dist = np.hypot(dif[..., 0], dif[..., 1])

    i, j = np.where(np.abs(dist - sep) < 1e-6)
    aristas = [(int(a) + 1, int(b) + 1) for a, b in zip(i, j) if a < b]
    return nodos, aristas


def grados(nodos, aristas):
    G = nx.Graph()
    G.add_nodes_from(nodos["id"])
    G.add_edges_from(aristas)
    return dict(G.degree)


def exportar_malla_gams(nodos, aristas, carpeta):
    """Listas para GAMS: n1, n2, ... y aristas n1.n2 (cada una una sola vez)."""
    (carpeta / "nodos_tile.inc").write_text(
        "\n".join(f"n{i}" for i in nodos["id"]) + "\n", encoding="utf-8")
    (carpeta / "aristas_tile.inc").write_text(
        "\n".join(f"n{a}.n{b}" for a, b in aristas) + "\n", encoding="utf-8")


def graficar_malla(nodos, aristas, ruta):
    xy = nodos.set_index("id")
    fig, ax = plt.subplots(figsize=(8, 8))
    for a, b in aristas:
        ax.plot(xy.loc[[a, b], "x"], xy.loc[[a, b], "y"], color="lightgray", linewidth=0.5)
    ax.scatter(nodos["x"], nodos["y"], s=5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Malla tresbolillo de 1 ha: {len(nodos)} nodos")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
