"""Malla de plantación a tresbolillo (triangular) y su exportación a GAMS."""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def separacion_tresbolillo(densidad_ha):
    """Separación (m) con la que el tresbolillo tiene densidad_ha nodos por ha.

    Cada nodo ocupa un rombo de sep x sep*sqrt(3)/2 m², así que
    densidad_ha = 10000 / (sep² sqrt(3)/2).
    """
    return np.sqrt(2 * 10000 / (np.sqrt(3) * densidad_ha))


def dimensiones_toroide(ancho, alto, sep):
    """Columnas y filas (número par) del toroide más cercano a ancho x alto."""
    dy = sep * np.sqrt(3) / 2
    return int(round(ancho / sep)), 2 * int(round(alto / dy / 2))


def factorizaciones_toroide(n_nodos):
    """(columnas, filas) con columnas*filas = n_nodos, filas par y ambos >= 3 (grado 6 sin repetir vecinos)."""
    return [(n_nodos // f, f) for f in range(4, n_nodos + 1, 2)
            if n_nodos % f == 0 and n_nodos // f >= 3]


def dimensiones_toroide_exacto(n_nodos, sep):
    """Columnas y filas del toroide con exactamente n_nodos, de forma lo más cuadrada posible.

    Para 658 = 2·7·47 solo hay 47x14 y 7x94; se elige 47x14 (196.9 x 50.8 m).
    """
    dy = sep * np.sqrt(3) / 2
    return min(factorizaciones_toroide(n_nodos),
               key=lambda cf: abs(np.log(cf[0] * sep / (cf[1] * dy))))


def generar_malla_toroide_exacto(densidad_ha):
    """Toroide de exactamente 1 ha: densidad_ha nodos, todos de grado 6 (para 658: 47x14).

    Alternativa a generar_malla_tresbolillo(100, 100, sep, toroidal=True), que redondea
    a un toroide casi cuadrado y no siempre tiene densidad_ha nodos (para 658 da 24x28 = 672).
    """
    sep = separacion_tresbolillo(densidad_ha)
    n_col, n_fil = dimensiones_toroide_exacto(densidad_ha, sep)
    return generar_malla_tresbolillo(n_col * sep, n_fil * sep * np.sqrt(3) / 2, sep, toroidal=True)


def generar_malla_tresbolillo(ancho, alto, sep=3.4, toroidal=False):
    """Nodos (id, x, y) a tresbolillo y aristas entre vecinos a distancia sep.

    Filas separadas sep*sqrt(3)/2; las impares se desplazan sep/2.
    Con toroidal=True los bordes se pegan; el número de filas se redondea a par
    para que el desplazamiento de las filas cierre al dar la vuelta. La densidad
    del toroide es exactamente 10000 / (sep² sqrt(3)/2) nodos por ha.
    """
    dy = sep * np.sqrt(3) / 2
    if toroidal:
        n_col, n_fil = dimensiones_toroide(ancho, alto, sep)
    else:
        n_col, n_fil = int(ancho // sep) + 1, int(alto // dy) + 1

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
