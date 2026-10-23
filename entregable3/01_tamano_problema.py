"""Tamaño del problema de acomodo en 1 ha: especies, demanda, malla, variables y parámetros.

Usa el código del Entregable 2 (datos, competencia, distribuciones, malla) sin
modificarlo y guarda tamano_problema.md en esta carpeta. Solo construye la
malla; no corre el recocido ni el Monte Carlo.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CARPETA = Path(__file__).parent
sys.path.insert(0, str(CARPETA.parent))

import competencia  # noqa: E402
import datos  # noqa: E402
import distribuciones  # noqa: E402
import malla  # noqa: E402
import recocido as rc  # noqa: E402

DENSIDAD_HA = 658
ARISTAS_ESPERADAS = DENSIDAD_HA * 3          # grado 6 en el toroide: 6N/2 = 3N


# ---------------------------------------------------------------- Mallas

def malla_codigo(sep):
    """Malla toroidal de "1 ha" tal como la construye main.py / comparacion.py."""
    nodos, aristas = malla.generar_malla_tresbolillo(100, 100, sep, toroidal=True)
    n_col, n_fil = malla.dimensiones_toroide(100, 100, sep)
    return describir_malla("Malla del código (100x100 m)", nodos, aristas, n_col, n_fil, sep)


def malla_exacta(sep, n_nodos):
    """Toroide con exactamente n_nodos: la factorización con forma más cercana a un cuadrado."""
    n_col, n_fil = malla.dimensiones_toroide_exacto(n_nodos, sep)
    nodos, aristas = malla.generar_malla_toroide_exacto(n_nodos)
    return describir_malla(f"Malla exacta de {n_nodos} nodos", nodos, aristas, n_col, n_fil, sep)


def describir_malla(nombre, nodos, aristas, n_col, n_fil, sep):
    dy = sep * np.sqrt(3) / 2
    ancho, alto = n_col * sep, n_fil * dy
    return {
        "nombre": nombre,
        "n_col": n_col,
        "n_fil": n_fil,
        "ancho_m": ancho,
        "alto_m": alto,
        "area_ha": ancho * alto / 10000,
        "nodos": len(nodos),
        "aristas": len(aristas),
        "grados": sorted(set(malla.grados(nodos, aristas).values())),
    }


# ---------------------------------------------------------------- Variables y parámetros

def contar_variables(n_nodos, n_aristas, n_especies):
    """a_is binaria por nodo y especie; y_ijst por arista (i<j) y par de especies; n_s entera."""
    variables = {
        "a_is": n_nodos * n_especies,
        "y_ijst": n_aristas * n_especies ** 2,
        "n_s": n_especies,
    }
    variables["total"] = sum(variables.values())
    return variables


def listar_parametros(especies, p_s, c_s, D_s, W, normal):
    """Parámetros del modelo agrupados por tipo: (símbolo, descripción, número de valores)."""
    S = len(especies)
    escalares = [
        ("N", f"nodos (plantas) por ha = {DENSIDAD_HA}", 1),
        ("B", f"presupuesto por ha = ${datos.PRESUPUESTO_HA:,}", 1),
        ("E_ha", f"plantas existentes por ha = {rc.TOTAL_EXISTENTES}", 1),
        ("mu", f"media de la Normal de existentes/ha = {normal['media']:.2f}", 1),
        ("sigma", f"desv. est. de la Normal = {normal['desv_muestral']:.2f} "
                  f"(muestral; {normal['desv']:.2f} poblacional en el código)", 1),
    ]
    vectores = [
        ("p_s", "proporción objetivo por especie", len(p_s)),
        ("c_s", "costo unitario por especie", len(c_s)),
        ("D_s", "demanda por ha = p_s * 658", len(D_s)),
        ("X_s", "existentes esperados por especie (X_esperado.inc)", S),
    ]
    matrices = [
        ("w_st", f"competencia ponderada {W.shape[0]}x{W.shape[1]} "
                 f"(simétrica: {S * (S + 1) // 2} valores distintos)", W.size),
    ]
    return {"Escalares": escalares, "Vectores": vectores, "Matrices": matrices}


# ---------------------------------------------------------------- Reporte

def tabla_md(encabezados, filas):
    lineas = ["| " + " | ".join(encabezados) + " |", "|" + "---|" * len(encabezados)]
    lineas += ["| " + " | ".join(str(c) for c in fila) + " |" for fila in filas]
    return lineas


def explicar_aristas(malla_cod, malla_exa):
    if malla_cod["aristas"] == ARISTAS_ESPERADAS:
        return [f"✔ La malla del código tiene {malla_cod['aristas']:,} aristas = 658·3."]
    otras = ", ".join(f"{c}×{f}" for c, f in malla.factorizaciones_toroide(DENSIDAD_HA))
    return [
        f"**⚠ La malla toroidal de 1 ha del código NO tiene 1,974 aristas: tiene "
        f"{malla_cod['aristas']:,}.**",
        "",
        f"- Todos los nodos tienen grado {', '.join(map(str, malla_cod['grados']))}, así que aristas = 6·N/2 = 3·N "
        f"= 3·{malla_cod['nodos']} = {3 * malla_cod['nodos']:,}. La relación aristas = 3·N sí se cumple; "
        f"lo que cambia es N.",
        f"- `malla.dimensiones_toroide(100, 100, sep)` busca el toroide más cercano a 100×100 m con "
        f"número **par** de filas (para que el desplazamiento de las filas impares cierre al dar la vuelta): "
        f"{malla_cod['n_col']} columnas × {malla_cod['n_fil']} filas = {malla_cod['nodos']} nodos.",
        f"- Ese toroide mide {malla_cod['ancho_m']:.2f} × {malla_cod['alto_m']:.2f} m = "
        f"{malla_cod['area_ha']:.4f} ha. La densidad es exactamente 658 nodos/ha, pero el área es "
        f"{(malla_cod['area_ha'] - 1) * 100:.2f}% mayor a 1 ha, por eso hay "
        f"{malla_cod['nodos'] - DENSIDAD_HA} nodos de más.",
        f"- Para tener exactamente 658 nodos se necesita columnas × filas = 658 = 2·7·47 con filas par "
        f"y ambos ≥ 3. Las únicas opciones son {otras} (columnas×filas); ninguna es cuadrada. La más "
        f"cercana es {malla_exa['n_col']}×{malla_exa['n_fil']} = {malla_exa['ancho_m']:.1f} × "
        f"{malla_exa['alto_m']:.1f} m (exactamente {malla_exa['area_ha']:.4f} ha), con "
        f"{malla_exa['aristas']:,} aristas.",
        "- Ambas mallas se construyen con la misma función `malla.generar_malla_tresbolillo`; las dos "
        "columnas de las tablas siguientes muestran el tamaño del problema con cada una.",
    ]


def main():
    # Datos del Entregable 2
    especies = datos.cargar_especies()
    monitoreo = datos.cargar_monitoreo()
    supervivencia = datos.calcular_supervivencia(monitoreo)
    prop_base = datos.calcular_prop_base(monitoreo)
    p_s = prop_base / prop_base.sum()
    c_s = especies["costo"]
    D_s = p_s * DENSIDAD_HA
    D_s_entero = rc.redondear_a_total(p_s, DENSIDAD_HA)
    W = competencia.calcular_W(competencia.matriz_base_C(especies),
                               competencia.calcular_vulnerabilidad(supervivencia["mortalidad_acum"]))

    plantas_por_ha = distribuciones.calcular_plantas_por_ha(datos.cargar_plantas(), datos.cargar_areas())
    _, ajuste, _ = distribuciones.ajustar_densidad(plantas_por_ha)
    normal = dict(ajuste["parametros"], desv_muestral=plantas_por_ha.std(ddof=1))

    # Mallas
    sep = malla.separacion_tresbolillo(DENSIDAD_HA)
    malla_cod = malla_codigo(sep)
    malla_exa = malla_exacta(sep, DENSIDAD_HA)
    assert malla_exa["nodos"] == DENSIDAD_HA and malla_exa["aristas"] == ARISTAS_ESPERADAS
    mallas = [malla_cod, malla_exa]

    S = len(especies)
    variables = [contar_variables(m["nodos"], m["aristas"], S) for m in mallas]
    parametros = listar_parametros(especies.index, p_s, c_s, D_s, W, normal)

    # Reporte
    texto = ["# Tamaño del problema (1 ha, malla tresbolillo toroidal)", "",
             "Generado con `01_tamano_problema.py` a partir del código del Entregable 2 "
             "(`datos.py`, `competencia.py`, `distribuciones.py`, `malla.py`).", "",
             f"## 1. Especies y demanda por ha (S = {S})", "",
             f"p_s = vivos 2021 normalizados (`datos.calcular_prop_base`), suma = {p_s.sum():.6f}. "
             f"D_s = p_s · {DENSIDAD_HA}; D_s entero por redondeo de mayor residuo "
             f"(`recocido.redondear_a_total`), suma = {D_s_entero.sum()}.", ""]
    filas = [(s, especies.loc[s, "nombre"], f"{p_s[s]:.4f}", f"{c_s[s]:.2f}", f"{D_s[s]:.2f}", D_s_entero[s])
             for s in especies.index]
    filas.append(("**Total**", "", f"{p_s.sum():.4f}", "", f"{D_s.sum():.2f}", D_s_entero.sum()))
    texto += tabla_md(["Especie", "Nombre", "p_s", "c_s ($)", "D_s = p_s·658", "D_s entero"], filas)

    texto += ["", "## 2. Malla", "",
              f"Separación sep = √(2·10000 / (√3·{DENSIDAD_HA})) = {sep:.4f} m; "
              f"entre filas {sep * np.sqrt(3) / 2:.4f} m.", ""]
    filas = [
        ("Columnas × filas", *[f"{m['n_col']} × {m['n_fil']}" for m in mallas]),
        ("Dimensiones (m)", *[f"{m['ancho_m']:.2f} × {m['alto_m']:.2f}" for m in mallas]),
        ("Área (ha)", *[f"{m['area_ha']:.4f}" for m in mallas]),
        ("Nodos", *[f"{m['nodos']:,}" for m in mallas]),
        ("Nodos por ha", *[f"{m['nodos'] / m['area_ha']:.2f}" for m in mallas]),
        ("Grado de los nodos", *[", ".join(map(str, m["grados"])) for m in mallas]),
        ("Aristas", *[f"{m['aristas']:,}" for m in mallas]),
        ("¿Aristas = 658·3 = 1,974?", *["sí" if m["aristas"] == ARISTAS_ESPERADAS else "no" for m in mallas]),
    ]
    texto += tabla_md(["", malla_cod["nombre"], malla_exa["nombre"]], filas)
    texto += ["", "### Verificación de aristas", ""] + explicar_aristas(malla_cod, malla_exa)

    texto += ["", "## 3. Variables", "",
              "a_is ∈ {0,1}: el nodo i lleva la especie s. y_ijst ≥ 0: la arista (i,j), i<j, une "
              "las especies s y t (una por arista y par ordenado de especies, como en "
              "`comparacion.construir_milp_fuerte`). n_s ∈ ℤ⁺: plantas nuevas de la especie s.", ""]
    filas = [
        ("a_is", "N · S", *[f"{v['a_is']:,}" for v in variables]),
        ("y_ijst", "\\|E\\| · S²", *[f"{v['y_ijst']:,}" for v in variables]),
        ("n_s", "S", *[f"{v['n_s']:,}" for v in variables]),
        ("**Total**", "", *[f"**{v['total']:,}**" for v in variables]),
    ]
    texto += tabla_md(["Variable", "Fórmula", malla_cod["nombre"], malla_exa["nombre"]], filas)

    texto += ["", "## 4. Parámetros", "",
              "No dependen de la malla. La lista de aristas E es un conjunto (estructura del grafo), "
              "no un parámetro numérico.", ""]
    filas = [(tipo, simbolo, descripcion, valores)
             for tipo, lista in parametros.items() for simbolo, descripcion, valores in lista]
    texto += tabla_md(["Tipo", "Símbolo", "Descripción", "Valores"], filas)
    texto += [""]
    filas = [(tipo, len(lista), sum(v for *_, v in lista)) for tipo, lista in parametros.items()]
    filas.append(("**Total**", sum(f[1] for f in filas), sum(f[2] for f in filas)))
    texto += tabla_md(["Tipo", "Parámetros", "Valores numéricos"], filas)

    texto += ["", "### Nota sobre la Normal", "",
              f"`distribuciones.ajustar_densidad` ajusta la Normal por máxima verosimilitud: "
              f"μ = {normal['media']:.2f}, σ = {normal['desv']:.4f} (desviación poblacional, ddof = 0). "
              f"El valor σ = 3.35 corresponde a la desviación muestral (ddof = 1): "
              f"{normal['desv_muestral']:.4f}. El Monte Carlo del Entregable 2 usa σ = {normal['desv']:.2f}.", ""]

    markdown = "\n".join(texto)
    (CARPETA / "tamano_problema.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
