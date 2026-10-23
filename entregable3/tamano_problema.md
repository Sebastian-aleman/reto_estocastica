# Tamaño del problema (1 ha, malla tresbolillo toroidal)

Generado con `01_tamano_problema.py` a partir del código del Entregable 2 (`datos.py`, `competencia.py`, `distribuciones.py`, `malla.py`).

## 1. Especies y demanda por ha (S = 10)

p_s = vivos 2021 normalizados (`datos.calcular_prop_base`), suma = 1.000000. D_s = p_s · 658; D_s entero por redondeo de mayor residuo (`recocido.redondear_a_total`), suma = 658.

| Especie | Nombre | p_s | c_s ($) | D_s = p_s·658 | D_s entero |
|---|---|---|---|---|---|
| LE | Agave lechuguilla | 0.0638 | 26.00 | 42.00 | 42 |
| SA | Agave salmiana | 0.2979 | 26.00 | 196.00 | 196 |
| SC | Agave scabra | 0.0638 | 26.00 | 42.00 | 42 |
| ST | Agave striata | 0.0638 | 26.00 | 42.00 | 42 |
| OC | Opuntia cantabrigiensis | 0.0745 | 18.00 | 49.00 | 49 |
| OE | Opuntia engelmannii | 0.0578 | 21.00 | 38.00 | 38 |
| OR | Opuntia robusta | 0.1109 | 18.00 | 73.00 | 73 |
| OS | Opuntia streptacantha | 0.0973 | 18.00 | 64.00 | 64 |
| PL | Prosopis laevigata | 0.1307 | 26.50 | 86.00 | 86 |
| YF | Yucca filifera | 0.0395 | 26.00 | 26.00 | 26 |
| **Total** |  | 1.0000 |  | 658.00 | 658 |

## 2. Malla

Separación sep = √(2·10000 / (√3·658)) = 4.1891 m; entre filas 3.6279 m.

|  | Malla del código (100x100 m) | Malla exacta de 658 nodos |
|---|---|---|
| Columnas × filas | 24 × 28 | 47 × 14 |
| Dimensiones (m) | 100.54 × 101.58 | 196.89 × 50.79 |
| Área (ha) | 1.0213 | 1.0000 |
| Nodos | 672 | 658 |
| Nodos por ha | 658.00 | 658.00 |
| Grado de los nodos | 6 | 6 |
| Aristas | 2,016 | 1,974 |
| ¿Aristas = 658·3 = 1,974? | no | sí |

### Verificación de aristas

**⚠ La malla toroidal de 1 ha del código NO tiene 1,974 aristas: tiene 2,016.**

- Todos los nodos tienen grado 6, así que aristas = 6·N/2 = 3·N = 3·672 = 2,016. La relación aristas = 3·N sí se cumple; lo que cambia es N.
- `malla.dimensiones_toroide(100, 100, sep)` busca el toroide más cercano a 100×100 m con número **par** de filas (para que el desplazamiento de las filas impares cierre al dar la vuelta): 24 columnas × 28 filas = 672 nodos.
- Ese toroide mide 100.54 × 101.58 m = 1.0213 ha. La densidad es exactamente 658 nodos/ha, pero el área es 2.13% mayor a 1 ha, por eso hay 14 nodos de más.
- Para tener exactamente 658 nodos se necesita columnas × filas = 658 = 2·7·47 con filas par y ambos ≥ 3. Las únicas opciones son 47×14, 7×94 (columnas×filas); ninguna es cuadrada. La más cercana es 47×14 = 196.9 × 50.8 m (exactamente 1.0000 ha), con 1,974 aristas.
- Ambas mallas se construyen con la misma función `malla.generar_malla_tresbolillo`; las dos columnas de las tablas siguientes muestran el tamaño del problema con cada una.

## 3. Variables

a_is ∈ {0,1}: el nodo i lleva la especie s. y_ijst ≥ 0: la arista (i,j), i<j, une las especies s y t (una por arista y par ordenado de especies, como en `comparacion.construir_milp_fuerte`). n_s ∈ ℤ⁺: plantas nuevas de la especie s.

| Variable | Fórmula | Malla del código (100x100 m) | Malla exacta de 658 nodos |
|---|---|---|---|
| a_is | N · S | 6,720 | 6,580 |
| y_ijst | \|E\| · S² | 201,600 | 197,400 |
| n_s | S | 10 | 10 |
| **Total** |  | **208,330** | **203,990** |

## 4. Parámetros

No dependen de la malla. La lista de aristas E es un conjunto (estructura del grafo), no un parámetro numérico.

| Tipo | Símbolo | Descripción | Valores |
|---|---|---|---|
| Escalares | N | nodos (plantas) por ha = 658 | 1 |
| Escalares | B | presupuesto por ha = $12,500 | 1 |
| Escalares | E_ha | plantas existentes por ha = 129 | 1 |
| Escalares | mu | media de la Normal de existentes/ha = 129.18 | 1 |
| Escalares | sigma | desv. est. de la Normal = 3.35 (muestral; 3.29 poblacional en el código) | 1 |
| Vectores | p_s | proporción objetivo por especie | 10 |
| Vectores | c_s | costo unitario por especie | 10 |
| Vectores | D_s | demanda por ha = p_s * 658 | 10 |
| Vectores | X_s | existentes esperados por especie (X_esperado.inc) | 10 |
| Matrices | w_st | competencia ponderada 10x10 (simétrica: 55 valores distintos) | 100 |

| Tipo | Parámetros | Valores numéricos |
|---|---|---|
| Escalares | 5 | 5 |
| Vectores | 4 | 40 |
| Matrices | 1 | 100 |
| **Total** | 10 | 145 |

### Nota sobre la Normal

`distribuciones.ajustar_densidad` ajusta la Normal por máxima verosimilitud: μ = 129.18, σ = 3.2896 (desviación poblacional, ddof = 0). El valor σ = 3.35 corresponde a la desviación muestral (ddof = 1): 3.3458. El Monte Carlo del Entregable 2 usa σ = 3.29.
