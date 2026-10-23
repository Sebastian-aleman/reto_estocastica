# Sensibilidad del recocido a L (30 semillas, 5 procesos)

Mismas semillas (y mismas existentes) para todo L; Δ positivo = más competencia que con L = 2000. Resto de parámetros: α = 0.995, T_min = 1e-3, paro tras 100 niveles sin mejora en frío. Estimado = tiempo medio de recocido × 1,000 / 5 procesos.

| L | z_final media | z_final desv | Δ vs L=2000 (%) | Peor corrida vs L=2000 (%) | Niveles | Tiempo medio (s) | Tiempo desv (s) | Estimado 1,000 corridas (min) |
|---|---|---|---|---|---|---|---|---|
| 500 | 746.366 | 2.179 | +0.290 | +0.540 | 1621 | 0.102 | 0.010 | 0.3 |
| 1,000 | 745.091 | 2.154 | +0.118 | +0.481 | 1620 | 0.199 | 0.010 | 0.7 |
| 2,000 | 744.209 | 2.125 | +0.000 | +0.000 | 1621 | 0.411 | 0.029 | 1.4 |
| 4,000 | 743.633 | 1.971 | -0.077 | +0.113 | 1621 | 0.826 | 0.029 | 2.8 |
