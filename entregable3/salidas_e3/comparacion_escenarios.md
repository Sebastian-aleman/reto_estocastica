# Escenario base vs costo mínimo (1,000 corridas factibles de 1,000)

Media [IC 95%] con IC = media ± 1.96·s/√n (Wald para la probabilidad). Mismas semillas: mismas existentes en ambos escenarios. Costo mínimo: min Σ c_s q_s s.a. Σ q_s = 658 − x_tot, ceil(0.9·p_s·658) ≤ x_s + q_s ≤ floor(1.1·p_s·658), q_s ≥ 0 entero (PuLP/CBC); después recocido con L = 2000.

| concepto | base (q_s = d_s − x_s) | costo mínimo (ILP) | diferencia |
|---|---|---|---|
| Costo por ha ($) | 12,437.93 [12,432.38, 12,443.48] | 12,282.93 [12,277.38, 12,288.48] | -155.00 |
| z_final | 743.98 [743.83, 744.14] | 752.68 [752.52, 752.83] | +8.69 |
| P(costo > 12,500) | 0.241 [0.214, 0.268] | 0.008 [0.002, 0.014] | -0.233 |
| Reducción de z vs orden actual (%) | 15.36 [15.28, 15.43] | 14.37 [14.29, 14.44] | -0.99 |
| q_s LE (proporción final costo mín. 0.0578 vs p_s 0.0638) | 33.90 | 29.90 | -4.00 |
| q_s SA (proporción final costo mín. 0.3009 vs p_s 0.2979) | 157.71 | 159.71 | +2.00 |
| q_s SC (proporción final costo mín. 0.0578 vs p_s 0.0638) | 33.70 | 29.70 | -4.00 |
| q_s ST (proporción final costo mín. 0.0578 vs p_s 0.0638) | 33.59 | 29.59 | -4.00 |
| q_s OC (proporción final costo mín. 0.0805 vs p_s 0.0745) | 39.57 | 43.57 | +4.00 |
| q_s OE (proporción final costo mín. 0.0623 vs p_s 0.0578) | 30.38 | 33.38 | +3.00 |
| q_s OR (proporción final costo mín. 0.1216 vs p_s 0.1109) | 58.88 | 65.88 | +7.00 |
| q_s OS (proporción final costo mín. 0.1064 vs p_s 0.0973) | 51.32 | 57.32 | +6.00 |
| q_s PL (proporción final costo mín. 0.1185 vs p_s 0.1307) | 69.00 | 61.00 | -8.00 |
| q_s YF (proporción final costo mín. 0.0365 vs p_s 0.0395) | 20.94 | 18.94 | -2.00 |
| q total | 528.98 | 528.98 | +0.00 |

Corridas infactibles: 0.
Tiempo del escenario de costo mínimo: 113.3 s (5 procesos).
Las especies de $26 (LE, SA, SC, ST, YF) empatan en costo: el ILP puede repartir entre ellas de varias formas óptimas y CBC elige una.
