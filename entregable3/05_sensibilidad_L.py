"""Sensibilidad del recocido a L (movimientos por nivel de temperatura).

Corre las primeras 30 semillas del experimento con L ∈ {500, 1000, 2000, 4000}
y guarda salidas_e3/sensibilidad_L.{csv,md} y sensibilidad_L_corridas.csv.
"""
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import sim_siembra as ss

SALIDAS = Path(__file__).parent / "salidas_e3"
N_SEMILLAS = 30
N_PROCESOS = 5
VALORES_L = [500, 1000, 2000, 4000]
L_BASE = ss.PARAMETROS_RECOCIDO["L"]
N_OBJETIVO = 1000


def main():
    SALIDAS.mkdir(exist_ok=True)
    semillas = ss.generar_semillas(N_SEMILLAS)
    filas, tiempos_lote = [], {}
    with Pool(N_PROCESOS, initializer=ss.calentar) as pool:
        for L in VALORES_L:
            inicio = time.perf_counter()
            resultados = pool.map(partial(ss.simular_hectarea, L=L), semillas)
            tiempos_lote[L] = time.perf_counter() - inicio
            filas += [{"L": L, "corrida": k, **ss.fila_resultado(r)} for k, r in enumerate(resultados)]
            print(f"L = {L}: {tiempos_lote[L]:.1f} s")

    corridas = pd.DataFrame(filas)
    corridas.to_csv(SALIDAS / "sensibilidad_L_corridas.csv", index=False)

    # Con L = L_BASE deben salir las mismas corridas que en el experimento
    ruta_experimento = SALIDAS / "corridas.csv"
    if ruta_experimento.exists():
        experimento = pd.read_csv(ruta_experimento, index_col="corrida")["z_final"].iloc[:N_SEMILLAS]
        base = corridas[corridas["L"] == L_BASE].set_index("corrida")["z_final"]
        assert np.allclose(base, experimento, rtol=0, atol=1e-9), "L base no reproduce el experimento"

    z_base = corridas[corridas["L"] == L_BASE].set_index("corrida")["z_final"]
    resumen = []
    for L, grupo in corridas.groupby("L"):
        z = grupo.set_index("corrida")["z_final"]
        cambio = (z - z_base) / z_base * 100
        resumen.append({
            "L": L,
            "z_final_media": z.mean(),
            "z_final_desv": z.std(),
            "cambio_vs_L2000_%": (z.mean() - z_base.mean()) / z_base.mean() * 100,
            "peor_corrida_vs_L2000_%": cambio.max(),
            "niveles_media": grupo["niveles"].mean(),
            "tiempo_media_s": grupo["tiempo_recocido"].mean(),
            "tiempo_desv_s": grupo["tiempo_recocido"].std(),
            # Tiempo de cómputo, sin el arranque de los procesos (que pesa en el primer lote)
            "estimado_1000_min": grupo["tiempo_recocido"].mean() * N_OBJETIVO / N_PROCESOS / 60,
        })
    resumen = pd.DataFrame(resumen)
    resumen.to_csv(SALIDAS / "sensibilidad_L.csv", index=False)

    encabezados = ["L", "z_final media", "z_final desv", "Δ vs L=2000 (%)", "Peor corrida vs L=2000 (%)",
                   "Niveles", "Tiempo medio (s)", "Tiempo desv (s)", f"Estimado {N_OBJETIVO:,} corridas (min)"]
    lineas = ["| " + " | ".join(encabezados) + " |", "|" + "---|" * len(encabezados)]
    for f in resumen.itertuples(index=False):
        lineas.append(f"| {f.L:,} | {f.z_final_media:.3f} | {f.z_final_desv:.3f} | {f[3]:+.3f} | {f[4]:+.3f} | "
                      f"{f.niveles_media:.0f} | {f.tiempo_media_s:.3f} | {f.tiempo_desv_s:.3f} | {f[8]:.1f} |")
    texto = [f"# Sensibilidad del recocido a L ({N_SEMILLAS} semillas, {N_PROCESOS} procesos)", "",
             "Mismas semillas (y mismas existentes) para todo L; Δ positivo = más competencia que con L = 2000. "
             "Resto de parámetros: α = 0.995, T_min = 1e-3, paro tras 100 niveles sin mejora en frío. "
             f"Estimado = tiempo medio de recocido × {N_OBJETIVO:,} / {N_PROCESOS} procesos.", "",
             *lineas, ""]
    (SALIDAS / "sensibilidad_L.md").write_text("\n".join(texto), encoding="utf-8")
    print("\n".join(texto))


if __name__ == "__main__":
    main()
