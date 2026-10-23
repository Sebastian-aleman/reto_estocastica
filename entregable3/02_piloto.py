"""Piloto: tiempo de 1 ha y de 10 ha en paralelo, estimación para 1,000 corridas.

Si la estimación pasa de LIMITE_HORAS, prueba L más chicos en las mismas semillas
para ver si empeoran z_final más de TOLERANCIA_PCT. Guarda salidas_e3/piloto.csv.
"""
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

import sim_siembra as ss

SALIDAS = Path(__file__).parent / "salidas_e3"
N_PILOTO = 10
N_PROCESOS = 5
N_OBJETIVO = 1000
LIMITE_HORAS = 3
L_BASE = ss.PARAMETROS_RECOCIDO["L"]
L_ALTERNATIVOS = [500, 1000]
TOLERANCIA_PCT = 0.5
COLUMNAS_RESUMEN = ["x_tot", "z_existentes", "z_orden_actual", "z_final", "costo", "tiempo_recocido"]


def fila_corrida(resultado, corrida, L, modo):
    """Resultado de una corrida sin conteos ni asignación."""
    return {"modo": modo, "L": L, "corrida": corrida, **ss.fila_resultado(resultado)}


def correr_paralelo(semillas, L):
    inicio = time.perf_counter()
    with Pool(N_PROCESOS, initializer=ss.calentar) as pool:
        resultados = pool.map(partial(ss.simular_hectarea, L=L), semillas)
    return resultados, time.perf_counter() - inicio


def horas(segundos):
    return f"{segundos / 3600:.2f} h ({segundos / 60:.1f} min)"


def main():
    SALIDAS.mkdir(exist_ok=True)
    semillas = ss.generar_semillas(N_PILOTO)
    filas, tiempos_lote = [], {}

    # Calentamiento: compila (o carga de caché) el núcleo de numba fuera de la medición
    ss.calentar()

    # 1 ha secuencial
    inicio = time.perf_counter()
    resultado = ss.simular_hectarea(semillas[0])
    tiempo_1ha = time.perf_counter() - inicio
    filas.append(fila_corrida(resultado, 0, L_BASE, "1 ha secuencial"))
    print(f"1 ha (L = {L_BASE}): {tiempo_1ha:.2f} s en total, {resultado['tiempo_recocido']:.2f} s de recocido, "
          f"T0 = {resultado['T0']:.3f}, {resultado['niveles']} niveles")

    # 10 ha en paralelo
    resultados, tiempos_lote[L_BASE] = correr_paralelo(semillas, L_BASE)
    filas += [fila_corrida(r, k, L_BASE, "paralelo") for k, r in enumerate(resultados)]
    estimado = tiempos_lote[L_BASE] / N_PILOTO * N_OBJETIVO
    print(f"{N_PILOTO} ha en paralelo ({N_PROCESOS} procesos): {tiempos_lote[L_BASE]:.2f} s")
    print(f"Estimación de {N_OBJETIVO:,} corridas con {N_PROCESOS} procesos: {horas(estimado)}; "
          f"secuencial: {horas(tiempo_1ha * N_OBJETIVO)}")

    # L más chicos solo si la estimación es demasiado larga
    if estimado > LIMITE_HORAS * 3600:
        print(f"\nLa estimación supera {LIMITE_HORAS} h: se prueban L = {L_ALTERNATIVOS} en las mismas semillas")
        for L in L_ALTERNATIVOS:
            resultados, tiempos_lote[L] = correr_paralelo(semillas, L)
            filas += [fila_corrida(r, k, L, "paralelo") for k, r in enumerate(resultados)]
    else:
        print(f"\nLa estimación no supera {LIMITE_HORAS} h: no se prueban L más chicos")

    tabla = pd.DataFrame(filas)
    tabla.to_csv(SALIDAS / "piloto.csv", index=False)

    paralelo = tabla[tabla["modo"] == "paralelo"]
    resumen = paralelo.groupby("L")[COLUMNAS_RESUMEN].agg(["mean", "std"])
    print("\nRESUMEN (corridas en paralelo, media y desviación)")
    print(resumen.round(3).T.to_string())

    if len(tiempos_lote) > 1:
        z_base = paralelo[paralelo["L"] == L_BASE].set_index("corrida")["z_final"]
        filas_L = []
        for L, tiempo_lote in sorted(tiempos_lote.items()):
            z_L = paralelo[paralelo["L"] == L].set_index("corrida")["z_final"]
            empeora = (z_L.mean() - z_base.mean()) / z_base.mean() * 100
            filas_L.append({
                "L": L,
                "z_final_media": z_L.mean(),
                "z_final_desv": z_L.std(),
                "empeora_vs_L2000_%": empeora,
                "peor_corrida_%": ((z_L - z_base) / z_base * 100).max(),
                "tiempo_recocido_medio_s": paralelo.loc[paralelo["L"] == L, "tiempo_recocido"].mean(),
                "tiempo_lote_s": tiempo_lote,
                "estimado_1000_h": tiempo_lote / N_PILOTO * N_OBJETIVO / 3600,
                f"aceptable (<= {TOLERANCIA_PCT}%)": "sí" if empeora <= TOLERANCIA_PCT else "no",
            })
        print(f"\nCOMPARACIÓN DE L (mismas {N_PILOTO} semillas)")
        print(pd.DataFrame(filas_L).round(3).to_string(index=False))

    print(f"\nGuardado: {SALIDAS / 'piloto.csv'}")


if __name__ == "__main__":
    main()
