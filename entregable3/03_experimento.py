"""Experimento completo: 1,000 hectáreas simuladas con simular_hectarea.

Guarda en salidas_e3/:
  corridas.csv           una fila por corrida (sin conteos)
  conteos.npz            suma de los conteos de vecindad y recorrido (recocido y orden actual)
  ejemplo_asignacion.csv asignación final de la corrida con z_final más cercano a la mediana
  experimento.json       tiempo total, parámetros y datos de la corrida ejemplo
"""
import json
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import sim_siembra as ss

SALIDAS = Path(__file__).parent / "salidas_e3"
N_CORRIDAS = 1000
N_PROCESOS = 5
COLUMNAS_VERIFICACION = ["x_tot", "costo", "z_existentes", "z_orden_actual", "z_final"]
CONTEOS = ["conteo_vecindad_final", "conteo_vecindad_actual",
           "conteo_recorrido_final", "conteo_recorrido_actual"]


def verificar_piloto(corridas):
    """Las primeras corridas deben reproducir exactamente las del piloto (mismas semillas)."""
    piloto = pd.read_csv(SALIDAS / "piloto.csv")
    piloto = piloto[(piloto["modo"] == "paralelo") & (piloto["L"] == ss.PARAMETROS_RECOCIDO["L"])]
    piloto = piloto.set_index("corrida")[COLUMNAS_VERIFICACION]
    actuales = corridas.loc[piloto.index, COLUMNAS_VERIFICACION]
    iguales = np.isclose(actuales.to_numpy(), piloto.to_numpy(), rtol=0, atol=1e-9).all()
    assert iguales, f"las primeras {len(piloto)} corridas no coinciden con el piloto"
    return len(piloto)


def main():
    SALIDAS.mkdir(exist_ok=True)
    semillas = ss.generar_semillas(N_CORRIDAS)
    filas, asignaciones, fijos = [], [], []
    conteos = {nombre: 0 for nombre in CONTEOS}

    inicio = time.perf_counter()
    with Pool(N_PROCESOS, initializer=ss.calentar) as pool:
        for k, resultado in enumerate(pool.imap(ss.simular_hectarea, semillas, chunksize=10)):
            filas.append({"corrida": k, **ss.fila_resultado(resultado)})
            asignaciones.append(resultado["asignacion_final"])
            fijos.append(resultado["fijo"])
            for nombre in CONTEOS:
                conteos[nombre] = conteos[nombre] + resultado[nombre]
            if (k + 1) % 100 == 0:
                print(f"  {k + 1:>5,} corridas, {time.perf_counter() - inicio:6.1f} s")
    tiempo_total = time.perf_counter() - inicio

    corridas = pd.DataFrame(filas).set_index("corrida")
    n_verificadas = verificar_piloto(corridas)
    corridas.to_csv(SALIDAS / "corridas.csv")

    especies = ss.cargar_modelo()["especies"]
    np.savez(SALIDAS / "conteos.npz", especies=np.array(especies), n_corridas=N_CORRIDAS,
             **{nombre.removeprefix("conteo_"): matriz for nombre, matriz in conteos.items()})

    # Ejemplo: la corrida con z_final más cercano a la mediana
    mediana = corridas["z_final"].median()
    ejemplo = int((corridas["z_final"] - mediana).abs().idxmin())
    pd.DataFrame({
        "nodo": np.arange(1, len(asignaciones[ejemplo]) + 1),
        "especie": np.array(especies)[asignaciones[ejemplo]],
        "existente": fijos[ejemplo],
    }).to_csv(SALIDAS / "ejemplo_asignacion.csv", index=False)

    meta = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "n_corridas": N_CORRIDAS,
        "n_procesos": N_PROCESOS,
        "semilla_maestra": ss.SEMILLA_MAESTRA,
        "parametros_recocido": ss.PARAMETROS_RECOCIDO,
        "tiempo_total_s": tiempo_total,
        "corridas_verificadas_con_piloto": n_verificadas,
        "ejemplo": {"corrida": ejemplo, "z_final": corridas.loc[ejemplo, "z_final"],
                    "mediana_z_final": mediana, "x_tot": int(corridas.loc[ejemplo, "x_tot"])},
    }
    (SALIDAS / "experimento.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")

    print(f"\n{N_CORRIDAS:,} corridas en {tiempo_total:.1f} s ({tiempo_total / 60:.2f} min) "
          f"con {N_PROCESOS} procesos")
    print(f"Las primeras {n_verificadas} corridas coinciden con el piloto")
    print(f"Ajustes de q_s necesarios: {(corridas['ajuste_q'] > 0).sum()}")
    print(f"Ejemplo: corrida {ejemplo}, z_final = {corridas.loc[ejemplo, 'z_final']:.4f} "
          f"(mediana {mediana:.4f})")


if __name__ == "__main__":
    main()
