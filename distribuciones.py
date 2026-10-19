"""Ajuste de distribuciones: plantas por ha, ocurrencia por especie y supervivencia."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln

N_NODOS_HA = 658
VAR_MIN_SUPERVIVENCIA = 0.0005


# ---------------------------------------------------------------- A) Plantas por ha

def calcular_plantas_por_ha(plantas, areas):
    return (plantas.sum() / areas).rename("plantas_por_ha")


def chi2_agrupada(datos_enteros, dist, n_param, n_clases=5):
    """Chi-cuadrada con clases de probabilidad ~igual según la distribución ajustada."""
    cortes = np.unique(dist.ppf(np.linspace(0, 1, n_clases + 1)[1:-1]))
    observados = np.histogram(datos_enteros, bins=np.concatenate([[-np.inf], cortes + 0.5, [np.inf]]))[0]
    esperados = len(datos_enteros) * np.diff(np.concatenate([[0], dist.cdf(cortes), [1]]))
    estadistico = ((observados - esperados) ** 2 / esperados).sum()
    gl = len(observados) - 1 - n_param
    return estadistico, gl, stats.chi2.sf(estadistico, gl)


def ajustar_densidad(plantas_por_ha):
    """Ajusta Normal, Poisson y Binomial(658, p) por máxima verosimilitud.

    Poisson y Binomial son discretas, así que se evalúan sobre la densidad redondeada.
    """
    x = plantas_por_ha.to_numpy()
    x_ent = np.round(x).astype(int)
    media, desv, lam = float(x.mean()), float(x.std()), float(x_ent.mean())

    candidatas = {
        "Normal": (stats.norm(media, desv), {"media": media, "desv": desv}, x),
        "Poisson": (stats.poisson(lam), {"lambda": lam}, x_ent),
        "Binomial": (stats.binom(N_NODOS_HA, lam / N_NODOS_HA),
                     {"n": N_NODOS_HA, "p": lam / N_NODOS_HA}, x_ent),
    }

    filas = []
    for nombre, (dist, parametros, datos_dist) in candidatas.items():
        k = 2 if nombre == "Normal" else 1
        log_ver = (dist.logpdf(datos_dist) if nombre == "Normal" else dist.logpmf(datos_dist)).sum()
        ks = stats.kstest(datos_dist, dist.cdf)
        fila = {
            "distribucion": nombre,
            "parametros": ", ".join(f"{p}={v:.4g}" for p, v in parametros.items()),
            "log_ver": log_ver,
            "aic": 2 * k - 2 * log_ver,
            "ks_D": ks.statistic,
            "ks_p": ks.pvalue,
        }
        if nombre != "Normal":
            fila["chi2"], fila["chi2_gl"], fila["chi2_p"] = chi2_agrupada(x_ent, dist, n_param=k)
        filas.append(fila)

    resultados = pd.DataFrame(filas).set_index("distribucion")
    mejor = resultados["aic"].idxmin()
    ajuste = {"distribucion": mejor, "parametros": candidatas[mejor][1]}
    return resultados, ajuste, {n: c[0] for n, c in candidatas.items()}


def guardar_ajuste(ajuste, ruta):
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(ajuste, f, indent=2, default=float)


def graficar_densidad(plantas_por_ha, dists, ruta):
    x = plantas_por_ha.to_numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(x, bins=10, density=True, color="lightgray", edgecolor="white", label="Observado")
    rejilla = np.arange(int(x.mean() - 40), int(x.mean() + 41))
    ax1.plot(rejilla, dists["Normal"].pdf(rejilla), label="Normal", linewidth=2)
    ax1.plot(rejilla, dists["Poisson"].pmf(rejilla), label="Poisson", linewidth=2)
    ax1.plot(rejilla, dists["Binomial"].pmf(rejilla), label="Binomial(658, p)", linewidth=2)
    ax1.set_xlabel("Plantas existentes por ha")
    ax1.set_ylabel("Densidad")
    ax1.set_title("Plantas por ha en los 30 polígonos")
    ax1.legend()

    stats.probplot(x, dist="norm", plot=ax2)
    ax2.set_title("QQ-plot contra la Normal")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- B) Ocurrencia por especie

def comparar_proporciones(plantas, prop_base):
    p_poligonos = plantas.sum(axis=1) / plantas.to_numpy().sum()
    tabla = pd.DataFrame({"p_poligonos": p_poligonos, "prop_base": prop_base})
    tabla["diferencia"] = tabla["p_poligonos"] - tabla["prop_base"]
    return tabla


def prueba_homogeneidad(plantas):
    """Chi-cuadrada de homogeneidad sobre la tabla polígonos x especies (30x10)."""
    resultado = stats.chi2_contingency(plantas.T.to_numpy())
    return resultado.statistic, resultado.dof, resultado.pvalue


def ajustar_dirichlet_multinomial(plantas):
    """Estima alpha0 por momentos y compara Dirichlet-multinomial vs multinomial.

    Para la DM con alpha = p*alpha0: Var(prop_s) = p_s(1-p_s)[rho + (1-rho)/N],
    con rho = 1/(1+alpha0). rho se despeja igualando la suma de cuadrados
    observada de las proporciones entre polígonos con su valor esperado.
    """
    conteos = plantas.T.to_numpy()          # 30 polígonos x 10 especies
    N = conteos.sum(axis=1)
    p = conteos.sum(axis=0) / N.sum()
    proporciones = conteos / N[:, None]

    suma_cuadrados = ((proporciones - p) ** 2).sum()
    A = (p * (1 - p)).sum()
    H = (1 / N).sum()
    rho = (suma_cuadrados / A - H) / (len(N) - H)
    alpha0 = 1 / rho - 1 if rho > 0 else np.inf

    log_ver_mult = stats.multinomial.logpmf(conteos, N, p).sum()
    if np.isfinite(alpha0):
        alpha = alpha0 * p
        log_ver_dm = (gammaln(N + 1) + gammaln(alpha0) - gammaln(N + alpha0)
                      + (gammaln(conteos + alpha) - gammaln(alpha) - gammaln(conteos + 1)).sum(axis=1)).sum()
    else:
        log_ver_dm = log_ver_mult

    # Factor de inflación de la varianza respecto a la multinomial en un polígono típico
    factor_sobredispersion = 1 + (N.mean() - 1) * max(rho, 0)
    aic_mult = 2 * 9 - 2 * log_ver_mult
    aic_dm = 2 * 10 - 2 * log_ver_dm
    usar_dm = factor_sobredispersion >= 1.2 and aic_dm < aic_mult - 2
    return {
        "p": pd.Series(p, index=plantas.index),
        "rho": rho,
        "alpha0": alpha0,
        "factor_sobredispersion": factor_sobredispersion,
        "N_medio": N.mean(),
        "log_ver_mult": log_ver_mult,
        "log_ver_dm": log_ver_dm,
        "aic_mult": aic_mult,
        "aic_dm": aic_dm,
        "distribucion": "Dirichlet-multinomial" if usar_dm else "Multinomial",
    }


def graficar_proporciones(tabla, ruta):
    x = np.arange(len(tabla))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, tabla["p_poligonos"], width=0.4, label="p_poligonos (30 polígonos)")
    ax.bar(x + 0.2, tabla["prop_base"], width=0.4, label="prop_base (monitoreo 2021)")
    ax.set_xticks(x, tabla.index)
    ax.set_ylabel("Proporción")
    ax.set_title("Proporción por especie: polígonos vs monitoreo")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- C) Supervivencia

def ajustar_beta_supervivencia(supervivencia, var_min=VAR_MIN_SUPERVIVENCIA):
    """Beta por momentos con las dos tasas anuales observadas de cada especie."""
    tasas = supervivencia[["sup_2022_2021", "sup_2023_2022"]]
    media = tasas.mean(axis=1)
    varianza = tasas.var(axis=1).clip(lower=var_min)
    k = media * (1 - media) / varianza - 1
    return pd.DataFrame({"media": media, "varianza": varianza, "a": media * k, "b": (1 - media) * k})


def supervivencia_markov(beta, supervivencia, anios=(1, 2, 3)):
    """Cadena vivo/muerto: P = [[s, 1-s], [0, 1]]; supervivencia a t años = (P^t)[vivo, vivo]."""
    tabla = pd.DataFrame(index=beta.index)
    for t in anios:
        tabla[f"sup_{t}"] = [np.linalg.matrix_power(np.array([[s, 1 - s], [0, 1]]), t)[0, 0]
                             for s in beta["media"]]
    # sup_acum = vivos 2023 / vivos 2021 abarca 2 transiciones anuales
    tabla["sup_acum_obs"] = supervivencia["sup_acum"]
    tabla["error_2_anios"] = tabla["sup_2"] - tabla["sup_acum_obs"]
    return tabla
