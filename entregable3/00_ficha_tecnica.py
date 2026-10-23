"""Ficha técnica del equipo donde se corren los experimentos del Entregable 3.

Detecta hardware, sistema operativo y versiones de software y guarda
ficha_tecnica.json y ficha_tecnica.md en esta carpeta. No corre simulaciones.
"""
import ctypes
import glob
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

CARPETA = Path(__file__).parent
NO_INSTALADO = "no instalado"
NO_DETECTADO = "no detectado"
PAQUETES = ["numpy", "pandas", "scipy", "networkx", "pulp", "numba", "matplotlib"]
SISTEMA = platform.system()          # "Windows", "Darwin" o "Linux"


# ---------------------------------------------------------------- Utilidades

def ejecutar(comando, tiempo_limite=20):
    """Salida estándar de un comando, o "" si falla."""
    try:
        resultado = subprocess.run(comando, capture_output=True, text=True,
                                   timeout=tiempo_limite, errors="ignore")
        return (resultado.stdout + resultado.stderr).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def powershell(instruccion):
    return ejecutar(["powershell", "-NoProfile", "-Command", instruccion])


def cim(clase, propiedad):
    """Propiedad de una clase CIM de Windows (p. ej. Win32_ComputerSystem.Model)."""
    return powershell(f"(Get-CimInstance {clase} | Select-Object -First 1).{propiedad}")


def sysctl(clave):
    return ejecutar(["sysctl", "-n", clave])


def leer_archivo(ruta):
    try:
        return Path(ruta).read_text(errors="ignore").strip()
    except OSError:
        return ""


def a_gb(n_bytes):
    return round(n_bytes / 1024 ** 3, 2)


# ---------------------------------------------------------------- Hardware y sistema

def modelo_pc():
    if SISTEMA == "Windows":
        fabricante = cim("Win32_ComputerSystem", "Manufacturer")
        modelo = cim("Win32_ComputerSystem", "Model")
        return " ".join(p for p in [fabricante, modelo] if p) or NO_DETECTADO
    if SISTEMA == "Darwin":
        texto = ejecutar(["system_profiler", "SPHardwareDataType"])
        nombre = re.search(r"Model Name:\s*(.+)", texto)
        identificador = re.search(r"Model Identifier:\s*(.+)", texto)
        partes = [m.group(1).strip() for m in [nombre, identificador] if m]
        return " ".join(partes) or NO_DETECTADO
    fabricante = leer_archivo("/sys/devices/virtual/dmi/id/sys_vendor")
    modelo = leer_archivo("/sys/devices/virtual/dmi/id/product_name")
    return " ".join(p for p in [fabricante, modelo] if p) or NO_DETECTADO


def sistema_operativo():
    if SISTEMA == "Windows":
        # platform.release() puede decir "10" en Windows 11; el nombre comercial viene de CIM
        nombre = cim("Win32_OperatingSystem", "Caption")
        version = cim("Win32_OperatingSystem", "Version")
        return {"nombre": nombre or f"Windows {platform.release()}",
                "version": version or platform.version()}
    if SISTEMA == "Darwin":
        return {"nombre": f"macOS {platform.mac_ver()[0]}", "version": platform.release()}
    os_release = leer_archivo("/etc/os-release")
    nombre = re.search(r'PRETTY_NAME="?([^"\n]+)', os_release)
    return {"nombre": nombre.group(1) if nombre else "Linux", "version": platform.release()}


def disco():
    """Capacidad del disco donde está el proyecto."""
    raiz = Path(CARPETA.anchor or "/")
    uso = shutil.disk_usage(raiz)
    return {"unidad": str(raiz), "total_gb": a_gb(uso.total), "libre_gb": a_gb(uso.free)}


def ram_total_gb():
    try:
        import psutil
        return a_gb(psutil.virtual_memory().total)
    except ImportError:
        pass
    if SISTEMA == "Windows":
        class EstadoMemoria(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        estado = EstadoMemoria()
        estado.dwLength = ctypes.sizeof(EstadoMemoria)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(estado))
        return a_gb(estado.ullTotalPhys)
    if SISTEMA == "Darwin":
        return a_gb(int(sysctl("hw.memsize") or 0))
    encontrado = re.search(r"MemTotal:\s*(\d+)", leer_archivo("/proc/meminfo"))
    return a_gb(int(encontrado.group(1)) * 1024) if encontrado else None


def procesador():
    """Nombre, frecuencia (GHz) y núcleos físicos/lógicos."""
    nombre, frecuencia_ghz, fisicos = "", None, None
    logicos = os.cpu_count()

    if SISTEMA == "Windows":
        nombre = cim("Win32_Processor", "Name")
        mhz = cim("Win32_Processor", "MaxClockSpeed")
        frecuencia_ghz = round(int(mhz) / 1000, 2) if mhz.isdigit() else None
        nucleos = cim("Win32_Processor", "NumberOfCores")
        fisicos = int(nucleos) if nucleos.isdigit() else None
    elif SISTEMA == "Darwin":
        nombre = sysctl("machdep.cpu.brand_string")
        hz = sysctl("hw.cpufrequency")        # no existe en Apple Silicon
        frecuencia_ghz = round(int(hz) / 1e9, 2) if hz.isdigit() else None
        fisicos = int(sysctl("hw.physicalcpu") or 0) or None
    else:
        cpuinfo = leer_archivo("/proc/cpuinfo")
        encontrado = re.search(r"model name\s*:\s*(.+)", cpuinfo)
        nombre = encontrado.group(1).strip() if encontrado else ""
        mhz = leer_archivo("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")   # kHz
        if mhz.isdigit():
            frecuencia_ghz = round(int(mhz) / 1e6, 2)
        else:
            encontrado = re.search(r"cpu MHz\s*:\s*([\d.]+)", cpuinfo)
            frecuencia_ghz = round(float(encontrado.group(1)) / 1000, 2) if encontrado else None
        pares = set(re.findall(r"physical id\s*:\s*(\d+)\n(?:.*\n)*?core id\s*:\s*(\d+)", cpuinfo))
        fisicos = len(pares) or None

    if fisicos is None:
        try:
            import psutil
            fisicos = psutil.cpu_count(logical=False)
        except ImportError:
            pass
    return {"nombre": nombre or platform.processor() or NO_DETECTADO,
            "frecuencia_ghz": frecuencia_ghz,
            "nucleos_fisicos": fisicos,
            "nucleos_logicos": logicos}


# ---------------------------------------------------------------- Software

def version_paquete(nombre):
    try:
        return importlib.metadata.version(nombre)
    except importlib.metadata.PackageNotFoundError:
        return NO_INSTALADO


def version_cbc():
    """CBC que usa pulp (COIN_CMD) o, si no, el que esté en el PATH."""
    ruta = None
    try:
        import pulp
        resolutor = pulp.COIN_CMD() if hasattr(pulp, "COIN_CMD") else pulp.PULP_CBC_CMD()
        if resolutor.available():
            ruta = resolutor.path
    except Exception:
        pass
    ruta = ruta or shutil.which("cbc")
    if not ruta:
        return NO_INSTALADO
    encontrado = re.search(r"Version:\s*(\S+)|CBC\s+(\S+)", ejecutar([ruta, "-quit"]))
    version = next((g for g in encontrado.groups() if g), "") if encontrado else ""
    # El CBC de pulp 4 viene en el paquete cbcbox y se reporta como "devel"
    empaquetado = version_paquete("cbcbox")
    if empaquetado != NO_INSTALADO:
        version = f"{version or 'desconocida'} (cbcbox {empaquetado})"
    return version or "instalado (versión no detectada)"


def ruta_gams():
    ruta = shutil.which("gams")
    if ruta:
        return Path(ruta)
    # GAMS no siempre se agrega al PATH; se busca en las carpetas de instalación usuales
    if SISTEMA == "Windows":
        patrones = ["C:/GAMS/*/gams.exe", "C:/GAMS/win64/*/gams.exe", "C:/Program Files/GAMS*/gams.exe"]
    elif SISTEMA == "Darwin":
        patrones = ["/Library/Frameworks/GAMS.framework/Versions/*/Resources/gams",
                    "/Applications/GAMS*/sysdir/gams"]
    else:
        patrones = ["/opt/gams/*/gams", "/opt/gams*/gams", str(Path.home() / "gams*/gams")]
    candidatos = sorted(c for patron in patrones for c in glob.glob(patron))
    return Path(candidatos[-1]) if candidatos else None


def version_gams():
    ruta = ruta_gams()
    if ruta is None:
        return NO_INSTALADO
    # gamsstmp.txt trae la versión sin tener que correr GAMS
    texto = leer_archivo(ruta.parent / "gamsstmp.txt") or ejecutar([str(ruta), "?"])
    encontrado = re.search(r"(\d+\.\d+\.\d+)", texto)
    return encontrado.group(1) if encontrado else "instalado (versión no detectada)"


def versiones_software():
    versiones = {"Python": platform.python_version()}
    versiones.update({p: version_paquete(p) for p in PAQUETES})
    versiones["CBC"] = version_cbc()
    versiones["GAMS"] = version_gams()
    return versiones


# ---------------------------------------------------------------- Ficha

def construir_ficha():
    return {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "modelo_pc": modelo_pc(),
        "sistema_operativo": sistema_operativo(),
        "disco": disco(),
        "ram_total_gb": ram_total_gb(),
        "procesador": procesador(),
        "software": versiones_software(),
        "interprete": sys.executable,
    }


def ficha_a_markdown(ficha):
    so, dd, cpu = ficha["sistema_operativo"], ficha["disco"], ficha["procesador"]
    frecuencia = f"{cpu['frecuencia_ghz']} GHz" if cpu["frecuencia_ghz"] else NO_DETECTADO
    filas_hw = [
        ("Modelo de la PC", ficha["modelo_pc"]),
        ("Sistema operativo", so["nombre"]),
        ("Versión del sistema operativo", so["version"]),
        (f"Disco duro ({dd['unidad']}) capacidad total", f"{dd['total_gb']:,.2f} GB"),
        (f"Disco duro ({dd['unidad']}) espacio libre", f"{dd['libre_gb']:,.2f} GB"),
        ("Memoria RAM total", f"{ficha['ram_total_gb']:,.2f} GB" if ficha["ram_total_gb"] else NO_DETECTADO),
        ("Procesador", cpu["nombre"]),
        ("Frecuencia del procesador (máxima)", frecuencia),
        ("Núcleos físicos", cpu["nucleos_fisicos"] or NO_DETECTADO),
        ("Núcleos lógicos (hilos)", cpu["nucleos_logicos"] or NO_DETECTADO),
    ]
    lineas = ["# Ficha técnica del equipo", "",
              f"Generada el {ficha['fecha']} con `00_ficha_tecnica.py`.", "",
              "## Hardware y sistema operativo", "",
              "| Característica | Valor |", "|---|---|"]
    lineas += [f"| {k} | {v} |" for k, v in filas_hw]
    lineas += ["", "## Software", "", "| Software | Versión |", "|---|---|"]
    lineas += [f"| {k} | {v} |" for k, v in ficha["software"].items()]
    lineas += ["", f"Intérprete: `{ficha['interprete']}`", ""]
    return "\n".join(lineas)


def main():
    ficha = construir_ficha()
    (CARPETA / "ficha_tecnica.json").write_text(
        json.dumps(ficha, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown = ficha_a_markdown(ficha)
    (CARPETA / "ficha_tecnica.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
