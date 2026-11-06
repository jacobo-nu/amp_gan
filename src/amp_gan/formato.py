"""Formato numérico en castellano para los informes y notebooks.

Reglas aplicadas (ortografía académica española):

* coma decimal;
* separador de miles solo a partir de **cinco** cifras enteras (``2000`` pero
  ``21 754``), con espacio duro (U+00A0) para que el número no se parta en dos
  líneas; la parte decimal no se agrupa;
* un espacio duro entre el número y el signo de porcentaje (``79,6 %``).

No importa Keras ni ninguna librería pesada.
"""

from __future__ import annotations

import math

#: Espacio duro (no se parte entre líneas), usado como separador de miles y antes de «%».
NBSP = " "

_SUPERSCRIPT = str.maketrans("0123456789-+", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺")


def _agrupar_miles(entera: str) -> str:
    """Agrupa de tres en tres desde la derecha si hay al menos cinco cifras."""
    if len(entera) < 5:
        return entera
    grupos = []
    while entera:
        grupos.append(entera[-3:])
        entera = entera[:-3]
    return NBSP.join(reversed(grupos))


def _no_finito(x: float) -> bool:
    return math.isnan(x) or math.isinf(x)


def num(x: float, d: int = 2, sign: bool = False) -> str:
    """Número con ``d`` decimales, coma decimal y separador de miles correcto.

    ``NaN`` e infinitos dan ``"n/d"``. Con ``sign=True`` se escribe ``+`` en los
    positivos; un valor que se redondea a cero no lleva signo.
    """
    x = float(x)
    if _no_finito(x):
        return "n/d"
    redondeado = f"{abs(x):.{d}f}"
    entera, _, decimal = redondeado.partition(".")
    out = _agrupar_miles(entera) + ("," + decimal if decimal else "")
    if float(redondeado) != 0:
        out = ("-" if x < 0 else "+" if sign else "") + out
    return out


def numg(x: float, d: int = 3) -> str:
    """Número con ``d`` cifras significativas y coma decimal (útil para valores muy pequeños).

    Los exponentes negativos se escriben como ``1,5·10⁻⁷`` en lugar de ``1.5e-07``;
    las magnitudes grandes se escriben enteras (``123 000``).
    """
    x = float(x)
    if _no_finito(x):
        return "n/d"
    texto = f"{x:.{d}g}"
    if "e+" in texto:  # magnitud grande con pocas cifras significativas: se escribe entera (123 000)
        return num(float(texto), 0)
    if "e-" in texto:
        mantisa, exponente = texto.split("e")
        return f"{mantisa.replace('.', ',')}·10{str(int(exponente)).translate(_SUPERSCRIPT)}"
    entera, punto, decimal = texto.lstrip("-").partition(".")
    out = _agrupar_miles(entera) + ("," + decimal if punto else "")
    return ("-" if x < 0 else "") + out


def pct(x: float, d: int = 1) -> str:
    """Fracción (0-1) como porcentaje: ``0,796`` -> ``«79,6 %»`` (con espacio duro)."""
    texto = num(100 * float(x), d)
    return texto if texto == "n/d" else texto + NBSP + "%"
