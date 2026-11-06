"""Tests de ``amp_gan.formato``: coma decimal, separador de miles y porcentajes en castellano."""

import math

import pytest

from amp_gan.formato import NBSP, num, numg, pct


@pytest.mark.parametrize(
    ("valor", "d", "esperado"),
    [
        (0.5, 2, "0,50"),
        (3.14159, 3, "3,142"),
        (0, 0, "0"),
        (169, 0, "169"),
        (2000, 0, "2000"),  # cuatro cifras: sin separador
        (9999.4, 0, "9999"),
        (21754, 0, f"21{NBSP}754"),  # cinco cifras: separador de miles
        (1234567, 0, f"1{NBSP}234{NBSP}567"),
        (12345.678, 1, f"12{NBSP}345,7"),  # la parte decimal no se agrupa
        (-21754, 0, f"-21{NBSP}754"),
        (-0.004, 2, "0,00"),  # redondeado a cero: sin signo
    ],
)
def test_num(valor, d, esperado):
    assert num(valor, d) == esperado


def test_num_signo_explicito():
    assert num(1.5, 1, sign=True) == "+1,5"
    assert num(-1.5, 1, sign=True) == "-1,5"
    assert num(0.0, 1, sign=True) == "0,0"


def test_num_no_finitos():
    assert num(math.nan) == "n/d"
    assert num(math.inf) == "n/d"
    assert numg(math.nan) == "n/d"
    assert pct(math.nan) == "n/d"


def test_numg_cifras_significativas_y_exponente():
    assert numg(0.123456, 3) == "0,123"
    assert numg(-0.5, 3) == "-0,5"
    assert numg(1.2e-7, 3) == "1,2·10⁻⁷"
    assert numg(123456.0, 3) == f"123{NBSP}000"
    assert numg(2000.0, 3) == "2000"


def test_pct_con_espacio_duro():
    assert pct(0.796) == f"79,6{NBSP}%"
    assert pct(1.0, 0) == f"100{NBSP}%"
    assert pct(0.0, 1) == f"0,0{NBSP}%"


def test_sin_punto_decimal_en_ninguna_salida():
    for texto in (num(1234.5678, 3), numg(0.00123, 2), pct(0.1234)):
        assert "." not in texto
