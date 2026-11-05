"""Tests de las correcciones por código por encima de la salida del modelo (sin red ni modelo):

- línea de los hallazgos del revisor (reasignada a la de su cita literal);
- recorte del resumen a ``max_words`` y coma decimal;
- su integración en la evaluación (salida cruda frente a corregida).
"""

import json

import pytest

from amp_agents import eval_roles as E
from amp_agents import verifiers as V

SOURCE = "\n".join(
    [
        "import math",  # 1
        "",  # 2
        "def a(x):",  # 3
        "    return x + 1",  # 4
        "",  # 5
        "def b(x):",  # 6
        "    y = x * 2",  # 7
        "    return x + 1",  # 8
        "",  # 9
        "def c(x):",  # 10
        "    return math.sqrt(x)",  # 11
        "",  # 12
    ]
)


def finding(linea, cita, problema="p", gravedad="media"):
    return {"linea": linea, "cita": cita, "problema": problema, "gravedad": gravedad}


# -- apariciones de una cita ---------------------------------------------------------------------------------------
def test_find_quote_occurrences_lists_every_appearance_in_order():
    assert V.find_quote_occurrences(SOURCE, "return x + 1") == [(4, 4), (8, 8)]
    assert V.find_quote("\n".join(["a", "b", "a"]), "a") == (1, 1)  # find_quote conserva la primera
    assert V.find_quote(SOURCE, "return x + 1") == (4, 4)
    assert V.find_quote_occurrences(SOURCE, "no está") == [] and V.find_quote_occurrences(SOURCE, "  ") == []
    spaced = "uno  dos\nx\nuno   dos\n"
    assert V.find_quote_occurrences(spaced, "uno dos") == [(1, 1), (3, 3)]  # con los espacios colapsados


# -- corrección de línea del revisor ----------------------------------------------------------------------------------
def test_line_shifted_by_the_model_is_reassigned_to_the_line_of_the_quote():
    obj = {"hallazgos": [finding(13, "return math.sqrt(x)")]}  # línea inexistente: el modelo la contó mal
    fixed = V.correct_review_lines(obj, SOURCE)
    assert fixed["obj"]["hallazgos"][0]["linea"] == 11 and fixed["obj"]["hallazgos"][0]["linea_modelo"] == 13
    assert fixed["obj"]["hallazgos"][0]["verificable"] is True
    assert fixed["items"] == [{"item": 0, "linea_modelo": 13, "linea": 11, "estado": "corregida", "apariciones": 1}]
    assert (fixed["verifiable"], fixed["corrected"], fixed["unverifiable"]) == (1, 1, 0)
    shifted_down = V.correct_review_lines({"hallazgos": [finding(2, "return math.sqrt(x)")]}, SOURCE)
    assert shifted_down["obj"]["hallazgos"][0]["linea"] == 11


def test_a_line_already_correct_is_left_alone():
    fixed = V.correct_review_lines({"hallazgos": [finding(11, "return math.sqrt(x)")]}, SOURCE)
    assert fixed["obj"]["hallazgos"][0]["linea"] == 11 and "linea_modelo" not in fixed["obj"]["hallazgos"][0]
    assert fixed["items"][0]["estado"] == "igual" and fixed["corrected"] == 0


def test_a_quote_that_does_not_appear_is_unverifiable_and_gets_no_invented_line():
    obj = {"hallazgos": [finding(5, "return x - 1"), finding(4, ""), {"linea": 3, "problema": "sin cita"}]}
    fixed = V.correct_review_lines(obj, SOURCE)
    first = fixed["obj"]["hallazgos"][0]
    assert first["verificable"] is False and first["linea"] == 5 and "linea_modelo" not in first  # conserva la del modelo
    assert [i["estado"] for i in fixed["items"]] == ["no_verificable"] * 3
    assert fixed["unverifiable"] == 3 and fixed["verifiable"] == 0
    assert V.verifiable_findings(fixed["obj"]) == {"hallazgos": []}


def test_a_repeated_quote_takes_the_occurrence_nearest_to_the_models_line():
    # «return x + 1» está en las líneas 4 y 8
    pick = lambda line: V.correct_review_lines({"hallazgos": [finding(line, "return x + 1")]}, SOURCE)["obj"]["hallazgos"][0]["linea"]
    assert pick(5) == 4 and pick(7) == 8 and pick(1) == 4 and pick(11) == 8
    assert pick(6) == 4  # a igual distancia (2 y 2), la primera
    assert pick(8) == 8 and pick(4) == 4  # ya correcta
    fixed = V.correct_review_lines({"hallazgos": [finding(7, "return x + 1")]}, SOURCE)
    assert fixed["items"][0]["apariciones"] == 2


def test_a_non_integer_model_line_uses_the_first_occurrence_and_out_of_range_ones_the_nearest():
    for bad in ("x", None, True, 4.5):
        fixed = V.correct_review_lines({"hallazgos": [finding(bad, "return x + 1")]}, SOURCE)
        assert fixed["obj"]["hallazgos"][0]["linea"] == 4 and fixed["obj"]["hallazgos"][0]["verificable"] is True
    pick = lambda line: V.correct_review_lines({"hallazgos": [finding(line, "return x + 1")]}, SOURCE)["obj"]["hallazgos"][0]["linea"]
    assert pick(999) == 8 and pick(0) == 4 and pick(-3) == 4  # fuera del fichero: la aparición más cercana


def test_a_multiline_quote_is_kept_when_the_model_line_falls_inside_it():
    cita = "def b(x):\n    y = x * 2"  # líneas 6 y 7
    assert V.correct_review_lines({"hallazgos": [finding(7, cita)]}, SOURCE)["obj"]["hallazgos"][0]["linea"] == 7
    assert V.correct_review_lines({"hallazgos": [finding(9, cita)]}, SOURCE)["obj"]["hallazgos"][0]["linea"] == 6


def test_correction_does_not_modify_the_input_and_tolerates_odd_objects():
    obj = {"hallazgos": [finding(13, "return math.sqrt(x)")], "otra": 1}
    snapshot = json.dumps(obj, sort_keys=True)
    fixed = V.correct_review_lines(obj, SOURCE)
    assert json.dumps(obj, sort_keys=True) == snapshot and fixed["obj"]["otra"] == 1
    for odd in (None, [], "texto", {}, {"hallazgos": "x"}, {"hallazgos": [3, None]}):
        result = V.correct_review_lines(odd, SOURCE)
        assert result["items"] == [] and result["unverifiable"] == 0


def test_verify_code_reviewer_judges_the_raw_output_and_reports_the_correction():
    obj = {"hallazgos": [finding(13, "return math.sqrt(x)"), finding(5, "return x - 1")]}
    result = V.verify("code_reviewer", json.dumps(obj), source=SOURCE)
    assert not result.valid  # la validez sigue siendo la de la salida cruda (línea inexistente y cita inventada)
    correction = result.details["line_correction"]
    assert correction["corrected"] == 1 and correction["unverifiable"] == 1
    assert [c["valid"] for c in result.details["citations"]] == [False, False]
    only_good = V.verify("code_reviewer", json.dumps({"hallazgos": [finding(11, "return math.sqrt(x)")]}), source=SOURCE)
    assert only_good.valid and only_good.details["line_correction"]["corrected"] == 0
    assert "line_correction" not in V.verify("code_reviewer", json.dumps(obj)).details  # sin fichero no se corrige


def test_corrected_lines_recover_seeds_that_the_raw_lines_miss():
    seeds = [{"id": "a", "line": 4}, {"id": "b", "line": 11}]
    obj = {"hallazgos": [finding(20, "return math.sqrt(x)"), finding(30, "return x + 1"), finding(7, "texto inventado")]}
    raw = E.match_code_findings(obj, seeds)
    assert raw["recovered"] == [] and raw["false_positives"] == [0, 1, 2]
    fixed = V.correct_review_lines(obj, SOURCE)
    match = E.match_code_findings(V.verifiable_findings(fixed["obj"]), seeds)
    # la cita única cae en 11 (acierta «b»); «return x + 1» está en 4 y 8 y la más cercana a 30 es la 8 (no es «a»);
    # la cita inventada no se puede verificar y no cuenta
    assert match["recovered"] == ["b"] and match["n_findings"] == 2 and match["false_positives"] == [1]


# -- integración en la evaluación del revisor ------------------------------------------------------------------------------
def test_code_summary_reports_raw_and_corrected_metrics_with_the_same_thresholds():
    def run(corrected, verifiable, unverifiable, n):
        details = {"citations": [{"valid": True}] * n, "line_correction": {"corrected": corrected, "verifiable": verifiable, "unverifiable": unverifiable, "items": []}}
        return E.RunRecord("code_reviewer", "m", "seeded", 1, True, "final", 1, None, None, 1.0, details=details)

    runs = [run(2, 3, 1, 4), run(1, 2, 0, 2)]
    per_rep = [
        {"rep": 1, "seeded": {"recovered": 1, "findings": 4, "valid_output": True, "recovered_corrected": 3, "findings_verifiable": 3}, "clean": {"recovered": 0, "findings": 1, "valid_output": True, "findings_verifiable": 1}},
        {"rep": 2, "seeded": {"recovered": 0, "findings": 2, "valid_output": True, "recovered_corrected": 4, "findings_verifiable": 2}, "clean": {"recovered": 0, "findings": 0, "valid_output": True, "findings_verifiable": 0}},
    ]
    summary = E._code_summary(runs, per_rep)
    rows = {name: (threshold, measured, ok) for name, threshold, measured, ok in summary["criteria"]}
    raw_name = "Defectos sembrados recuperados (media por repetición, de 5)"
    fixed_name = "[línea corregida por código] " + raw_name
    assert rows[raw_name][0] == rows[fixed_name][0] == ">= 3"
    assert rows[raw_name][2] is False and rows[fixed_name][2] is True  # 0,5 de media frente a 3,5
    assert rows[raw_name][1].startswith("0.50") and rows[fixed_name][1].startswith("3.50") and "3 líneas corregidas" in rows[fixed_name][1] and "1 no verificables" in rows[fixed_name][1]
    assert summary["metrics"]["recovered_raw_mean"] == 0.5 and summary["metrics"]["recovered_corrected_mean"] == 3.5
    assert summary["metrics"]["lines_corrected"] == 3 and summary["metrics"]["findings_unverifiable"] == 1 and summary["metrics"]["findings_verifiable"] == 5
    assert len(summary["criteria"]) == 6 and not summary["criteria"][0][0].startswith("[")  # los tres criterios originales, intactos


# -- recorte del resumen ----------------------------------------------------------------------------------------------------
def words(n):
    return " ".join(f"p{chr(97 + i % 26)}" for i in range(n))


def test_a_summary_that_fits_is_returned_untouched():
    text = "Primera frase corta. Segunda frase.\n\nTercera."
    assert V.trim_summary(text, 150) == text and V.trim_summary_with_mode(text, 150) == (text, "intacto")
    assert V.trim_summary_with_mode(text, V.count_words(text))[1] == "intacto"
    assert V.trim_summary("", 150) == ""


def test_trim_cuts_at_the_last_sentence_end_that_fits():
    text = "Uno dos tres cuatro. Cinco seis siete. Ocho nueve diez once doce."
    trimmed, mode = V.trim_summary_with_mode(text, 8)
    assert trimmed == "Uno dos tres cuatro. Cinco seis siete." and mode == "frase"
    assert V.trim_summary(text, 4) == "Uno dos tres cuatro."
    assert V.trim_summary(text, 6) == "Uno dos tres cuatro."  # la segunda frase llega justo a 7 palabras
    assert V.trim_summary(text, 7) == "Uno dos tres cuatro. Cinco seis siete."
    assert V.trim_summary_with_mode(text, 3) == ("Uno dos tres", "palabras")  # ninguna frase cabe


def test_trim_respects_the_limit_in_words_and_keeps_paragraphs():
    first = "Primer párrafo con " + words(30) + "."
    second = "Segundo párrafo con " + words(30) + "."
    third = "Tercero con " + words(60) + "."
    text = f"{first}\n\n{second}\n\n{third}"
    for limit in (10, 40, 70, 75, 100, 130):
        trimmed = V.trim_summary(text, limit)
        assert V.count_words(trimmed) <= limit, limit
        assert text.startswith(trimmed)
    trimmed = V.trim_summary(text, 75)
    assert trimmed == f"{first}\n\n{second}" and "\n\n" in trimmed and trimmed.endswith(".")


def test_trim_does_not_split_figures_or_stop_at_decimal_points_or_abbreviations():
    # un decimal y un punto de abreviatura no son finales de frase; el recorte por palabras no parte «21 754» ni «0,43»
    text = "El valor fue 0.9 en la época 3 según p. ej. el fichero run_1.5.csv y siguió así sin parar nunca más."
    trimmed, mode = V.trim_summary_with_mode(text, 8)
    assert mode == "palabras" and V.count_words(trimmed) <= 8 and trimmed.endswith("3")
    figures = "uno dos tres cuatro 21 754 cinco seis"
    assert V.trim_summary(figures, 5) == "uno dos tres cuatro"  # no deja «21» sin sus «754»
    assert V.trim_summary("uno dos tres 37,5 % cuatro cinco", 4) == "uno dos tres 37,5 %"
    assert V.trim_summary("uno dos tres 0,43 cuatro cinco seis", 4) == "uno dos tres 0,43"
    assert V.trim_summary("Valor 0,43. Otro valor 0,9 y más cosas sin acabar", 4) == "Valor 0,43."  # el final «0,43.» sí es final de frase
    assert V.trim_summary("1. Uno dos tres. 2. Cuatro cinco seis siete.", 5) == "1. Uno dos tres."  # «2.» es un marcador de lista


def test_trim_cuts_by_words_when_no_sentence_fits_and_strips_loose_separators():
    text = "Esta frase es larguísima " + words(20) + "."
    trimmed, mode = V.trim_summary_with_mode(text, 6)
    assert mode == "palabras" and V.count_words(trimmed) == 6 and not trimmed.endswith((",", ";", ":"))
    assert V.trim_summary("uno dos tres, cuatro cinco", 3) == "uno dos tres"
    with pytest.raises(ValueError):
        V.trim_summary("a b", 0)


@pytest.mark.parametrize("limit", [1, 2, 5, 17, 40, 80])
def test_trim_never_exceeds_the_limit_whatever_the_text(limit):
    text = "Un resumen. Con 0,9 de media, 21 754 filas y un 37,5 % (p. ej. esto). " * 12 + "Final sin punto"
    assert V.count_words(V.trim_summary(text, limit)) <= limit


# -- coma decimal ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("La pérdida fue 0.9 al final.", "La pérdida fue 0,9 al final."),
        ("Cayó de 0.4321 a 0.05.", "Cayó de 0,4321 a 0,05."),
        ("Valor -0.5 y (0.25) y 12.75%", "Valor -0,5 y (0,25) y 12,75%"),
        ("Un 37.5 % de las muestras.", "Un 37,5 % de las muestras."),
        ("Terminó en 0.9.", "Terminó en 0,9."),
        ("Entre 0.1 y 0.9, ambos incluidos", "Entre 0,1 y 0,9, ambos incluidos"),
        ("Ya hay coma: 0,9 y 1,5.", "Ya hay coma: 0,9 y 1,5."),
        ("Sin cifras aquí.", "Sin cifras aquí."),
        ("", ""),
    ],
)
def test_decimal_comma_converts_isolated_numbers(text, expected):
    assert V.decimal_comma(text) == expected
    assert V.decimal_comma(expected) == expected  # idempotente


@pytest.mark.parametrize(
    "text",
    [
        "Ver src/amp_gan/properties.py y src/a/1.5.py.",
        "El fichero cgan_hydro_metrics.csv y run_0.9.csv.",
        "El modelo v1.2 y la versión 1.2.3 del 2026.11.05.",
        "Usa `valor = 0.9` tal cual y `x.0.5`.",
        "Más en https://ejemplo.org/a/0.9/b y el id model-0.9.",
        "El identificador x0.5 y épocas_0.5 no cambian.",
        "Un millar: 21.754 filas y 1.234 más.",  # ambiguo (podría ser 21754): se deja
        "Rutas C:\\datos\\0.9\\a.csv",
    ],
)
def test_decimal_comma_leaves_paths_names_identifiers_and_ambiguous_numbers(text):
    assert V.decimal_comma(text) == text


def test_ambiguous_three_digit_numbers_are_resolved_only_by_the_source():
    text = "La carga fue 2.609 y el tiempo 21.754 segundos."
    assert V.decimal_comma(text) == text  # sin la entrada no se puede saber
    # 2.609 es el redondeo de 2.6094 (solo la lectura decimal está respaldada); 21.754 sale como millar (21754)
    assert V.decimal_comma(text, "carga,2.6094\nfilas 21754") == "La carga fue 2,609 y el tiempo 21.754 segundos."
    # si la entrada respalda 21.7544 como decimal, también se cambia
    assert V.decimal_comma(text, "carga,2.6094\ntiempo 21.7544") == "La carga fue 2,609 y el tiempo 21,754 segundos."
    # respaldado como decimal y como millar a la vez, o por ninguna lectura: se deja
    assert V.decimal_comma(text, "carga,2.6094\nfilas 21.754 de 21754") == "La carga fue 2,609 y el tiempo 21.754 segundos."
    assert V.decimal_comma("Dato 5.123.", "nada") == "Dato 5.123."
    assert V.decimal_comma("Dato 5.123.", "x=5.1234") == "Dato 5,123."  # el punto final de frase no estorba
    # y el veredicto de las cifras no cambia con la conversión
    source = "carga,2.6094"
    assert V.check_numbers_supported("carga 2.609", source)["missing"] == [] == V.check_numbers_supported(V.decimal_comma("carga 2.609", source), source)["missing"]


def test_a_dash_between_digits_is_a_range_but_after_a_letter_it_is_part_of_a_name():
    assert V.decimal_comma("entre 19,3-20.3 y") == "entre 19,3-20,3 y"
    assert V.decimal_comma("el id run-0.9 y model_a-1.5") == "el id run-0.9 y model_a-1.5"


def test_decimal_comma_only_touches_the_number_inside_a_mixed_text():
    text = "En `src/a/0.9.py` la pérdida fue 0.9 (fichero run_0.9.csv)."
    assert V.decimal_comma(text) == "En `src/a/0.9.py` la pérdida fue 0,9 (fichero run_0.9.csv)."
    assert V.decimal_comma("Valor 0.375 y 3.14 y 12.3456") == "Valor 0,375 y 3,14 y 12,3456"  # empieza por 0 o no son tres cifras exactas
    assert V.decimal_comma("a 2.500 b y 3.141") == "a 2.500 b y 3.141"  # exactamente tres cifras tras el punto: podría ser un millar


# -- las cifras se siguen comparando bien tras la coma ---------------------------------------------------------------------
@pytest.mark.parametrize(
    ("summary", "source"),
    [
        ("La pérdida fue 0.43 al final y el 37.5 % se acertó.", "epoch,loss,acc\n200,0.4321,0.375\n"),
        ("Hubo 2000 secuencias y una media de 9.7964.", "n,2000,2000,1,0.0115\nlength_mean=9.7963982"),
        ("El valor 1.2e-05 y el 0.0115.", "lr=1.2e-05\nstd,0.0115"),
        ("Pasó de 0.9 a 0.05 en 3 épocas.", "a=0.9\nb=0.05\nepocas=3"),
    ],
)
def test_check_numbers_supported_gives_the_same_verdict_before_and_after_the_comma(summary, source):
    before = V.check_numbers_supported(summary, source)
    after = V.check_numbers_supported(V.decimal_comma(summary), source)
    assert before["missing"] == [] and after["missing"] == []
    assert (before["total"], before["supported"]) == (after["total"], after["supported"])
    assert V.decimal_comma(summary) != summary  # y de verdad se cambió algo


def test_the_comma_does_not_hide_an_unsupported_figure_nor_invent_one():
    source = "loss=0.4321"
    for wrong in ("La pérdida fue 0.5.", "La pérdida fue 0,5."):
        assert V.check_numbers_supported(V.decimal_comma(wrong), source)["missing"] == ["0,5"]
    # una lista de cifras con coma de separación sigue leyéndose como cifras sueltas
    assert V.check_numbers_supported("Las épocas 100, 200 y 300.", "e=100 e=200 e=300")["missing"] == []


def test_postprocess_summary_trims_converts_and_reports_before_and_after():
    source = "epoch,loss\n200,0.4321\n"
    body = "El experimento llegó a la época 200 con una pérdida de 0.43 en el conjunto de datos. " + " ".join(["De la y."] * 60)
    fixed, info = V.postprocess_summary(body, source, 40)
    assert "0,43" in fixed and "0.43" not in fixed
    assert info["mode"] == "frase" and info["words_before"] > 150 and info["words_after"] <= 40
    assert info["numbers_before"]["missing"] == [] and info["numbers_after"]["missing"] == []
    clean, info = V.postprocess_summary("Texto corto de 0,43.", source, 150)
    assert clean == "Texto corto de 0,43." and info["mode"] == "intacto" and "numbers_before" in info
    assert "numbers_after" not in V.postprocess_summary("x", None)[1]


def test_trimming_can_only_remove_unsupported_figures_never_add_them():
    source = "loss=0.4321"
    body = "La pérdida fue 0.43. " + "Relleno de la muestra en el sistema. " * 3 + "Además llegó a 0.99."
    fixed, info = V.postprocess_summary(body, source, 12)
    assert info["numbers_before"]["missing"] == ["0.99"] and info["numbers_after"]["missing"] == []
    assert info["numbers_after"]["total"] < info["numbers_before"]["total"]


# -- integración en la evaluación del resumidor ---------------------------------------------------------------------------------
def test_summary_corrected_metrics_records_words_numbers_and_language():
    source = "epoch,loss\n200,0.4321\n"
    long_body = "El experimento llegó a la época 200 con una pérdida de 0.43 en el conjunto de datos. " + " ".join(["Es un dato de la prueba."] * 60)
    raw = V.verify("summarizer", long_body, source=source)
    assert not raw.valid and raw.details["words"] > 150
    fixed = E.summary_corrected_metrics(long_body, source)
    assert fixed["words"] <= 150 and fixed["mode"] == "frase" and fixed["text"].startswith("El experimento llegó")
    assert fixed["numbers"]["missing"] == [] and fixed["numbers_raw"]["missing"] == [] and fixed["spanish"]["is_spanish"]
    assert V.verify("summarizer", fixed["text"], source=source).valid  # el resumen corregido ya pasa el verificador
    assert E.summary_corrected_metrics(long_body, source, max_words=20)["words"] <= 20


def test_summary_criteria_keep_the_raw_judgement_and_add_corrected_rows():
    source = "epoch,loss\n200,0.4321\n"
    body = "El experimento llegó a la época 200 con una pérdida de 0.43 en el conjunto de datos. " + " ".join(["Es un dato de la prueba."] * 60)

    def record(case):
        d = V.verify("summarizer", body, source=source).details
        details = {"case": case, "words": d["words"], "numbers": d["numbers"], "spanish": d["spanish"], "corrected": E.summary_corrected_metrics(body, source)}
        return E.RunRecord("summarizer", "m", case, 1, False, "final", 1, None, None, 1.0, details=details)

    summary = E._summarizer_summary([record("hydro"), record("length")])
    rows = {name: (threshold, measured, ok) for name, threshold, measured, ok in summary["criteria"]}
    words_name = "Palabras por resumen (todas las ejecuciones)"
    assert len(summary["criteria"]) == 6
    assert rows[words_name][2] is False and rows["[corregido por código] " + words_name][2] is True  # >150 crudo, <=150 corregido
    assert rows[words_name][0] == rows["[corregido por código] " + words_name][0] == "<= 150"
    assert rows["[corregido por código] En castellano (todas las ejecuciones)"][2] is True
    numbers_name = "Cifras del resumen presentes en la entrada (agregado)"
    assert rows[numbers_name][2] is True and rows["[corregido por código] " + numbers_name][2] is True
    corrected = summary["metrics"]["corrected"]
    assert corrected["modes"] == ["frase", "frase"] and all(w <= 150 for w in corrected["words"]) and all(w > 150 for w in summary["metrics"]["words"])
    # sin ejecuciones válidas no se da nada por cumplido
    empty = E._summarizer_summary([])
    assert not any(ok for _, _, _, ok in empty["criteria"])


def test_declared_criteria_are_untouched_by_the_corrections():
    assert E.CRITERIA["summarizer"] == {"numbers_min": 0.95, "max_words": 150}
    assert E.CRITERIA["code_reviewer"] == {"recovered_mean_min": 3.0, "valid_citations_min": 0.90, "false_positives_max": 2}


def test_new_modules_have_no_forbidden_mentions():
    from pathlib import Path

    forbidden = ["cla" + "ude", "anthro" + "pic", "asistente de " + "ia", "/ho" + "me/"]
    src = Path(__file__).resolve().parents[1]
    for file in (src / "amp_agents" / "locate.py", src / "tests" / "test_agents_corrections.py", src / "tests" / "test_agents_locate.py"):
        text = file.read_text(encoding="utf-8").lower().replace(".claude", "")
        for word in forbidden:
            assert word not in text, f"{file.name} menciona {word!r}"
