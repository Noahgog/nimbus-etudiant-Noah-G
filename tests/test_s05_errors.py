from nimbus.s03_algebraic_types import Outlier
from nimbus.s05_errors import (
    Failures, Err, Ok, Success, load, combine, attempt, traverse, validate, validate_line,
)


def inverse(x: float):
    return Err("division par zéro") if x == 0 else Ok(1 / x)


def test_functor_and_monad():
    assert Ok(2).map(lambda x: x + 1) == Ok(3)
    assert Err("e").map(lambda x: x + 1) == Err("e")
    assert Ok(4).flat_map(inverse) == Ok(0.25)
    assert Ok(0).flat_map(inverse).map(lambda x: x * 2) == Err("division par zéro")


def test_functor_laws():
    for r in (Ok(3), Err("x")):
        assert r.map(lambda x: x) == r
        f, g = (lambda x: x + 1), (lambda x: x * 2)
        assert r.map(f).map(g) == r.map(lambda x: g(f(x)))


def test_attempt():
    assert attempt(lambda: int("42")) == Ok(42)
    r = attempt(lambda: int("quarante"))
    assert isinstance(r, Err) and r.error.startswith("ValueError")


def test_traverse_stops_at_first_failure():
    assert traverse([1, 2, 4], inverse) == Ok([1.0, 0.5, 0.25])
    assert traverse([1, 0, 0], inverse) == Err("division par zéro")


def test_combine_accumulates():
    assert combine(Success(1), Success("a")) == Success((1, "a"))
    assert combine(validate(False, 1, "e1"), Success(2), validate(False, 3, "e2")) == Failures(("e1", "e2"))


def test_validate_line():
    ok = validate_line("2025-01-01,BREST,10,80,1010,5,0".split(","))
    assert isinstance(ok, Success) and ok.value.station == "BREST"
    # une mesure aberrante ne fait PAS échouer la ligne
    ab = validate_line("2025-01-01,BREST,999,80,1010,5,0".split(","))
    assert isinstance(ab, Success) and isinstance(ab.value.temperature_c, Outlier)
    # deux erreurs structurelles sont TOUTES les deux signalées
    ko = validate_line("2025/01/01,ATLANTIS,10,80,1010,5,0".split(","))
    assert isinstance(ko, Failures) and len(ko.errors) == 2
    assert validate_line(["a", "b"]) == Failures(("attendu 7 colonnes, reçu 2",))


def test_load(csv_full):
    ch = load(csv_full)
    assert len(ch.readings) + len(ch.rejected) == 3 * 365
    assert len(ch.rejected) > 0
    assert all(n >= 2 for n, _ in ch.rejected)


# ---------------------------------------------------------------------------
# Partie B (bloc 3) — remplacer l'import à exceptions
# ---------------------------------------------------------------------------

import contextlib  # noqa: E402
import io  # noqa: E402
import tracemalloc  # noqa: E402

from nimbus.datagen import generate_massive_lines  # noqa: E402
from nimbus.s03_algebraic_types import Missing, Valid  # noqa: E402
from nimbus.s05_errors import error_profile, measurement_quality, rejection_lines, validate_stream  # noqa: E402
from nimbus.s05_legacy_import import import_csv  # noqa: E402

TWO_ERRORS = ("date,station,temperature_c,humidity_pct,pressure_hpa,wind_kmh,rain_mm\n"
              "2025-01-01,BREST,10,80,1010,5,0\n"
              "2025/01/02,ATLANTIS,10,80,1010,5,0\n"
              "2025-01-03,LYON,NA,80,1010,5,0\n"
              "2025-01-04,LYON,4\n")


def legacy(text):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        readings = import_csv(text)
    return readings, out.getvalue().splitlines()


def test_rejection_lines_donne_toutes_les_raisons():
    assert rejection_lines(load(TWO_ERRORS)) == [
        "ligne 3 : date invalide '2025/01/02' ; station inconnue 'ATLANTIS'",
        "ligne 5 : attendu 7 colonnes, reçu 3",
    ]
    _, printed = legacy(TWO_ERRORS)
    assert printed == ["ligne 3 ignorée", "ligne 4 ignorée", "ligne 5 ignorée"]   # sans raison


def test_le_legacy_perd_des_lignes_et_garde_des_aberrations(csv_full):
    old, _ = legacy(csv_full)
    new = load(csv_full)
    assert len(new.readings) > len(old)                       # une mesure illisible ne coûte plus la ligne
    assert any(t > 50 for (_, _, t, *_) in old)               # 999.9 °C accepté par le legacy
    kept = {(r.date, r.station) for r in new.readings}
    assert {(d, s) for (d, s, *_) in old} <= kept             # rien de ce que gardait le legacy n'est perdu


def test_measurement_quality(csv_full):
    q = measurement_quality(load(csv_full))
    assert set(q) == {"temperature_c", "humidity_pct", "pressure_hpa", "wind_kmh", "rain_mm"}
    assert all(set(v) == {"valid", "missing", "outlier"} for v in q.values())
    n = len(load(csv_full).readings)
    assert all(sum(v.values()) == n for v in q.values())
    assert q["temperature_c"]["outlier"] > 0 and q["humidity_pct"]["outlier"] > 0
    assert measurement_quality(load(TWO_ERRORS))["temperature_c"] == {"valid": 1, "missing": 1, "outlier": 0}


# ---------------------------------------------------------------------------
# Partie C (bloc 3) — profil d'erreurs sur un flux
# ---------------------------------------------------------------------------

N_STATIONS, N_DAYS = 20, 3650
KNOWN = {f"ST{k:04d}" for k in range(1, N_STATIONS + 1)}


def test_validate_stream_est_lazy_et_numerote():
    def lines():
        yield "header"
        yield "2016-01-01,ST0001,1,50,1000,3,0"
        yield ""
        yield "2016-01-02,ST0001,1,50"
        raise AssertionError("trop loin")

    it = validate_stream(lines(), KNOWN)
    n, v = next(it)
    assert n == 2 and isinstance(v, Success)
    n, v = next(it)
    assert n == 4 and v == Failures(("attendu 7 colonnes, reçu 4",))


def test_error_profile_en_memoire_constante():
    tracemalloc.start()
    try:
        profile = error_profile(generate_massive_lines(5, N_STATIONS, N_DAYS), KNOWN)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert set(profile) == {"lignes valides", "date invalide", "station inconnue", "attendu 7 colonnes"}
    assert sum(profile.values()) >= N_STATIONS * N_DAYS
    assert profile["lignes valides"] > 0.85 * N_STATIONS * N_DAYS
    assert peak < 4 * 1024 * 1024, f"pic mémoire {peak / 1e6:.0f} Mo"
