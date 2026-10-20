"""Séance 3 — Types algébriques de données (ADT) et filtrage par motif.

Type produit  : « A ET B »  -> dataclass (un relevé a une date ET une station ET ...)
Type somme    : « A OU B »  -> union de dataclasses (une mesure est valide OU manquante OU aberrante)

Principe : « make illegal states unrepresentable » (Yaron Minsky).
Le `match` + `assert_never` laisse le vérificateur de types (mypy --strict)
signaler tout cas oublié.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import assert_never
from unittest import case

# ---------------------------------------------------------------------------
# Le type somme `Measurement` (fourni)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Valid:
    value: float


@dataclass(frozen=True)
class Missing:
    pass


@dataclass(frozen=True)
class Outlier:
    raw: str
    reason: str


type Measurement = Valid | Missing | Outlier

MISSING_VALUES = frozenset({"", "NA", "-", "null"})

# Bornes physiquement plausibles par variable (fourni)
BOUNDS: dict[str, tuple[float, float]] = {
    "temperature_c": (-40.0, 50.0),
    "humidity_pct": (0.0, 100.0),
    "pressure_hpa": (900.0, 1080.0),
    "wind_kmh": (0.0, 250.0),
    "rain_mm": (0.0, 300.0),
}


@dataclass(frozen=True)
class TypedReading:
    """Type produit dont les champs sont des types sommes."""

    date: str
    station: str
    temperature_c: Measurement
    humidity_pct: Measurement
    pressure_hpa: Measurement
    wind_kmh: Measurement
    rain_mm: Measurement


# ---------------------------------------------------------------------------
# Exercices
# ---------------------------------------------------------------------------


def parse_measurement(raw: str, lower: float, upper: float) -> Measurement:
    """Exercice 3.1 — transforme une chaîne brute en Measurement. Ne lève JAMAIS d'exception.

    - chaîne (strippée) dans MISSING_VALUES             -> Missing()
    - non convertible en float                          -> Outlier(raw, "non numérique")
    - hors [lower, upper]                               -> Outlier(raw, "hors bornes")
    - sinon                                             -> Valid(value)
    """
    cleaned = raw.strip()
    if cleaned in MISSING_VALUES:
        return Missing()
    try:
        value = float(cleaned)
    except ValueError:
        return Outlier(raw, "non numérique")
    if lower <= value <= upper:
        return Valid(value)
    return Outlier(raw, "hors bornes")


def value_or(m: Measurement, default: float) -> float:
    """Exercice 3.2 — la valeur si Valid, sinon `default`. Utiliser `match`."""
    match m:
        case Valid(value) :
            return value
        case _:
            return default


def describe(m: Measurement) -> str:
    """Exercice 3.3 — description lisible, `match` exhaustif terminé par assert_never.

    Valid(12.5)                   -> "12.5"
    Missing()                     -> "manquante"
    Outlier("999", "hors bornes") -> "aberrante (hors bornes) : '999'"
    """
    match m:
        case Valid(value) :
            return str(value)
        case Missing() :
            return "manquante"
        case Outlier(raw, reason) :
            return f"aberrante ({reason}) : {raw!r}"
        case _:
            assert_never(m)


def parse_reading(date: str, station: str, fields: dict[str, str]) -> TypedReading:
    """Exercice 3.4 — construit un TypedReading à partir des champs bruts.

    `fields` associe chaque nom de BOUNDS à sa chaîne brute.
    """
    return TypedReading(
        date=date,
        station=station,
        temperature_c=parse_measurement(fields["temperature_c"], *BOUNDS["temperature_c"]),
        humidity_pct=parse_measurement(fields["humidity_pct"], *BOUNDS["humidity_pct"]),
        pressure_hpa=parse_measurement(fields["pressure_hpa"], *BOUNDS["pressure_hpa"]),
        wind_kmh=parse_measurement(fields["wind_kmh"], *BOUNDS["wind_kmh"]),
        rain_mm=parse_measurement(fields["rain_mm"], *BOUNDS["rain_mm"]),
    )


def measurements(r: TypedReading) -> tuple[Measurement, ...]:
    """Fourni : les 5 mesures d'un relevé, dans l'ordre de BOUNDS."""
    return (r.temperature_c, r.humidity_pct, r.pressure_hpa, r.wind_kmh, r.rain_mm)


def is_complete(r: TypedReading) -> bool:
    """Exercice 3.5 — vrai si toutes les mesures du relevé sont Valid."""
    return all(isinstance(m, Valid) for m in measurements(r))


def count_by_kind(ms: Iterable[Measurement]) -> dict[str, int]:
    """Exercice 3.6 — {"valid": n1, "missing": n2, "outlier": n3} (toujours les 3 clés)."""
    return {
        "valid": sum(1 for m in ms if isinstance(m, Valid)),
        "missing": sum(1 for m in ms if isinstance(m, Missing)),
        "outlier": sum(1 for m in ms if isinstance(m, Outlier)),
    }

# ===========================================================================
# PARTIE B — Remplacer des valeurs magiques par un ADT (bloc 2)
# ===========================================================================
# Lisez `nimbus/s03_legacy_alerts.py` : -999 pour « pas de valeur », des
# chaînes pour le type d'alerte, None pour « rien ». Ci-dessous, le même domaine
# modélisé par un sum type. Les seuils sont ceux du legacy.

HEATWAVE_THRESHOLD = 25.0
FROST_THRESHOLD = 0.0
STORM_THRESHOLD = 28.0
FLOOD_THRESHOLD = 15.0


@dataclass(frozen=True)
class Storm:
    wind_kmh: float


@dataclass(frozen=True)
class Flood:
    rain_mm: float


@dataclass(frozen=True)
class Heatwave:
    temperature_c: float


@dataclass(frozen=True)
class Frost:
    temperature_c: float


@dataclass(frozen=True)
class NoAlert:
    pass


type Alert = Storm | Flood | Heatwave | Frost | NoAlert


def detect(r: TypedReading) -> Alert:
    """Exercice 3.7 — l'alerte d'un relevé, avec la priorité du legacy :
    Storm > Flood > Heatwave > Frost > NoAlert.

    Seules les mesures Valid comptent : une valeur Missing ou Outlier ne
    déclenche jamais d'alerte (c'est ce qui supprime les fausses alertes du legacy).
    Indice : `match (r.wind_kmh, r.rain_mm, r.temperature_c)` avec des guards
    (`case (Valid(v), _, _) if v >= STORM_THRESHOLD:`).
    """
    match (r.wind_kmh, r.rain_mm, r.temperature_c) :
        case (Valid(v), _, _) if v >= STORM_THRESHOLD:
            return Storm(wind_kmh=v)
        case (_, Valid(v), _) if v >= FLOOD_THRESHOLD:
            return Flood(rain_mm=v)
        case (_, _, Valid(v)) if v >= HEATWAVE_THRESHOLD:
            return Heatwave(temperature_c=v)
        case (_, _, Valid(v)) if v <= FROST_THRESHOLD:
            return Frost(temperature_c=v)
        case _:
            return NoAlert()


def message(date: str, station: str, a: Alert) -> str | None:
    """Renvoie le texte d'une alerte, ou None si aucune alerte n'est présente."""
    match a:
        case Storm(wind_kmh):
            return f"{date} {station} : tempête, vent {wind_kmh:.1f} km/h"
        case Flood(rain_mm):
            return f"{date} {station} : risque d'inondation, {rain_mm:.1f} mm"
        case Heatwave(temperature_c):
            return f"{date} {station} : canicule, {temperature_c:.1f} °C"
        case Frost(temperature_c):
            return f"{date} {station} : gel, {temperature_c:.1f} °C"
        case NoAlert():
            return None
        case _:
            assert_never(a)


def alert_report(lines: Iterable[list[str]]) -> list[str]:
    """Exercice 3.9 — les messages d'alerte de toutes les lignes à 7 colonnes, dans l'ordre.

    `lines` est le résultat de `split_csv` (bloc 2, partie A) : en-tête déjà retiré.
    Utilisez `parse_reading`, `detect` et `message`.
    """
    return [
        msg
        for line in lines
        if len(line) == 7
        for msg in [message(line[0], line[1], detect(parse_reading(line[0], line[1], {
            "temperature_c": line[2],
            "humidity_pct": line[3],
            "pressure_hpa": line[4],
            "wind_kmh": line[5],
            "rain_mm": line[6],
        })))]
        if msg is not None
    ]