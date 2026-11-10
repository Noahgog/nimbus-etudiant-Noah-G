"""Séance 5 — Gestion fonctionnelle des erreurs : Option, Result, Validation.

Une exception est un `goto` invisible dans la signature. Ici l'échec devient une
VALEUR : `Result[T, E] = Ok[T] | Err[E]` (Either en Scala/Haskell).

- `map`      : transformer la valeur si succès       -> foncteur
- `flat_map` : enchaîner une opération qui peut échouer -> monade (s'arrête au 1er échec)
- `Validation` : combiner des vérifications INDÉPENDANTES en accumulant TOUTES les
  erreurs -> foncteur applicatif. Indispensable pour un rapport de qualité de données.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Iterator
from dataclasses import dataclass
from datetime import date
from typing import Any

from nimbus.datagen import STATIONS
from nimbus.s02_higher_order import COLUMNS, split_csv
from nimbus.s03_algebraic_types import BOUNDS, TypedReading, count_by_kind, parse_reading

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ok[T]:
    value: T

    def map[U](self, f: Callable[[T], U]) -> Ok[U]:
        """Exercice 5.1a"""
        return Ok(f(self.value))

    def flat_map[U, E](self, f: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Exercice 5.1b"""
        return f(self.value)

    def get_or(self, default: Any) -> T:
        return self.value

    def is_ok(self) -> bool:
        return True


@dataclass(frozen=True)
class Err[E]:
    error: E

    def map(self, f: Callable[[Any], Any]) -> Err[E]:
        """Exercice 5.1c — un échec reste un échec."""
        return self

    def flat_map(self, f: Callable[[Any], Any]) -> Err[E]:
        """Exercice 5.1d"""
        return self

    def get_or[T](self, default: T) -> T:
        return default

    def is_ok(self) -> bool:
        return False


type Result[T, E] = Ok[T] | Err[E]

def attempt[T](f: Callable[[], T]) -> Result[T, str]:
    """Exercice 5.2 — pont avec le monde à exceptions : exécute f, capture l'exception en Err(message)."""
    try:
        return Ok(f())
    except Exception as e:
        return Err(f"{type(e).__name__}: {e}")

def traverse[A, B, E](xs: Iterable[A], f: Callable[[A], Result[B, E]]) -> Result[list[B], E]:
    """Exercice 5.3 — applique f à chaque élément ; Ok(liste) si tout réussit, sinon le PREMIER Err."""
    result: list[B] = []
    for x in xs:
        match f(x):
            case Ok(value):
                result.append(value)
            case Err(error):
                return Err(error)
    return Ok(result)

# ---------------------------------------------------------------------------
# Validation (accumule les erreurs)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Success[T]:
    value: T


@dataclass(frozen=True)
class Failures:
    errors: tuple[str, ...]


type Validation[T] = Success[T] | Failures


def validate[T](condition: bool, value: T, message: str) -> Validation[T]:
    """Fourni."""
    return Success(value) if condition else Failures((message,))


def combine(*validations: Validation[Any]) -> Validation[tuple[Any, ...]]:
    """Exercice 5.4 — combine N validations INDÉPENDANTES.

    Tous Success -> Success(tuple des valeurs) ; sinon Failures(concaténation de TOUTES les erreurs).
    """
    values: list[Any] = []
    errors: list[str] = []
    for v in validations:
        match v:
            case Success(value):
                values.append(value)
            case Failures(errs):
                errors.extend(errs)
    if errors:
        return Failures(tuple(errors))
    else:
        return Success(tuple(values))

def _date_iso(s: str) -> Validation[str]:
    """Fourni (utilise `attempt`)."""
    match attempt(lambda: date.fromisoformat(s)):
        case Ok(_):
            return Success(s)
        case Err(_):
            return Failures((f"date invalide {s!r}",))
    raise AssertionError  # inatteignable

def validate_line(fields: list[str], stations: Collection[str] = STATIONS) -> Validation[TypedReading]:
    """Exercice 5.5 — valide la STRUCTURE d'une ligne et construit le TypedReading.

    Vérifications (toutes, sans s'arrêter à la première) :
      - nombre de colonnes == len(COLUMNS)         sinon "attendu 7 colonnes, reçu N"
        (si ce test échoue, inutile d'aller plus loin : renvoyer directement l'échec)
      - date au format ISO (utiliser `_date_iso`)
      - station dans `stations`                       sinon "station inconnue 'X'"
    Les mesures elles-mêmes ne font PAS échouer la ligne : elles deviennent des
    `Measurement` via `parse_reading` (séance 3).
    """
    if len(fields) != len(COLUMNS):
        return Failures((f"attendu {len(COLUMNS)} colonnes, reçu {len(fields)}",))
    date_str, station, *measurements = fields
    date_validation = _date_iso(date_str)
    station_validation = validate(station in stations, station, f"station inconnue {station!r}")
    combined = combine(date_validation, station_validation)
    match combined:
        case Success((valid_date, valid_station)):
            reading_fields = dict(zip(BOUNDS, measurements))
            return Success(parse_reading(valid_date, valid_station, reading_fields))
        case Failures(errors):
            return Failures(errors)
    raise AssertionError("résultat de validation inattendu")

@dataclass(frozen=True)
class Loaded:
    readings: tuple[TypedReading, ...]
    rejected: tuple[tuple[int, tuple[str, ...]], ...]  # (numéro de ligne dans le fichier, errors)


def load(csv_text: str) -> Loaded:
    """Exercice 5.6 — découpe (séance 2) puis valide chaque ligne et partitionne.

    Numéro de ligne : 2 pour la première ligne de données (l'en-tête est la ligne 1).
    """
    lines = split_csv(csv_text)
    readings: list[TypedReading] = []
    rejected: list[tuple[int, tuple[str, ...]]] = []
    for line_number, fields in enumerate(lines, start=2):
        validation = validate_line(fields)
        match validation:
            case Success(reading):
                readings.append(reading)
            case Failures(errors):
                rejected.append((line_number, errors))
    return Loaded(tuple(readings), tuple(rejected))

# ===========================================================================
# Partie B — remplacer un import à exceptions (bloc 3)
# ===========================================================================
# Lisez `s05_legacy_import.py` : un `try/except` qui avale tout, une ligne entière
# jetée pour une seule mesure illisible, des valeurs aberrantes acceptées, et un
# simple print sans la raison. Ici, chaque problème devient une valeur.


def rejection_lines(loaded: Loaded) -> list[str]:
    """Exercice 5.7 — une ligne de rapport par ligne rejetée, dans l'ordre du fichier :

        "ligne 12 : date invalide '2025/01/03' ; station inconnue 'ATLANTIS'"

    (les erreurs d'une même ligne séparées par " ; ").
    """
    lines: list[str] = []
    for line_number, errors in loaded.rejected:
        error_messages = " ; ".join(errors)
        lines.append(f"ligne {line_number} : {error_messages}")
    return lines

def measurement_quality(loaded: Loaded) -> dict[str, dict[str, int]]:
    """Exercice 5.8 — pour chaque colonne de mesure (clés de BOUNDS), le décompte
    {"valid": .., "missing": .., "outlier": ..} sur les relevés acceptés.

    Réutilisez `count_by_kind` (bloc 2) et `getattr(r, column)`.
    """
    return {
        column: count_by_kind([getattr(reading, column) for reading in loaded.readings])
        for column in BOUNDS
    }

# ===========================================================================
# Partie C — un profil d'erreurs sur un flux, en mémoire constante
# ===========================================================================


def error_kind(message: str) -> str:
    """Fourni : la nature d'une erreur, sans la valeur fautive.
    "date invalide '2016/01/03'" -> "date invalide" ; "attendu 7 colonnes, reçu 5" -> "attendu 7 colonnes"."""
    return message.split(" '")[0].split(",")[0]


def validate_stream(lines: Iterable[str], stations: Collection[str]) -> Iterator[tuple[int, Validation[TypedReading]]]:
    """Exercice 5.9 — (numéro de ligne, validation) pour chaque ligne de données, en lazy.

    `lines` commence par l'en-tête (ligne 1, ignorée). Une ligne vide est ignorée.
    Aucune liste intermédiaire : un generator.
    """
    for line_number, line in enumerate(lines, start=1):
        if line_number == 1 or not line.strip():
            continue
        fields = line.split(",")
        yield (line_number, validate_line(fields, stations))


def error_profile(lines: Iterable[str], stations: Collection[str]) -> dict[str, int]:
    """Exercice 5.10 — nombre d'erreurs par nature (`error_kind`) sur tout le flux, plus
    la clé "lignes valides". Une seule passe, mémoire proportionnelle au nombre de natures."""
    profile: dict[str, int] = {"lignes valides": 0}
    for line_number, line in enumerate(lines, start=1):
        if line_number == 1 or not line.strip():
            continue
        fields = line.split(",")
        validation = validate_line(fields, stations)
        match validation:
            case Success(_):
                profile["lignes valides"] += 1
            case Failures(errors):
                for error in errors:
                    kind = error_kind(error)
                    profile[kind] = profile.get(kind, 0) + 1
    return profile