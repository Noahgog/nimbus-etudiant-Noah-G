r"""Bloc 2 — DÉFI ★★★ (facultatif) : un langage de règles d'alerte, avec des parser combinators.

Objectif : que l'ingénieur météo écrive ses règles en texte, par exemple

    temperature_c >= 30 and (wind_kmh > 60 or not rain_mm < 10)

et que le programme les transforme en un algebraic data type (l'AST), puis les évalue
sur des relevés.

Un parser est une FONCTION : (text, position) -> Parsed(value, new_position) | Failed(position, expected).
Les parser combinators sont des higher-order functions qui prennent des parsers et
renvoient un nouveau parser (séquence, alternative, répétition...). C'est la technique
de Parsec (Haskell), de FastParse (Scala) ou de nom (Rust).

Grammaire (du moins prioritaire au plus prioritaire) :

    expr        := term ("or" term)*              -- associatif à gauche
    term        := factor ("and" factor)*         -- associatif à gauche
    factor      := "not" factor | "(" expr ")" | comparison
    comparison  := variable comparator number
    variable    := une clé de BOUNDS (temperature_c, humidity_pct, pressure_hpa, wind_kmh, rain_mm)
    comparator  := ">=" | "<=" | "==" | "!=" | ">" | "<"
    number      := -?[0-9]+(\.[0-9]+)?

Les espaces sont ignorés avant chaque élément. Difficulté : D1-D3 ★★, D4-D5 ★★★, D7 ★★★★
(le test de round-trip est une propriété Hypothesis sur des AST aléatoires).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, assert_never

from nimbus.s03_algebraic_types import BOUNDS, TypedReading, Valid

# ---------------------------------------------------------------------------
# Résultat d'un parser et AST (fournis)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Parsed[T]:
    value: T
    position: int


@dataclass(frozen=True)
class Failed:
    position: int
    expected: str


type ParseResult[T] = Parsed[T] | Failed
type Parser[T] = Callable[[str, int], ParseResult[T]]


@dataclass(frozen=True)
class Comparison:
    variable: str
    comparator: str
    value: float


@dataclass(frozen=True)
class And:
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Or:
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Not:
    expr: Expr


type Expr = Comparison | And | Or | Not

OPERATORS = (">=", "<=", "==", "!=", ">", "<")      # l'ordre compte : ">=" avant ">"


def skip_spaces(text: str, pos: int) -> int:
    """Fourni."""
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


# ---------------------------------------------------------------------------
# D1 à D4 — les combinators
# ---------------------------------------------------------------------------


def literal(s: str) -> Parser[str]:
    """D1 ★★ — reconnaît exactement `s` (après les espaces). Échec : Failed(pos après espaces, repr(s))."""
    def parse_literal(text: str, position: int) -> ParseResult[str]:
        position = skip_spaces(text, position)
        if text.startswith(s, position):
            return Parsed(s, position + len(s))
        return Failed(position, repr(s))

    return parse_literal


def pattern(regex: str, name: str) -> Parser[str]:
    """D1 ★★ — reconnaît l'expression régulière `regex` (après les espaces), échec nommé `name`.
    Indice : `re.compile(regex).match(text, pos)`."""
    compiled = re.compile(regex)

    def parse_pattern(text: str, position: int) -> ParseResult[str]:
        position = skip_spaces(text, position)
        match = compiled.match(text, position)
        if match is None:
            return Failed(position, name)
        return Parsed(match.group(), match.end())

    return parse_pattern


def fmap[A, B](p: Parser[A], f: Callable[[A], B]) -> Parser[B]:
    """D2 ★★ — applique f à la valeur lue (le « map » des parsers)."""
    def mapped(text: str, position: int) -> ParseResult[B]:
        result = p(text, position)
        if isinstance(result, Failed):
            return result
        return Parsed(f(result.value), result.position)

    return mapped


def sequence(*ps: Parser[Any]) -> Parser[tuple[Any, ...]]:
    """D2 ★★ — applique les parsers l'un après l'autre ; renvoie le tuple des valeurs,
    ou le premier échec."""
    def sequenced(text: str, position: int) -> ParseResult[tuple[Any, ...]]:
        values: list[Any] = []
        for parser in ps:
            result = parser(text, position)
            if isinstance(result, Failed):
                return result
            values.append(result.value)
            position = result.position
        return Parsed(tuple(values), position)

    return sequenced


def alternative[A](*ps: Parser[A]) -> Parser[A]:
    """D3 ★★ — le premier parser qui réussit. Si tous échouent, renvoyer l'échec qui est
    allé le plus loin (position maximale) : c'est lui qui donne le meilleur message d'erreur."""
    def choose(text: str, position: int) -> ParseResult[A]:
        failures: list[Failed] = []
        for parser in ps:
            result = parser(text, position)
            if isinstance(result, Parsed):
                return result
            failures.append(result)
        if not failures:
            return Failed(skip_spaces(text, position), "alternative")
        return max(failures, key=lambda failure: failure.position)

    return choose


def chain_left(element: Parser[Expr], separator: Parser[str],
                  build: Callable[[Expr, Expr], Expr]) -> Parser[Expr]:
    """D4 ★★★ — element (separateur element)*, combiné de façon associative à gauche :
    a or b or c  ->  build(build(a, b), c). C'est un fold (bloc 3) !"""
    def chained(text: str, position: int) -> ParseResult[Expr]:
        result = element(text, position)
        if isinstance(result, Failed):
            return result
        value = result.value
        position = result.position
        while True:
            separator_result = separator(text, position)
            if isinstance(separator_result, Failed):
                return Parsed(value, position)
            next_result = element(text, separator_result.position)
            if isinstance(next_result, Failed):
                return Parsed(value, position)
            value = build(value, next_result.value)
            position = next_result.position

    return chained


# ---------------------------------------------------------------------------
# D5 — la grammaire complète
# ---------------------------------------------------------------------------


def parse(text: str) -> Expr | Failed:
    """D5 ★★★ — texte -> AST, ou Failed si le texte n'est pas une règle valide.

    Tout le texte doit être consommé (espaces finaux compris) ; sinon Failed(position, "end of rule").
    Les mots-clés "and", "or", "not" ne doivent pas être confondus avec le début d'une variable.
    La récursion (factor -> "(" expr ")") se règle avec une fonction qui appelle `expr` au moment
    du parsing, pas au moment de la construction.
    """
    def keyword(word: str) -> Parser[str]:
        word_parser = literal(word)

        def parse_keyword(source: str, position: int) -> ParseResult[str]:
            result = word_parser(source, position)
            if isinstance(result, Failed):
                return result
            if result.position < len(source) and (source[result.position].isalnum() or source[result.position] == "_"):
                return Failed(skip_spaces(source, position), repr(word))
            return result

        return parse_keyword

    variable_pattern = r"(?:" + "|".join(re.escape(name) for name in sorted(BOUNDS, key=len, reverse=True)) + r")(?![A-Za-z0-9_])"
    variable_parser = pattern(variable_pattern, "variable")
    comparator_parser = alternative(*(literal(operator) for operator in OPERATORS))
    number_parser = fmap(pattern(r"-?[0-9]+(?:\.[0-9]+)?", "number"), float)

    def comparison_parser(source: str, position: int) -> ParseResult[Expr]:
        result = sequence(variable_parser, comparator_parser, number_parser)(source, position)
        if isinstance(result, Failed):
            return result
        variable, comparator, value = result.value
        return Parsed(Comparison(variable, comparator, value), result.position)

    def factor_parser(source: str, position: int) -> ParseResult[Expr]:
        position = skip_spaces(source, position)
        not_result = keyword("not")(source, position)
        if isinstance(not_result, Parsed):
            nested = factor_parser(source, not_result.position)
            if isinstance(nested, Failed):
                return nested
            return Parsed(Not(nested.value), nested.position)

        if position < len(source) and source[position] == "(":
            nested = expr_parser(source, position + 1)
            if isinstance(nested, Failed):
                return nested
            closing = literal(")")(source, nested.position)
            if isinstance(closing, Failed):
                return closing
            return Parsed(nested.value, closing.position)

        return comparison_parser(source, position)

    and_parser = fmap(keyword("and"), lambda _: "and")
    or_parser = fmap(keyword("or"), lambda _: "or")
    term_parser = chain_left(factor_parser, and_parser, And)
    expr_parser = chain_left(term_parser, or_parser, Or)

    result = expr_parser(text, 0)
    if isinstance(result, Failed):
        return result
    position = skip_spaces(text, result.position)
    if position != len(text):
        return Failed(position, "end of rule")
    return result.value


# ---------------------------------------------------------------------------
# D6, D7 — interpréter et afficher l'AST
# ---------------------------------------------------------------------------


def evaluate(e: Expr, r: TypedReading) -> bool:
    """D6 ★★ — vrai si le relevé vérifie la règle. Une comparaison sur une mesure qui
    n'est pas Valid est fausse (et donc `negation` de cette comparaison est vraie)."""
    match e:
        case Comparison(variable, comparator, value):
            measurement = getattr(r, variable)
            if not isinstance(measurement, Valid):
                return False
            actual = measurement.value
            match comparator:
                case ">=":
                    return actual >= value
                case "<=":
                    return actual <= value
                case "==":
                    return actual == value
                case "!=":
                    return actual != value
                case ">":
                    return actual > value
                case "<":
                    return actual < value
                case _:
                    raise ValueError(f"comparateur inconnu : {comparator!r}")
        case And(left, right):
            return evaluate(left, r) and evaluate(right, r)
        case Or(left, right):
            return evaluate(left, r) or evaluate(right, r)
        case Not(expr):
            return not evaluate(expr, r)
        case _:
            assert_never(e)


def render(e: Expr) -> str:
    """D7 ★★★★ — AST -> texte, tel que `parse(render(e)) == e` pour TOUT AST.

    Les nombres doivent se relire à l'identique (`repr(float)` convient). Le plus simple :
    parenthéser chaque And et chaque Or. Bonus : n'écrire que les parenthèses nécessaires.
    """
    match e:
        case Comparison(variable, comparator, value):
            return f"{variable} {comparator} {value!r}"
        case And(left, right):
            return f"({render(left)} and {render(right)})"
        case Or(left, right):
            return f"({render(left)} or {render(right)})"
        case Not(expr):
            return f"not {render(expr)}"
        case _:
            assert_never(e)
