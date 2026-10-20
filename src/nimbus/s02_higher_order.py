"""Séance 2 — Fonctions d'ordre supérieur, composition, curryfication, paresse.

Une fonction est une valeur : on peut la passer, la renvoyer, la combiner.
`map`, `filter` et `reduce` remplacent la plupart des boucles ; les générateurs
apportent l'évaluation paresseuse (on ne calcule que ce qu'on consomme).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from itertools import islice
from typing import Any

from nimbus.datagen import HEADER

type Fn = Callable[[Any], Any]


def compose(*fns: Fn) -> Fn:
    """Exercice 2.1 — composition mathématique, de droite à gauche.

    compose(f, g, h)(x) == f(g(h(x))) ; compose() est l'identité.
    """
    def composed(x: Any) -> Any:
        for fn in reversed(fns):
            x = fn(x)
        return x
    return composed


def pipe(value: Any, *fns: Fn) -> Any:
    """Exercice 2.2 — application de gauche à droite : pipe(x, h, g, f) == f(g(h(x)))."""
    for fn in fns:
        value = fn(value)
    return value


def curry2[A, B, C](f: Callable[[A, B], C]) -> Callable[[A], Callable[[B], C]]:
    """Exercice 2.3 — curryfication : curry2(f)(a)(b) == f(a, b)."""
    return lambda a: lambda b: f(a, b)


def my_map[A, B](f: Callable[[A], B], it: Iterable[A]) -> Iterator[B]:
    """Exercice 2.4 — `map` paresseux, écrit avec `yield` (sans utiliser map)."""
    for value in it:
        yield f(value)


def my_filter[A](predicate: Callable[[A], bool], it: Iterable[A]) -> Iterator[A]:
    """Exercice 2.5 — `filter` paresseux, écrit avec `yield`."""
    for value in it:
        if predicate(value):
            yield value


def my_reduce[A, B](f: Callable[[B, A], B], it: Iterable[A], initial: B) -> B:
    """Exercice 2.6 — `reduce` (fold gauche) : f(f(f(initial, x0), x1), x2)..."""
    accumulator = initial
    for value in it:
        accumulator = f(accumulator, value)
    return accumulator


def take[A](n: int, it: Iterable[A]) -> list[A]:
    """Exercice 2.7 — les n premiers éléments ; doit fonctionner sur un itérable INFINI."""
    return list(islice(it, max(0, n)))


def sliding_windows[A](it: Iterable[A], size: int) -> Iterator[tuple[A, ...]]:
    """Exercice 2.8 — fenêtres glissantes paresseuses.

    sliding_windows([1, 2, 3, 4], 2) -> (1, 2), (2, 3), (3, 4)
    Indice : `collections.deque(maxlen=taille)`.
    """
    if size <= 0:
        raise ValueError("size doit être strictement positif")
    window: deque[A] = deque(maxlen=size)
    for value in it:
        window.append(value)
        if len(window) == size:
            yield tuple(window)


def moving_average(vals: Iterable[float], size: int) -> Iterator[float]:
    """Exercice 2.9 — moyenne mobile = composition de `sliding_windows` et `my_map`."""
    return my_map(lambda window: sum(window) / len(window), sliding_windows(vals, size))


def memoize[A, B](f: Callable[[A], B]) -> Callable[[A], B]:
    """Exercice 2.10 — fermeture (closure) qui met en cache les résultats de f.

    N'est légitime QUE parce que f est pure : pourquoi ?
    """
    cache: dict[A, B] = {}

    def cached(value: A) -> B:
        if value not in cache:
            cache[value] = f(value)
        return cache[value]

    return cached


# ---------------------------------------------------------------------------
# Application au fil rouge : découper le CSV brut en dictionnaires, paresseusement
# ---------------------------------------------------------------------------

COLUMNS: tuple[str, ...] = tuple(HEADER.split(","))


def non_empty_lines(text: str) -> Iterator[str]:
    """Fourni."""
    return (l for l in text.splitlines() if l.strip())


def split_csv(text: str) -> Iterator[list[str]]:
    """Exercice 2.11 — ignore l'en-tête et renvoie chaque ligne découpée sur ','.

    À écrire comme un pipeline : non_empty_lines -> sauter la 1re -> split.
    Indice : `itertools.islice`.
    """
    return (line.split(",") for line in islice(non_empty_lines(text), 1, None))
