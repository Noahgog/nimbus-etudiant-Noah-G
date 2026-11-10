"""Bloc 3 — DÉFI ★★★ (facultatif) : évaluer un arbre de profondeur 100 000.

Une expression arithmétique est un sum type récursif. L'évaluation récursive naturelle
lève RecursionError dès quelques milliers de niveaux : CPython limite la pile
(sys.getrecursionlimit(), 1 000 par défaut). Augmenter la limite ne fait que repousser
le crash du processus.

Deux techniques fonctionnelles permettent d'aller aussi profond qu'on veut :
- le continuation-passing style (CPS) : au lieu de renvoyer un résultat, chaque appel
  reçoit la suite du calcul (une fonction) ; combiné au trampoline du bloc 3, plus
  aucun appel ne s'empile ;
- un catamorphisme (fold sur l'arbre) avec une pile explicite, en parcours postfixe.

C'est le problème que résolvent les interpréteurs (et ZIO, au bloc 4) : ne jamais
dépendre de la pile du langage hôte.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import assert_never

from nimbus.s04_recursion import Bounce, Done, Trampoline, run_trampoline


@dataclass(frozen=True, eq=False)
class Num:
    value: float


@dataclass(frozen=True, eq=False)
class Add:
    left: Expr
    right: Expr


@dataclass(frozen=True, eq=False)
class Mul:
    left: Expr
    right: Expr


@dataclass(frozen=True, eq=False)
class Neg:
    expr: Expr


type Expr = Num | Add | Mul | Neg
# eq=False : l'égalité générée par dataclass serait elle-même récursive.


def eval_naive(e: Expr) -> float:
    """Fourni : la version récursive directe. Juste, mais RecursionError sur un arbre profond."""
    match e:
        case Num(v):
            return v
        case Add(a, b):
            return eval_naive(a) + eval_naive(b)
        case Mul(a, b):
            return eval_naive(a) * eval_naive(b)
        case Neg(x):
            return -eval_naive(x)
        case _:
            assert_never(e)


def eval_cps(e: Expr) -> float:
    """D1 ★★★ — même résultat que eval_naive, sur n'importe quelle profondeur.

    Écrire go(e, k) -> Trampoline[float], où k : float -> Trampoline[float] est la suite.
    - Num(v)    : Bounce(lambda: k(v))
    - Add(a, b) : évaluer a avec une suite qui évalue b avec une suite qui appelle k(x + y)
    Chaque appel récursif est enveloppé dans Bounce, pour que run_trampoline le déroule
    dans une boucle. Résultat : run_trampoline(go(e, Done)).
    """
    raise NotImplementedError("S04 — à compléter")


def cata[R](e: Expr, num: Callable[[float], R], add: Callable[[R, R], R],
            mul: Callable[[R, R], R], neg: Callable[[R], R]) -> R:
    """D2 ★★★ — catamorphisme (fold) sur l'arbre, avec une pile explicite (liste Python).

    Parcours postfixe : on empile les nœuds à visiter ; quand les résultats des enfants
    sont prêts (sur une seconde pile de résultats), on applique la fonction du nœud.
    Indice : empiler ("visit", noeud) puis, pour un nœud interne, ("build", noeud) suivi
    de ses enfants ; un ("build", Add) dépile deux résultats.
    """
    raise NotImplementedError("S04 — à compléter")


def depth(e: Expr) -> int:
    """D3 ★ — profondeur de l'arbre (une feuille : 1), avec `cata`."""
    raise NotImplementedError("S04 — à compléter")


def render(e: Expr) -> str:
    """D3 ★ — texte entièrement parenthésé, avec `cata` : Add(Num(1), Neg(Num(2))) -> "(1.0 + -(2.0))"."""
    raise NotImplementedError("S04 — à compléter")
