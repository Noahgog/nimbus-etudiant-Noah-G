"""Séance 1 — DÉFI ★★★ (facultatif, pour les plus curieux) : le vecteur persistant.

Question laissée ouverte en cours : « l'immutabilité, c'est bien, mais copier un
tuple d'un million d'éléments pour en changer UN, c'est ruineux ». Réponse des
langages fonctionnels (Clojure, Scala, Haskell...) : le VECTEUR PERSISTANT.

Idée (Phil Bagwell, Rich Hickey) : stocker les éléments dans les feuilles d'un
arbre où chaque nœud a jusqu'à 32 enfants. Pour modifier l'élément i, on ne
recopie QUE le chemin racine -> feuille (au plus ~4 nœuds pour un million
d'éléments) ; tout le reste est PARTAGÉ entre l'ancienne et la nouvelle version.
Les deux versions restent valides et immuables.

            racine (décalage = 5)
           /        |         \\
     [0..31]    [32..63]   [64..65]        <- feuilles : tuples de ≤ 32 valeurs

Pour trouver l'élément i : à chaque niveau, l'enfant à suivre est donné par
5 bits de i :   (i >> decalage) & 31,  puis decalage -= 5, jusqu'à la feuille
où l'on prend i & 31.

Représentation imposée (les tests en dépendent) :
- une feuille est un `tuple` de valeurs ; un nœud interne est un `tuple` de nœuds ;
- `offset` = 5 × (hauteur - 1) : 0 quand la racine est elle-même une feuille ;
- aucun nœud vide ne doit subsister, et la racine n'a jamais un seul enfant
  si decalage > 0 (sauf transitoirement : voir `pop_last`).

Difficulté : D1, D2, D5 ★★ ; D3, D4 ★★★ ; D6 ★★★★ (hors barème). Les tests vérifient le partage
structurel : une copie complète échoue, même si le résultat est juste.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import reduce
from typing import Any, cast

BITS = 5
WIDTH = 1 << BITS      # 32
MASK = WIDTH - 1     # 0b11111

type Node = tuple[Any, ...]


@dataclass(frozen=True)
class Vector[T]:
    size: int
    offset: int
    root: Node

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[T]:
        return iter(to_tuple(self))


EMPTY: Vector[Any] = Vector(0, 0, ())


def depth(v: Vector[Any]) -> int:
    """Fourni : nombre de niveaux de l'arbre."""
    return v.offset // BITS + 1


def lookup[T](v: Vector[T], i: int) -> T:
    """D1 ★★ — l'élément d'indice i en O(log32 n). IndexError si i hors de [0, size)."""
    if i < 0 or i >= v.size:
        raise IndexError(i)
    node = v.root
    level = v.offset
    while level > 0:
        node = node[(i >> level) & MASK]
        level -= BITS
    return cast(T, node[i & MASK])


def update[T](v: Vector[T], i: int, x: T) -> Vector[T]:
    """D2 ★★ — nouvelle version où l'élément i vaut x ; seul le chemin est recopié."""
    if i < 0 or i >= v.size:
        raise IndexError(i)

    def replace(node: Node, level: int) -> Node:
        index = (i >> level) & MASK if level > 0 else i & MASK
        children = list(node)
        if level == 0:
            children[index] = x
        else:
            children[index] = replace(node[index], level - BITS)
        return tuple(children)

    return Vector(v.size, v.offset, replace(v.root, v.offset))


def _new_path[T](level: int, x: T) -> Node:
    if level == 0:
        return (x,)
    return (_new_path(level - BITS, x),)


def _append_node[T](node: Node, level: int, size: int, x: T) -> Node:
    if level == 0:
        return (*node, x)
    child_index = (size >> level) & MASK
    children = list(node)
    if child_index < len(children):
        children[child_index] = _append_node(children[child_index], level - BITS, size, x)
    else:
        children.append(_new_path(level - BITS, x))
    return tuple(children)


def append[T](v: Vector[T], x: T) -> Vector[T]:
    """D3 ★★★ — ajoute x à la fin.

    Trois cas : vecteur vide ; racine pleine (size == 1 << (offset + BITS)) : il
    faut une NOUVELLE racine, un niveau plus haut, dont le 2e enfant est un chemin neuf ;
    sinon : descendre en recopiant le chemin, et créer un chemin neuf là où l'enfant
    n'existe pas encore.
    """
    if v.size == 0:
        return Vector(1, 0, (x,))
    if v.size == 1 << (v.offset + BITS):
        root = (v.root, _new_path(v.offset, x))
        return Vector(v.size + 1, v.offset + BITS, root)
    return Vector(v.size + 1, v.offset, _append_node(v.root, v.offset, v.size, x))


def _pop_node(node: Node, level: int, index: int) -> Node:
    child_index = (index >> level) & MASK if level > 0 else index & MASK
    children = list(node)
    if level == 0:
        children.pop()
    else:
        updated_child = _pop_node(node[child_index], level - BITS, index)
        if updated_child:
            children[child_index] = updated_child
        else:
            children.pop()
    return tuple(children)


def pop_last[T](v: Vector[T]) -> Vector[T]:
    """D4 ★★★ — supprime le dernier élément (IndexError si vide).

    Attention aux nœuds devenus vides (à supprimer) et à la racine qui n'a plus
    qu'un enfant (l'arbre doit perdre un niveau).
    """
    if v.size == 0:
        raise IndexError("pop from empty vector")
    if v.size == 1:
        return EMPTY

    root = _pop_node(v.root, v.offset, v.size - 1)
    offset = v.offset
    while offset > 0 and len(root) == 1:
        root = root[0]
        offset -= BITS
    return Vector(v.size - 1, offset, root)


def to_tuple[T](v: Vector[T]) -> tuple[T, ...]:
    """D5 ★★ — tous les éléments dans l'ordre (parcours de l'arbre, sans appeler obtenir n fois)."""
    values: list[T] = []

    def visit(node: Node, level: int) -> None:
        if level == 0:
            values.extend(node)
        else:
            for child in node:
                visit(child, level - BITS)

    if v.size:
        visit(v.root, v.offset)
    return tuple(values)


def from_iterable[T](xs: Iterable[T]) -> Vector[T]:
    """Fourni : un fold d'`append` (séance 4 !)."""
    return reduce(append, xs, EMPTY)


# ---------------------------------------------------------------------------
# D6 ★★★★ (hors barème) : `concatener(a, b)` en mieux que O(len(b)).
# Indice : chercher « RRB-tree » (Bagwell & Rompf, 2011). Aucun test fourni :
# écrivez les vôtres, avec Hypothesis, contre le modèle `tuple`.
# ---------------------------------------------------------------------------
