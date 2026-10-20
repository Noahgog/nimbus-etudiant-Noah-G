"""Séance 1 Pureté, transparence référentielle, immutabilité.

Idée clé : une fonction pure ne dépend que de ses arguments et ne fait rien
d'autre que renvoyer un résultat. On peut donc remplacer tout appel par sa
valeur (transparence référentielle) : c'est ce qui rend le code testable,
parallélisable et différentiable (cf. JAX en séance 7).

Les blocs entre « >>> SOLUTION » et « <<< SOLUTION » sont retirés dans la
version étudiante.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

# ---------------------------------------------------------------------------
# Contre-exemple (fourni) : le style impératif que l'on va remplacer.
# Repérez les trois effets de bord de cette fonction.
# ---------------------------------------------------------------------------

_JOURNAL: list[str] = []


def nettoyer_imperatif(temperatures: list[float]) -> float:
    """NE PAS IMITER. Modifie son argument, écrit dans un état global, imprime."""
    i = 0
    while i < len(temperatures):
        if temperatures[i] > 60:
            del temperatures[i]          # 1. mutation de l'argument
        else:
            i += 1
    _JOURNAL.append("nettoyage")         # 2. écriture dans un état global
    print("moyenne calculée")            # 3. entrée/sortie
    return sum(temperatures) / len(temperatures)


# ---------------------------------------------------------------------------
# Exercices
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Releve:
    """Un relevé immuable. `frozen=True` interdit toute affectation après création."""

    date: str
    station: str
    temperature_c: float
    humidite_pct: float


def celsius_vers_fahrenheit(c: float) -> float:
    """Exercice 1.1 — échauffement : F = C × 9/5 + 32."""
    return c * 9/5 + 32


def filtrer_plausibles(temperatures: Sequence[float], seuil: float = 60.0) -> tuple[float, ...]:
    """Exercice 1.2 — version pure de la boucle de `nettoyer_imperatif`.

    Ne modifie pas l'entrée ; renvoie un tuple (immuable) des valeurs <= seuil.
    """
    return tuple(temperature for temperature in temperatures if temperature <= seuil)


def moyenne(valeurs: Sequence[float]) -> float | None:
    """Exercice 1.3 — moyenne, ou None si la séquence est vide (pas d'exception !)."""
    if not valeurs:
        return None
    return sum(valeurs) / len(valeurs)


def normaliser(valeurs: Sequence[float]) -> tuple[float, ...]:
    """Exercice 1.4 — normalisation min-max dans [0, 1].

    Si toutes les valeurs sont égales, renvoyer des 0.0. Séquence vide -> tuple vide.
    """
    if not valeurs:
        return ()
    minimum = min(valeurs)
    maximum = max(valeurs)
    if minimum == maximum:
        return tuple(0.0 for _ in valeurs)
    return tuple((valeur - minimum) / (maximum - minimum) for valeur in valeurs)


def corriger_temperature(r: Releve, delta: float) -> Releve:
    """Exercice 1.5 — renvoie un NOUVEAU relevé dont la température est décalée de delta.

    Indice : `dataclasses.replace`.
    """
    return replace(r, temperature_c=r.temperature_c + delta)


def ajouter(historique: tuple[Releve, ...], r: Releve) -> tuple[Releve, ...]:
    """Exercice 1.6 — ajoute un relevé à un historique sans le modifier."""
    return historique + (r,)


def nettoyer_pur(temperatures: Sequence[float]) -> tuple[float | None, tuple[str, ...]]:
    """Exercice 1.7 — réécriture pure de `nettoyer_imperatif`.

    Au lieu d'écrire dans un journal global, on RENVOIE les messages de journal
    avec le résultat : (moyenne des plausibles, ("nettoyage",)).
    C'est le germe de la monade Writer — et de l'idée « l'effet devient une valeur ».
    """
    return moyenne(filtrer_plausibles(temperatures)), ("nettoyage",)


# ===========================================================================
# PARTIE B — Refactoring d'un script hérité (≈ 30 min)
# ===========================================================================
# Lisez `nimbus/s01_legacy.py`. Il produit un rapport par station en s'appuyant
# sur trois variables globales, des mutations et des `print`. Réécrivez-le en
# fonctions pures. Contrainte : `rapport(texte)` doit renvoyer EXACTEMENT le
# texte que le script hérité affiche (un `\n` à la fin de chaque ligne).

BORNES_TEMPERATURE = (-40.0, 50.0)

# mypy: ignore-errors

def lire_temperatures(texte: str) -> tuple[tuple[tuple[str, float], ...], int]:
    """Exercice 1.8 — ((station, température), ...) des lignes valides + nombre de rejets.

    Mêmes règles que le script hérité : 7 colonnes, température convertible en float,
    comprise dans BORNES_TEMPERATURE (bornes incluses). L'en-tête est ignoré.
    """
    paires: list[tuple[str, float]] = []
    rejets = 0
    for ligne in texte.splitlines()[1:]:
        champs = ligne.split(",")
        if len(champs) != 7: #une ligne contient 7 colonnes
            rejets += 1
            continue
        try:
            temperature = float(champs[2]) #3eme ligne comprend température
        except ValueError:
            rejets += 1
            continue #si la conversion en float échoue
        if not BORNES_TEMPERATURE[0] <= temperature <= BORNES_TEMPERATURE[1]: #selon la variable préalablement définie
            rejets += 1
            continue
        paires.append((champs[1], temperature))
    return tuple(paires), rejets

def regrouper(paires: Sequence[tuple[str, float]]) -> Mapping[str, tuple[float, ...]]:
    """Exercice 1.9 — températures TRIÉES par station : {"BREST": (2.5, 3.1, ...), ...}."""
    groupes: dict[str, list[float]] = {} #dictionnaire ayant pour clé le nom de la station
    for station, temperature in paires:
        groupes.setdefault(station, []).append(temperature) #ajout de la température si station existe déjà, sinon on l'ajoute en créant une liste vide, objectif est de regrouper toutes les températures d'une même station
    return {station: tuple(sorted(temperatures)) for station, temperatures in groupes.items()}


def ligne_station(station: str, temperatures: Sequence[float]) -> str:
    """Exercice 1.10 — une ligne du rapport, au format exact du script hérité.

    `temperatures` est supposée triée et non vide. Médiane = élément d'indice len // 2.
    """
    moyenne_station = sum(temperatures) / len(temperatures) # moyenne des températures d'une station
    mediane = temperatures[len(temperatures) // 2] # valeur médiane pour élément central, valeur du "milieu"
    return (
        f"{station:<8} n={len(temperatures):<4} moyenne={moyenne_station:6.2f}  "
        f"médiane={mediane:6.2f}  min={temperatures[0]:6.1f}  max={temperatures[-1]:6.1f}"
    ) #Affichage pour chacune des stations, avec nb valeurs, min, max, moyenne, mediane, etc ...


def rapport(texte: str) -> str:
    """Exercice 1.11 — le rapport complet, par composition des trois fonctions précédentes."""
    paires, rejets = lire_temperatures(texte)
    groupes = regrouper(paires)
    lignes = [ligne_station(station, groupes[station]) for station in sorted(groupes)]
    lignes.append(f"Rejets : {rejets}")
    return "".join(ligne + "\n" for ligne in lignes) #Affichage final avec rejets et nb de valeurs


# ===========================================================================
# PARTIE C — Le hasard, purement (≈ 25 min)
# ===========================================================================
# `random.random()` est impur : deux appels identiques donnent deux résultats.
# Solution fonctionnelle : l'état du générateur devient une VALEUR qu'on passe
# en argument et qu'on renvoie mise à jour. C'est exactement le modèle de JAX :
#     key, sous_cle = jax.random.split(key); x = jax.random.normal(sous_cle)

_A = 6364136223846793005
_C = 1442695040888963407
_MASQUE = (1 << 64) - 1

def suivant(etat: int) -> int:
    """Fourni : un pas de générateur congruentiel linéaire 64 bits."""
    return (_A * etat + _C) & _MASQUE #Cela produit un nouvel état pseudo-aléatoire

def alea(etat: int) -> tuple[float, int]:
    """Exercice 1.12 — (flottant uniforme dans [0, 1), nouvel état).

    Calculer e2 = suivant(etat) puis renvoyer ((e2 >> 11) / 2**53, e2).
    """
    nouvel_etat = suivant(etat)
    return (nouvel_etat >> 11) / 2**53, nouvel_etat #on décale de 11 bits


def alea_n(etat: int, n: int) -> tuple[tuple[float, ...], int]:
    """Exercice 1.13 — n tirages successifs : on « fait passer » l'état de tirage en tirage."""
    tirages: list[float] = []
    for _ in range(n):
        tirage, etat = alea(etat)
        tirages.append(tirage)
    return tuple(tirages), etat # renvoie les tirages effectués et une liste de sortie


def melanger[T](xs: Sequence[T], etat: int) -> tuple[tuple[T, ...], int]:
    """Exercice 1.14 — permutation aléatoire (Fisher-Yates) SANS modifier xs.

    Pour i de len-1 à 1 : tirer u = alea(etat), j = int(u * (i + 1)), échanger i et j
    dans une COPIE locale. Renvoyer (tuple mélangé, état final).
    """
    melange = list(xs)
    for indice in range(len(melange) - 1, 0, -1):
        tirage, etat = alea(etat)
        autre_indice = int(tirage * (indice + 1))
        melange[indice], melange[autre_indice] = melange[autre_indice], melange[indice]
    return tuple(melange), etat


def separer(etat: int) -> tuple[int, int]:
    """Exercice 1.15 — deux états indépendants à partir d'un seul (comme jax.random.split).

    Proposition : e1 = suivant(etat) ; e2 = suivant(e1) XOR 0x9E3779B97F4A7C15.
    Pourquoi ne faut-il JAMAIS réutiliser `etat` après l'avoir séparé ?
    """
    premier = suivant(etat)
    second = suivant(premier) ^ 0x9E3779B97F4A7C15 #décorrélation des 2 résultats
    return premier, second


def decouper_apprentissage_test[T](xs: Sequence[T], part_test: float, etat: int) -> tuple[tuple[T, ...], tuple[T, ...], int]:
    """Exercice 1.16 — découpage train/test reproductible : (apprentissage, test, nouvel état).

    Mélanger, puis les round(len * part_test) premiers éléments forment le test.
    Même graine => même découpage, sur toutes les machines : c'est ça, la reproductibilité.
    """
    if not 0.0 <= part_test <= 1.0:
        raise ValueError("part_test doit être comprise entre 0 et 1")
    melange, nouvel_etat = melanger(xs, etat)
    taille_test = round(len(melange) * part_test)
    return melange[taille_test:], melange[:taille_test], nouvel_etat #données d'apprentissage, données de test et état final sont renvoyés