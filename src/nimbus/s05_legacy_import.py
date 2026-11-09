# mypy: ignore-errors
"""Bloc 3, partie B — import CSV « robuste » à base d'exceptions. NE PAS MODIFIER.

Il ne plante jamais... parce qu'il avale tout :
- une seule mesure illisible ("NA", "vingt") fait jeter la ligne entière ;
- une température de 999.9 °C est acceptée comme une vraie mesure ;
- la raison du rejet n'est jamais dite, et un `except:` nu attrape aussi les bugs ;
- le seul compte rendu est un print.

Réécrivez le compte rendu avec Result / Validation dans `s05_errors.py` (5.7 et 5.8).
"""

from datetime import date

from nimbus.datagen import STATIONS


def import_csv(text):
    readings = []
    for i, line in enumerate(text.splitlines()[1:], start=2):
        try:
            d, st, t, h, p, w, r = line.split(",")
            date.fromisoformat(d)
            if st not in STATIONS:
                raise ValueError(st)
            readings.append((d, st, float(t), float(h), float(p), float(w), float(r)))
        except:  # noqa: E722
            print(f"ligne {i} ignorée")
    return readings
