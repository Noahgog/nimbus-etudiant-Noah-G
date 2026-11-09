"""Défi bloc 3 (facultatif) : uv run pytest tests/test_s04_challenge_stack.py"""

import sys

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nimbus.s04_challenge_stack import Add, Mul, Neg, Num, cata, depth, eval_cps, eval_naive, render

DEEP = 100_000

small_floats = st.integers(-5, 5).map(float)
exprs = st.recursive(
    st.builds(Num, small_floats),
    lambda sub: st.one_of(st.builds(Add, sub, sub), st.builds(Mul, sub, sub), st.builds(Neg, sub)),
    max_leaves=25,
)


def left_comb(n):
    e = Num(1.0)
    for i in range(n):
        e = Add(e, Num(1.0))
    return e


def right_comb(n):
    e = Num(0.0)
    for i in range(n):
        e = Add(Num(1.0), Neg(Neg(e))) if i % 2 else Add(Num(1.0), e)
    return e


@settings(max_examples=200, deadline=None)
@given(exprs)
def test_memes_resultats_que_la_version_naive(e):
    assert eval_cps(e) == eval_naive(e)
    assert cata(e, lambda v: v, lambda a, b: a + b, lambda a, b: a * b, lambda a: -a) == eval_naive(e)


def test_la_version_naive_casse_bien():
    with pytest.raises(RecursionError):
        eval_naive(left_comb(DEEP))


@pytest.mark.parametrize("build", [left_comb, right_comb])
def test_profondeur_100000(build):
    limit = sys.getrecursionlimit()
    e = build(DEEP)
    assert eval_cps(e) == (DEEP + 1 if build is left_comb else DEEP)
    assert cata(e, lambda v: v, lambda a, b: a + b, lambda a, b: a * b, lambda a: -a) == eval_cps(e)
    assert sys.getrecursionlimit() == limit           # pas de triche avec setrecursionlimit


def test_depth_et_render():
    e = Add(Num(1.0), Neg(Num(2.0)))
    assert render(e) == "(1.0 + -(2.0))"
    assert depth(e) == 3
    assert depth(left_comb(DEEP)) == DEEP + 1
    assert len(render(left_comb(DEEP))) > DEEP
