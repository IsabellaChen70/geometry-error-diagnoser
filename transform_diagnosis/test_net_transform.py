"""Exhaustive contract tests for the v6 canonical net-affine representation."""

from __future__ import annotations

import itertools

import pytest

from transform_diagnosis import net_transform as nt
from transform_diagnosis import transform_core as tc


def test_vocabulary_is_exactly_the_eight_generator_d4_maps():
    assert nt.D4_LINEAR_NAMES == (
        "identity",
        "rot_ccw_90",
        "rot_180",
        "rot_ccw_270",
        "reflect_x_axis",
        "reflect_y_axis",
        "reflect_y_eq_x",
        "reflect_y_eq_neg_x",
    )
    assert len(nt.LINEAR_TO_MATRIX) == len(nt.MATRIX_TO_LINEAR) == 8
    assert set(nt.LINEAR_TO_MATRIX.values()) == set(tc.ALL_LINEAR_MAPS)


@pytest.mark.parametrize("linear", nt.D4_LINEAR_NAMES)
@pytest.mark.parametrize("tx,ty", [(0, 0), (2, -3), (-11, 17)])
def test_all_linear_maps_and_translations_round_trip(linear, tx, ty):
    net = {"linear": linear, "tx": tx, "ty": ty}
    affine = nt.net_to_affine(net)
    assert nt.affine_to_net(affine) == net
    assert affine.matrix == nt.LINEAR_TO_MATRIX[linear]
    assert affine.vec == (tx, ty)


def test_every_d4_pair_composition_round_trips_losslessly():
    primitives = [tc.Transform(m, (0, 0)) for m in tc.ALL_LINEAR_MAPS]
    for first, second in itertools.product(primitives, repeat=2):
        seq = [first, tc.translate(3, -4), second]
        net = nt.sequence_to_net(seq)
        assert nt.net_to_affine(net) == tc.compose(seq)
        assert nt.affine_to_net(nt.net_to_affine(net)) == net


def test_sequence_helper_uses_transform_core_composition_order():
    seq = [tc.translate(3, -2), tc.rotate(90, "ccw")]
    assert nt.sequence_to_net(seq) == {
        "linear": "rot_ccw_90",
        "tx": 2,
        "ty": 3,
    }
    assert nt.net_to_affine(nt.sequence_to_net(seq)) == tc.compose(seq)


def test_strict_canonical_equality_and_description():
    value = {"linear": "reflect_y_eq_neg_x", "tx": -2, "ty": 5}
    assert nt.canonical_net_equal(value, dict(value))
    assert not nt.canonical_net_equal(value, {**value, "ty": 6})
    desc = nt.describe_net(value)
    assert "y = -x" in desc and "(-2, 5)" in desc


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        "rot_ccw_90",
        {},
        {"linear": "identity", "tx": 0},
        {"linear": "identity", "tx": 0, "ty": 0, "extra": 1},
        {"linear": "rotate_90", "tx": 0, "ty": 0},
        {"linear": 3, "tx": 0, "ty": 0},
        {"linear": "identity", "tx": "2", "ty": 0},
        {"linear": "identity", "tx": 2.0, "ty": 0},
        {"linear": "identity", "tx": True, "ty": 0},
        {"linear": "identity", "tx": 0, "ty": False},
    ],
)
def test_invalid_values_are_rejected(bad):
    with pytest.raises(ValueError):
        nt.validate_net(bad)
    with pytest.raises(ValueError):
        nt.net_to_affine(bad)
    assert not nt.is_net(bad)


def test_affine_outside_d4_is_rejected():
    with pytest.raises(ValueError):
        nt.affine_to_net(tc.Transform(((2, 0), (0, 1)), (0, 0)))


def test_affine_helper_requires_transform_instance():
    with pytest.raises(TypeError):
        nt.affine_to_net("rotate 90 degrees counterclockwise")  # type: ignore[arg-type]


def test_diagnose_nets_is_the_transform_core_oracle():
    correct = nt.sequence_to_net([tc.rotate(90), tc.translate(2, -1)])
    student = nt.sequence_to_net([tc.reflect("x"), tc.translate(2, -1)])
    assert nt.diagnose_nets(correct, student) == "reflection_instead_of_rotation"
    assert nt.diagnose_nets(correct, correct) == "correct"
