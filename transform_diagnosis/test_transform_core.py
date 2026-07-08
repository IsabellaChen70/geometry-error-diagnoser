"""Contract test for transform_core (ACCEPTANCE TEST #1).

This is THE contract test for the single canonical implementation. It must always pass
unchanged. If judging behaviour ever changes, change ``transform_core`` AND this test
together — never weaken this test to paper over a failure elsewhere.
"""

from __future__ import annotations

import random

import pytest

from transform_diagnosis import transform_core as tc


# --------------------------------------------------------------------------------------
# Factories & matrices
# --------------------------------------------------------------------------------------

def test_identity():
    i = tc.identity()
    assert i.matrix == ((1, 0), (0, 1))
    assert i.vec == (0, 0)
    assert i.apply([(3, 4), (-2, 5)]) == [(3, 4), (-2, 5)]
    assert i.det() == 1


def test_rotation_matrices_ccw():
    assert tc.rotate(90, "ccw").apply([(1, 0)]) == [(0, 1)]
    assert tc.rotate(180, "ccw").apply([(1, 0)]) == [(-1, 0)]
    assert tc.rotate(270, "ccw").apply([(1, 0)]) == [(0, -1)]
    assert tc.rotate(90, "ccw").apply([(2, 3)]) == [(-3, 2)]


def test_rotation_direction_equivalences():
    assert tc.rotate(90, "cw") == tc.rotate(270, "ccw")
    assert tc.rotate(270, "cw") == tc.rotate(90, "ccw")
    assert tc.rotate(180, "cw") == tc.rotate(180, "ccw")
    assert tc.rotate(90, "cw").apply([(1, 0)]) == [(0, -1)]


def test_rotation_det_is_plus_one():
    for deg in (90, 180, 270):
        assert tc.rotate(deg, "ccw").det() == 1


def test_reflection_matrices():
    assert tc.reflect("x").apply([(2, 3)]) == [(2, -3)]
    assert tc.reflect("y").apply([(2, 3)]) == [(-2, 3)]
    assert tc.reflect("y=x").apply([(2, 3)]) == [(3, 2)]
    assert tc.reflect("y=-x").apply([(2, 3)]) == [(-3, -2)]
    for line in ("x", "y", "y=x", "y=-x"):
        assert tc.reflect(line).det() == -1


def test_reflect_line_aliases():
    assert tc.reflect("x axis") == tc.reflect("x")
    assert tc.reflect("the x-axis") == tc.reflect("x")
    assert tc.reflect("line y = x") == tc.reflect("y=x")
    assert tc.reflect("y = -x") == tc.reflect("y=-x")


def test_translate():
    t = tc.translate(7, -3)
    assert t.matrix == ((1, 0), (0, 1))
    assert t.vec == (7, -3)
    assert t.apply([(0, 0), (1, 2)]) == [(7, -3), (8, -1)]
    assert t.det() == 1


# --------------------------------------------------------------------------------------
# Matrix helpers (exposed on the Transform contract)
# --------------------------------------------------------------------------------------

def test_mat_helpers():
    a = ((1, 2), (3, 4))
    b = ((5, 6), (7, 8))
    assert tc.Transform.mat_mul(a, b) == ((19, 22), (43, 50))
    assert tc.mat_mul(a, b) == ((19, 22), (43, 50))
    assert tc.Transform.mat_vec(a, (1, 1)) == (3, 7)
    assert tc.mat_vec(a, (1, 1)) == (3, 7)
    assert tc.Transform.det_of(a) == -2
    assert tc.det(a) == -2


def test_orientations_map():
    assert tc.ORIENTATIONS[tc.rotate(90, "ccw").det()] == "rotation"
    assert tc.ORIENTATIONS[tc.reflect("x").det()] == "reflection"
    assert tc.ORIENTATIONS == {1: "rotation", -1: "reflection"}


# --------------------------------------------------------------------------------------
# Composition (seq[0] applies FIRST)
# --------------------------------------------------------------------------------------

def test_compose_order_is_seq0_first():
    original = [(1, 0)]
    # rotate 90 ccw first -> (0,1); then translate (3,-2) -> (3,-1)
    assert tc.compose([tc.rotate(90, "ccw"), tc.translate(3, -2)]).apply(original) == [(3, -1)]
    # translate first -> (4,-2); then rotate 90 ccw -> (2,4)
    assert tc.compose([tc.translate(3, -2), tc.rotate(90, "ccw")]).apply(original) == [(2, 4)]


def test_compose_not_commutative():
    seq_a = [tc.rotate(90, "ccw"), tc.translate(3, -2)]
    seq_b = [tc.translate(3, -2), tc.rotate(90, "ccw")]
    assert tc.compose(seq_a) != tc.compose(seq_b)


def test_compose_empty_is_identity():
    assert tc.compose([]) == tc.identity()


def test_compose_accepts_strings():
    a = tc.compose(["rotate 90 degrees counterclockwise", "translate 3 left"])
    b = tc.compose([tc.rotate(90, "ccw"), tc.translate(-3, 0)])
    assert a == b


# --------------------------------------------------------------------------------------
# describe / parse round-trip
# --------------------------------------------------------------------------------------

def test_describe_canonical_strings():
    assert tc.describe_transform(tc.rotate(90, "ccw")) == "rotate 90 degrees counterclockwise"
    assert tc.describe_transform(tc.rotate(90, "ccw"), rotation_style="cw") == "rotate 270 degrees clockwise"
    assert tc.describe_transform(tc.reflect("x")) == "reflect across x axis"
    assert tc.describe_transform(tc.reflect("y=-x")) == "reflect across line y = -x"
    assert tc.describe_transform(tc.translate(7, 0)) == "translate 7 right"
    assert tc.describe_transform(tc.translate(-7, 0)) == "translate 7 left"
    assert tc.describe_transform(tc.translate(0, 10)) == "translate 10 up"
    assert tc.describe_transform(tc.translate(0, -10)) == "translate 10 down"
    assert tc.describe_transform(tc.translate(3, -2)) == "translate by (3, -2)"


def test_parse_round_trip():
    prims = [
        tc.rotate(90, "ccw"), tc.rotate(180, "ccw"), tc.rotate(270, "ccw"),
        tc.reflect("x"), tc.reflect("y"), tc.reflect("y=x"), tc.reflect("y=-x"),
        tc.translate(7, 0), tc.translate(-7, 0), tc.translate(0, 5), tc.translate(0, -5),
        tc.translate(3, -2), tc.translate(-4, -6),
    ]
    for t in prims:
        assert tc.parse_transform(tc.describe_transform(t)) == t
        assert tc.parse_transform(tc.describe_transform(t, rotation_style="cw")) == t


def test_parse_reworded_rotation_equal_matrix():
    assert tc.parse_transform("rotate 270 degrees clockwise") == tc.rotate(90, "ccw")
    assert tc.parse_transform("rotate 90 degrees clockwise") == tc.rotate(270, "ccw")


# --------------------------------------------------------------------------------------
# grade
# --------------------------------------------------------------------------------------

def test_grade_true_and_false():
    original = [(1, 0), (2, 3), (-1, 1)]
    seq = [tc.rotate(90, "ccw"), tc.translate(3, -2)]
    image = tc.compose(seq).apply(original)
    assert tc.grade(original, image, seq) is True
    assert tc.grade(original, image, [tc.reflect("x"), tc.translate(3, -2)]) is False
    # strings work too
    assert tc.grade(original, image, ["rotate 90 degrees counterclockwise", "translate 3 left"]) is False


# --------------------------------------------------------------------------------------
# diagnose — every label, plus totality
# --------------------------------------------------------------------------------------

_ORIG = [(0, 0), (3, 0), (3, 1), (1, 1), (1, 2)]

_CASES = {
    "correct": ([tc.rotate(90, "ccw"), tc.translate(7, 0)],
                [tc.rotate(90, "ccw"), tc.translate(7, 0)]),
    "reflection_instead_of_rotation": ([tc.rotate(90, "ccw"), tc.translate(7, 0)],
                                        [tc.reflect("x"), tc.translate(7, 0)]),
    "rotation_instead_of_reflection": ([tc.reflect("x"), tc.translate(7, 0)],
                                       [tc.rotate(90, "ccw"), tc.translate(7, 0)]),
    "wrong_rotation_angle": ([tc.rotate(90, "ccw"), tc.translate(7, 0)],
                             [tc.rotate(180, "ccw"), tc.translate(7, 0)]),
    "wrong_reflection_line": ([tc.reflect("x"), tc.translate(7, 0)],
                              [tc.reflect("y"), tc.translate(7, 0)]),
    "wrong_translation": ([tc.rotate(90, "ccw"), tc.translate(7, 0)],
                          [tc.rotate(90, "ccw"), tc.translate(3, 2)]),
    "opposite_translation": ([tc.rotate(90, "ccw"), tc.translate(7, 0)],
                             [tc.rotate(90, "ccw"), tc.translate(-7, 0)]),
    "completely_wrong": ([tc.rotate(90, "ccw"), tc.translate(7, 0)],
                         [tc.reflect("x"), tc.translate(3, 2)]),
}


@pytest.mark.parametrize("label,seqs", list(_CASES.items()))
def test_diagnose_each_label(label, seqs):
    correct, student = seqs
    assert tc.diagnose(_ORIG, correct, student) == label


def test_diagnose_accepts_strings():
    assert tc.diagnose(
        _ORIG,
        ["rotate 90 degrees counterclockwise", "translate 7 right"],
        ["reflect across x axis", "translate 7 right"],
    ) == "reflection_instead_of_rotation"


def test_diagnose_is_total_and_deterministic():
    rng = random.Random(12345)

    def rand_seq():
        prims = [tc.rotate(rng.choice((90, 180, 270)), "ccw"),
                 tc.reflect(rng.choice(("x", "y", "y=x", "y=-x"))),
                 tc.translate(rng.randint(-6, 6), rng.randint(-6, 6))]
        rng.shuffle(prims)
        return prims[:rng.choice((1, 2))]

    for _ in range(2000):
        c, s = rand_seq(), rand_seq()
        label = tc.diagnose(_ORIG, c, s)
        assert label in tc.DIAGNOSIS_LABELS
        assert tc.diagnose(_ORIG, c, s) == label  # deterministic


def test_diagnosis_labels_closed_set():
    assert tc.DIAGNOSIS_LABELS == [
        "correct", "reflection_instead_of_rotation", "rotation_instead_of_reflection",
        "wrong_rotation_angle", "wrong_reflection_line", "wrong_translation",
        "opposite_translation", "completely_wrong",
    ]


def test_correct_only_when_identical_net_map():
    # Same net map reached by different sequences must still read as 'correct'.
    correct = [tc.rotate(180, "ccw"), tc.translate(2, 3)]
    student = [tc.rotate(90, "ccw"), tc.rotate(90, "ccw"), tc.translate(2, 3)]
    assert tc.compose(correct) == tc.compose(student)
    assert tc.diagnose(_ORIG, correct, student) == "correct"


# --------------------------------------------------------------------------------------
# is_asymmetric
# --------------------------------------------------------------------------------------

def test_is_asymmetric_true_for_chiral_shape():
    assert tc.is_asymmetric([(0, 0), (3, 0), (3, 1), (1, 1), (1, 2)]) is True


def test_is_asymmetric_false_for_square():
    assert tc.is_asymmetric([(0, 0), (2, 0), (2, 2), (0, 2)]) is False


def test_is_asymmetric_false_for_180_symmetric():
    # A parallelogram has 180-degree rotational symmetry -> not asymmetric.
    assert tc.is_asymmetric([(0, 0), (3, 0), (4, 2), (1, 2)]) is False


# --------------------------------------------------------------------------------------
# recover_map (uniqueness / identifiability)
# --------------------------------------------------------------------------------------

def test_recover_map_returns_intended_net_map():
    original = [(0, 0), (3, 0), (3, 1), (1, 1), (1, 2)]
    for seq in (
        [tc.rotate(90, "ccw"), tc.translate(2, 1)],
        [tc.reflect("y=x"), tc.translate(-3, 4)],
        [tc.translate(2, -1), tc.rotate(270, "ccw")],
        [tc.rotate(90, "ccw"), tc.reflect("x")],
    ):
        image = tc.compose(seq).apply(original)
        assert tc.recover_map(original, image) == tc.compose(seq)


def test_recover_map_none_when_not_isometry():
    original = [(0, 0), (3, 0), (3, 1), (1, 1), (1, 2)]
    scaled = [(2 * x, 2 * y) for x, y in original]  # not a lattice isometry
    assert tc.recover_map(original, scaled) is None


def test_recover_map_unique_for_asymmetric():
    original = [(0, 0), (3, 0), (3, 1), (1, 1), (1, 2)]
    assert tc.is_asymmetric(original)
    seq = [tc.reflect("y"), tc.translate(1, 1)]
    image = tc.compose(seq).apply(original)
    rec = tc.recover_map(original, image)
    # exactly one lattice isometry reproduces the image
    matches = [m for m in tc.ALL_LINEAR_MAPS
               if (lambda mm: all(
                   (tc.mat_vec(mm, p)[0] + (image[0][0] - tc.mat_vec(mm, original[0])[0]),
                    tc.mat_vec(mm, p)[1] + (image[0][1] - tc.mat_vec(mm, original[0])[1])) == q
                   for p, q in zip(original, image)))(m)]
    assert len(matches) == 1
    assert rec == tc.compose(seq)


# --------------------------------------------------------------------------------------
# Transform value semantics
# --------------------------------------------------------------------------------------

def test_transform_equality_is_structural_and_hashable():
    assert tc.rotate(90, "ccw") == tc.rotate(90, "ccw")
    assert tc.rotate(90, "ccw") == tc.rotate(270, "cw")
    assert len({tc.rotate(90, "ccw"), tc.rotate(270, "cw")}) == 1
    assert tc.translate(1, 2) != tc.translate(2, 1)


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        tc.rotate(45, "ccw")
    with pytest.raises(ValueError):
        tc.reflect("diagonal")
    with pytest.raises(ValueError):
        tc.parse_transform("frobnicate the shape")
