"""Distance kernels: textbook formulas, and the ordering/score contract."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyvec.core.distance import (
    as_matrix,
    as_vector,
    cosine_similarity,
    distance,
    distance_from_score,
    dot_product,
    l2_distance,
    normalize,
    pairwise_distance,
    score_from_distance,
    squared_norms,
)
from pyvec.core.errors import InvalidDimensionError
from pyvec.core.types import Metric


class TestRawMetrics:
    def test_cosine_of_identical_vectors_is_one(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(a, a) == pytest.approx(1.0, abs=1e-6)

    def test_cosine_of_orthogonal_is_zero(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_cosine_of_opposite_is_minus_one(self):
        a = np.array([1.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, -a) == pytest.approx(-1.0, abs=1e-6)

    def test_cosine_ignores_magnitude(self):
        a = np.array([1.0, 2.0], dtype=np.float32)
        assert cosine_similarity(a, a * 100) == pytest.approx(1.0, abs=1e-6)

    def test_dot_product_matches_manual_sum(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
        assert dot_product(a, b)[0] == pytest.approx(32.0)

    def test_l2_is_pythagorean(self):
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([[3.0, 4.0]], dtype=np.float32)
        assert l2_distance(a, b)[0] == pytest.approx(5.0)
        assert l2_distance(a, b, squared=True)[0] == pytest.approx(25.0)

    def test_l2_of_self_is_zero_to_within_float32_cancellation(self):
        """Pins the documented accuracy bound of the expanded L2 form.

        ``|x|^2 - 2x.q + |q|^2`` cancels badly when the true distance is small
        relative to the norms, leaving error of order ``||x|| * sqrt(eps_f32)``.
        Asserting exact zero here would be asserting something false; asserting
        this bound catches a genuine regression (e.g. losing the ``maximum(0)``
        clamp, or the terms drifting out of sync) while documenting the real
        behaviour of the reporting path.
        """
        rng = np.random.default_rng(0)
        x = rng.normal(size=(20, 16)).astype(np.float32)
        eps = float(np.finfo(np.float32).eps)
        for row in x:
            bound = float(np.linalg.norm(row)) * math.sqrt(eps) * 4
            assert 0.0 <= l2_distance(row, row.reshape(1, -1))[0] <= bound

    def test_l2_squared_of_self_is_tiny(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(20, 16)).astype(np.float32)
        for row in x:
            assert l2_distance(row, row.reshape(1, -1), squared=True)[0] < 1e-5

    def test_l2_is_never_negative(self):
        """Clamping matters: sqrt of a negative would produce NaN."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=(20, 16)).astype(np.float32)
        assert np.all(l2_distance(x[0], x, squared=True) >= 0.0)
        assert np.all(np.isfinite(l2_distance(x[0], x)))

    def test_ordering_path_error_is_bounded(self):
        """`distance()` stays in float32 and is allowed to be slightly off, but
        only by far less than any gap that could reorder two distinct vectors."""
        rng = np.random.default_rng(0)
        q = rng.normal(size=128).astype(np.float32)
        x = rng.normal(size=(50, 128)).astype(np.float32)
        fast = distance(Metric.L2, q, x)
        exact = l2_distance(q, x, squared=True)
        assert np.max(np.abs(fast - exact)) < 1e-2
        assert list(np.argsort(fast)) == list(np.argsort(exact))

    def test_matches_scipy_style_bruteforce(self):
        rng = np.random.default_rng(1)
        q = rng.normal(size=8).astype(np.float32)
        x = rng.normal(size=(30, 8)).astype(np.float32)
        expected = np.array([np.sqrt(np.sum((q - row) ** 2)) for row in x])
        np.testing.assert_allclose(l2_distance(q, x), expected, rtol=1e-4)


class TestNormalize:
    def test_produces_unit_vectors(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(10, 5)).astype(np.float32)
        np.testing.assert_allclose(
            np.linalg.norm(normalize(x), axis=1), 1.0, rtol=1e-5
        )

    def test_zero_vector_stays_zero_and_does_not_divide_by_zero(self):
        z = np.zeros((1, 4), dtype=np.float32)
        assert np.all(np.isfinite(normalize(z)))
        assert np.all(normalize(z) == 0.0)

    def test_cosine_equals_dot_after_normalizing(self):
        """LEARNING.md layer 0: the reason we normalise on insert."""
        rng = np.random.default_rng(3)
        a = normalize(rng.normal(size=6).astype(np.float32))
        b = normalize(rng.normal(size=(5, 6)).astype(np.float32))
        np.testing.assert_allclose(
            cosine_similarity(a, b), dot_product(a, b), rtol=1e-5
        )


class TestOrderingDistance:
    """`distance()` must always be lower-is-better, whatever the metric."""

    @pytest.mark.parametrize("metric", list(Metric))
    def test_nearest_vector_has_smallest_distance(self, metric):
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        x = np.array(
            [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        if metric is Metric.COSINE:
            x = normalize(x)
        d = distance(metric, q, x)
        assert np.argmin(d) == 0
        assert list(np.argsort(d)) == [0, 1, 2, 3]

    @pytest.mark.parametrize("metric", list(Metric))
    def test_score_round_trips_through_distance(self, metric):
        for d in (0.0, 0.25, 1.0, 2.0):
            score = score_from_distance(metric, d)
            assert distance_from_score(metric, score) == pytest.approx(d, abs=1e-6)

    def test_cosine_score_recovers_the_cosine(self):
        a = normalize(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        b = normalize(np.array([[3.0, 2.0, 1.0]], dtype=np.float32))
        d = distance(Metric.COSINE, a, b)[0]
        assert score_from_distance(Metric.COSINE, d) == pytest.approx(
            float(cosine_similarity(a, b)[0]), abs=1e-5
        )

    def test_l2_score_recovers_euclidean_distance(self):
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([[3.0, 4.0]], dtype=np.float32)
        d = distance(Metric.L2, a, b)[0]
        assert score_from_distance(Metric.L2, d) == pytest.approx(5.0, abs=1e-5)

    def test_higher_is_better_flag_matches_reality(self):
        assert Metric.COSINE.higher_is_better
        assert Metric.DOT.higher_is_better
        assert not Metric.L2.higher_is_better

    def test_accepts_a_single_1d_candidate(self):
        q = np.array([1.0, 0.0], dtype=np.float32)
        d = distance(Metric.L2, q, np.array([0.0, 1.0], dtype=np.float32))
        assert d.shape == (1,)


class TestSquaredNormFastPath:
    """The cache must be a pure optimisation — identical numbers, no drift."""

    def test_precomputed_norms_give_identical_results(self):
        rng = np.random.default_rng(4)
        q = rng.normal(size=16).astype(np.float32)
        x = rng.normal(size=(40, 16)).astype(np.float32)
        np.testing.assert_allclose(
            distance(Metric.L2, q, x),
            distance(Metric.L2, q, x, squared_norms(x)),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_squared_norms_matches_manual(self):
        x = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(squared_norms(x), [25.0, 1.0])


class TestPairwise:
    @pytest.mark.parametrize("metric", list(Metric))
    def test_shape_and_agreement_with_per_row_distance(self, metric):
        rng = np.random.default_rng(5)
        a = rng.normal(size=(6, 4)).astype(np.float32)
        b = rng.normal(size=(9, 4)).astype(np.float32)
        if metric is Metric.COSINE:
            a, b = normalize(a), normalize(b)
        m = pairwise_distance(metric, a, b)
        assert m.shape == (6, 9)
        for i in range(6):
            np.testing.assert_allclose(
                m[i], distance(metric, a[i], b), rtol=1e-4, atol=1e-5
            )

    def test_self_distance_diagonal_is_zero_for_l2(self):
        rng = np.random.default_rng(6)
        a = rng.normal(size=(8, 4)).astype(np.float32)
        m = pairwise_distance(Metric.L2, a, a)
        np.testing.assert_allclose(np.diag(m), 0.0, atol=1e-4)


class TestCoercion:
    def test_as_vector_flattens_and_casts(self):
        v = as_vector([[1, 2, 3]])
        assert v.shape == (3,) and v.dtype == np.float32

    def test_as_vector_enforces_dimension(self):
        with pytest.raises(InvalidDimensionError, match="expected dimension 4"):
            as_vector([1, 2, 3], dim=4)

    def test_as_matrix_promotes_1d(self):
        assert as_matrix([1, 2, 3]).shape == (1, 3)

    def test_as_matrix_enforces_dimension(self):
        with pytest.raises(InvalidDimensionError):
            as_matrix([[1, 2, 3]], dim=5)


class TestMetricParsing:
    @pytest.mark.parametrize("raw", ["cosine", "COSINE", "Cosine"])
    def test_parse_is_case_insensitive(self, raw):
        assert Metric.parse(raw) is Metric.COSINE

    def test_unknown_metric_is_rejected_with_the_api_error(self):
        from pyvec.core.errors import InvalidMetricError

        with pytest.raises(InvalidMetricError) as exc:
            Metric.parse("manhattan")
        assert exc.value.code == "INVALID_METRIC"
        assert exc.value.status == 400

    def test_only_cosine_normalizes_on_insert(self):
        assert Metric.COSINE.normalize_on_insert
        assert not Metric.L2.normalize_on_insert
        assert not Metric.DOT.normalize_on_insert
