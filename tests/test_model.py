"""Dimension bookkeeping only; no pricing or filtering assumptions."""

from dataclasses import FrozenInstanceError

import pytest

from kalmanfilter.model import ModelDimensions


def test_per_time_counts_follow_documented_state_decomposition():
    previous = ModelDimensions(n_p_t=3, n_c_t=2, n_u_t=4, n_z_t=3)
    current = ModelDimensions(n_p_t=3, n_c_t=1, n_u_t=4, n_z_t=0)

    assert (previous.n_s_t, previous.n_x_t) == (5, 9)
    assert (current.n_s_t, current.n_x_t) == (4, 8)
    assert current.n_z_t == 0


def test_empty_blocks_are_representable_without_prescribing_filter_behavior():
    dimensions = ModelDimensions(n_p_t=0, n_c_t=0, n_u_t=0, n_z_t=0)
    assert (dimensions.n_s_t, dimensions.n_x_t) == (0, 0)


@pytest.mark.parametrize("field", ["n_p_t", "n_c_t", "n_u_t", "n_z_t"])
@pytest.mark.parametrize(
    ("invalid", "error"), [(-1, ValueError), (1.5, TypeError), (True, TypeError)]
)
def test_invalid_dimension_counts_are_rejected(field, invalid, error):
    counts = dict(n_p_t=1, n_c_t=1, n_u_t=1, n_z_t=1)
    counts[field] = invalid
    with pytest.raises(error, match=field):
        ModelDimensions(**counts)


def test_dimension_metadata_is_immutable():
    dimensions = ModelDimensions(n_p_t=1, n_c_t=1, n_u_t=1, n_z_t=1)
    with pytest.raises(FrozenInstanceError):
        dimensions.n_p_t = 2
