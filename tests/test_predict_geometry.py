import numpy as np
import pytest
from alphalens.predict_geometry import check_ca_geometry


def test_normal_backbone_and_separate_chain_boundaries():
    xyz = [[0,0,0],[3.8,0,0],[7.6,0,0],[25,0,0],[28.8,0,0]]
    result = check_ca_geometry(xyz, list('AAABB'))
    assert result['ca_pairs_under_2A'] == result['adjacent_ca_outliers'] == 0
    assert result['adjacent_ca_median'] == pytest.approx(3.8)


def test_collapsed_or_broken_coordinates_fail_even_with_high_confidence():
    with pytest.raises(ValueError, match='collapsed or broken'):
        check_ca_geometry(np.zeros((20,3)), ['A']*20)
    with pytest.raises(ValueError, match='geometry checks'):
        check_ca_geometry([[i*8,0,0] for i in range(20)], ['A']*20)
