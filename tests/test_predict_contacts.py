import csv
import numpy as np
import pytest

from alphalens.predict import crop_chains
from alphalens.predict_contacts import export, region_pairs, stats, summary
from alphalens.predict_inputs import resolve_entries, sequence_chains


def test_pairs_match_original_rectangle_with_multiple_chains_and_overlapping_axes():
    rows = [dict(chain_id=c, chain_index=i, aa='A', number=i+10, insertion_code='')
            for c in 'ABC' for i in range(4)]
    region = (2, 6, 5, 10)
    chains = crop_chains(rows, region, 1)
    mapping = [r['source_row'] for c in chains for r in c['residue_map']]
    observed = {tuple(sorted((mapping[i], mapping[j]))) for i, j in region_pairs(chains, region)}
    expected = {tuple(sorted((i, j))) for i in range(2, 7) for j in range(5, 11)
                if rows[i]['chain_id'] != rows[j]['chain_id']}
    assert observed == expected
    assert len(region_pairs(chains, region)) == len(expected)


def test_contact_csv_contains_union_and_original_residue_labels(tmp_path):
    chains, _ = sequence_chains(resolve_entries('>a\nACD\n>b\nEFG'), ['2-3', '2-3'])
    pairs = region_pairs(chains)
    source = dict(probability=[.8, .1, .2, .1], cutoff_A=8., distance_A=[10., 12., 6., 15.])
    predicted = dict(probability=[.2, .9, .3, .1], cutoff_A=8., distance_A=[11., 12., 10., 15.])
    report = export(tmp_path, chains, pairs, predicted, source)
    assert report['pairs_evaluated'] == 4 and report['csv_rows'] == 3
    assert report['source']['count'] == report['prediction']['count'] == 1
    assert report['strongest'][0] == dict(pair='A:2–B:3', probability=.9)
    with (tmp_path/'region_contacts.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert [(r['residue_1'], r['residue_2']) for r in rows] == [('2','2'), ('2','3'), ('3','2')]
    assert rows[0]['source_probability'] == '0.8' and rows[0]['prediction_probability'] == '0.2'
    assert 'Top-10 mean contact P: 0.300 → 0.375' in summary(report)


def test_missing_probabilities_remain_unavailable_not_zero(tmp_path):
    chains, _ = sequence_chains(resolve_entries('>a\nA\n>b\nA'))
    values = dict(probability=None, cutoff_A=None, distance_A=[4.])
    report = export(tmp_path, chains, region_pairs(chains), values)
    assert report['prediction']['count'] is None
    assert report['prediction']['top10_mean'] is None
    assert report['prediction']['structural_count'] == 1
    assert report['source']['structural_count'] is None
    assert 'unavailable' in summary(report)
    with (tmp_path/'region_contacts.csv').open() as stream:
        row = next(csv.DictReader(stream))
    assert row['prediction_probability'] == ''
    assert stats(dict(values, probability=[0.], cutoff_A=8), .5)['count'] == 0


def test_snapshot_size_mismatch_is_rejected(tmp_path):
    chains, _ = sequence_chains(resolve_entries('>a\nA\n>b\nA'))
    with pytest.raises(ValueError, match='snapshot'):
        export(tmp_path, chains, region_pairs(chains), dict(probability=None, distance_A=[]))
