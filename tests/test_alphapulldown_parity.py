"""AlphaPulldown's region and numbering scenarios, as AlphaLens handles them.

Each test names the AlphaPulldown test (KosinskiLab/AlphaPulldown, test/) or README example it
mirrors. Where AlphaLens differs on purpose, the test says so: it keeps the regions of a chain in
sequence order without overlap, and reads gaps of unknown size as far apart.
"""
import json

import pytest
from alphalens import predict_inputs as inputs
from alphalens import predict_features as pf
from alphalens.predict import (crop_a3m, crop_chains, duplicate_insertion_codes, engine_numbers,
                               renumber_outputs, segment_count, validate_chains)

SEQUENCE = 'ACDEFGHIKLMNPQRSTVWY' * 15          # 300 residues


def fetch(identifier):
    return dict(label=identifier, sequence=SEQUENCE, input_kind='uniprot', accession=identifier)


def resolve(text):
    entries = inputs.resolve_entries(text, fetch)
    return inputs.sequence_chains(entries, [e.get('range', '') for e in entries])[0]


# ---- Input notation: README "Advanced protein specification options" and the manual's lists

@pytest.mark.parametrize('text, expected', [
    ('Q8I2G6:1-100', [('Q8I2G6', 1, [(1, 100)])]),
    ('Q8I2G6:1-100:150-200', [('Q8I2G6', 1, [(1, 100), (150, 200)])]),
    ('Q8I2G6:2', [('Q8I2G6', 2, [])]),
    ('Q8I2G6:2:1-100+Q8I5K4', [('Q8I2G6', 2, [(1, 100)]), ('Q8I5K4', 1, [])]),
    ('Q8I2G6:2:1-100:150-200+Q8I5K4', [('Q8I2G6', 2, [(1, 100), (150, 200)]), ('Q8I5K4', 1, [])]),
    # The manual's protein lists: "," between fields, ";" between proteins
    ('P0A836,4,1-100;P0AGE9', [('P0A836', 4, [(1, 100)]), ('P0AGE9', 1, [])]),
    ('P0AGE9,2;P0A836', [('P0AGE9', 2, []), ('P0A836', 1, [])]),
])
def test_readme_and_manual_notation(text, expected):
    assert inputs.parse_alphapulldown(text) == expected


def test_copies_of_discontinuous_regions_are_gapped_chains_of_one_entity():
    """test_af3_keeps_discontinuous_chopped_regions_in_one_gapped_chain, with two copies."""
    chains = resolve('Q8I2G6:2:1-100:150-200+Q8I5K4')
    assert len(chains) == 3
    ids = list(range(1, 101)) + list(range(150, 201))
    assert chains[0]['residue_ids'] == chains[1]['residue_ids'] == ids
    assert chains[0]['sequence'] == chains[1]['sequence'] == SEQUENCE[:100] + SEQUENCE[149:200]
    assert [segment_count(c) for c in chains] == [2, 2, 1] and 'residue_ids' not in chains[2]
    assert engine_numbers(chains[2]) == list(range(1, 301))


@pytest.mark.parametrize('text', ['P0A836+A0A075B6L2:1-10:2-5:12-15', 'A0A075B6L2:2:8-10:2-5'])
def test_overlapping_or_out_of_order_regions_are_refused_with_the_reason(text):
    """AlphaPulldown folds these (test_af3_keeps_two_out_of_order_gapped_copies_as_two_chains:
    residue ids 8, 9, 10, 2, 3, 4, 5; ..._discontinuous_chopped_regions_...: 1-10, 2-5, 12-15).
    AlphaLens keeps residue numbers increasing along a chain and says how to fix the input."""
    with pytest.raises(ValueError, match='in sequence order, without overlap'):
        inputs.parse_alphapulldown(text)


def test_identifiers_that_are_not_uniprot_are_refused():
    """AlphaPulldown names feature files freely (TEST, proteinA); AlphaLens looks names up in UniProt."""
    with pytest.raises(ValueError, match='UniProt accession'):
        inputs.parse_alphapulldown('proteinA:1-10+proteinB')


# ---- Alignments: test_a3m_region_slice_keeps_only_the_insertions_inside_it

def test_a3m_crop_keeps_only_the_insertions_inside_the_region(tmp_path):
    text = '>q\nACDF\n>h\nAbCDeF\n'
    # Lowercase letters are insertions between two aligned residues: kept only when both are kept
    assert pf.crop_a3m_text(text, 0, 3, 'ACDF') == '>q\nACD\n>h\nAbCD\n'     # AlphaPulldown: "AbCD"
    assert pf.crop_a3m_text(text, 1, 2, 'ACDF') == '>q\nCD\n>h\nCD\n'        # AlphaPulldown: "CD"
    # Two regions: each keeps its own insertions, none across the gap
    assert pf.crop_a3m_text(text, [0, 1, 3], None, 'ACDF') == '>q\nACF\n>h\nAbCF\n'
    source = tmp_path / 'source.a3m'
    source.write_text(text)
    crop_a3m(source, tmp_path / 'cut.a3m', 'ACDF', [0, 1, 3])                # The source-MSA crop agrees
    assert (tmp_path / 'cut.a3m').read_text() == '>q\nACF\n>h\nAbCF\n'


# ---- Re-predicting from an AlphaPulldown model of discontinuous regions

def model_rows(numbers, breaks, chain='A'):
    """A chain of an AlphaPulldown model: no unresolved residues, gaps only where breaks are."""
    rows = [dict(chain_id=chain, chain_index=k, aa='A', number=n, insertion_code='', break_before=k in breaks)
            for k, n in enumerate(numbers)]
    partner = [dict(chain_id='B', chain_index=k, aa='A', number=k + 1, insertion_code='', break_before=False)
               for k in range(6)]
    return rows + partner


def test_repredicting_overlapping_regions_keeps_order_and_tells_repeats_apart(tmp_path):
    """An AlphaPulldown model of A0A075B6L2:1-10:2-5:12-15 numbers its residues 1-10, 2-5, 12-15.
    The engine gets increasing residue numbers (the repeat as far apart, the last gap its size);
    the written structure keeps the model's numbers, repeats with insertion codes as
    AlphaPulldown's AF3 output does (test_af3_viewer_output_uses_insertion_codes_for_duplicate_residue_ids)."""
    numbers = list(range(1, 11)) + list(range(2, 6)) + list(range(12, 16))
    rows = model_rows(numbers, breaks={10, 14})
    chain = crop_chains(rows, (0, len(numbers) - 1, len(numbers), len(numbers) + 5), 0)[0]
    assert chain['residue_ids'] == list(range(1, 11)) + [43, 44, 45, 46] + [53, 54, 55, 56]
    validate_chains([chain])
    pdb = ''.join(f'ATOM  {k + 1:5d}  CA  ALA A{n:4d}    {k * 3.:8.3f}{0.:8.3f}{0.:8.3f}{1.:6.2f}{90.:6.2f}           C\n'
                  for k, n in enumerate(chain['residue_ids'])) + 'END\n'
    (tmp_path / 'fold.pdb').write_text(pdb)
    assert renumber_outputs(tmp_path, [chain])['scheme'] == 'source'
    labels = [(l[22:26].strip() + l[26].strip()) for l in (tmp_path / 'fold.pdb').read_text().splitlines()
              if l.startswith('ATOM')]
    assert labels == [str(n) for n in range(1, 11)] + ['2A', '3A', '4A', '5A'] + [str(n) for n in range(12, 16)]


def test_repredicting_out_of_order_regions_reads_the_gap_as_far_apart():
    """A model of A0A075B6L2:8-10:2-5 (AlphaPulldown residue ids 8, 9, 10, 2, 3, 4, 5): AlphaLens
    keeps the residue order and puts the segments beyond the relative-position range (33)."""
    chain = crop_chains(model_rows([8, 9, 10, 2, 3, 4, 5], breaks={3}), (0, 6, 7, 12), 0)[0]
    assert chain['residue_ids'] == [1, 2, 3, 36, 37, 38, 39]
    assert [r['number'] for r in chain['residue_map']] == [8, 9, 10, 2, 3, 4, 5]


def test_insertion_codes_for_repeats_stop_after_z():
    """test_duplicate_occurrence_and_author_id_helpers_handle_overflow: 2nd -> A, 27th -> Z."""
    labels = duplicate_insertion_codes([(10, '')] * 27)
    assert [c for _, c in labels] == [''] + [chr(ord('A') + k) for k in range(26)]
    assert duplicate_insertion_codes([(10, '')] * 28) is None
    assert duplicate_insertion_codes([(10, 'A'), (10, '')]) == [(10, 'A'), (10, '')]   # Source codes kept


# ---- Precomputed features: a heteromer of a full chain and a gapped chain, and gapped homomers

def test_gapped_homomer_features_are_not_paired(tmp_path):
    """AlphaFold-Multimer pairs heteromers only; two copies of one gapped region are a homomer
    (test_multimeric_object_pairs_full_then_chopped_with_correct_features covers the heteromer)."""
    from test_predict_features import lab_database
    root = lab_database(tmp_path)
    chain = dict(name='P11111', sequence='MKAW', start_hint=0, segments=[['MK', 0], ['AW', 4]])
    args = pf.prepare(dict(directory=str(tmp_path / 'out'), root=str(root), backend='af2', job=None,
                           chains=[dict(chain, chain_id='A'), dict(chain, chain_id='B')]))
    assert not any(a.startswith('--paired-a3m') for a in args)
    for cid in 'AB':
        assert (tmp_path / 'out' / f'chain_{cid}.a3m').read_text().split('\n')[1] == 'MKAW'
    provenance = json.loads((tmp_path / 'out' / 'features.json').read_text())
    assert [p['segments'] for p in provenance] == [[[1, 2], [5, 6]]] * 2
