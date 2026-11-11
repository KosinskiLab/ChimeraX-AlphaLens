import hashlib
from urllib.error import HTTPError

import pytest
from alphalens import predict_inputs as inputs
from alphalens.predict import PredictionJob, MONOMER_MODEL


def test_raw_sequence_and_fasta_preserve_positions():
    entries = inputs.resolve_entries('acde\nfghi')
    chains, source = inputs.sequence_chains(entries, ['3-7', ''])
    assert chains[0]['sequence'] == 'DEFGH'
    assert [r['number'] for r in chains[0]['residue_map']] == [3, 4, 5, 6, 7]
    assert source['inputs'][0]['full_sequence_sha256'] == hashlib.sha256(b'ACDEFGHI').hexdigest()
    entries = inputs.resolve_entries('>first\nACDE\n>second\nFGHI\n')
    chains, _ = inputs.sequence_chains(entries)
    assert [c['sequence'] for c in chains] == ['ACDE', 'FGHI']
    assert chains[1]['residue_map'][0]['source_row'] == 4


@pytest.mark.parametrize('text', ['', '>empty\n', 'TP53', 'https://example.org'])
def test_invalid_input(text):
    with pytest.raises(ValueError):
        inputs.resolve_entries(text)


def test_uniprot_accessions_isoforms_and_entry_ids():
    seen = []
    def fetch(identifier):
        seen.append(identifier)
        return dict(label=identifier, sequence='ACDE', input_kind='uniprot')
    assert len(inputs.resolve_entries('p05067-2 HBA_HUMAN', fetch)) == 2
    assert seen == ['P05067-2', 'HBA_HUMAN']
    inputs.resolve_entries('A0A024RBG1', fetch)
    assert seen[-1] == 'A0A024RBG1'


def test_multiple_chains_from_fasta_and_uniprot():
    entries = inputs.resolve_entries('>a\nACDE\n>b\nFGHI\n>c\nKLMN')
    chains, _ = inputs.sequence_chains(entries, ['', '2-3', '3-4'])
    assert [c['sequence'] for c in chains] == ['ACDE', 'GH', 'MN']
    assert [c['source_chain'] for c in chains] == list('ABC')
    assert chains[2]['residue_map'][0]['source_row'] == 10
    seen = []
    def fetch(identifier):
        seen.append(identifier)
        return dict(label=identifier, sequence='ACDE', input_kind='uniprot')
    assert len(inputs.resolve_entries('P69905 P68871 P00648', fetch)) == 3
    assert seen == ['P69905', 'P68871', 'P00648']


@pytest.mark.parametrize('interval', ['0-3', '3-2', '1-999', '1:3', '-3'])
def test_bad_ranges(interval):
    with pytest.raises(ValueError):
        inputs.sequence_chains(inputs.resolve_entries('ACDE'), [interval, ''])


def test_large_uniprot_sequence_is_allowed_and_can_be_cropped():
    entries = [dict(label='large', sequence='A'*1800, input_kind='uniprot')]
    assert len(inputs.sequence_chains(entries)[0][0]['sequence']) == 1800
    assert len(inputs.sequence_chains(entries, ['500-700', ''])[0][0]['sequence']) == 201
    with pytest.raises(ValueError, match='missing chain'):
        inputs.sequence_chains(entries, ['1-5', '1-5'])


def test_uniprot_not_found_and_identifier_validation(monkeypatch):
    def missing(*args, **kw):
        raise HTTPError('url', 404, 'not found', {}, None)
    monkeypatch.setattr(inputs, 'urlopen', missing)
    with pytest.raises(ValueError, match='could not find P69905'):
        inputs.fetch_uniprot('P69905')
    with pytest.raises(ValueError, match='accession'):
        inputs.fetch_uniprot('../../file')


def test_uniprot_fasta_metadata(monkeypatch):
    class Response:
        headers = {'X-UniProt-Release': '2026_05'}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, n): return b'>sp|P69905|HBA_HUMAN Alpha globin\nACDE\n'
    seen = []
    def open_(request, timeout):
        seen.append((request.full_url, timeout))
        return Response()
    monkeypatch.setattr(inputs, 'urlopen', open_)
    entry = inputs.fetch_uniprot('hba_human')
    assert entry['accession'] == 'P69905'
    assert entry['requested_id'] == 'HBA_HUMAN'
    assert entry['release'] == '2026_05'
    assert seen == [('https://rest.uniprot.org/uniprotkb/HBA_HUMAN.fasta', 20)]


def test_monomer_uses_monomer_model(tmp_path):
    chains, source = inputs.sequence_chains(inputs.resolve_entries('ACDE'))
    executable = tmp_path/'engine'; executable.write_text('#!/bin/sh\nexit 0\n'); executable.chmod(0o755)
    job = PredictionJob(tmp_path/'job', chains, source=source, executable=executable, search=True)
    assert job.model == MONOMER_MODEL


def test_alphapulldown_notation_sets_copies_and_ranges():
    seen = []
    def fetch(identifier):
        seen.append(identifier)
        return dict(label=identifier, sequence='ACDEFGHIKL', input_kind='uniprot', accession=identifier)
    entries = inputs.resolve_entries('P0A836:2:3-6+P0AGE9', fetch)
    assert seen == ['P0A836', 'P0AGE9']                # one lookup for both copies
    assert [(e['accession'], e.get('range')) for e in entries] == [('P0A836', '3-6'), ('P0A836', '3-6'),
                                                                   ('P0AGE9', None)]
    chains, source = inputs.sequence_chains(entries, [e.get('range', '') for e in entries])
    assert [c['sequence'] for c in chains] == ['DEFG', 'DEFG', 'ACDEFGHIKL']
    assert [r['number'] for r in chains[1]['residue_map']] == [3, 4, 5, 6]
    assert source['inputs'][1]['range'] == [3, 6]
    assert inputs.parse_alphapulldown('q8i2g6:2:1-100:150-200 P69905') == [
        ('Q8I2G6', 2, [(1, 100), (150, 200)]), ('P69905', 1, [])]


@pytest.mark.parametrize('text', ['P0A836:0', 'P0A836:5-2', 'P0A836:1-5:2', 'TP53:1-10', 'P0A836:x'])
def test_alphapulldown_notation_errors(text):
    with pytest.raises(ValueError):
        inputs.parse_alphapulldown(text)


def test_several_ranges_make_one_gapped_chain():
    chains, source = inputs.sequence_chains(inputs.resolve_entries('ACDEFGHIKL'), ['1-3,6-8'])
    assert chains[0]['sequence'] == 'ACDGHI'
    assert chains[0]['residue_ids'] == [1, 2, 3, 6, 7, 8]
    assert [r['number'] for r in chains[0]['residue_map']] == [1, 2, 3, 6, 7, 8]
    assert source['inputs'][0]['ranges'] == [[1, 3], [6, 8]] and source['inputs'][0]['range'] == [1, 8]
    joined, _ = inputs.sequence_chains(inputs.resolve_entries('ACDEFGHIKL'), ['1-3:4-6'])
    assert 'residue_ids' not in joined[0]                    # adjacent ranges are one segment
    for bad in ('5-8,1-3', '1-5,4-8', '1-3,x'):
        with pytest.raises(ValueError):
            inputs.sequence_chains(inputs.resolve_entries('ACDEFGHIKL'), [bad])
