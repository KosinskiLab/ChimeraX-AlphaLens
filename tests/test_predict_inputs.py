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


@pytest.mark.parametrize('text', ['', '>empty\n', '>a\nAA\n>b\nAA\n>c\nAA', 'P69905 P68871 P00648', 'TP53', 'https://example.org'])
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


@pytest.mark.parametrize('interval', ['0-3', '3-2', '1-999', '1:3', '-3'])
def test_bad_ranges(interval):
    with pytest.raises(ValueError):
        inputs.sequence_chains(inputs.resolve_entries('ACDE'), [interval, ''])


def test_large_uniprot_sequence_can_be_cropped_before_limit_check():
    entries = [dict(label='large', sequence='A'*800, input_kind='uniprot')]
    with pytest.raises(ValueError, match='400'):
        inputs.sequence_chains(entries)
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
