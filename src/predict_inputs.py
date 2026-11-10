"""Sequence/FASTA/UniProt input for local prediction; independent of ChimeraX."""
from datetime import datetime, timezone
import hashlib
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .predict import chain_ids, validate_chains

_ACCESSION = r'(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-[1-9][0-9]*)?'
_IDENTIFIER = re.compile(r'(?:' + _ACCESSION + r'|[A-Z0-9]{1,15}_[A-Z0-9]{1,10})\Z')


def parse_fasta(text):
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            label = line[1:].strip()
            if not label:
                raise ValueError('Give each FASTA entry a name after >.')
            records.append(dict(label=label, sequence='', input_kind='fasta'))
        elif not records:
            raise ValueError('FASTA must start with a >name line.')
        else:
            records[-1]['sequence'] += ''.join(line.split()).upper()
    if not records or any(not r['sequence'] for r in records):
        raise ValueError('Each FASTA entry needs a nonempty protein sequence.')
    return records


def fetch_uniprot(identifier):
    """Retrieve a single named entry/isoform; never search an ambiguous gene symbol."""
    identifier = identifier.strip().upper()
    if not _IDENTIFIER.fullmatch(identifier):
        raise ValueError('Use a UniProt accession (P69905), isoform (P05067-2), or entry ID (HBA_HUMAN).')
    url = f'https://rest.uniprot.org/uniprotkb/{identifier}.fasta'
    request = Request(url, headers={'User-Agent': 'AlphaLens-local-prediction/0.11', 'Accept': 'text/plain'})
    try:
        with urlopen(request, timeout=20) as response:
            body = response.read(1024 * 1024 + 1)
            release = response.headers.get('X-UniProt-Release')
    except HTTPError as e:
        if e.code in (400, 404, 410):
            raise ValueError(f'UniProt could not find {identifier}. Check the accession or isoform ID.') from e
        raise ValueError(f'UniProt returned HTTP {e.code}; try again or paste a sequence.') from e
    except (URLError, TimeoutError) as e:
        raise ValueError(f'Could not reach UniProt for {identifier}; try again or paste a sequence. {e}') from e
    if len(body) > 1024 * 1024:
        raise ValueError('UniProt returned an unexpectedly large entry.')
    records = parse_fasta(body.decode('utf-8'))
    if len(records) != 1:
        raise ValueError('UniProt did not return exactly one sequence for that identifier.')
    record = records[0]
    parts = record['label'].split('|', 2)
    record.update(input_kind='uniprot', requested_id=identifier,
                  accession=parts[1] if len(parts) == 3 else identifier,
                  url=url, release=release, fetched_at=datetime.now(timezone.utc).isoformat())
    if '-' in identifier and record['accession'] != identifier:
        raise ValueError(f'UniProt did not return the requested isoform {identifier}.')
    return record


def resolve_entries(text, fetcher=None):
    """One raw sequence, FASTA entries, or whitespace-separated UniProt IDs."""
    text = text.strip()
    if not text:
        raise ValueError('Paste a protein sequence, FASTA entries, or UniProt IDs.')
    if len(text) > 1024 * 1024:
        raise ValueError('Input is too large for this prototype.')
    if text.startswith('>'):
        entries = parse_fasta(text)
    else:
        words = text.upper().split()
        if all(_IDENTIFIER.fullmatch(w) for w in words):
            chain_ids(len(words))
            fetch = fetcher or fetch_uniprot
            entries = [fetch(w) for w in words]
        else:
            sequence = ''.join(words)
            if not sequence.isalpha() or not sequence.isascii():
                raise ValueError('Use a raw amino-acid sequence, FASTA, or valid UniProt IDs. Use FASTA for multiple sequences.')
            entries = [dict(label='sequence_A', sequence=sequence, input_kind='sequence')]
    chain_ids(len(entries))
    return entries


def sequence_chains(entries, ranges=None):
    """Apply optional 1-based inclusive ranges and record original sequence positions."""
    chains, provenance = [], []
    ranges = list(ranges or [])
    if any(interval.strip() for interval in ranges[len(entries):]):
        raise ValueError('A range was supplied for a missing chain.')
    ranges += [''] * max(0, len(entries)-len(ranges))
    offset = 0
    for cid, entry, interval in zip(chain_ids(len(entries)), entries, ranges):
        seq = entry['sequence']
        start, end = 1, len(seq)
        if interval.strip():
            match = re.fullmatch(r'\s*(\d+)\s*-\s*(\d+)\s*', interval)
            if not match:
                raise ValueError(f'Chain {cid}: enter a range such as 25-120, or leave it blank.')
            start, end = map(int, match.groups())
            if not 1 <= start <= end <= len(seq):
                raise ValueError(f'Chain {cid}: the range must be within 1-{len(seq)}.')
        mapping = [dict(chain_id=cid, number=i+1, insertion_code='', chain_index=i,
                        aa=seq[i], source_row=offset+i) for i in range(start-1, end)]
        chains.append(dict(source_chain=cid, sequence=seq[start-1:end], residue_map=mapping,
                           core_rows=[r['source_row'] for r in mapping]))
        info = {k: v for k, v in entry.items() if k != 'sequence'}
        info.update(chain_id=cid, full_length=len(seq), range=[start, end],
                    full_sequence_sha256=hashlib.sha256(seq.encode()).hexdigest())
        provenance.append(info)
        offset += len(seq)
    validate_chains(chains)
    return chains, dict(input_mode='sequence', inputs=provenance)
