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
    request = Request(url, headers={'User-Agent': 'AlphaLens-local-prediction', 'Accept': 'text/plain'})
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


def parse_alphapulldown(text):
    """Chains in AlphaPulldown's fold notation, as [(identifier, copies, [(start, end), ...])].

    ID[:copies][:start-end[:start-end...]], chains joined by "+" (or whitespace): P69905:2,
    P0A836:1-100+P0AGE9, Q8I2G6:2:1-100:150-200. Ranges are 1-based and inclusive.
    """
    text = text.strip()
    if ';' in text or (',' in text and ':' not in text):
        # AlphaPulldown's older protein lists: proteinA,4,1-100;proteinB
        text = text.replace(';', '+').replace(',', ':')
    chains = []
    for token in re.split(r'[\s+]+', text.upper()):
        if not token:
            continue
        identifier, *parts = token.split(':')
        if not _IDENTIFIER.fullmatch(identifier):
            raise ValueError(f'{identifier or token}: use a UniProt accession or entry ID in '
                             'AlphaPulldown notation, such as P0A836:1-100+P0AGE9.')
        copies, ranges = 1, []
        for i, part in enumerate(parts):
            interval = re.fullmatch(r'(\d+)-(\d+)', part)
            if interval:
                start, end = map(int, interval.groups())
                if not 1 <= start <= end:
                    raise ValueError(f'{token}: a range runs from a residue to a later one, such as 25-120.')
                ranges.append((start, end))
            elif part.isdigit() and i == 0 and int(part) >= 1:
                copies = int(part)
            else:
                raise ValueError(f'{token}: after the ID come the number of copies (ID:2) and/or '
                                 'ranges (ID:25-120:150-200).')
        for (a, b), (c, d) in zip(ranges, ranges[1:]):
            if c <= b:
                raise ValueError(f'{token}: AlphaLens takes the regions of a chain in sequence order, without '
                                 f'overlap ({a}-{b} then {c}-{d}). AlphaPulldown also folds regions out of order '
                                 'or overlapping, with repeated or decreasing residue numbers; reorder or merge them.')
        chains.append((identifier, copies, ranges))
    return chains


def resolve_entries(text, fetcher=None):
    """One raw sequence, FASTA entries, whitespace-separated UniProt IDs, or AlphaPulldown notation."""
    text = text.strip()
    if not text:
        raise ValueError('Paste a protein sequence, FASTA entries, or UniProt IDs.')
    if len(text) > 1024 * 1024:
        raise ValueError('Input is too large for this prototype.')
    if text.startswith('>'):
        entries = parse_fasta(text)
    elif any(c in text for c in ':+;') or (',' in text and not text.replace(',', '').replace(' ', '').isalpha()):
        specs = parse_alphapulldown(text)
        chain_ids(sum(copies for _, copies, _ in specs))
        fetch = fetcher or fetch_uniprot
        fetched, entries = {}, []
        for identifier, copies, ranges in specs:
            if identifier not in fetched:
                fetched[identifier] = fetch(identifier)
            for _ in range(copies):
                entry = dict(fetched[identifier])
                if ranges:
                    entry['range'] = ','.join(f'{a}-{b}' for a, b in ranges)
                entries.append(entry)
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


def parse_ranges(interval, length, cid):
    """'25-120' or several, '1-100,150-200' (also ':' or spaces between them): [(start, end)], 1-based."""
    text = interval.strip()
    if not text:
        return [(1, length)]
    parts = [p for p in re.split(r'[\s,;:]+', text) if p]
    ranges = []
    for part in parts:
        match = re.fullmatch(r'(\d+)-(\d+)', part)
        if not match:
            raise ValueError(f'Chain {cid}: enter a range such as 25-120 (several as 1-100,150-200), '
                             'or leave it blank.')
        a, b = map(int, match.groups())
        if not 1 <= a <= b <= length:
            raise ValueError(f'Chain {cid}: the range must be within 1-{length}.')
        if ranges and a <= ranges[-1][1]:
            raise ValueError(f'Chain {cid}: ranges must follow one another along the sequence without overlap.')
        ranges.append((a, b))
    return ranges


def sequence_chains(entries, ranges=None):
    """Apply optional 1-based inclusive ranges and record original sequence positions.

    Several ranges in one chain make one chain with gaps (residue_ids: the positions), as
    AlphaPulldown predicts discontinuous regions; adjacent ranges simply join.
    """
    chains, provenance = [], []
    ranges = list(ranges or [])
    if any(interval.strip() for interval in ranges[len(entries):]):
        raise ValueError('A range was supplied for a missing chain.')
    ranges += [''] * max(0, len(entries)-len(ranges))
    offset = 0
    for cid, entry, interval in zip(chain_ids(len(entries)), entries, ranges):
        seq = entry['sequence']
        intervals = parse_ranges(interval, len(seq), cid)
        positions = [i for a, b in intervals for i in range(a - 1, b)]
        mapping = [dict(chain_id=cid, number=i+1, insertion_code='', chain_index=i,
                        aa=seq[i], source_row=offset+i) for i in positions]
        chain = dict(source_chain=cid, sequence=''.join(seq[i] for i in positions), residue_map=mapping,
                     core_rows=[r['source_row'] for r in mapping])
        if any(b != a + 1 for a, b in zip(positions, positions[1:])):
            chain['residue_ids'] = [i + 1 for i in positions]
        chains.append(chain)
        info = {k: v for k, v in entry.items() if k not in ('sequence', 'range')}
        info.update(chain_id=cid, full_length=len(seq), range=[intervals[0][0], intervals[-1][1]],
                    full_sequence_sha256=hashlib.sha256(seq.encode()).hexdigest())
        if len(intervals) > 1:
            info['ranges'] = [list(r) for r in intervals]
        provenance.append(info)
        offset += len(seq)
    validate_chains(chains)
    return chains, dict(input_mode='sequence', inputs=provenance)
