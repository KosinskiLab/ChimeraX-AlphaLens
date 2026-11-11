"""Precomputed AlphaPulldown / AlphaFold features as the alignments of a local prediction.

No ChimeraX imports and no package-relative imports: the plugin imports this module to find
feature files, and the prediction process runs it as a script that converts them and then
execs LocalFold, so a slow read over a network mount never blocks the ChimeraX window:

    python -B predict_features.py <spec.json> -- <LocalFold command ...>

Formats, one file a protein:
  * AF2: a pickle of AlphaPulldown's MonomericObject (or ChoppedObject), whose feature_dict
    holds AlphaFold's per-chain features, or such a feature dict itself (AlphaFold's own
    features.pkl of a monomer). Plain, .xz, .gz or .bz2. Read with an unpickler that
    builds numpy arrays and stands in for the AlphaPulldown/AlphaFold classes; anything else
    in the pickle is refused, so neither package needs to be installed.
  * AF3: AlphaPulldown's <id>_af3_input.json (AlphaFold 3's data-pipeline JSON), whose
    unpairedMsa and pairedMsa go into the job inline (AF3 model only).

AF2 features are written as one A3M a chain (`msa`) plus, for a heteromer, row-aligned paired
A3Ms made from `msa_all_seq` by species, as AlphaFold-Multimer pairs them (msa_pairing.py);
LocalFold then merges, crops and deduplicates them as it does a ColabFold search. Templates are
not used, as with the search.
"""

import bz2
import gzip
import hashlib
import io
import json
import lzma
import os
from pathlib import Path
import pickle
import re
import sys
import time

import numpy as np

LAB_ROOT = '/g/alphafold/input_features'                   # EMBL, through the /g mount
PUBLIC_ROOT = 'https://s3.embl.de/alphapulldown'            # The same database, public
ROOT_VARIABLE = 'ALPHALENS_FEATURES_DIR'
AF2_SUFFIXES = ('.pkl.xz', '.pkl', '.pkl.gz', '.pkl.bz2')
AF3_SUFFIXES = ('_af3_input.json.xz', '_af3_input.json', '_af3_input.json.gz')
MAX_ROWS_PER_SPECIES = 600          # AlphaFold-Multimer skips species with more hits in a chain
# UniProt accession, optionally an isoform (https://www.uniprot.org/help/accession_numbers)
ACCESSION = re.compile(r'(?<![A-Z0-9])([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})'
                       r'(-\d+)?(?![A-Z0-9])')
# AlphaFold's MSA encodings: HHblits order (data pipeline) and residue_constants order
# (processed features); 20 = unknown, 21 = gap.
HHBLITS = 'ACDEFGHIKLMNPQRSTVWYX-'
RESTYPES = 'ARNDCQEGHILKMFPSTWYVX-'


def default_root():
    '''$ALPHALENS_FEATURES_DIR, else the lab copy when /g is mounted, else the public one.'''
    return os.environ.get(ROOT_VARIABLE, '') or (LAB_ROOT if mounted(LAB_ROOT) else PUBLIC_ROOT)


def mounted(path):
    '''
    Whether a mounted file system holds path, from the mount table, without touching the
    share: a stat of a stale network mount can hang for minutes, and this runs in the ChimeraX
    window. A top-level symlink (macOS /g -> /Users/<me>/g) is resolved locally.
    '''
    import subprocess
    top = '/' + path.strip('/').split('/')[0]
    if os.path.islink(top):         # lstat of a top-level entry is on the local disk
        target = os.readlink(top)
        path = (target if target.startswith('/') else '/' + target) + path[len(top):]
    try:
        table = subprocess.run(['mount'], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    for line in table.splitlines():
        m = re.search(r' on (/\S*(?: \S+)*?) (?:\(|type )', line)
        point = m.group(1) if m else None
        if point and point != '/' and (path == point or path.startswith(point.rstrip('/') + '/')):
            return True
    return False


def is_url(root):
    return str(root).startswith(('https://', 'http://'))


def cache_directory():
    return Path.home() / '.cache' / 'alphalens' / 'features'


# -----------------------------------------------------------------------------
# Finding feature files
# -----------------------------------------------------------------------------

def accessions_in(text):
    '''UniProt accessions in a name such as "Q15014+Q9NV56", "q15014_and_q9nv56" or
    "Q15014:385-460+P12345", in order (an isoform keeps its suffix).'''
    return [m.group(1) + (m.group(2) or '') for m in ACCESSION.finditer(text.upper())]


def guess_names(path, chain_ids):
    '''
    {chain id: UniProt ID} from a model's path, as AlphaPulldown and AlphaFold 3 name folds:
    ".../Q15014+Q9NV56/ranked_0.cif", ".../q15014_and_q9nv56_model.cif", ".../P1:20-80+P2/...".
    The nearest name with one ID a chain (or one ID for all) wins; {} if none fits.
    '''
    p = Path(path)
    for part in [p.name.split('.')[0]] + [q.name for q in list(p.parents)[:3]]:
        found = accessions_in(part)
        if len(found) == len(chain_ids):
            return dict(zip(chain_ids, found))
        if len(found) == 1:
            return {cid: found[0] for cid in chain_ids}
    return {}


def format_names(mapping):
    return ', '.join(f'{cid}: {name}' for cid, name in mapping.items())


def parse_chain_accessions(text, chain_ids):
    '''
    "A: P0A836, B: P0AGE9" (explicit) or "P0A836 P0AGE9" (in chain order) -> {chain id: id}.
    A single bare ID applies to every chain (a homo-oligomer).
    '''
    text = text.strip()
    if not text:
        return {}
    pairs = re.findall(r'([A-Za-z0-9]+)\s*[:=]\s*([A-Za-z0-9_.-]+)', text)
    if pairs:
        rest = re.sub(r'([A-Za-z0-9]+)\s*[:=]\s*([A-Za-z0-9_.-]+)', ' ', text)
        if rest.replace(',', ' ').replace(';', ' ').strip():
            raise ValueError('Write the feature IDs as "A: P0A836, B: P0AGE9" or as a list in chain order.')
        mapping = {}
        for cid, name in pairs:
            if cid not in chain_ids:
                raise ValueError(f'There is no chain {cid}; the chains are {", ".join(chain_ids)}.')
            mapping[cid] = name
        return mapping
    names = [w for w in re.split(r'[\s,;+]+', text) if w]
    if len(names) == 1:
        return {cid: names[0] for cid in chain_ids}
    if len(names) != len(chain_ids):
        raise ValueError(f'{len(names)} feature IDs for {len(chain_ids)} chains ({", ".join(chain_ids)}); '
                         'give one a chain, or write them as "A: ID, B: ID".')
    return dict(zip(chain_ids, names))


def feature_directories(root):
    '''Where feature files may be, AF2 and AF3 apart: the lab database's
    alphafold2|alphafold3/<organism>/features, AlphaPulldown's own output folders
    (<root>/<name>/ or <root> itself, both formats side by side).'''
    root = Path(root).expanduser()
    if not root.is_dir():
        raise ValueError(f'Feature folder {root} not found' +
                         (' (is /g mounted?).' if str(root).startswith('/g/') else '.'))
    af2 = sorted(root.glob('alphafold2/*/features'))
    af3 = sorted(root.glob('alphafold3/*/features'))
    if not af2 and not af3:          # An AlphaPulldown feature folder, perhaps with subfolders
        both = [root] + sorted(p for p in root.iterdir() if p.is_dir() and not p.name.endswith('_env'))
        return both, both
    return af2, af3


# -- The public database (https://s3.embl.de/alphapulldown) -----------------------------

REMOTE_LAYOUT = (('af2', 'input_features', '{}.pkl.xz'), ('af3', 'input_features_af3', '{}_af3_input.json.xz'))


def _request(url, method='GET'):
    from urllib.request import Request
    return Request(url, method=method, headers={'User-Agent': 'AlphaLens'})


def remote_organisms(base):
    '''{'af2': [organism, ...], 'af3': [...]} from the database's index pages.'''
    from urllib.request import urlopen
    found = {}
    for kind, folder, _ in REMOTE_LAYOUT:
        try:
            with urlopen(_request(f'{base}/{folder}/index.html'), timeout=30) as r:
                page = r.read(4 * 1024 * 1024).decode('utf-8', 'replace')
        except OSError as e:
            raise ValueError(f'Could not reach the feature database {base}: {e}') from None
        found[kind] = sorted(set(re.findall(r'href="([A-Za-z0-9_]+)\.html"', page)) - {'index'})
    return found


def find_remote(base, name, organisms):
    '''{'af2': URL or None, 'af3': URL or None}: one HEAD request an organism, in parallel.'''
    from concurrent.futures import ThreadPoolExecutor
    from urllib.request import urlopen

    def exists(url):
        try:
            with urlopen(_request(url, 'HEAD'), timeout=30) as r:
                return r.status == 200
        except OSError:
            return False
    found = {}
    for kind, folder, pattern in REMOTE_LAYOUT:
        urls = [f'{base}/{folder}/{o}/features/{pattern.format(name.upper())}' for o in organisms[kind]]
        with ThreadPoolExecutor(8) as pool:
            hits = [u for u, ok in zip(urls, pool.map(exists, urls)) if ok]
        found[kind] = hits[0] if hits else None
    return found


def download(url):
    '''The cached local copy of a database file; downloaded once.'''
    from urllib.request import urlopen
    from urllib.parse import urlparse
    target = cache_directory() / urlparse(url).path.lstrip('/')
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + '.part')
    try:
        with urlopen(_request(url), timeout=60) as r, part.open('wb') as out:
            size = int(r.headers.get('Content-Length') or 0)
            say(f'downloading {url.rsplit("/", 1)[-1]}' + (f' ({size / 1e6:.1f} MB)' if size else ''))
            while True:
                block = r.read(1 << 20)
                if not block:
                    break
                out.write(block)
    except OSError as e:
        part.unlink(missing_ok=True)
        raise ValueError(f'Could not download {url}: {e}') from None
    part.replace(target)
    return target


def find_features(root, name, directories=None):
    '''{'af2': path or None, 'af3': path or None} of the features named name (UniProt ID).'''
    af2_dirs, af3_dirs = directories or feature_directories(root)
    found = dict(af2=None, af3=None)
    names = tuple(dict.fromkeys((name.upper(), name)))    # Database names are upper case
    for kind, dirs, suffixes in (('af2', af2_dirs, AF2_SUFFIXES), ('af3', af3_dirs, AF3_SUFFIXES)):
        # Most likely name first: each test is a round trip on a network mount
        found[kind] = next((d / (n + s) for s in suffixes for n in names for d in dirs
                            if (d / (n + s)).is_file()), None)
    return found


def choose_features(root, chains, backend):
    '''
    The feature file of each chain (dicts with 'name'), all of one kind: AF3 uses AF3 JSON
    features when every chain has them, else AF2 features; AF2 uses AF2 features only.
    '''
    remote = is_url(root)
    if remote:
        root = str(root).rstrip('/')
        organisms = remote_organisms(root)
    else:
        directories = feature_directories(root)
    found = {}
    for n in dict.fromkeys(c['name'] for c in chains):
        say(f'looking for {n} in {root}')
        found[n] = find_remote(root, n, organisms) if remote else find_features(root, n, directories)
    use_af3 = backend == 'af3' and all(f['af3'] for f in found.values())
    kind = 'af3' if use_af3 else 'af2'
    absent = [n for n, f in found.items() if f[kind] is None]
    if absent:
        only_af3 = [n for n in absent if found[n]['af3']]
        hint = (f' {", ".join(only_af3)}: AlphaFold 3 features only, choose the AlphaFold 3 model.'
                if only_af3 and backend == 'af2' else '')
        raise ValueError(f'No {"AlphaFold 2 " if backend == "af2" else ""}features for {", ".join(absent)} '
                         f'in {root}.{hint}')
    return [dict(c, kind=kind, path=str(found[c['name']][kind])) for c in chains]


# -----------------------------------------------------------------------------
# Reading
# -----------------------------------------------------------------------------

def read_bytes(path):
    '''Decompressed contents and the sha256 of the file as stored.'''
    path = Path(path)
    raw = path.read_bytes()
    opener = {'.xz': lzma.decompress, '.gz': gzip.decompress, '.bz2': bz2.decompress}.get(path.suffix)
    try:
        data = opener(raw) if opener else raw
    except (lzma.LZMAError, OSError, EOFError) as e:
        raise ValueError(f'{path} is damaged or was read incompletely ({e}). Through the /g network '
                         'mount, copy it to a local folder and use that.') from None
    return data, hashlib.sha256(raw).hexdigest()


class _Stand_in:
    '''An AlphaPulldown or AlphaFold object in a pickle (MonomericObject, Jackhmmer, ...): keeps its
    attributes, runs none of its code.'''
    def __init__(self, *args, **kw):
        pass

    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) == 2:
            state, slots = state
            state = dict(state or {}, **(slots or {}))
        if isinstance(state, dict):
            self.__dict__.update(state)


class _FeatureUnpickler(pickle.Unpickler):
    NUMPY = {('numpy', 'ndarray'), ('numpy', 'dtype'), ('numpy.core.multiarray', '_reconstruct'),
             ('numpy._core.multiarray', '_reconstruct'), ('numpy.core.multiarray', 'scalar'),
             ('numpy._core.multiarray', 'scalar'), ('numpy.core.numeric', '_frombuffer'),
             ('numpy._core.numeric', '_frombuffer')}
    BUILTINS = {('builtins', 'set'), ('builtins', 'frozenset'), ('builtins', 'bytearray'),
                ('collections', 'OrderedDict')}

    def find_class(self, module, name):
        if (module, name) in self.NUMPY:
            if module.startswith('numpy.core') and not hasattr(np, 'core'):
                module = module.replace('numpy.core', 'numpy._core')
            return super().find_class(module, name)
        if module == 'numpy.dtypes' and name.endswith('DType'):
            return super().find_class(module, name)
        if (module, name) in self.BUILTINS:
            return super().find_class(module, name)
        if module.split('.')[0] in ('alphapulldown', 'alphafold'):
            return type(name, (_Stand_in,), {'__module__': module})
        raise pickle.UnpicklingError(f'the feature file holds {module}.{name}, which AlphaLens does not read')


def load_af2(path):
    '''Per-chain AlphaFold features from a feature pickle: dict(name, sequence, msa (N x L, HHblits
    order), deletions, all_msa / all_deletions / all_species (None without msa_all_seq), sha256).'''
    data, sha = read_bytes(path)
    try:
        obj = _FeatureUnpickler(io.BytesIO(data)).load()
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, ValueError, TypeError) as e:
        raise ValueError(f'Could not read {path} as AlphaFold features: {e}') from None
    fd = obj if isinstance(obj, dict) else getattr(obj, 'feature_dict', None)
    if not isinstance(fd, dict) or 'msa' not in fd:
        raise ValueError(f'{path} holds no AlphaFold MSA features.')
    if 'asym_id' in fd and len(np.unique(np.asarray(fd['asym_id']))) > 1:
        raise ValueError(f'{path} holds the features of a whole complex (AlphaFold-Multimer); '
                         'AlphaLens reads per-protein features, one file a chain.')
    sequence = getattr(obj, 'sequence', None)
    if not isinstance(sequence, str) and 'sequence' in fd:
        s = np.asarray(fd['sequence']).ravel()[0]
        sequence = s.decode() if isinstance(s, bytes) else str(s)
    if not isinstance(sequence, str) or not sequence:
        raise ValueError(f'{path} holds no sequence.')
    sequence = sequence.strip().upper()
    msa = np.asarray(fd['msa'], np.int32)
    deletions = np.asarray(fd.get('deletion_matrix_int', fd.get('deletion_matrix', np.zeros(msa.shape))))
    if msa.ndim != 2 or msa.shape[1] != len(sequence) or deletions.shape != msa.shape:
        raise ValueError(f'{path}: the MSA does not have the length of the sequence ({len(sequence)}).')
    # Pipeline features use the HHblits order, processed ones residue_constants'; row 0 is the query
    table = next((t for t in (HHBLITS, RESTYPES) if decode(msa[:1], t)[0] == canonical(sequence, t)), None)
    if table is None:
        raise ValueError(f'{path}: the first MSA row is not the sequence; not AlphaFold features?')
    if table != HHBLITS:
        msa = recode(msa, table)
    out = dict(name=getattr(obj, 'description', None) or Path(path).name, sequence=sequence, msa=msa,
               deletions=np.round(deletions).astype(np.int32), all_msa=None, all_deletions=None,
               all_species=None, sha256=sha)
    if 'msa_all_seq' in fd and 'msa_species_identifiers_all_seq' in fd:
        a = np.asarray(fd['msa_all_seq'], np.int32)
        if a.ndim == 2 and a.shape[1] == len(sequence):
            out['all_msa'] = recode(a, table) if table != HHBLITS else a
            out['all_deletions'] = np.round(np.asarray(fd.get('deletion_matrix_int_all_seq',
                                                              np.zeros(a.shape)))).astype(np.int32)
            out['all_species'] = [s.decode() if isinstance(s, bytes) else str(s)
                                  for s in np.asarray(fd['msa_species_identifiers_all_seq']).ravel()]
            if len(out['all_species']) != len(a) or out['all_deletions'].shape != a.shape:
                out['all_msa'] = out['all_deletions'] = out['all_species'] = None
    return out


def load_af3(path):
    '''The protein entry of an AlphaFold 3 data-pipeline JSON: dict(sequence, unpaired, paired, sha256).'''
    data, sha = read_bytes(path)
    try:
        job = json.loads(data.decode('utf-8'))
    except (UnicodeDecodeError, ValueError) as e:
        raise ValueError(f'{path} is not JSON: {e}') from None
    entries = job.get('sequences') if isinstance(job, dict) else None
    if not entries or len(entries) != 1 or 'protein' not in entries[0]:
        raise ValueError(f'{path}: expected AlphaFold 3 features of one protein.')
    p = entries[0]['protein']
    texts = {}
    for key in ('unpairedMsa', 'pairedMsa'):
        text = p.get(key)
        if text is None and p.get(key + 'Path'):
            ref = Path(p[key + 'Path'])
            text = read_bytes(ref if ref.is_absolute() else Path(path).parent / ref)[0].decode('utf-8')
        texts[key] = text or ''
    return dict(name=job.get('name') or Path(path).name, sequence=p['sequence'].upper(),
                unpaired=texts['unpairedMsa'], paired=texts['pairedMsa'], sha256=sha)


# -----------------------------------------------------------------------------
# Converting
# -----------------------------------------------------------------------------

def decode(msa, table=HHBLITS):
    lut = np.frombuffer(table.encode(), np.uint8)
    return [r.tobytes().decode() for r in lut[np.clip(msa, 0, len(table) - 1)]]


def canonical(sequence, table):
    '''The sequence as it reads back after encoding (U as C, B as D, ... in AlphaFold's maps).'''
    hh = {'B': 'D', 'J': 'X', 'O': 'X', 'U': 'C', 'Z': 'E'}
    return ''.join(c if c in table[:21] else hh.get(c, 'X') for c in sequence)


def recode(msa, table):
    '''Codes in table's order -> HHblits order.'''
    m = np.array([HHBLITS.index(c) for c in table], np.int32)
    return m[np.clip(msa, 0, len(table) - 1)]


def locate(feature_sequence, sequence, hint=None, name=''):
    '''Start of the chain's sequence within the features' sequence.'''
    if hint is not None and 0 <= hint and feature_sequence[hint:hint + len(sequence)] == sequence:
        return hint
    starts = [m.start() for m in re.finditer(f'(?={re.escape(sequence)})', feature_sequence)]
    if len(starts) == 1:
        return starts[0]
    if not starts:
        raise ValueError(f'The chain sequence ({len(sequence)} residues) is not part of the sequence in the '
                         f'features of {name} ({len(feature_sequence)} residues): another isoform or version?')
    raise ValueError(f'The chain sequence occurs {len(starts)} times in the features of {name}.')


def a3m_rows(msa, deletions, headers):
    '''A3M text: deletion counts become lowercase insertions before their column.'''
    lut = np.frombuffer(HHBLITS.encode(), np.uint8)
    codes = lut[np.clip(msa, 0, 21)]
    lines = []
    for h, row, dels in zip(headers, codes, deletions):
        if dels.any():
            text = bytearray()
            for c, d in zip(row, dels):
                if d > 0:
                    text += b'x' * int(d)
                text.append(c)
            body = text.decode()
        else:
            body = row.tobytes().decode()
        lines.append(f'>{h}\n{body}\n')
    return ''.join(lines)


def crop_columns(msa, deletions, start, length):
    m = msa[:, start:start + length]
    d = deletions[:, start:start + length].copy()
    if start:
        d[:, 0] = 0          # Insertions before the first kept column belong to the cut part
    return m, d


def unpaired_a3m(features, start, length, label):
    m, d = crop_columns(features['msa'], features['deletions'], start, length)
    keep = np.flatnonzero((m != 21).any(1))
    keep = np.concatenate([[0], keep[keep != 0]])
    return a3m_rows(m[keep], d[keep], [label] + [f'{label}_{i}' for i in keep[1:]]), len(keep)


def pair_rows(chains):
    '''
    AlphaFold-Multimer's species pairing (msa_pairing.pair_sequences and reorder_paired_rows) of
    msa_all_seq: per species found in at least two chains and at most MAX_ROWS_PER_SPECIES rows in
    each, rows by similarity to the query, paired k-th with k-th; -1 for a chain without that species.
    chains: per chain (msa, species) with the msa cropped to the predicted residues; rows with no
    residue left in the crop take no part. Returns an (R, n_chains) int array, the query row first.
    '''
    n = len(chains)
    groups = []
    for msa, species in chains:
        if msa is None:
            groups.append({})
            continue
        sim = (msa == msa[0]).sum(1) / msa.shape[1]
        present = (msa != 21).any(1)
        g = {}
        for row in range(1, len(msa)):
            if species[row] and present[row]:
                g.setdefault(species[row], []).append(row)
        groups.append({s: sorted(rows, key=lambda r: -sim[r]) for s, rows in g.items()})
    by_count = {k: [] for k in range(2, n + 1)}
    for s in sorted(set().union(*groups)):
        have = [g.get(s) for g in groups]
        count = sum(h is not None for h in have)
        if count <= 1 or max(len(h) for h in have if h is not None) > MAX_ROWS_PER_SPECIES:
            continue
        take = min(len(h) for h in have if h is not None)
        for k in range(take):
            by_count[count].append([h[k] if h is not None else -1 for h in have])
    rows = [[0] * n]
    for count in sorted(by_count, reverse=True):
        block = by_count[count]
        if block:
            order = np.argsort([abs(np.prod(r, dtype=np.float64)) for r in block], kind='stable')
            rows.extend(block[i] for i in order)
    return np.array(rows, int).reshape(-1, n)


def paired_a3ms(chain_features, starts, lengths, labels):
    '''Row-aligned paired A3Ms, one a chain (LocalFold --paired-a3m).'''
    cropped = []
    for f, s, l in zip(chain_features, starts, lengths):
        if f['all_msa'] is None:
            cropped.append((None, None, None))
        else:
            m, d = crop_columns(f['all_msa'], f['all_deletions'], s, l)
            cropped.append((m, d, f['all_species']))
    rows = pair_rows([(m, sp) for m, _, sp in cropped])
    texts = []
    for c, ((m, d, _), l, label) in enumerate(zip(cropped, lengths, labels)):
        idx = rows[:, c]
        if m is None:      # Query row only; gaps where the others have partners
            m = np.full((1, l), 21, np.int32); d = np.zeros((1, l), np.int32)
            q, _ = crop_columns(chain_features[c]['msa'][:1], chain_features[c]['deletions'][:1],
                                starts[c], l)
            m[0] = q[0]
            idx = np.where(idx == 0, 0, -1)
        gap_m = np.vstack([m, np.full((1, l), 21, np.int32)])
        gap_d = np.vstack([d, np.zeros((1, l), np.int32)])
        texts.append(a3m_rows(gap_m[idx], gap_d[idx],
                              [label] + [f'{label}_paired_{k}' for k in range(1, len(idx))]))
    return texts, len(rows) - 1


def crop_a3m_text(text, start, length, query):
    '''One chain's A3M (AF3 JSON) cropped to columns start..start+length; headers kept, rows
    with no residue left dropped (the query is kept).'''
    out = []
    header, parts = None, []

    def flush():
        if header is None:
            return
        aligned, inserts, pending = [], [], ''
        for ch in ''.join(parts):
            if ch.islower():
                pending += ch
            elif ch == '.':
                continue
            else:
                aligned.append(ch); inserts.append(pending); pending = ''
        if len(aligned) != len(query):
            raise ValueError(f'An alignment row has {len(aligned)} columns for a {len(query)}-residue sequence.')
        cols = range(start, start + length)
        row = ''.join((inserts[c] if c > start else '') + aligned[c] for c in cols)
        if not out or any(aligned[c] != '-' for c in cols):
            out.append(f'{header}\n{row}\n')

    for line in text.splitlines():
        if line.startswith('#') or not line.strip():
            continue
        if line.startswith('>'):
            flush()
            header, parts = line, []
        elif header is not None:
            parts.append(line.strip())
    flush()
    if out:
        first = out[0].split('\n')[1]
        if ''.join(c for c in first if not c.islower()) != query[start:start + length]:
            raise ValueError('The first alignment row is not the query sequence.')
    return ''.join(out)


# -----------------------------------------------------------------------------
# The step before LocalFold (run as a script)
# -----------------------------------------------------------------------------

def say(message):
    print(f'features: {message}', flush=True)


def prepare(spec):
    '''
    Find and convert the features of spec's chains; returns the LocalFold arguments that name
    the alignments (none when they went into the AF3 job). Provenance goes to features.json.
    spec: directory, root, backend, job (input.json), chains [{chain_id, name, sequence, start_hint}].
    '''
    out_dir = Path(spec['directory'])
    out_dir.mkdir(parents=True, exist_ok=True)
    chains = choose_features(spec['root'], spec['chains'], spec['backend'])
    provenance = []
    loaded = {}
    for c in chains:
        if is_url(c['path']):
            c['url'], c['path'] = c['path'], str(download(c['path']))
        if c['path'] not in loaded:
            size = Path(c['path']).stat().st_size
            say(f'reading {c["name"]} ({size / 1e6:.1f} MB) from {c["path"]}')
            t0 = time.monotonic()
            try:
                loaded[c['path']] = (load_af3 if c['kind'] == 'af3' else load_af2)(c['path'])
            except ValueError:
                if c.get('url'):            # A damaged download is fetched again next time
                    Path(c['path']).unlink(missing_ok=True)
                raise
            say(f'read {c["name"]} in {time.monotonic() - t0:.1f} s')
    feats = [loaded[c['path']] for c in chains]
    starts = [locate(f['sequence'], c['sequence'], c.get('start_hint'), c['name']) for f, c in zip(feats, chains)]
    lengths = [len(c['sequence']) for c in chains]
    for c, f, s in zip(chains, feats, starts):
        provenance.append(dict(chain_id=c['chain_id'], name=c['name'], kind=c['kind'], path=c['path'],
                               url=c.get('url'), sha256=f['sha256'], residues=[s + 1, s + len(c['sequence'])],
                               feature_length=len(f['sequence'])))
    args = []
    if chains[0]['kind'] == 'af3':
        job_path = Path(spec['job'])
        job = json.loads(job_path.read_text(encoding='utf-8'))
        for entry, c, f, s, l in zip(job['sequences'], chains, feats, starts, lengths):
            p = entry['protein']
            p['unpairedMsa'] = crop_a3m_text(f['unpaired'], s, l, f['sequence']) or f'>query\n{c["sequence"]}\n'
            p['pairedMsa'] = crop_a3m_text(f['paired'], s, l, f['sequence']) if f['paired'] else ''
            p['templates'] = []
        part = job_path.with_suffix('.json.part')
        part.write_text(json.dumps(job, indent=1) + '\n', encoding='utf-8')
        part.replace(job_path)
        say(f'AlphaFold 3 alignments of {len(chains)} chain(s) put into {job_path.name}')
    else:
        a3ms = [out_dir / f'chain_{c["chain_id"]}.a3m' for c in chains]
        for c, f, s, l, p, path in zip(chains, feats, starts, lengths, provenance, a3ms):
            text, rows = unpaired_a3m(f, s, l, c['chain_id'])
            path.write_text(text, encoding='utf-8')
            p['unpaired_rows'] = rows
        args.append('--a3m=' + ','.join(str(a) for a in a3ms))
        # AlphaFold-Multimer pairs the chains of a heteromer only
        if len(chains) > 1 and len({c['sequence'] for c in chains}) > 1:
            texts, rows = paired_a3ms(feats, starts, lengths, [c['chain_id'] for c in chains])
            paired = [out_dir / f'paired_{c["chain_id"]}.a3m' for c in chains]
            for text, p, path in zip(texts, provenance, paired):
                path.write_text(text, encoding='utf-8')
                p['paired_rows'] = rows
            args.append('--paired-a3m=' + ','.join(str(a) for a in paired))
            without = [c['name'] for c, f in zip(chains, feats) if f['all_msa'] is None]
            say(f'{rows} rows paired by species' +
                (f'; no msa_all_seq to pair in {", ".join(without)}' if without else ''))
        say('alignments of ' + ', '.join(f'{c["chain_id"]} {p["unpaired_rows"]} rows' for c, p in zip(chains, provenance)))
    (out_dir / 'features.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    return args


ERROR_PREFIX = 'Error: precomputed features: '


def main(argv):
    if len(argv) < 3 or argv[1] != '--':
        print('usage: predict_features.py <spec.json> -- <command ...>', file=sys.stderr)
        return 2
    try:
        args = prepare(json.loads(Path(argv[0]).read_text(encoding='utf-8')))
    except Exception as e:     # One line, as LocalFold reports a refusal
        print(f'{ERROR_PREFIX}{e}', flush=True)
        return 1
    sys.stdout.flush()
    os.execv(argv[2], argv[2:] + args)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
