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
  * A3M files of one chain in AlphaFold 3's form, an unpaired and a paired alignment (the
    chain's UniProt hits, paired by species when the job is folded), as the AlphaFold Server
    puts them in a download's msas folder, or as AlphaFold 3 writes them for each chain into a
    prediction's <name>_data.json. Inline in an AF3 job; for AF2, converted to the features
    above, species from the UniProt headers.
  * A whole complex's features, the features.pkl of a native AlphaFold 2 run (AlphaFold-Multimer's
    merged MSA, paired rows first): its columns are cropped into one merged A3M (LocalFold --a3m).

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
# An AlphaFold Server download: msas/<job>_unpaired_msa_chains_<ids>.a3m and
# <job>_paired_msa_chains_<ids>.a3m, ids the chain's lower-case ID (several joined by _ assumed
# for identical chains)
SERVER_MSA = re.compile(r'(?P<job>.+)_(?P<kind>unpaired|paired)_msa_chains_(?P<ids>[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)\.a3m')
# AlphaFold 3's data-pipeline output beside <name>_model.cif, each chain's MSAs inline
AF3_DATA_SUFFIXES = ('_data.json', '_data.json.xz', '_data.json.gz')
# A native AlphaFold 2 run's features of the whole complex, beside its models
AF2_COMPLEX_FEATURES = ('features.pkl', 'features.pkl.xz', 'features.pkl.gz')
# AlphaFold 2's chain IDs, in order (protein.PDB_CHAIN_IDS)
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
# AlphaFold's species of a UniProt hit (msa_identifiers.py): the mnemonic after the entry name's _
UNIPROT_SPECIES = re.compile(r'(?:tr|sp)\|[A-Za-z0-9]{6,10}(?:_\d)?\|[A-Za-z0-9]+_(?P<species>[A-Za-z0-9]{1,5})(?:_\d+)?')


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


def guess_names(path, chain_ids, sequences=None):
    '''
    {chain id: UniProt ID} from a model's path, as AlphaPulldown and AlphaFold 3 name folds:
    ".../Q15014+Q9NV56/ranked_0.cif", ".../q15014_and_q9nv56_model.cif", ".../P1:20-80+P2/...",
    ".../Q15014_af3_input.json_and_Q9NV56_af3_input.json_model.cif". The nearest name with one
    ID a chain wins; else one ID for all chains, only when they are one protein (sequences, one
    a chain, all the same; None: assumed). {} if none fits.
    '''
    p = Path(path)
    found = [accessions_in(part) for part in [p.stem] + [q.name for q in list(p.parents)[:3]]]
    for ids in found:
        if len(ids) == len(chain_ids):
            return dict(zip(chain_ids, ids))
    if sequences is None or len(set(sequences)) <= 1:
        for ids in found:
            if len(ids) == 1:
                return {cid: ids[0] for cid in chain_ids}
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


def query_sequence(path):
    '''The residues of an A3M file's first record, its query (insertions left out); reads no further.'''
    with Path(path).open(encoding='utf-8', errors='surrogateescape') as stream:
        return first_record(stream)


def server_msas(model_path):
    '''
    The AlphaFold Server alignments of the prediction whose model is model_path, from the msas
    folder beside it: {chain ID (upper case): dict(unpaired=path, paired=path or None,
    sequence=query)}, {} when there are none. Reads only each unpaired file's query.
    '''
    model = Path(model_path)
    folder = model.parent / 'msas'
    try:
        names = sorted(p.name for p in folder.iterdir())
    except OSError:
        return {}
    jobs = {}
    for name in names:
        m = SERVER_MSA.fullmatch(name)
        if m:
            jobs.setdefault(m['job'], {}).setdefault(m['ids'], {})[m['kind']] = str(folder / name)
    job = re.sub(r'_model_\d+$', '', model.stem)
    if job not in jobs:                 # A renamed model: the folder's only job
        if len(jobs) != 1:
            return {}
        job = next(iter(jobs))
    found = {}
    for ids, files in jobs[job].items():
        if 'unpaired' in files:
            entry = dict(unpaired=files['unpaired'], paired=files.get('paired'),
                         sequence=query_sequence(files['unpaired']))
            for cid in ids.split('_'):
                found[cid.upper()] = entry
    return found


def open_text(path):
    '''A text stream of a file, decompressed (.xz, .gz).'''
    opener = {'.xz': lzma.open, '.gz': gzip.open}.get(Path(path).suffix, open)
    return opener(path, 'rt', encoding='utf-8', errors='surrogateescape')


def af3_data_json(model_path):
    '''
    The AlphaFold 3 data-pipeline JSON of the prediction whose model is model_path, holding each
    chain's MSAs: <name>_data.json (.xz, .gz) beside <name>_model.cif, or in the folder above a
    sample's seed-<S>_sample-<K>/model.cif; else a folder's only one. None when there is none.
    Reads only its start.
    '''
    model = Path(model_path)
    stem = model.stem[:-len('_model')] if model.stem.endswith('_model') else None
    folders = [model.parent]
    if re.fullmatch(r'seed-\d+_sample-\d+', model.parent.name):
        folders.append(model.parent.parent)
    for folder in folders:
        named = [p for p in (folder / f'{stem}{x}' for x in AF3_DATA_SUFFIXES) if p.is_file()] if stem else []
        try:
            found = named or sorted(p for x in AF3_DATA_SUFFIXES for p in folder.glob('*' + x))
        except OSError:
            continue
        if len(found) != 1 and not named:
            continue
        try:
            with open_text(found[0]) as stream:
                head = stream.read(1 << 16)
        except (OSError, EOFError, lzma.LZMAError, UnicodeDecodeError):
            continue
        if '"sequences"' in head and 'Msa' in head:        # unpairedMsa(Path), pairedMsa(Path)
            return found[0]
    return None


def af2_complex_features(model_path):
    '''
    A native AlphaFold 2 run's features of the whole complex beside its model: (features.pkl,
    [sequence of each chain in order]) with the sequences from msas/chain_id_map.json, which
    AlphaFold-Multimer writes; None when either is missing. Reads only the chain map.
    '''
    folder = Path(model_path).parent
    path = next((folder / n for n in AF2_COMPLEX_FEATURES if (folder / n).is_file()), None)
    if path is None:
        return None
    try:
        chains = json.loads((folder / 'msas' / 'chain_id_map.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not isinstance(chains, dict) or not chains:
        return None
    order = sorted(chains, key=lambda c: (PDB_CHAIN_IDS.find(c) if c in PDB_CHAIN_IDS else len(PDB_CHAIN_IDS), c))
    sequences = [chains[c].get('sequence') if isinstance(chains[c], dict) else None for c in order]
    return (path, sequences) if all(isinstance(q, str) and q for q in sequences) else None


def snakemake_features(model_path, chain_ids):
    '''
    The AlphaPulldown features of a prediction laid out as AlphaPulldownSnakemake writes it:
    <out>/predictions/<fold>/<model>, each protein's feature file in <out>/features. Returns
    (that folder, {chain ID: feature name}) when every chain's file is there, else None. The
    names come from the fold's name, as AlphaPulldown writes it: chains joined by + or "_and_",
    a homo-oligomer <name>_homo_<n>er, residue ranges as <name>:<start>-<end> or
    <name>_<start>-<end>_... (left out: the features are of the whole protein).
    '''
    fold = Path(model_path).parent
    if fold.parent.name != 'predictions':
        return None
    folder = fold.parent.parent / 'features'
    names = []
    for part in re.split(r'\+|_and_', fold.name):
        homo = re.fullmatch(r'(.+)_homo_(\d+)er', part)
        if homo:
            names += [homo[1]] * int(homo[2])
        elif part:
            names.append(re.sub(r'(?:_\d+-\d+)+$', '', part.split(':')[0]))
    if len(names) != len(chain_ids):
        return None
    suffixes = AF2_SUFFIXES + AF3_SUFFIXES
    for name in dict.fromkeys(names):
        if not any((folder / (name + x)).is_file() for x in suffixes):
            return None
    return folder, dict(zip(chain_ids, names))


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
    texts = _msa_texts(p, path)
    return dict(name=job.get('name') or Path(path).name, sequence=p['sequence'].upper(),
                unpaired=texts['unpairedMsa'], paired=texts['pairedMsa'], sha256=sha)


def _msa_texts(protein, path):
    '''A protein entry's unpairedMsa and pairedMsa, inline or from their ...Path files.'''
    texts = {}
    for key in ('unpairedMsa', 'pairedMsa'):
        text = protein.get(key)
        if text is None and protein.get(key + 'Path'):
            ref = Path(protein[key + 'Path'])
            text = read_bytes(ref if ref.is_absolute() else Path(path).parent / ref)[0].decode('utf-8')
        texts[key] = text or ''
    return texts


def load_af3_chains(path):
    '''Each protein chain's MSAs in an AlphaFold 3 data-pipeline JSON of a whole job:
    {chain ID: dict as load_a3m_pair}.'''
    data, sha = read_bytes(path)
    try:
        job = json.loads(data.decode('utf-8'))
    except (UnicodeDecodeError, ValueError) as e:
        raise ValueError(f'{path} is not JSON: {e}') from None
    out = {}
    for entry in (job.get('sequences') if isinstance(job, dict) else None) or []:
        p = entry.get('protein') if isinstance(entry, dict) else None
        if not p:
            continue
        texts = _msa_texts(p, path)
        ids = p['id'] if isinstance(p['id'], list) else [p['id']]
        for cid in ids:
            out[cid] = dict(name=f'{Path(path).name} chain {cid}', sequence=p['sequence'].upper(),
                            unpaired=texts['unpairedMsa'], paired=texts['pairedMsa'], sha256=sha,
                            paired_sha256=None)
    return out


def first_record(lines):
    '''The residues of the first record of an A3M's lines (insertions left out).'''
    rows, started = [], False
    for line in lines:
        if line.startswith('#') or not line.strip():
            continue
        if line.startswith('>'):
            if started:
                break
            started = True
        elif started:
            rows.append(''.join(c for c in line.strip() if not c.islower() and c != '.'))
    return ''.join(rows)


def load_a3m_pair(unpaired, paired=None):
    '''One chain's A3M files in AlphaFold 3's form, as load_af3: dict(name, sequence, unpaired,
    paired, sha256, paired_sha256).'''
    data, sha = read_bytes(unpaired)
    text = data.decode('utf-8', errors='surrogateescape')
    sequence = first_record(text.splitlines())
    if not sequence:
        raise ValueError(f'{unpaired} holds no alignment.')
    out = dict(name=Path(unpaired).name, sequence=sequence.upper(), unpaired=text, paired='',
               sha256=sha, paired_sha256=None)
    if paired:
        data, out['paired_sha256'] = read_bytes(paired)
        out['paired'] = data.decode('utf-8', errors='surrogateescape')
        if first_record(out['paired'].splitlines()).upper() != out['sequence']:
            raise ValueError(f'The first sequences of {Path(unpaired).name} and {Path(paired).name} differ.')
    return out


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


def columns(start, length=None):
    """The feature columns kept: start..start+length, or start itself as a list of columns
    (a chain of discontinuous segments)."""
    return np.arange(start, start + length) if length is not None else np.asarray(start, dtype=int)


def crop_columns(msa, deletions, start, length=None):
    cols = columns(start, length)
    m = msa[:, cols]
    d = deletions[:, cols].copy()
    # Insertions before a kept column that follows a cut belong to the cut part
    d[:, np.flatnonzero(np.diff(np.concatenate([[-1], cols])) != 1)] = 0
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
    cols = [columns(s, l) for s, l in zip(starts, lengths)]
    lengths = [len(c) for c in cols]
    for f, s, l in zip(chain_features, cols, lengths):
        if f['all_msa'] is None:
            cropped.append((None, None, None))
        else:
            m, d = crop_columns(f['all_msa'], f['all_deletions'], s)
            cropped.append((m, d, f['all_species']))
    rows = pair_rows([(m, sp) for m, _, sp in cropped])
    texts = []
    for c, ((m, d, _), l, label) in enumerate(zip(cropped, lengths, labels)):
        idx = rows[:, c]
        if m is None:      # Query row only; gaps where the others have partners
            m = np.full((1, l), 21, np.int32); d = np.zeros((1, l), np.int32)
            q, _ = crop_columns(chain_features[c]['msa'][:1], chain_features[c]['deletions'][:1], cols[c])
            m[0] = q[0]
            idx = np.where(idx == 0, 0, -1)
        gap_m = np.vstack([m, np.full((1, l), 21, np.int32)])
        gap_d = np.vstack([d, np.zeros((1, l), np.int32)])
        texts.append(a3m_rows(gap_m[idx], gap_d[idx],
                              [label] + [f'{label}_paired_{k}' for k in range(1, len(idx))]))
    return texts, len(rows) - 1


def species_of(header):
    '''AlphaFold's species identifier of an A3M header (UniProt hits), '' for none.'''
    words = header.lstrip('>').split()
    m = UNIPROT_SPECIES.fullmatch(words[0].partition('/')[0]) if words else None
    return m['species'] if m else ''


def a3m_arrays(text):
    '''An A3M text as AlphaFold features (parsers.parse_a3m): (msa N x L in HHblits order,
    deletions, headers).'''
    lut = np.full(256, 20, np.int32)                    # Anything else is unknown, as X
    for i, c in enumerate(HHBLITS):
        lut[ord(c)] = i
    for c, same in (('B', 'D'), ('U', 'C'), ('Z', 'E')):
        lut[ord(c)] = HHBLITS.index(same)
    headers, rows, header, parts = [], [], None, []

    def flush():
        if header is None:
            return
        row = np.frombuffer(''.join(parts).replace('.', '').encode('ascii', 'replace'), np.uint8)
        lower = (row >= 97) & (row <= 122)
        at = np.flatnonzero(~lower)
        inserted = np.concatenate([[0], np.cumsum(lower)])  # Insertions before each position
        dels = inserted[at] - np.concatenate([[0], inserted[at[:-1] + 1]]) if len(at) else at
        headers.append(header)
        rows.append((lut[row[at]], dels))

    for line in text.splitlines():
        if line.startswith('#') or not line.strip():
            continue
        if line.startswith('>'):
            flush()
            header, parts = line[1:], []
        elif header is not None:
            parts.append(line.strip())
    flush()
    if not rows or len({len(m) for m, _ in rows}) != 1:
        raise ValueError('An alignment is empty, or its rows have different lengths.')
    return (np.array([m for m, _ in rows], np.int32), np.array([d for _, d in rows], np.int32), headers)


def a3m_features(feats, columns_of):
    '''
    AF2 features (as load_af2) of the predicted columns from A3M texts in AlphaFold 3's form:
    each chain's unpaired and paired texts are cropped to its columns first, so only the
    predicted part is converted. Returns the features and their columns (all of them).
    '''
    out = []
    for f, cols in zip(feats, columns_of):
        cols = [int(c) for c in cols]
        query = ''.join(f['sequence'][c] for c in cols)
        msa, dels, _ = a3m_arrays(crop_a3m_text(f['unpaired'], cols, None, f['sequence']) or f'>query\n{query}\n')
        g = dict(name=f['name'], sequence=query, msa=msa, deletions=dels,
                 all_msa=None, all_deletions=None, all_species=None, sha256=f['sha256'])
        if f['paired']:
            m, d, headers = a3m_arrays(crop_a3m_text(f['paired'], cols, None, f['sequence']))
            g.update(all_msa=m, all_deletions=d, all_species=[species_of(h) for h in headers])
        out.append(g)
    return out, [list(range(len(g['sequence']))) for g in out]


def crop_a3m_text(text, start, length, query):
    '''One chain's A3M (AF3 JSON) cropped to columns start..start+length (or the list of columns
    start, length None); headers kept, rows with no residue left dropped (the query is kept).'''
    cols = [int(c) for c in columns(start, length)]
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
        row = ''.join((inserts[c] if k and c == cols[k - 1] + 1 else '') + aligned[c] for k, c in enumerate(cols))
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
        if ''.join(c for c in first if not c.islower()) != ''.join(query[c] for c in cols):
            raise ValueError('The first alignment row is not the query sequence.')
    return ''.join(out)


# -----------------------------------------------------------------------------
# The step before LocalFold (run as a script)
# -----------------------------------------------------------------------------

def say(message):
    print(f'features: {message}', flush=True)


def prepare_merged(spec):
    '''
    The merged MSA of a whole complex's AlphaFold 2 features (a native run's features.pkl, as
    AlphaFold-Multimer used it, paired rows first), cropped to the predicted columns into one
    A3M; returns its LocalFold argument. spec: directory, merged (the features), source_query
    (the source's whole sequence), columns (the predicted residues' columns, in order), chains.
    '''
    out_dir = Path(spec['directory'])
    out_dir.mkdir(parents=True, exist_ok=True)
    path, query = spec['merged'], spec['source_query']
    cols = [int(c) for c in spec['columns']]
    say(f'reading {Path(path).name} ({Path(path).stat().st_size / 1e6:.1f} MB) from {path}')
    t0 = time.monotonic()
    data, sha = read_bytes(path)
    try:
        obj = _FeatureUnpickler(io.BytesIO(data)).load()
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, ValueError, TypeError) as e:
        raise ValueError(f'Could not read {path} as AlphaFold features: {e}') from None
    fd = obj if isinstance(obj, dict) else getattr(obj, 'feature_dict', None)
    if not isinstance(fd, dict) or 'msa' not in fd:
        raise ValueError(f'{path} holds no AlphaFold MSA features.')
    msa = np.asarray(fd['msa'], np.int32)
    deletions = np.asarray(fd.get('deletion_matrix', fd.get('deletion_matrix_int', np.zeros(msa.shape))))
    if msa.ndim != 2 or msa.shape[1] != len(query) or deletions.shape != msa.shape:
        raise ValueError(f'{path}: the MSA has {msa.shape[-1]} columns, the prediction {len(query)} residues.')
    # Processed features use residue_constants' order, pipeline features HHblits'; row 0 is the query
    table = next((t for t in (RESTYPES, HHBLITS) if decode(msa[:1], t)[0] == canonical(query, t)), None)
    if table is None:
        raise ValueError(f'The first MSA row of {path} is not the sequence of this prediction.')
    if table != HHBLITS:
        msa = recode(msa, table)
    real = np.ones(len(msa), bool)                  # Rows padded to AlphaFold's minimum are left out
    mask = np.asarray(fd['msa_mask']) if 'msa_mask' in fd else None
    if mask is not None and mask.shape == msa.shape:
        real = mask.sum(1) > 0
    elif 'num_alignments' in fd:
        real[int(np.asarray(fd['num_alignments']).ravel()[0]):] = False
    msa, deletions = msa[real], np.round(deletions[real]).astype(np.int32)
    if any(not 0 <= c < len(query) for c in cols) or len(set(cols)) != len(cols):
        raise ValueError('Invalid MSA crop column mapping.')
    if ''.join(query[c] for c in cols) != ''.join(c['sequence'] for c in spec['chains']):
        raise ValueError('The predicted residues are not those columns of the source sequence.')
    m, d = crop_columns(msa, deletions, cols)
    keep = np.flatnonzero((m != 21).any(1))
    keep = np.concatenate([[0], keep[keep != 0]])
    a3m = out_dir / 'merged.a3m'
    a3m.write_text(a3m_rows(m[keep], d[keep], ['query'] + [f'row_{i}' for i in keep[1:]]), encoding='utf-8')
    say(f'read {Path(path).name} in {time.monotonic() - t0:.1f} s; {len(keep)} of {len(msa)} rows of '
        f'its merged MSA kept for the {len(cols)} predicted residues')
    provenance = [dict(kind='af2_complex', path=str(path), sha256=sha, rows=int(len(keep)),
                       source_rows=int(len(msa)), columns=len(cols))]
    (out_dir / 'features.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    return [f'--a3m={a3m}']


def prepare(spec):
    '''
    Find and convert the features of spec's chains; returns the LocalFold arguments that name
    the alignments (none when they went into the AF3 job). Provenance goes to features.json.
    spec: directory, root, backend, job (input.json), chains [{chain_id, name, sequence, start_hint}],
    a chain with unpaired (and paired) naming its A3M files in AlphaFold 3's form, or with af3_json
    and af3_chain naming its chain in an AlphaFold 3 data-pipeline JSON, instead of a feature
    file to find under root. With merged (a whole complex's features), prepare_merged.
    '''
    if spec.get('merged'):
        return prepare_merged(spec)
    out_dir = Path(spec['directory'])
    out_dir.mkdir(parents=True, exist_ok=True)
    given = [c.get('unpaired') or c.get('af3_json') for c in spec['chains']]
    if all(given):
        chains = [dict(c, kind='a3m', path=g) for c, g in zip(spec['chains'], given)]
    else:
        chains = choose_features(spec['root'], spec['chains'], spec['backend'])
    provenance = []
    loaded = {}
    for c in chains:
        if is_url(c['path']):
            c['url'], c['path'] = c['path'], str(download(c['path']))
        if c['path'] not in loaded:
            files = [c['path']] + ([c['paired']] if c['kind'] == 'a3m' and c.get('paired') else [])
            size = sum(Path(f).stat().st_size for f in files)
            say(f'reading {Path(c["path"]).name if c.get("af3_json") else c["name"]} ({size / 1e6:.1f} MB) '
                f'from {", ".join(files)}')
            t0 = time.monotonic()
            try:
                if c.get('af3_json'):
                    loaded[c['path']] = load_af3_chains(c['path'])
                elif c['kind'] == 'a3m':
                    loaded[c['path']] = load_a3m_pair(c['path'], c.get('paired'))
                else:
                    loaded[c['path']] = (load_af3 if c['kind'] == 'af3' else load_af2)(c['path'])
            except ValueError:
                if c.get('url'):            # A damaged download is fetched again next time
                    Path(c['path']).unlink(missing_ok=True)
                raise
            say(f'read {c["name"]} in {time.monotonic() - t0:.1f} s')
    feats = []
    for c in chains:
        f = loaded[c['path']]
        if c.get('af3_json'):
            if c.get('af3_chain') not in f:
                raise ValueError(f'{Path(c["path"]).name} has no protein chain {c.get("af3_chain")}.')
            f = f[c['af3_chain']]
        feats.append(f)
    # Each chain's columns in its full-length features: one located run, or one per segment of a
    # chain of discontinuous regions (kept as one chain, as AlphaPulldown does)
    starts = []
    for f, c in zip(feats, chains):
        segments = c.get('segments') or [[c['sequence'], c.get('start_hint')]]
        cols = []
        for piece, hint in segments:
            at = locate(f['sequence'], piece, hint, c['name'])
            cols.extend(range(at, at + len(piece)))
        if any(b <= a for a, b in zip(cols, cols[1:])):
            raise ValueError(f'The segments of chain {c["chain_id"]} are not in order in the features of {c["name"]}.')
        starts.append(cols)
    lengths = [None] * len(chains)
    for c, f, cols in zip(chains, feats, starts):
        runs = [[cols[0] + 1, cols[0] + 1]]
        for a, b in zip(cols, cols[1:]):
            if b == a + 1:
                runs[-1][1] = b + 1
            else:
                runs.append([b + 1, b + 1])
        provenance.append(dict(chain_id=c['chain_id'], name=c['name'], kind=c['kind'], path=c['path'],
                               url=c.get('url'), sha256=f['sha256'],
                               residues=runs[0] if len(runs) == 1 else [runs[0][0], runs[-1][1]],
                               **({'segments': runs} if len(runs) > 1 else {}),
                               feature_length=len(f['sequence']),
                               **({'paired': c.get('paired'), 'paired_sha256': f['paired_sha256']}
                                  if c['kind'] == 'a3m' and not c.get('af3_json') else {}),
                               **({'af3_chain': c['af3_chain']} if c.get('af3_json') else {})))
    args = []
    if chains[0]['kind'] == 'af3' or (chains[0]['kind'] == 'a3m' and spec['backend'] == 'af3'):
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
        if chains[0]['kind'] == 'a3m':
            feats, starts = a3m_features(feats, starts)
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
