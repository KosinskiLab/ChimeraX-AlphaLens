"""
Locate and read AlphaFold2 / AlphaFold3 prediction outputs (PAE and distograms) and
derive per-pair matrices from the distogram.

Nothing in this module touches the ChimeraX GUI, so it is safe to call from a worker thread.

Supported layouts
-----------------
AlphaFold2 / AlphaPulldown directory
    ranked_<i>.pdb|cif, (un)relaxed_<model>.pdb, ranking_debug.json,
    pae_<model>.json and/or result_<model>.pkl[.gz]
    The distogram is ``result['distogram']['logits']`` (L, L, 64) with ``bin_edges`` (63,),
    present only if the run kept it in the result pickle.

AlphaFold3 directory
    <name>_model.cif + <name>_confidences.json (top ranked sample), ranking_scores.csv,
    seed-<s>_sample-<k>/model.cif + confidences.json,
    seed-<s>_distogram/<name>_seed-<s>_distogram.npz with key ``distogram`` holding
    logits (L, L, 64), present only if AF3 was run with distogram saving enabled.
"""

import contextlib
import glob
import gzip
import json
import os
import pickle
import re
import time

import numpy as np

# AlphaFold2 and AlphaFold3 distogram heads use the same 64 bins:
# 63 edges from 2.3125 to 21.6875 A; first bin is < 2.3125, last bin is > 21.6875.
AF_BIN_EDGES = np.linspace(2.3125, 21.6875, 63).astype(np.float32)

PICKLE_SUFFIXES = ('.pkl', '.pkl.gz', '.pkl.xz', '.pickle', '.pickle.gz', '.pickle.xz')
JSON_SUFFIXES = ('.json', '.json.gz', '.json.xz')
NPZ_SUFFIXES = ('.npz', '.npz.gz', '.npz.xz')
NPY_SUFFIXES = ('.npy', '.npy.gz', '.npy.xz')


class PredictionFiles:
    """Paths of the files that belong to one predicted model."""

    def __init__(self, structure_path, kind='unknown', model_name=None,
                 pae_path=None, distogram_path=None):
        self.structure_path = structure_path
        self.kind = kind                    # 'af2', 'af3' or 'unknown'
        self.model_name = model_name        # e.g. 'model_3_multimer_v3_pred_0' or 'seed-1_sample-0'
        self.pae_path = pae_path
        self.distogram_path = distogram_path

    def __repr__(self):
        return (f'PredictionFiles(kind={self.kind!r}, model={self.model_name!r}, '
                f'pae={self.pae_path!r}, distogram={self.distogram_path!r})')


# -----------------------------------------------------------------------------
# Finding files
# -----------------------------------------------------------------------------

def structure_path_for_directory(directory, rank=0):
    """Return the structure file of the model with the given rank in a prediction directory."""
    d = os.path.abspath(os.path.expanduser(directory))
    if not os.path.isdir(d):
        raise ValueError(f'{directory} is not a directory')

    # AlphaFold3
    tops = sorted(glob.glob(os.path.join(d, '*_model.cif')))
    if tops or glob.glob(os.path.join(d, 'seed-*_sample-*', 'model.cif')):
        if rank == 0 and len(tops) == 1:
            return tops[0]
        ranked = _af3_ranked_samples(d)
        if rank >= len(ranked):
            raise ValueError(f'AlphaFold3 directory {d} has {len(ranked)} samples, '
                             f'rank {rank} requested')
        seed, sample = ranked[rank][:2]
        p = os.path.join(d, f'seed-{seed}_sample-{sample}', 'model.cif')
        if not os.path.exists(p):
            raise ValueError(f'Missing {p}')
        return p

    # AlphaFold2 / AlphaPulldown
    for ext in ('pdb', 'cif'):
        p = os.path.join(d, f'ranked_{rank}.{ext}')
        if os.path.exists(p):
            return p
    order = _af2_ranking_order(d)
    if order and rank < len(order):
        for prefix in ('relaxed_', 'unrelaxed_'):
            for ext in ('pdb', 'cif'):
                p = os.path.join(d, f'{prefix}{order[rank]}.{ext}')
                if os.path.exists(p):
                    return p

    # ColabFold
    hits = sorted(glob.glob(os.path.join(d, f'*_relaxed_rank_{rank+1:03d}_*.pdb')) +
                  glob.glob(os.path.join(d, f'*_unrelaxed_rank_{rank+1:03d}_*.pdb')))
    if hits:
        return hits[0]

    raise ValueError(f'Could not find a rank {rank} model in {d} '
                     '(expected AlphaFold2 ranked_N.pdb/cif or AlphaFold3 *_model.cif)')


def find_prediction_files(structure_path):
    """Guess PAE and distogram files belonging to a structure file."""
    path = os.path.abspath(structure_path)
    d = os.path.dirname(path)
    base = os.path.basename(path)

    # ---- AlphaFold3 sample directory: seed-S_sample-K/model.cif
    m = re.match(r'seed-(-?\d+)_sample-(\d+)$', os.path.basename(d))
    if m and base == 'model.cif':
        seed = m.group(1)
        parent = os.path.dirname(d)
        conf = _first_existing(os.path.join(d, f'confidences{s}') for s in JSON_SUFFIXES)
        return PredictionFiles(path, 'af3', f'seed-{seed}_sample-{m.group(2)}', conf,
                               _af3_distogram(parent, seed))

    # ---- AlphaFold3 top level: <name>_model.cif
    if base.endswith('_model.cif'):
        prefix = base[:-len('_model.cif')]
        conf = _first_existing(os.path.join(d, f'{prefix}_confidences{s}') for s in JSON_SUFFIXES)
        if conf or os.path.exists(os.path.join(d, 'ranking_scores.csv')):
            ranked = _af3_ranked_samples(d, fallback=False)
            seed = ranked[0][0] if ranked else None
            model_name = f'seed-{ranked[0][0]}_sample-{ranked[0][1]}' if ranked else 'top'
            # Directory order and a lone available distogram do not identify the top seed.
            distogram = _af3_distogram(d, seed) if seed is not None else None
            return PredictionFiles(path, 'af3', model_name, conf, distogram)

    # ---- AlphaFold2 / AlphaPulldown
    model_name = None
    m = re.match(r'ranked_(\d+)\.(pdb|cif)$', base)
    if m:
        order = _af2_ranking_order(d)
        i = int(m.group(1))
        if order and i < len(order):
            model_name = order[i]
    else:
        m = re.match(r'(?:un)?relaxed_(.+)\.(pdb|cif)$', base)
        if m:
            model_name = m.group(1)
    if model_name is not None:
        pae = _first_existing(os.path.join(d, f'pae_{model_name}{s}') for s in JSON_SUFFIXES)
        pkl = _first_existing(os.path.join(d, f'result_{model_name}{s}') for s in PICKLE_SUFFIXES)
        if pae is None:
            pae = pkl
        disto = pkl if (pkl and pickle_has_distogram(pkl)) else None
        return PredictionFiles(path, 'af2', model_name, pae, disto)

    # ---- ColabFold: X_unrelaxed_rank_001_<tag>.pdb, X_scores_rank_001_<tag>.json,
    #      X_all_rank_001_<tag>.pickle (only with --save-all)
    m = re.match(r'(.*)_(?:un)?relaxed_(rank_\d+_.+)\.(pdb|cif)$', base)
    if m:
        stem, tag = m.group(1), m.group(2)
        pae = _first_existing([os.path.join(d, f'{stem}_scores_{tag}.json')])
        allp = _first_existing(os.path.join(d, f'{stem}_all_{tag}{s}') for s in PICKLE_SUFFIXES)
        disto = allp if (allp and pickle_has_distogram(allp)) else None
        return PredictionFiles(path, 'colabfold', tag, pae, disto)

    # ---- Fall back to ChimeraX's own PAE file guessing
    pae = None
    try:
        from chimerax.alphafold.pae import _matching_pae_file
        pae = _matching_pae_file(path)
    except Exception:
        pass
    return PredictionFiles(path, 'unknown', None, pae, None)


def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _af2_ranking_order(d):
    p = os.path.join(d, 'ranking_debug.json')
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f).get('order')
    except Exception:
        return None


def _af3_ranked_samples(d, fallback=True):
    """[(seed, sample, score), ...] best first, from ranking_scores.csv or sample dirs."""
    rows = []
    p = os.path.join(d, 'ranking_scores.csv')
    if os.path.exists(p):
        with open(p) as f:
            header = f.readline().strip().split(',')
            for line in f:
                vals = line.strip().split(',')
                if len(vals) != len(header):
                    continue
                r = dict(zip(header, vals))
                try:
                    rows.append((r['seed'], r['sample'], float(r['ranking_score'])))
                except (KeyError, ValueError):
                    pass
    if not rows and fallback:
        for sd in sorted(glob.glob(os.path.join(d, 'seed-*_sample-*'))):
            m = re.match(r'seed-(-?\d+)_sample-(\d+)$', os.path.basename(sd))
            if m:
                rows.append((m.group(1), m.group(2), 0.0))
    rows.sort(key=lambda r: -r[2])
    return rows


def _af3_distogram(d, seed=None):
    # A known seed must never fall back to a different seed's predictions.
    pattern = '*' if seed is None else str(seed)
    hits = [p for suffix in NPZ_SUFFIXES
            for p in sorted(glob.glob(os.path.join(d, f'seed-{pattern}_distogram', '*' + suffix)))]
    if not hits:
        return None
    if seed is None and len({os.path.dirname(p) for p in hits}) != 1:
        return None                    # The top model's seed cannot be inferred safely
    return hits[0]                     # Prefer an uncompressed copy if one exists


# -----------------------------------------------------------------------------
# Cheap availability checks (no full load)
# -----------------------------------------------------------------------------

class LoadProgress:
    """How far the read of one or several distograms has come. The reading thread writes it,
    the plot polls it."""

    READ_SHARE = 0.9            # Part of a file's share for reading it; the rest for computing

    def __init__(self, steps=1):
        self.steps = steps      # Files to read
        self.step = 0           # Files finished
        self.label = 'distogram'
        self.total = 0          # Size of the current file in bytes (0: not known yet)
        self.bytes_read = 0
        self.computed = None    # Fraction of its matrices computed, once the file has been read
        self.started = time.monotonic()

    def next_file(self, step, label):
        """Start file number step (from 0) of several; label says what it is, e.g. 'seed 7'."""
        self.total = self.bytes_read = 0
        self.computed = None
        self.step = step
        self.label = f'{label} ({step + 1} of {self.steps})'

    def _file_fraction(self):
        if self.computed is not None:
            return self.READ_SHARE + (1 - self.READ_SHARE) * self.computed
        if not self.total:
            return None
        return self.READ_SHARE * min(1.0, self.bytes_read / self.total)

    def fraction(self):
        """Overall progress from 0 to 1, or None while the size of a single file is not known."""
        f = self._file_fraction()
        if f is None and self.steps == 1:
            return None
        return (self.step + (f or 0)) / self.steps

    def seconds_left(self):
        """Estimated time to finish, or None if it is too early to tell."""
        if self.steps > 1:
            done = self.fraction()
        elif not self.total or self.computed is not None:
            return None
        else:
            done = min(1.0, self.bytes_read / self.total)
        elapsed = time.monotonic() - self.started
        if done < 0.01 or elapsed < 3:
            return None
        return elapsed * (1 - done) / done

    def message(self):
        if self.computed is not None:
            what = 'contact probabilities' if self.steps > 1 else 'contact probabilities and spread'
            return f'Loading {self.label}: computing {what}'
        if not self.total:
            return f'Loading {self.label} ...'
        text = (f'Loading {self.label}: {_file_size(min(self.bytes_read, self.total))} of '
                f'{_file_size(self.total)}')
        left = self.seconds_left()
        if left is None:
            return text
        if left < 60:
            return f'{text}, less than a minute left'
        return f'{text}, about {round(left / 60)} min left'


def _file_size(n):
    return f'{n / 1e9:.2f} GB' if n >= 1e9 else f'{n / 1e6:.0f} MB'


class _ProgressFile:
    """A binary file that counts in a LoadProgress how many bytes have been read from it."""

    CHUNK = 16 << 20            # Large reads are split so that the bar keeps moving

    def __init__(self, path, progress):
        self._f = open(path, 'rb')
        self._progress = progress
        progress.total = os.fstat(self._f.fileno()).st_size

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._f.close()

    def __getattr__(self, name):
        return getattr(self._f, name)

    def read(self, n=-1):
        if n is None:
            n = -1
        if 0 <= n <= self.CHUNK:
            data = self._f.read(n)
            self._progress.bytes_read += len(data)
            return data
        parts = []
        while n != 0:
            data = self._f.read(self.CHUNK if n < 0 else min(n, self.CHUNK))
            if not data:
                break
            self._progress.bytes_read += len(data)
            parts.append(data)
            if n > 0:
                n -= len(data)
        return b''.join(parts)

    def readinto(self, buffer):
        view = memoryview(buffer).cast('B')
        done = 0
        while done < len(view):
            n = self._f.readinto(view[done:done + self.CHUNK])
            if not n:
                break
            self._progress.bytes_read += n
            done += n
        return done


@contextlib.contextmanager
def _open_with_progress(path, progress):
    with _ProgressFile(path, progress) as raw:
        if path.endswith('.gz'):
            with gzip.open(raw, 'rb') as f:
                yield f
        elif path.endswith('.xz'):
            import lzma
            with lzma.open(raw, 'rb') as f:
                yield f
        else:
            yield raw


def _open_maybe_gz(path, progress=None):
    if progress is not None:
        return _open_with_progress(path, progress)
    if path.endswith('.gz'):
        return gzip.open(path, 'rb')
    if path.endswith('.xz'):
        import lzma
        return lzma.open(path, 'rb')
    return open(path, 'rb')


def pickle_has_distogram(path, peek_bytes=4 << 20):
    """
    True if an AlphaFold result pickle stores a distogram.

    Reads only the start of the (decompressed) stream: AlphaFold writes the result dict with
    'distogram' as its first key, so the key name appears within the first bytes.
    """
    try:
        with _open_maybe_gz(path) as f:
            head = f.read(peek_bytes)
    except Exception:
        return False
    return b'distogram' in head


def distogram_file_ok(path):
    if path is None or not os.path.exists(path):
        return False
    if path.endswith(PICKLE_SUFFIXES):
        return pickle_has_distogram(path)
    return path.endswith(NPZ_SUFFIXES + NPY_SUFFIXES)


def contact_metrics(contact_probs, source, cutoff=8.0):
    """Metrics dict when only contact probabilities (no full distogram) are available."""
    return {'contact': np.asarray(contact_probs, np.float32), 'cutoff': float(cutoff),
            'path': source, 'contact_only': True}


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def _load_pickle(path, progress=None):
    with _open_maybe_gz(path, progress) as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as e:
            if 'jax' not in str(e):
                raise
    from chimerax.alphafold.pae import _fix_alphafold_pickle_jax_dependency
    _fix_alphafold_pickle_jax_dependency()
    if progress is not None:
        progress.bytes_read = 0
    with _open_maybe_gz(path, progress) as f:
        return pickle.load(f)


def read_pae(path):
    """
    Returns (pae, contact_probs or None) with float32 N x N matrices.

    Reads AF2 pae json / result pickle, AF3 confidences json (which also carries
    contact_probs), ColabFold scores json, npy/npz; .gz and .xz compressed files too.
    """
    if path.endswith(PICKLE_SUFFIXES):
        p = _load_pickle(path)
        for key in ('predicted_aligned_error', 'pae'):
            if isinstance(p, dict) and key in p:
                return np.asarray(p[key], dtype=np.float32), None
        raise ValueError(f'{path} has no predicted_aligned_error')
    if path.endswith(JSON_SUFFIXES):
        with _open_maybe_gz(path) as f:
            j = json.load(f)
        if isinstance(j, list) and j and isinstance(j[0], dict):
            j = j[0]
        if not isinstance(j, dict):
            raise ValueError(f'{path} is not a PAE json file')
        if 'residue1' in j and 'distance' in j:          # AlphaFold DB v1/v2
            r1 = np.asarray(j['residue1'], int) - 1
            r2 = np.asarray(j['residue2'], int) - 1
            n = r1.max() + 1
            pae = np.zeros((n, n), np.float32)
            pae[r1, r2] = j['distance']
            return pae, None
        key = next((k for k in ('pae', 'predicted_aligned_error') if k in j), None)
        if key is None:
            raise ValueError(f'{path} has no "pae" or "predicted_aligned_error"')
        pae = np.asarray(j[key], dtype=np.float32)
        cp = j.get('contact_probs')
        cp = np.asarray(cp, dtype=np.float32) if cp is not None else None
        return pae, cp
    from chimerax.alphafold.pae import read_pae_matrix
    return np.asarray(read_pae_matrix(path), dtype=np.float32), None


def read_distogram(path, progress=None):
    """
    Returns (values, bin_edges, are_logits) with values of shape (N, N, nbins).
    progress: a LoadProgress to update while reading.
    """
    if path.endswith(PICKLE_SUFFIXES):
        p = _load_pickle(path, progress)
        dg = p.get('distogram') if isinstance(p, dict) else None
        if dg is None:
            raise ValueError(f'{path} contains no distogram')
        if isinstance(dg, dict):
            logits = dg.get('logits')
            edges = dg.get('bin_edges')
        else:
            logits, edges = dg, None
        values = np.asarray(logits)
        del p
        are_logits = True
    elif path.endswith(NPZ_SUFFIXES):
        with _open_maybe_gz(path, progress) as f, np.load(f, allow_pickle=False) as z:
            key = next((k for k in ('distogram', 'logits', 'distogram_logits', 'probs')
                        if k in z.files), z.files[0])
            values = z[key]
            edges = z['bin_edges'] if 'bin_edges' in z.files else None
        are_logits = None
    else:
        with _open_maybe_gz(path, progress) as f:
            values = np.load(f, allow_pickle=False)
        edges = None
        are_logits = None

    if values.ndim != 3 or values.shape[0] != values.shape[1]:
        raise ValueError(f'Distogram in {path} has shape {values.shape}, expected (N, N, bins)')
    nbins = values.shape[-1]
    if edges is None:
        if nbins != len(AF_BIN_EDGES) + 1:
            raise ValueError(f'Distogram in {path} has {nbins} bins and no bin_edges; '
                             f'expected the AlphaFold 64-bin layout')
        edges = AF_BIN_EDGES
    edges = np.asarray(edges, dtype=np.float32)
    if are_logits is None:
        # Probabilities are non-negative and sum to 1 along the bin axis.
        s = values[:2, :2].astype(np.float64)
        are_logits = not (s.min() >= 0 and np.allclose(s.sum(-1), 1, atol=1e-2))
    return values, edges, are_logits


def bin_centers(edges):
    edges = np.asarray(edges, dtype=np.float64)
    step = edges[1] - edges[0]
    mids = 0.5 * (edges[:-1] + edges[1:])
    return np.concatenate([[edges[0] - step / 2], mids, [edges[-1] + step / 2]])


def distogram_metrics(values, edges, are_logits=True, cutoff=8.0, chunk=32, progress=None):
    """
    Per-pair matrices from a distogram (N, N, nbins):

    contact  P(d < cutoff): probability mass in bins whose upper edge is <= cutoff
             (with cutoff 8 this reproduces AlphaFold3's ``contact_probs``)
    std      standard deviation of the predicted distance distribution (A)
    entropy  Shannon entropy of the bin probabilities (bits, max log2(nbins))
    mean     expected distance (A)

    far      probability mass in the last bin (d > last edge, 21.7 A for AlphaFold).
             The last bin collects every longer distance, so where far is high the
             spread is not meaningful; the plot greys those pairs out.
    """
    n, _, nbins = values.shape
    centers = bin_centers(edges).astype(np.float32)
    centers2 = centers * centers
    nbelow = int(np.searchsorted(edges, cutoff, side='right'))
    out = {k: np.empty((n, n), np.float32) for k in ('contact', 'std', 'entropy', 'mean', 'far')}
    tiny = np.float32(1e-12)
    for i0 in range(0, n, chunk):
        if progress is not None:
            progress.computed = i0 / n
        i1 = min(n, i0 + chunk)
        x = np.array(values[i0:i1], dtype=np.float32)
        if are_logits:
            x -= x.max(axis=-1, keepdims=True)
            np.exp(x, out=x)
        x /= x.sum(axis=-1, keepdims=True)
        out['contact'][i0:i1] = x[..., :nbelow].sum(axis=-1)
        out['far'][i0:i1] = x[..., -1]
        mean = x @ centers
        var = x @ centers2 - mean * mean
        out['mean'][i0:i1] = mean
        out['std'][i0:i1] = np.sqrt(np.maximum(var, 0))
        out['entropy'][i0:i1] = -(x * np.log2(x + tiny)).sum(axis=-1)
    out['cutoff'] = float(cutoff)
    out['far_edge'] = float(edges[-1])
    out['contact_upper_edge'] = float(edges[nbelow - 1]) if nbelow > 0 else float(edges[0])
    return out


def distogram_contact(values, edges, are_logits=True, cutoff=8.0, chunk=32, progress=None):
    """P(d < cutoff) only, computed as in distogram_metrics."""
    n = values.shape[0]
    nbelow = int(np.searchsorted(edges, cutoff, side='right'))
    out = np.empty((n, n), np.float32)
    for i0 in range(0, n, chunk):
        if progress is not None:
            progress.computed = i0 / n
        i1 = min(n, i0 + chunk)
        x = np.array(values[i0:i1], dtype=np.float32)
        if are_logits:
            x -= x.max(axis=-1, keepdims=True)
            np.exp(x, out=x)
        x /= x.sum(axis=-1, keepdims=True)
        out[i0:i1] = x[..., :nbelow].sum(axis=-1)
    return out


def load_distogram_metrics(path, cutoff=8.0, progress=None):
    values, edges, are_logits = read_distogram(path, progress)
    m = distogram_metrics(values, edges, are_logits, cutoff=cutoff, progress=progress)
    m['path'] = path
    m['nbins'] = values.shape[-1]
    return m


# -----------------------------------------------------------------------------
# All models of a prediction (for the model switcher)
# -----------------------------------------------------------------------------

def prediction_models(structure_path):
    """
    Models of the prediction a structure file belongs to, best first:
    [{'label', 'path' (None if not present locally), 'score'}].
    """
    path = os.path.abspath(structure_path)
    d = os.path.dirname(path)
    if re.match(r'seed-(-?\d+)_sample-(\d+)$', os.path.basename(d)):
        d = os.path.dirname(d)
    models = []

    if os.path.exists(os.path.join(d, 'ranking_debug.json')):         # AlphaFold2
        try:
            with open(os.path.join(d, 'ranking_debug.json')) as f:
                rd = json.load(f)
        except Exception:
            rd = {}
        scores = rd.get('iptm+ptm') or rd.get('plddts') or {}
        score_name = 'iptm+ptm' if 'iptm+ptm' in rd else 'pLDDT'
        for i, m in enumerate(rd.get('order', [])):
            p = _first_existing([os.path.join(d, f'ranked_{i}.pdb'), os.path.join(d, f'ranked_{i}.cif')] +
                                [os.path.join(d, f'{pre}{m}.{ext}') for pre in ('relaxed_', 'unrelaxed_')
                                 for ext in ('pdb', 'cif')])
            sc = scores.get(m)
            label = f'ranked_{i}  {m}' + (f'  {score_name} {sc:.3f}' if sc is not None else '')
            models.append({'label': label, 'path': p, 'score': sc})
        return models

    ranked = _af3_ranked_samples(d)                                   # AlphaFold3
    tops = sorted(glob.glob(os.path.join(d, '*_model.cif')))
    if ranked or tops:
        if tops:
            best = f' = seed {ranked[0][0]} sample {ranked[0][1]}' if ranked else ''
            models.append({'label': f'top model{best}', 'path': tops[0],
                           'score': ranked[0][2] if ranked else None})
        for k, (seed, sample, score) in enumerate(ranked):
            p = os.path.join(d, f'seed-{seed}_sample-{sample}', 'model.cif')
            models.append({'label': f'#{k + 1}  seed {seed} sample {sample}  score {score:.3f}',
                           'path': p if os.path.exists(p) else None, 'score': score})
    return models


# -----------------------------------------------------------------------------
# Independent runs of a prediction (for the agreement view)
# -----------------------------------------------------------------------------

def agreement_runs(structure_path):
    """
    Independent runs of the prediction a structure file belongs to, best first: one per
    AlphaFold3 seed (the samples of a seed share one distogram and the same contact_probs)
    or one per AlphaFold2 model. Each run is a dict with 'label', 'kind' ('seed' or 'model'),
    'source' ('distogram' or 'contact_probs') and 'path'. AlphaFold2 result pickles are
    listed whether or not they kept the distogram; run_contact_probs() finds out.
    """
    path = os.path.abspath(structure_path)
    d = os.path.dirname(path)
    if re.match(r'seed-(-?\d+)_sample-(\d+)$', os.path.basename(d)):
        d = os.path.dirname(d)
    runs = []

    if os.path.exists(os.path.join(d, 'ranking_debug.json')):         # AlphaFold2
        names = _af2_ranking_order(d) or []
        for name, label in zip(names, _af2_run_labels(names)):
            pkl = _first_existing(os.path.join(d, f'result_{name}{s}') for s in PICKLE_SUFFIXES)
            if pkl:
                runs.append({'label': label, 'kind': 'model', 'source': 'distogram',
                             'path': pkl})
        return runs

    shown = find_prediction_files(path)                                # AlphaFold3
    m = re.match(r'seed-(-?\d+)_sample-\d+$', shown.model_name or '') if shown.kind == 'af3' else None
    shown_seed = m.group(1) if m else None
    done = set()
    for seed, sample, _ in _af3_ranked_samples(d):
        if seed in done:
            continue
        distogram = _af3_distogram(d, seed)
        if distogram:
            source, p = 'distogram', distogram
        elif seed == shown_seed and shown.pae_path:
            source, p = 'contact_probs', shown.pae_path      # Already read for the plot
        else:
            p = _first_existing(os.path.join(d, f'seed-{seed}_sample-{sample}', f'confidences{s}')
                                for s in JSON_SUFFIXES)
            if p is None:
                continue                # Try the next sample of this seed
            source = 'contact_probs'
        done.add(seed)
        runs.append({'label': f'seed {seed}', 'kind': 'seed', 'source': source, 'path': p})
    return runs


def agreement_shortfall(structure_path, runs):
    """Why agreement_runs() found fewer than two runs, for messages."""
    n = len(runs)
    if structure_path:
        d = os.path.dirname(os.path.abspath(structure_path))
        if re.match(r'seed-(-?\d+)_sample-(\d+)$', os.path.basename(d)):
            d = os.path.dirname(d)
        seeds = {r[0] for r in _af3_ranked_samples(d)}
        if len(seeds) > n:
            return (f'{n} of the {len(seeds)} seeds have files here; copy the seed-*_sample-* '
                    f'folders too (scripts/fetch_pred.sh does)')
        if len(seeds) == 1:
            return 'the prediction has one seed'
    return f'found {n}'


def _af2_run_labels(names):
    """'model 3', or 'model 3 pred 1' when there are several predictions per model."""
    parsed = [re.match(r'model_(\d+)(?:_[a-z]+(?:_v\d+)?)?(?:_pred_(\d+))?$', n) for n in names]
    if not names or not all(parsed):
        return list(names)
    if len({p.group(2) for p in parsed}) <= 1:
        labels = [f'model {p.group(1)}' for p in parsed]
    else:
        labels = [f'model {p.group(1)} pred {p.group(2) or 0}' for p in parsed]
    return labels if len(set(labels)) == len(labels) else list(names)


def run_contact_probs(run, cutoff=8.0, progress=None):
    """P(d < cutoff) (N x N, float32) of one run from agreement_runs()."""
    path = run['path']
    if run['source'] == 'distogram':
        if path.endswith(PICKLE_SUFFIXES) and not pickle_has_distogram(path):
            raise ValueError(f'{os.path.basename(path)} has no distogram')
        values, edges, are_logits = read_distogram(path, progress)
        return distogram_contact(values, edges, are_logits, cutoff=cutoff, progress=progress)
    if abs(cutoff - 8.0) > 1e-6:
        raise ValueError('AlphaFold3 contact_probs are fixed at 8 Å; another cutoff needs '
                         'the distogram')
    cp = read_pae(path)[1]
    if cp is None:
        raise ValueError(f'{os.path.basename(path)} has no contact_probs')
    return cp
