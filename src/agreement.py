"""
Agreement between the independent runs of one prediction: the seeds of an AlphaFold3
prediction (the samples of a seed share one distogram, so they are not independent) or the
models of an AlphaFold2 prediction. For every residue pair it counts the runs in which the
contact probability P(d < cutoff) reaches a threshold.

Nothing in this module touches ChimeraX, so it is safe to call from a worker thread.
"""

import numpy as np

from . import data

TOP_N = 10                  # Interface score of a run: mean of its TOP_N highest inter-chain P
MAX_BYTES = 4 << 30         # Largest stack of per-run matrices kept in memory


class Agreement:

    def __init__(self, labels, kind, stack, cutoff):
        self.labels = list(labels)      # One per run, best first
        self.kind = kind                # 'seed' or 'model'
        self.stack = stack              # (runs, N, N) float16 P(d < cutoff), symmetric
        self.cutoff = cutoff
        self._counts = None             # (threshold, counts) of the last call

    @property
    def n(self):
        return len(self.labels)

    def counts(self, threshold):
        '''Number of runs with P >= threshold for every residue pair (float32 N x N).'''
        if self._counts is None or self._counts[0] != threshold:
            t = np.float16(threshold)
            c = np.zeros(self.stack.shape[1:], np.float32)
            for m in self.stack:
                c += m >= t
            self._counts = (threshold, c)
        return self._counts[1]

    def pair_values(self, i, j):
        '''P of residue pair (i, j) in each run.'''
        return self.stack[:, i, j].astype(np.float32)

    def chain_pair_scores(self, chains, threshold, top_n=TOP_N):
        '''
        {(chain a, chain b): (mean of the top_n highest P in each run, number of runs with a
        residue pair at P >= threshold)} for every pair of chains. chains: chain id per row.
        '''
        chains = np.asarray(chains)
        ids = [c for c in dict.fromkeys(chains.tolist()) if c]
        rows = {c: np.flatnonzero(chains == c) for c in ids}
        t = np.float16(threshold)
        out = {}
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                block = np.ix_(rows[ids[a]], rows[ids[b]])
                means, hits = [], 0
                for m in self.stack:
                    v = m[block].ravel()
                    k = min(top_n, v.size)
                    means.append(float(np.partition(v.astype(np.float32), v.size - k)[-k:].mean()))
                    hits += bool((v >= t).any())
                out[(ids[a], ids[b])] = (means, hits)
        return out


def load(runs, cutoff=8.0, full_size=None, rows=None, known=None):
    '''
    Read the contact probabilities of every run. Returns (Agreement, or None if fewer than
    two runs could be read, and messages about the runs that were skipped).

    runs       from data.agreement_runs(), best first
    full_size  tokens in the PAE matrix of the shown model; runs of another size are skipped
    rows       matrix rows kept for the structure (AlphaLensPAE._rows), or None for all rows
    known      {path: P} already computed for these rows (the shown model)
    '''
    n = len(rows) if rows is not None else full_size
    need = len(runs) * n * n * 2
    if need > MAX_BYTES:
        raise ValueError(f'{len(runs)} runs of {n} x {n} residues need {need / 2**30:.1f} GB; '
                         f'the limit is {MAX_BYTES / 2**30:.0f} GB')
    stack = np.empty((len(runs), n, n), np.float16)
    labels, msgs = [], []
    for run in runs:
        P = (known or {}).get(run['path'])
        if P is not None and P.shape != (n, n):
            P = None
        if P is None:
            try:
                P = data.run_contact_probs(run, cutoff)
            except Exception as e:
                msgs.append(f'{run["label"]}: {e}')
                continue
            if P.shape != (full_size, full_size):
                msgs.append(f'{run["label"]}: {P.shape[0]} tokens, the shown model has {full_size}')
                continue
            if rows is not None:
                P = P[rows, :][:, rows]
        np.maximum(P, P.T, out=stack[len(labels)])
        labels.append(run['label'])
    if len(labels) < 2:
        return None, msgs
    return Agreement(labels, runs[0]['kind'], stack[:len(labels)], cutoff), msgs
