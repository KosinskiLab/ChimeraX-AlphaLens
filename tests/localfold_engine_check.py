#!/usr/bin/env python3
"""Opt-in real-engine check without ChimeraX, for the Metal (macOS) and CUDA (Linux) builds alike.

Runs AlphaLens's own PredictionJob with an installed LocalFold engine on barnase-barstar:
AF2-Multimer (with AlphaLens's contact head), AF2 monomer (barnase, from a cropped MSA) and AF3.
Each structure is scored against the 1BRS crystal, which is never an input.

    python3 tests/localfold_engine_check.py --msa <barnase+barstar A3M> --crystal <1brs-crystal.pdb> \\
        --out <new folder> [--prefix ~/.local/share/alphalens] [--cache <weights cache>] [--reference <earlier --out>]

The A3M query must be 1BRS chains A then D (108 + 87 residues). The crystal is LocalFold's public
fixture tools/fixtures/1brs-crystal.pdb. --reference compares each case with an earlier run of this
script, for example Metal against CUDA at the same seed. Needs numpy. Writes <out>/summary.json and
exits 0 only when every case completes with a verified engine, contact probabilities, and a complex
CA RMSD to the crystal within --max-rmsd.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import types

SRC = Path(__file__).resolve().parents[1] / 'src'
package = types.ModuleType('alphalens')      # src/__init__.py needs ChimeraX; the predict modules do not
package.__path__ = [str(SRC)]
sys.modules.setdefault('alphalens', package)

import numpy as np                            # noqa: E402
from alphalens import predict                 # noqa: E402

THREE = dict(zip('ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR'.split(),
                 predict.AMINO_ACIDS))
BARNASE = 108
CASES = ('af2-multimer', 'af2-monomer', 'af3')


def ca_chains(path):
    """Per chain: one-letter sequence and CA coordinates (first alternate location only)."""
    rows = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith('ATOM') and line[12:16].strip() == 'CA' and line[16] in ' A':
            xyz = [float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))]
            rows.setdefault(line[21], []).append((THREE.get(line[17:20], 'X'), xyz))
    return {c: (''.join(a for a, _ in r), np.array([x for _, x in r])) for c, r in rows.items()}


def kabsch_rmsd(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    u, s, vt = np.linalg.svd(a.T @ b)
    if np.linalg.det(u @ vt) < 0:
        s[-1] = -s[-1]
    return float(math.sqrt(max((a * a).sum() + (b * b).sum() - 2 * s.sum(), 0) / len(a)))


def query_of(a3m):
    rows, started = [], False
    for line in Path(a3m).read_text(**predict.A3M_TEXT).splitlines():
        if line.startswith('#') or not line.strip():
            continue
        if line.startswith('>'):
            if started:
                break
            started = True
        elif started:
            rows.append(''.join(c for c in line.strip() if not c.islower() and c != '.'))
    return ''.join(rows)


def chain(cid, sequence):
    return dict(source_chain=cid, sequence=sequence,
                residue_map=[dict(chain_id=cid, number=i + 1) for i in range(len(sequence))])


def run(case, args, query, crystal):
    backend = 'af3' if case == 'af3' else 'af2'
    barnase, barstar = query[:BARNASE], query[BARNASE:]
    chains = [chain('A', barnase)] if case == 'af2-monomer' else [chain('A', barnase), chain('D', barstar)]
    crop = dict(source_query=query, columns=list(range(BARNASE))) if case == 'af2-monomer' else None
    job = predict.PredictionJob(args.out / case, chains, executable=args.prefix / 'bin' / f'localfold-{backend}',
                                msa=args.msa, msa_crop=crop, seed=args.seed, cache=args.cache, backend=backend,
                                accept_af3_terms=backend == 'af3', timeout=args.timeout)
    started = time.monotonic()
    job.start()
    stages, growth = [], []
    while job.poll() == 'running':
        stage = job.read_progress().stage
        if not stages or stages[-1] != stage:
            stages.append(stage)
        size = (job.directory / 'prediction.log').stat().st_size
        if size and (not growth or growth[-1][1] != size):
            growth.append((round(time.monotonic() - started, 2), size))
        time.sleep(.25)
    record = dict(case=case, state=job.state, error=job.error, seconds=round(time.monotonic() - started, 2),
                  engine=job.manifest.get('engine'), revision=job.manifest.get('revision'),
                  runtime_environment=job.manifest.get('runtime_environment'),
                  contact_head=job.manifest.get('contact_head_sha256') is not None,
                  stages=stages, log_updates=len(growth), first_output_s=growth[0][0] if growth else None)
    if job.state != 'complete':
        return record
    result = job.result
    record.update({k: result.get(k) for k in ('ptm', 'iptm', 'mean_plddt', 'has_contact_probabilities')})
    predicted = ca_chains(job.directory / 'fold.pdb')
    order = predict.chain_ids(len(chains))
    reference = ['A', 'D'][:len(chains)]
    if [predicted[c][0] for c in order] != [crystal[c][0] for c in reference]:
        record['error'] = 'Predicted sequences do not match 1BRS chains ' + '/'.join(reference)
        return record
    xyz = np.vstack([predicted[c][1] for c in order])
    record['rmsd_to_crystal'] = round(kabsch_rmsd(xyz, np.vstack([crystal[c][1] for c in reference])), 3)
    record['chain_rmsd_to_crystal'] = {r: round(kabsch_rmsd(predicted[c][1], crystal[r][1]), 3)
                                       for c, r in zip(order, reference)}
    confidences = json.loads((job.directory / 'fold_confidences.json').read_text())
    if len(chains) > 1 and confidences.get('contact_probs') is not None:
        p = np.array(confidences['contact_probs'])[:BARNASE, BARNASE:]
        a, b = crystal['A'][1], crystal['D'][1]
        near = np.linalg.norm(a[:, None] - b[None], axis=-1) < 10
        predicted_contacts = p >= .5
        record['interface_contacts_p50'] = int(predicted_contacts.sum())
        record['interface_precision_ca10'] = (round(float(near[predicted_contacts].mean()), 3)
                                              if predicted_contacts.any() else None)
    if args.reference:
        other = args.reference / case / 'fold.pdb'
        if other.is_file():
            theirs = ca_chains(other)
            record['rmsd_to_reference'] = round(kabsch_rmsd(xyz, np.vstack([theirs[c][1] for c in order])), 3)
            summary = json.loads((args.reference / case / 'fold_summary_confidences.json').read_text())
            record['reference_delta'] = {k: round(record[k] - summary[k], 4) for k in ('ptm', 'mean_plddt')
                                         if isinstance(summary.get(k), (int, float))}
            if record.get('iptm') is not None and isinstance(summary.get('iptm'), (int, float)):
                record['reference_delta']['iptm'] = round(record['iptm'] - summary['iptm'], 4)
    return record


def problems(record, max_rmsd):
    found = []
    if record['state'] != 'complete' or record.get('error'):
        found.append(f"{record['state']}: {record.get('error')}")
        return found
    if record['revision'] != predict.ENGINE_REVISION:
        found.append(f"engine revision {record['revision']}, expected the pinned {predict.ENGINE_REVISION}")
    if not record.get('has_contact_probabilities'):
        found.append('no contact probabilities')
    if record['case'] == 'af2-multimer' and not record['contact_head']:
        found.append('AF2-Multimer ran without the AlphaLens contact head')
    if record.get('rmsd_to_crystal') is None or record['rmsd_to_crystal'] > max_rmsd:
        found.append(f"CA RMSD to 1BRS {record.get('rmsd_to_crystal')} > {max_rmsd} A")
    if record['case'] != 'af2-monomer' and not (record.get('interface_contacts_p50') or 0) >= 10:
        found.append(f"only {record.get('interface_contacts_p50')} interface contacts at P >= 0.5")
    if (record.get('interface_precision_ca10') or 1) < .8:
        found.append(f"interface contact precision {record['interface_precision_ca10']} < 0.8")
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--msa', type=Path, required=True)
    parser.add_argument('--crystal', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='new folder')
    parser.add_argument('--prefix', type=Path, default=predict.data_directory())
    parser.add_argument('--cache', type=Path, help='weights cache (default: AlphaLens\'s)')
    parser.add_argument('--reference', type=Path, help='an earlier --out to compare with')
    parser.add_argument('--cases', default=','.join(CASES))
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--max-rmsd', type=float, default=1.0)
    args = parser.parse_args()
    args.out = args.out.expanduser().resolve()
    args.out.mkdir(parents=True)
    query = query_of(args.msa)
    crystal = ca_chains(args.crystal)
    if query != crystal['A'][0] + crystal['D'][0]:
        parser.error('The A3M query is not 1BRS chain A followed by chain D.')
    records, failed = [], False
    for case in args.cases.split(','):
        if case not in CASES:
            parser.error(f'Unknown case {case}; choose from {", ".join(CASES)}.')
        record = run(case, args, query, crystal)
        record['problems'] = problems(record, args.max_rmsd)
        failed |= bool(record['problems'])
        records.append(record)
        print(json.dumps(record), flush=True)
    summary = dict(native=predict.engine_platform(), engine_revision=predict.ENGINE_REVISION, seed=args.seed,
                   prefix=str(args.prefix), passed=not failed, cases=records)
    predict.write_json(args.out / 'summary.json', summary)
    print('LocalFold engine check ' + ('FAILED' if failed else 'PASSED'), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
