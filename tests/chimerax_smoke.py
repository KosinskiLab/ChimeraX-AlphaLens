"""Run with ChimeraX --nogui --exit --cmd 'open tests/chimerax_smoke.py'.

Loads the checkout in this process without installing it or changing user preferences.
The ordinary pytest suite uses stand-ins; this smoke check uses real ChimeraX structures,
command parsing, PAE objects and CSV exports for synthetic AF2 and AF3 predictions.
"""

import csv
import gzip
import importlib.util
import json
from pathlib import Path
import pickle
import shutil
import sys
import tempfile

import numpy as np
from chimerax.core.commands import quote_path_if_necessary as quote, run
from chimerax.core.errors import UserError


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def probabilities(p):
    values = np.zeros((4, 4, 64), np.float32)
    values[..., 0] = p
    values[..., -1] = 1 - p
    return values


def main(session):
    src = Path(__file__).resolve().parents[1] / 'src'
    name = 'chimerax.alphalens_smoke'
    spec = importlib.util.spec_from_file_location(name, src / '__init__.py',
                                                submodule_search_locations=[str(src)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    cmd = importlib.import_module(name + '.cmd')
    cmd.register_commands(session.logger)

    with tempfile.TemporaryDirectory(prefix='alphalens-smoke-') as root:
        d = Path(root)
        af2 = d / 'af2'
        af2.mkdir()
        pdb = af2 / 'ranked_0.pdb'
        lines = ['SEQRES   1 A    2  ALA ALA\n', 'SEQRES   1 B    2  ALA ALA\n']
        serial = 1
        for chain, y in (('A', 0.), ('B', 5.)):
            for residue in (1, 2):
                x = (residue - 1) * 3.8
                for atom, dx, dy, z in (('N', -1.2, .2, -.1), ('CA', 0., 0., 0.),
                                        ('C', 1.3, .5, .1), ('O', 1.8, 1.5, .15),
                                        ('CB', 0., -1.5, 1.)):
                    lines.append(f'ATOM  {serial:5d} {atom:^4s} ALA {chain}{residue:4d}    '
                                 f'{x + dx:8.3f}{y + dy:8.3f}{z:8.3f}'
                                 f'{1.:6.2f}{90.:6.2f}          {atom[0]:>2s}  \n')
                    serial += 1
            lines.append('TER\n')
        pdb.write_text(''.join(lines) + 'END\n')
        (af2 / 'ranking_debug.json').write_text(json.dumps({'order': ['model_1', 'model_2']}))
        for model, p in ((1, .9), (2, .1)):
            result = {'distogram': {'logits': np.log(np.maximum(probabilities(p), 1e-9))},
                      'predicted_aligned_error': np.full((4, 4), 5.)}
            (af2 / f'pae_model_{model}.json').write_text(json.dumps({'pae': [[5.] * 4] * 4}))
            with (af2 / f'result_model_{model}.pkl').open('wb') as f:
                pickle.dump(result, f)
        out = d / 'af2.csv'
        run(session, f'alphalens open {quote(str(af2))} mode agreement '
                     f'wait true saveContacts {quote(str(out))}')
        rows = read_csv(out)
        assert len(rows) == 4 and rows[0]['models'] == '2'
        assert rows[0]['p_contact'] == '0.900'

        af3 = d / 'af3'
        af3.mkdir()
        (af3 / 'ranking_scores.csv').write_text('seed,sample,ranking_score\n1,0,0.9\n2,0,0.1\n')
        cif = d / 'template.cif'
        run(session, f'save {quote(str(cif))} models #1')
        for seed, p in ((1, .9), (2, .1)):
            sample = af3 / f'seed-{seed}_sample-0'
            sample.mkdir()
            shutil.copyfile(cif, sample / 'model.cif')
            (sample / 'confidences.json').write_text(json.dumps({
                'pae': [[5.] * 4] * 4, 'contact_probs': [[p] * 4] * 4}))
        # Only seed 1 has a distogram, externally compressed.
        folder = af3 / 'seed-1_distogram'
        folder.mkdir()
        plain = d / 'distogram.npz'
        np.savez(plain, distogram=probabilities(.9))
        with gzip.open(folder / 'distogram.npz.gz', 'wb') as f:
            f.write(plain.read_bytes())
        out = d / 'af3.csv'
        run(session, f'alphalens open {quote(str(af3))} mode agreement '
                     f'wait true saveContacts {quote(str(out))}')
        rows = read_csv(out)
        assert len(rows) == 4 and rows[0]['seeds'] == '2'
        assert rows[0]['p_seed_1'] == '0.900' and rows[0]['p_seed_2'] == '0.100'

        # Seed 2 must use its own contact_probs, not seed 1's distogram.
        out = d / 'seed2.csv'
        run(session, f'alphalens open {quote(str(af3 / "seed-2_sample-0" / "model.cif"))} '
                     f'minProb 0.05 wait true saveContacts {quote(str(out))}')
        assert read_csv(out)[0]['p_contact'] == '0.100'

        # Corrupt AF2 data must fail a requested export through the real command parser.
        (af2 / 'result_model_1.pkl').write_bytes(b'distogram: corrupt pickle')
        failed = d / 'failed.csv'
        try:
            run(session, f'alphalens open {quote(str(af2))} '
                         f'wait true saveContacts {quote(str(failed))}')
        except UserError:
            pass
        else:
            raise AssertionError('Corrupt prediction did not raise UserError')
        assert not failed.exists()
        session.logger.info('AlphaLens ChimeraX smoke checks PASSED (AF2, AF3, export, seed isolation).')


main(session)
