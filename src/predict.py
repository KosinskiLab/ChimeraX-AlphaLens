"""Experimental local AF2 monomer/multimer jobs. No ChimeraX imports or shell commands."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ENGINE_REVISION = '68c59196fa7d32c6edfc061ed369889c1cdb680b'
MODEL = 'model_1_multimer_v3'
MONOMER_MODEL = 'model_1_ptm'
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
# Total across chains, after extending a selected region with flanks.
# LocalFold also checks its estimated allocation against the GPU working-set budget.
MAX_RESIDUES = 1024


def data_directory():
    return Path.home() / '.local' / 'share' / 'alphalens'


def default_executable():
    return os.environ.get('ALPHALENS_LOCALFOLD', '') or shutil.which('localfold-af2') or str(
        data_directory() / 'bin' / 'localfold-af2')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    part = path.with_suffix(path.suffix + '.part')
    part.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    part.replace(path)


def validate_chains(chains):
    if not 1 <= len(chains) <= 2:
        raise ValueError('The prototype needs one or two protein chains.')
    for chain in chains:
        seq = chain['sequence']
        if not seq or any(c not in AMINO_ACIDS for c in seq):
            raise ValueError('Use nonempty protein sequences with the 20 standard amino acids.')
        if len(chain['residue_map']) != len(seq):
            raise ValueError('The sequence and residue mapping have different lengths.')
    if sum(len(c['sequence']) for c in chains) > MAX_RESIDUES:
        raise ValueError(f'Local prediction is limited to {MAX_RESIDUES} total residues, '
                         'including flanks; select a smaller region.')


def crop_chains(rows, region, flanks=10):
    """Crop two contiguous protein intervals; retain source indices for comparison."""
    if region is None:
        raise ValueError('Drag a box across two chains in the PAE plot first.')
    if not isinstance(flanks, int) or not 0 <= flanks <= 100:
        raise ValueError('Flanks must be between 0 and 100 residues.')
    y0, y1, x0, x1 = region
    chains = []
    for lo, hi in ((y0, y1), (x0, x1)):
        if not 0 <= lo <= hi < len(rows):
            raise ValueError('The selected region no longer matches the structure.')
        core = rows[lo:hi + 1]
        if any(r is None for r in core) or len({r['chain_id'] for r in core}) != 1:
            raise ValueError('Each axis of the box must cover one protein chain.')
        cid = core[0]['chain_id']
        start, end = lo, hi
        for _ in range(flanks):
            if start and rows[start - 1] and rows[start - 1]['chain_id'] == cid:
                start -= 1
            if end + 1 < len(rows) and rows[end + 1] and rows[end + 1]['chain_id'] == cid:
                end += 1
        selected = rows[start:end + 1]
        if any(r is None for r in selected) or any(
                b['chain_index'] != a['chain_index'] + 1 for a, b in zip(selected, selected[1:])):
            raise ValueError('The crop crosses missing residues; choose a complete contiguous interval.')
        chains.append(dict(source_chain=cid, sequence=''.join(r['aa'] for r in selected),
                           residue_map=[dict(r, source_row=i) for i, r in enumerate(selected, start)],
                           core_rows=list(range(lo, hi + 1))))
    if chains[0]['source_chain'] == chains[1]['source_chain']:
        raise ValueError('Select an interface between two different protein chains.')
    validate_chains(chains)
    return chains


def validate_a3m(path, query):
    first = []
    started = False
    with Path(path).open() as stream:
        for line in stream:
            if line.startswith('#') or not line.strip():
                continue
            if line.startswith('>'):
                if started:
                    break
                started = True
            elif started:
                first.append(''.join(c for c in line.strip() if not c.islower() and c != '.'))
    if ''.join(first) != query:
        raise ValueError('The A3M query does not match the selected sequence(s) in order.')


def crop_a3m(path, destination, source_query, columns):
    """Keep selected query columns of an already paired/merged alignment.

    Lowercase insertions belong to the following aligned column. Insertions at a
    new crop boundary are discarded. Row pairing is preserved, never recomputed.
    """
    validate_a3m(path, source_query)
    if len(set(columns)) != len(columns) or any(not 0 <= i < len(source_query) for i in columns):
        raise ValueError('Invalid MSA crop column mapping.')
    records = []
    header, seq = None, []
    for line in Path(path).read_text().splitlines() + ['>END']:
        if line.startswith('#') or not line.strip():
            continue
        if line.startswith('>'):
            if header is not None:
                aligned, inserts, pending = [], [], ''
                for char in ''.join(seq):
                    if char.islower():
                        pending += char
                    elif char == '.':
                        continue
                    elif char == '-' or char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                        aligned.append(char); inserts.append(pending); pending = ''
                    else:
                        raise ValueError('Invalid A3M character.')
                if len(aligned) != len(source_query):
                    raise ValueError('A source MSA row has the wrong aligned length.')
                selected = ''.join((inserts[c] if i and c == columns[i-1]+1 else '') + aligned[c]
                                   for i, c in enumerate(columns))
                if not records or any(aligned[c] != '-' for c in columns):
                    records.append(header + '\n' + selected + '\n')
            header, seq = line, []
        elif header is not None:
            seq.append(line.strip())
    Path(destination).write_text(''.join(records))
    validate_a3m(destination, ''.join(source_query[i] for i in columns))


def validate_result(directory, chains):
    d = Path(directory)
    confidences = json.loads((d / 'fold_confidences.json').read_text())
    summary = json.loads((d / 'fold_summary_confidences.json').read_text())
    sizes = [len(c['sequence']) for c in chains]
    n = sum(sizes)
    expected_ids = [cid for cid, size in zip('AB', sizes) for _ in range(size)]
    expected_res = [i for size in sizes for i in range(1, size + 1)]
    if confidences.get('token_chain_ids') != expected_ids or confidences.get('token_res_ids') != expected_res:
        raise ValueError('Prediction token order does not match the requested chains and residues.')
    pae = confidences.get('pae', [])
    if len(pae) != n or any(len(r) != n for r in pae):
        raise ValueError('The returned PAE matrix has the wrong dimensions.')
    if any(not math.isfinite(v) or v < 0 for row in pae for v in row):
        raise ValueError('The returned PAE matrix contains invalid values.')
    for key in ('ptm', 'mean_plddt') + (('iptm',) if len(chains) == 2 else ()):
        v = summary.get(key)
        if not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= (100 if key == 'mean_plddt' else 1):
            raise ValueError(f'Invalid prediction confidence: {key}.')
    # Check the structure independently of the confidence file, including sequence identity.
    three = dict(zip('ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR'.split(), AMINO_ACIDS))
    residues = {}
    for line in (d / 'fold.pdb').read_text().splitlines():
        if line.startswith('ATOM'):
            xyz = [float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))]
            if not all(math.isfinite(v) for v in xyz):
                raise ValueError('Prediction contains nonfinite coordinates.')
            if line[12:16].strip() == 'CA':
                residues[(line[21], int(line[22:26]))] = three.get(line[17:20], 'X')
    expected = {(cid, i + 1): aa for cid, c in zip('AB', chains) for i, aa in enumerate(c['sequence'])}
    if residues != expected:
        raise ValueError('Returned structure sequence or numbering does not match the job.')
    cross = [pae[i][j] for i in range(sizes[0]) for j in range(sizes[0], n)]
    cross += [pae[j][i] for i in range(sizes[0]) for j in range(sizes[0], n)]
    if len(chains) == 1:
        summary['iptm'] = None  # There is no interface score for a monomer.
    return dict(summary, mean_cross_chain_pae=sum(cross) / len(cross) if cross else None,
                has_contact_probabilities='contact_probs' in confidences)


class PredictionJob:
    """One cancellable process; poll() is nonblocking and called by the GUI timer."""
    def __init__(self, directory, chains, *, executable=None, search=False, msa=None,
                 seed=0, recycles=3, max_msa=512, max_extra=1024, source=None,
                 cache=None, timeout=300, msa_crop=None):
        validate_chains(chains)
        if search and msa:
            raise ValueError('Choose remote search or an existing merged MSA, not both.')
        if not search and not msa:
            raise ValueError('Choose remote search or supply an MSA for this prediction.')
        for value, lo, hi, label in ((seed, 0, 2**31-1, 'seed'), (recycles, 0, 10, 'recycles'),
                                      (max_msa, 1, 512, 'MSA rows'), (max_extra, 0, 2048, 'extra MSA rows')):
            if not isinstance(value, int) or not lo <= value <= hi:
                raise ValueError(f'Invalid {label}: {value}.')
        self.directory = Path(directory).expanduser().resolve()
        self.chains = chains
        self.model = MODEL if len(chains) == 2 else MONOMER_MODEL
        self.cache = Path(cache or Path.home() / '.cache' / 'alphalens' / 'localfold')
        self.executable = Path(executable or default_executable()).expanduser().resolve()
        if not self.executable.is_file() or not os.access(self.executable, os.X_OK):
            raise ValueError('LocalFold is not installed. Run scripts/setup_localfold.py or choose localfold-af2.')
        self.timeout = timeout
        self.process = self.lock = self.log = None
        self.cancelled_at = None
        self.state = 'prepared'
        self.result = None
        self.error = None
        self.options = dict(seed=seed, recycles=recycles, max_msa=max_msa, max_extra=max_extra)
        self.source = source or {}
        self.search = search
        self.msa = Path(msa).expanduser().resolve() if msa else None
        self.msa_crop = msa_crop
        if msa_crop and not msa:
            raise ValueError('An MSA crop requires a source alignment.')
        self.api = os.environ.get('LOCALFOLD_MMSEQS_API', 'https://api.colabfold.com').rstrip('/')
        key = json.dumps([ENGINE_REVISION, self.model, self.api, [c['sequence'] for c in chains]])
        self.msa_cache = self.cache / 'msa' / (hashlib.sha256(key.encode()).hexdigest() + '.a3m')

    def start(self):
        import fcntl
        if self.state != 'prepared':
            raise ValueError('This job has already been started.')
        self.cache.mkdir(parents=True, exist_ok=True)
        self.lock = (self.cache / 'prediction.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close(); self.lock = None
            raise ValueError('Another local prediction is running; wait or cancel it first.')
        try:
            self.directory.mkdir(parents=True, exist_ok=False)
            self.started = time.monotonic()
            query = ''.join(c['sequence'] for c in self.chains)
            alignment = self.msa or (self.msa_cache if self.search and self.msa_cache.is_file() else None)
            if alignment:
                if self.msa_crop:
                    crop_a3m(alignment, self.directory / 'input.a3m', **self.msa_crop)
                else:
                    validate_a3m(alignment, query)
                    shutil.copyfile(alignment, self.directory / 'input.a3m')
                validate_a3m(self.directory / 'input.a3m', query)
            job = dict(name=self.directory.name, dialect='alphafold3', version=1,
                       modelSeeds=[self.options['seed']], sequences=[
                           {'protein': {'id': cid, 'sequence': c['sequence']}} for cid, c in zip('AB', self.chains)])
            write_json(self.directory / 'input.json', job)
            write_json(self.directory / 'residue_map.json', self.chains)
            args = [str(self.executable), f'--job={self.directory}/input.json', f'--model={self.model}',
                    f'--out={self.directory}/fold.pdb', f'--weights-dir={self.cache}']
            args += [f'--{k.replace("_", "-")}={v}' for k, v in self.options.items()]
            args += [f'--a3m={self.directory}/input.a3m'] if alignment else ['--search']
            provenance = self.executable.with_suffix('.json')
            engine_info = json.loads(provenance.read_text()) if provenance.exists() else {}
            binary_hash = digest(self.executable)
            revision = engine_info.get('revision') if engine_info.get('sha256') == binary_hash else 'unverified'
            self.manifest = dict(engine='LocalFold native Metal', revision=revision,
                                 executable_sha256=binary_hash, model=self.model,
                                 weights=f'LocalFold AF2 {self.model} int5 bundle (CC BY 4.0)',
                                 source=self.source, options=self.options, args=args,
                                 msa_crop=self.msa_crop,
                                 source_msa_sha256=digest(alignment) if alignment else None,
                                 msa_source=('supplied' if self.msa else 'cache') if alignment else self.api,
                                 input_sha256=digest(self.directory / 'input.json'))
            self.log = (self.directory / 'prediction.log').open('w')
            self.process = subprocess.Popen(args, stdout=self.log, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL, start_new_session=True)
            self.state = 'running'
            self._save()
        except Exception as e:
            self.state, self.error = 'failed', str(e)
            try:
                if self.process is not None and self.process.poll() is None:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.process.wait()
            finally:
                self._release()
            raise
        return self

    def _save(self):
        write_json(self.directory / 'job.json', dict(self.manifest, state=self.state,
                   elapsed_seconds=time.monotonic()-self.started, error=self.error, result=self.result))

    def _release(self):
        if self.log:
            self.log.close(); self.log = None
        if self.lock:
            self.lock.close(); self.lock = None

    def cancel(self):
        if self.state == 'running' and self.cancelled_at is None:
            self.cancelled_at = time.monotonic()
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def poll(self):
        if self.state != 'running':
            return self.state
        if time.monotonic()-self.started > self.timeout and self.cancelled_at is None:
            self.error = f'Prediction exceeded the {self.timeout}-second prototype limit.'
            self.cancel()
        if self.cancelled_at and time.monotonic()-self.cancelled_at > 1 and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        code = self.process.poll()
        if code is None:
            return self.state
        try:
            if self.cancelled_at:
                self.state = 'failed' if self.error else 'cancelled'
            elif code:
                self.state = 'failed'
                text = (self.directory / 'prediction.log').read_text(errors='replace')
                self.error = f'LocalFold exited with status {code}. ' + text[-700:]
            else:
                self.result = validate_result(self.directory, self.chains)
                searched = self.directory / 'fold.a3m'
                alignment = searched if searched.exists() else self.directory / 'input.a3m'
                # Supplied/cropped alignments are reusable too; no second remote query.
                validate_a3m(alignment, ''.join(c['sequence'] for c in self.chains))
                self.msa_cache.parent.mkdir(parents=True, exist_ok=True)
                part = self.msa_cache.with_suffix('.part')
                shutil.copyfile(alignment, part); part.replace(self.msa_cache)
                self.manifest['msa_sha256'] = digest(alignment)
                weights = self.cache / ('model-multimer' if len(self.chains) == 2 else 'model') / 'manifest.json'
                if weights.is_file():
                    self.manifest['weights_manifest_sha256'] = digest(weights)
                self.state = 'complete'
        except Exception as e:
            self.state, self.error = 'failed', str(e)
        finally:
            try:
                self._save()
            finally:
                self._release()
        return self.state
