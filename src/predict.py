"""Experimental local AF2/AF3 jobs. No ChimeraX imports or shell commands."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time

from .predict_setup import REVISION as ENGINE_REVISION, cuda_compute_capability
NATIVE_NAMES = dict(metal='Metal', cuda='CUDA')
MODEL = 'model_1_multimer_v3'
MONOMER_MODEL = 'model_1_ptm'
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
# A UI warning threshold only. LocalFold checks the GPU working-set budget.
LARGE_JOB_RESIDUES = 1024
CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
# ChimeraX's PDB reader leaves the process in the C locale, whose default text encoding is
# ASCII: after a first result has opened, a header with any other byte would stop the next
# job. Alignment rows are ASCII; header bytes of any encoding pass through unchanged.
A3M_TEXT = dict(encoding='utf-8', errors='surrogateescape')


def chain_ids(count):
    if not 1 <= count <= len(CHAIN_IDS):
        raise ValueError('Choose at least one chain. PDB output supports at most 62 distinct chain IDs.')
    return CHAIN_IDS[:count]


def random_seed():
    return secrets.randbelow(2**31 - 1) + 1


def data_directory():
    return Path.home() / '.local' / 'share' / 'alphalens'


def af3_terms_accepted():
    try:
        return json.loads((data_directory() / 'af3_terms.json').read_text()).get('accepted') is True
    except (OSError, ValueError):
        return False


def record_af3_terms_acceptance():
    from datetime import datetime, timezone
    root = data_directory()
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / 'af3_terms.json', dict(accepted=True, accepted_at=datetime.now(timezone.utc).isoformat(),
        terms='https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md',
        policy='https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_PROHIBITED_USE_POLICY.md'))


def engine_platform():
    """The native build that runs engines on this OS: Metal on macOS, CUDA on Linux.

    Decided by the OS, not by Python's architecture: an x86_64 Python under Rosetta on
    Apple Silicon still launches the arm64 Metal engine and needs its AF3 settings."""
    return {'darwin': 'metal', 'linux': 'cuda'}.get(sys.platform)


def default_executable(backend='af2'):
    name = f'localfold-{backend}'
    variable = 'ALPHALENS_LOCALFOLD_AF3' if backend == 'af3' else 'ALPHALENS_LOCALFOLD'
    return os.environ.get(variable, '') or shutil.which(name) or str(data_directory() / 'bin' / name)


def chimerax_python():
    '''The Python running ChimeraX, for the helper steps run before LocalFold.'''
    python = Path(sys.executable)
    if not python.name.startswith('python'):
        python = Path(sys.exec_prefix) / 'bin' / f'python{sys.version_info.major}.{sys.version_info.minor}'
    if not python.is_file():
        raise ValueError('Could not locate the ChimeraX Python interpreter for preparing the prediction.')
    return python


def check_engine_gpu(executable):
    """Refuse CUDA engines built for another GPU generation.

    Built on an A100 (8.0) and run on an H100 (9.0), the pinned engines gave NaN for AF2-Multimer
    and AF3, and an AF2 monomer 3.4 A from the crystal that passed every output check."""
    executable = Path(executable)
    if engine_platform() != 'cuda':
        return
    try:
        info = json.loads(executable.with_suffix('.json').read_text())
    except (OSError, ValueError):
        return      # An executable of the user's own, with no receipt: their choice.
    built_for = info.get('cuda_compute_capability')
    if not built_for or info.get('sha256') != digest(executable):
        return
    current = cuda_compute_capability()
    if current and current != built_for:
        raise ValueError(
            f'{executable.name} was built for an NVIDIA GPU of compute capability {built_for}, but this GPU '
            f'has {current}. Engines built for another GPU generation give wrong structures. Delete '
            f'{executable.parent}/localfold-* and click Install & Predict to build them for this GPU.')


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
    chain_ids(len(chains))
    for chain in chains:
        seq = chain['sequence']
        if not seq or any(c not in AMINO_ACIDS for c in seq):
            raise ValueError('Use nonempty protein sequences with the 20 standard amino acids.')
        if len(chain['residue_map']) != len(seq):
            raise ValueError('The sequence and residue mapping have different lengths.')


def size_warning(chains):
    total = sum(len(c['sequence']) for c in chains)
    if total > LARGE_JOB_RESIDUES:
        return (f'Large prediction: {total:,} residues including flanks. More GPU memory '
                'and time may be needed. You can proceed.')
    return ''


def crop_chains(rows, region, flanks=10):
    """Crop every protein chain touched by either axis; retain the exact core rows."""
    if region is None:
        raise ValueError('Drag a box across two chains in the PAE plot first.')
    if not isinstance(flanks, int) or not 0 <= flanks <= 100:
        raise ValueError('Flanks must be between 0 and 100 residues.')
    y0, y1, x0, x1 = region
    selected_cores = {}
    for lo, hi in ((y0, y1), (x0, x1)):
        if not 0 <= lo <= hi < len(rows):
            raise ValueError('The selected region no longer matches the structure.')
        core = rows[lo:hi + 1]
        if any(r is None for r in core):
            raise ValueError('The selected box must contain protein residues only.')
        for i in range(lo, hi + 1):
            selected_cores.setdefault(rows[i]['chain_id'], set()).add(i)
    if len(selected_cores) < 2:
        raise ValueError('Select an interface between different protein chains.')
    chains = []
    for cid, core_rows in selected_cores.items():
        start, end = min(core_rows), max(core_rows)
        for _ in range(flanks):
            if start and rows[start - 1] and rows[start - 1]['chain_id'] == cid:
                start -= 1
            if end + 1 < len(rows) and rows[end + 1] and rows[end + 1]['chain_id'] == cid:
                end += 1
        selected = rows[start:end + 1]
        if any(r is None or r['chain_id'] != cid for r in selected) or any(
                b['chain_index'] != a['chain_index'] + 1 for a, b in zip(selected, selected[1:])):
            raise ValueError('The crop crosses missing residues; choose a complete contiguous interval.')
        chains.append(dict(source_chain=cid, sequence=''.join(r['aa'] for r in selected),
                           residue_map=[dict(r, source_row=i) for i, r in enumerate(selected, start)],
                           core_rows=sorted(core_rows)))
    validate_chains(chains)
    return chains


def validate_a3m(path, query):
    first = []
    started = False
    with Path(path).open(**A3M_TEXT) as stream:
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
    for line in Path(path).read_text(**A3M_TEXT).splitlines() + ['>END']:
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
    Path(destination).write_text(''.join(records), **A3M_TEXT)
    validate_a3m(destination, ''.join(source_query[i] for i in columns))


def validate_result(directory, chains):
    d = Path(directory)
    confidences = json.loads((d / 'fold_confidences.json').read_text())
    summary = json.loads((d / 'fold_summary_confidences.json').read_text())
    sizes = [len(c['sequence']) for c in chains]
    n = sum(sizes)
    ids = chain_ids(len(chains))
    expected_ids = [cid for cid, size in zip(ids, sizes) for _ in range(size)]
    expected_res = [i for size in sizes for i in range(1, size + 1)]
    if confidences.get('token_chain_ids') != expected_ids or confidences.get('token_res_ids') != expected_res:
        raise ValueError('Prediction token order does not match the requested chains and residues.')
    pae = confidences.get('pae', [])
    if len(pae) != n or any(len(r) != n for r in pae):
        raise ValueError('The returned PAE matrix has the wrong dimensions.')
    if any(not math.isfinite(v) or v < 0 for row in pae for v in row):
        raise ValueError('The returned PAE matrix contains invalid values.')
    for key in ('ptm', 'mean_plddt') + (('iptm',) if len(chains) > 1 else ()):
        v = summary.get(key)
        if not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= (100 if key == 'mean_plddt' else 1):
            raise ValueError(f'Invalid prediction confidence: {key}.')
    # Check the structure independently of the confidence file, including sequence identity.
    three = dict(zip('ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR'.split(), AMINO_ACIDS))
    residues, coordinates = {}, {}
    for line in (d / 'fold.pdb').read_text().splitlines():
        if line.startswith('ATOM'):
            xyz = [float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))]
            if not all(math.isfinite(v) for v in xyz):
                raise ValueError('Prediction contains nonfinite coordinates.')
            if line[12:16].strip() == 'CA':
                residues[(line[21], int(line[22:26]))] = three.get(line[17:20], 'X')
                coordinates[(line[21], int(line[22:26]))] = xyz
    expected = {(cid, i + 1): aa for cid, c in zip(ids, chains) for i, aa in enumerate(c['sequence'])}
    if residues != expected:
        raise ValueError('Returned structure sequence or numbering does not match the job.')
    from .predict_geometry import check_ca_geometry
    geometry = check_ca_geometry([coordinates[key] for key in expected], [key[0] for key in expected])
    cross_sum, cross_count = 0., 0
    for i, row in enumerate(pae):
        for j, value in enumerate(row):
            if expected_ids[i] != expected_ids[j]:
                cross_sum += value; cross_count += 1
    contact = confidences.get('contact_probs')
    if contact is not None and (len(contact) != n or any(len(row) != n for row in contact)
            or any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
                   for row in contact for v in row)):
        raise ValueError('The returned contact-probability matrix is invalid.')
    if len(chains) == 1:
        summary['iptm'] = None  # There is no interface score for a monomer.
    return dict(summary, geometry=geometry, mean_cross_chain_pae=cross_sum/cross_count if cross_count else None,
                has_contact_probabilities='contact_probs' in confidences)


class PredictionProgress:
    """Read reported engine milestones; never estimate end-to-end percentages."""
    def __init__(self):
        self.stage = 'Preparing model and alignment'
        self.completed = 0
        self.total = None
        self._pending = ''

    def feed(self, text):
        lines = (self._pending + text).split('\n')
        self._pending = lines.pop()[-4096:]
        for line in lines:
            line = line.strip()
            passed = re.match(r'pass (\d+)/(\d+):', line)
            header = re.match(r'AF2 (?:monomer|multimer): .*?, (\d+) passes$', line)
            if passed:
                completed, total = map(int, passed.groups())
                if 0 < completed <= total:
                    self.completed = completed
                    self.total = total if completed < total else None
                    self.stage = 'Predicting' if completed < total else 'Writing prediction files'
            elif header and int(header[1]) > 0:
                self.stage, self.completed, self.total = 'Predicting', 0, int(header[1])
            elif line.startswith('search:'):
                self.stage = 'MSA search · ' + line[len('search:'):].strip()[:160]
                self.total = None
            elif line.startswith('features:'):
                self.stage = 'Precomputed features · ' + line[len('features:'):].strip()[:160]
                self.total = None
            elif re.match(r'(?:af3|monomer|multimer)(?:-\d+)?:', line):
                self.stage, self.total = 'Downloading model weights', None
            elif line.startswith(('input:', 'trunk:')):
                self.stage, self.total = 'Predicting with AF3', None
            elif (re.match(r'\d+ residues, .* passes ->', line)
                  or line.startswith('loaded in ')):
                self.stage, self.total = 'Loading model and preparing GPU', None


class PredictionJob:
    """One cancellable process; poll() is nonblocking and called by the GUI timer."""
    def __init__(self, directory, chains, *, executable=None, search=False, msa=None,
                 seed=None, recycles=None, max_msa=512, max_extra=1024, source=None,
                 cache=None, timeout=None, msa_crop=None, backend='af2', accept_af3_terms=False,
                 diffusion_steps=200, diffusion_samples=5, features=None):
        validate_chains(chains)
        if backend not in ('af2', 'af3'):
            raise ValueError('Choose the AF2 or AF3 prediction model.')
        if backend == 'af3' and not accept_af3_terms:
            raise ValueError('Confirm acceptance of DeepMind’s AF3 parameter terms before predicting with AF3.')
        if not isinstance(diffusion_steps, int) or not 1 <= diffusion_steps <= 1000:
            raise ValueError('AF3 diffusion steps must be between 1 and 1000.')
        if not isinstance(diffusion_samples, int) or not 1 <= diffusion_samples <= 10:
            raise ValueError('AF3 diffusion samples must be between 1 and 10.')
        if recycles is None:
            recycles = 10 if backend == 'af3' else 3
        if seed is None:
            seed = random_seed()
        # features: dict(root=folder or URL, names=[feature name (UniProt ID) of each chain])
        if features is not None and (not features.get('root') or len(features.get('names') or ()) != len(chains)
                                     or not all(features['names'])):
            raise ValueError('Give the feature folder or URL and a UniProt ID for every chain.')
        if sum(map(bool, (search, msa, features))) > 1:
            raise ValueError('Choose one alignment source: remote search, an existing merged MSA or '
                             'precomputed features.')
        if not search and not msa and not features:
            raise ValueError('Choose remote search, supply an MSA or use precomputed features for this prediction.')
        for value, lo, hi, label in ((seed, 0, 2**31-1, 'seed'), (recycles, 0, 10, 'recycles'),
                                      (max_msa, 1, 512, 'MSA rows'), (max_extra, 0, 2048, 'extra MSA rows')):
            if not isinstance(value, int) or not lo <= value <= hi:
                raise ValueError(f'Invalid {label}: {value}.')
        self.directory = Path(directory).expanduser().resolve()
        self.chains = chains
        self.backend = backend
        self.model = 'af3' if backend == 'af3' else MODEL if len(chains) > 1 else MONOMER_MODEL
        self.cache = Path(cache or Path.home() / '.cache' / 'alphalens' / 'localfold')
        self.executable = Path(executable or default_executable(backend)).expanduser().resolve()
        if not self.executable.is_file() or not os.access(self.executable, os.X_OK):
            raise ValueError(f'LocalFold is not installed. Run scripts/setup_localfold.py or choose localfold-{backend}.')
        check_engine_gpu(self.executable)
        if timeout is not None and (not isinstance(timeout, (float, int)) or
                                    not math.isfinite(timeout) or timeout <= 0):
            raise ValueError('The time limit must be positive, or None for no limit.')
        self.timeout = timeout
        self.process = self.lock = self.log = None
        self.cancelled_at = None
        self.state = 'prepared'
        self.result = None
        self.contact_report = None
        self.error = None
        self.progress = PredictionProgress()
        self.weights_cache = self.cache
        self._log_offset = 0
        self.options = dict(seed=seed, recycles=recycles, max_msa=max_msa, max_extra=max_extra)
        if backend == 'af3':
            self.options.pop('max_extra')
            self.options.update(steps=diffusion_steps, samples=diffusion_samples, update='euler', schedule='karras')
        self.source = source or {}
        self.search = search
        self.features = features
        self.msa = Path(msa).expanduser().resolve() if msa else None
        self.msa_crop = msa_crop
        if msa_crop and not msa:
            raise ValueError('An MSA crop requires a source alignment.')
        self.api = os.environ.get('LOCALFOLD_MMSEQS_API', 'https://api.colabfold.com').rstrip('/')
        sequences = [c['sequence'] for c in chains]
        key = json.dumps([ENGINE_REVISION, self.api, sequences])
        self.msa_cache = self.cache / 'msa' / (hashlib.sha256(key.encode()).hexdigest() + '.a3m')
        self.legacy_msa_caches = [self.cache / 'msa' / (hashlib.sha256(json.dumps(
            [ENGINE_REVISION, model, self.api, sequences]).encode()).hexdigest() + '.a3m')
            for model in (self.model, MODEL if len(chains) > 1 else MONOMER_MODEL)]

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
            cached = next((p for p in [self.msa_cache, *self.legacy_msa_caches] if p.is_file()), None)
            alignment = self.msa or (cached if self.search else None)
            if alignment:
                if self.msa_crop:
                    crop_a3m(alignment, self.directory / 'input.a3m', **self.msa_crop)
                else:
                    validate_a3m(alignment, query)
                    shutil.copyfile(alignment, self.directory / 'input.a3m')
                validate_a3m(self.directory / 'input.a3m', query)
            job = dict(name=self.directory.name, dialect='alphafold3', version=1,
                       modelSeeds=[self.options['seed']], sequences=[
                           {'protein': {'id': cid, 'sequence': c['sequence']}}
                           for cid, c in zip(chain_ids(len(self.chains)), self.chains)])
            write_json(self.directory / 'input.json', job)
            write_json(self.directory / 'residue_map.json', self.chains)
            args = [str(self.executable), f'--job={self.directory}/input.json', f'--model={self.model}',
                    f'--out={self.directory}/fold.pdb', f'--weights-dir={self.cache}']
            args += [f'--{k.replace("_", "-")}={v}' for k, v in self.options.items()]
            if not self.features:       # The features step adds its own alignment arguments
                args += [f'--a3m={self.directory}/input.a3m'] if alignment else ['--search']
            provenance = self.executable.with_suffix('.json')
            engine_info = json.loads(provenance.read_text()) if provenance.exists() else {}
            binary_hash = digest(self.executable)
            revision = engine_info.get('revision') if engine_info.get('sha256') == binary_hash else 'unverified'
            native = engine_platform()
            self.manifest = dict(engine='LocalFold native ' + NATIVE_NAMES.get(native, 'engine'), revision=revision,
                                 executable_sha256=binary_hash, model=self.model,
                                 weights=('LocalFold AF3 int8 parameters (DeepMind AF3 terms)'
                                          if self.backend == 'af3' else f'LocalFold AF2 {self.model} int5 bundle (CC BY 4.0)'),
                                 source=self.source, options=self.options, args=args,
                                 timeout_seconds=self.timeout,
                                 msa_crop=self.msa_crop,
                                 source_msa_sha256=digest(alignment) if alignment else None,
                                 msa_source=(f'precomputed features from {self.features["root"]}' if self.features
                                             else ('supplied' if self.msa else 'cache') if alignment else self.api),
                                 input_sha256=digest(self.directory / 'input.json'))
            self.log = (self.directory / 'prediction.log').open('w')
            env = os.environ.copy()
            if self.backend == 'af3':
                env['LOCALFOLD_ACCEPT_MODEL_TERMS'] = 'alphafold3'
                self.manifest['af3_terms_accepted'] = True
                if native == 'metal':
                    # The pinned engine's M5 tensor path collapses AF3 coordinates.
                    # Its conventional Metal kernels pass the 1BRS crystal control.
                    # Foundation otherwise buffers AF3 stdout until inference finishes.
                    runtime = dict(LOCALFOLD_GEMM_TENSOR='0', LOCALFOLD_ATTN_TENSOR='0', NSUnbufferedIO='YES')
                    env.update(runtime)
                    self.manifest['runtime_environment'] = runtime
            launch = args
            if self.model == MODEL and revision == ENGINE_REVISION:
                from .predict_weights import HEAD_SHA256
                python = chimerax_python()
                launch = [str(python), '-B', str(Path(__file__).with_name('predict_weights.py')), str(self.cache), *args]
                self.weights_cache = self.cache / 'alphalens-contacts'
                self.manifest['contact_head_sha256'] = HEAD_SHA256
                self.manifest['weights_directory'] = str(self.weights_cache)
                self.manifest['args'] = [f'--weights-dir={self.weights_cache}'
                                        if a.startswith('--weights-dir=') else a for a in args]
            if self.features:
                # Find, read (network mounts, downloads) and convert the features in the job's own
                # process, which then becomes LocalFold: the window never waits on them.
                spec = self.directory / 'features_spec.json'
                write_json(spec, dict(directory=str(self.directory / 'features'), root=str(self.features['root']),
                                      backend=self.backend, job=str(self.directory / 'input.json'),
                                      chains=[dict(chain_id=cid, name=name, sequence=c['sequence'],
                                                   start_hint=int(c['residue_map'][0].get('number', 1)) - 1)
                                              for cid, name, c in zip(chain_ids(len(self.chains)),
                                                                      self.features['names'], self.chains)]))
                launch = [str(chimerax_python()), '-B', str(Path(__file__).with_name('predict_features.py')),
                          str(spec), '--', *launch]
                self.manifest['features_root'] = str(self.features['root'])
            self.process = subprocess.Popen(launch, stdout=self.log, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL, start_new_session=True, env=env)
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

    def read_progress(self):
        # Read only new bytes, bounded per GUI tick. Keep milestones across long
        # diagnostic lines and partial writes instead of repeatedly tailing the log.
        with (self.directory / 'prediction.log').open('rb') as log:
            log.seek(self._log_offset)
            chunk = log.read(64 * 1024)
            self._log_offset = log.tell()
        self.progress.feed(chunk.decode(errors='replace'))
        return self.progress

    def poll(self):
        if self.state != 'running':
            return self.state
        if (self.timeout is not None and time.monotonic()-self.started > self.timeout
                and self.cancelled_at is None):
            self.error = f'Prediction exceeded the configured {self.timeout}-second time limit.'
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
                from .predict_features import ERROR_PREFIX
                failed = [line for line in text.splitlines() if line.startswith(ERROR_PREFIX)]
                self.error = (failed[-1][len('Error: '):] if self.features and failed
                              else f'LocalFold exited with status {code}. ' + text[-700:])
            else:
                self.result = validate_result(self.directory, self.chains)
                if self.features:
                    # The feature files are the record; their alignments are not a merged MSA
                    provenance = self.directory / 'features' / 'features.json'
                    self.manifest['features'] = json.loads(provenance.read_text()) if provenance.exists() else None
                else:
                    searched = self.directory / 'fold.a3m'
                    alignment = searched if searched.exists() else self.directory / 'input.a3m'
                    # Supplied/cropped alignments are reusable too; no second remote query.
                    validate_a3m(alignment, ''.join(c['sequence'] for c in self.chains))
                    self.msa_cache.parent.mkdir(parents=True, exist_ok=True)
                    part = self.msa_cache.with_suffix('.part')
                    shutil.copyfile(alignment, part); part.replace(self.msa_cache)
                    self.manifest['msa_sha256'] = digest(alignment)
                weights = self.weights_cache / ('model-multimer' if len(self.chains) > 1 else 'model') / 'manifest.json'
                if self.backend == 'af2' and weights.is_file():
                    self.manifest['weights_manifest_sha256'] = digest(weights)
                if self.backend == 'af3':
                    blob = self.cache / 'af3am-af3' / 'af3.int8.bin.zst'
                    if blob.is_file():
                        self.manifest['weights_sha256'] = digest(blob)
                        provenance = blob.parent / '.source'
                        self.manifest['weights_source'] = provenance.read_text() if provenance.exists() else None
                self.state = 'complete'
        except Exception as e:
            self.state, self.error = 'failed', str(e)
        finally:
            try:
                self._save()
            finally:
                self._release()
        return self.state
