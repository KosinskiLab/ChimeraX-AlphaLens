"""Prepare the pinned AF2-Multimer bundle with its own float32 distogram head."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
from urllib.request import urlopen

BUNDLE_URL = 'https://huggingface.co/sokrypton/localfold/resolve/b61b5a3ed56470025daeac7986f884bb5451d2fd/af2-multimer-int5/'
MANIFEST_SHA256 = 'e2817b8d6e4d4468dbbf95dc315bd71cacd4fe5b5cade0e0053f826da38c6b6f'
HEAD_SHA256 = 'f809845f39567c829a1bf829cb44eb91c2a69e3a7f879a16a70e506c3a8258ab'
HEAD_FILE = Path(__file__).parent / 'weights' / 'af2-multimer-v3-distogram.f32'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def download(url, target):
    temporary = target.with_suffix(target.suffix + '.part')
    with urlopen(url, timeout=60) as response, temporary.open('wb') as output:
        shutil.copyfileobj(response, output)
    temporary.replace(target)


def prepare(cache):
    """Use a separate enhanced cache; leave the original weight files untouched."""
    cache = Path(cache)
    original = cache / 'model-multimer'
    root = cache / 'alphalens-contacts'
    target = root / 'model-multimer'
    target.mkdir(parents=True, exist_ok=True)
    manifest_file = target / 'manifest.json'
    head = HEAD_FILE.read_bytes()
    if len(head) != 33024 or sha256(head) != HEAD_SHA256:
        raise ValueError('The bundled AF2-Multimer contact-head weights are damaged.')
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        if manifest.get('alphalens_contact_head_sha256') == HEAD_SHA256:
            files = {r['file'] for r in manifest['tensors'].values()}
            if (all((target / f).is_file() for f in files)
                    and sha256((target / 'alphalens-distogram.f32.bin').read_bytes()) == HEAD_SHA256):
                return root
    print('multimer: preparing weights and contact probabilities', flush=True)
    original_manifest = original / 'manifest.json'
    data = original_manifest.read_bytes() if original_manifest.exists() else b''
    if sha256(data) != MANIFEST_SHA256:
        download(BUNDLE_URL + 'manifest.json', target / 'manifest.upstream.json')
        data = (target / 'manifest.upstream.json').read_bytes()
    if sha256(data) != MANIFEST_SHA256:
        raise ValueError('The downloaded AF2-Multimer manifest does not match the pinned version.')
    manifest = json.loads(data)
    files = sorted({r['file'] for r in manifest['tensors'].values()})
    if any(Path(f).name != f or f in ('.', '..') for f in files):
        raise ValueError('Invalid model shard filename.')
    local_manifest_matches = original_manifest.exists() and sha256(original_manifest.read_bytes()) == MANIFEST_SHA256

    def copy_or_download(name):
        destination = target / name
        if destination.exists():
            return
        source = original / name
        if local_manifest_matches and source.is_file():
            try:
                os.link(source, destination)
            except OSError:
                temporary = destination.with_suffix(destination.suffix + '.part')
                shutil.copy2(source, temporary)
                temporary.replace(destination)
        else:
            download(BUNDLE_URL + name, destination)
        print(f'multimer: prepared {name}', flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(copy_or_download, files))
    head_name = 'alphalens-distogram.f32.bin'
    (target / head_name).write_bytes(head)
    for name, shape, offset in [('af2DistogramHalfLogitsWeights', [128, 64], 0),
                                ('af2DistogramHalfLogitsBias', [64], 32768)]:
        manifest['tensors'][name] = dict(file=head_name, shape=shape, byteOffset=offset, dtype='float32')
    manifest['distogramHead'] = dict(weights='af2DistogramHalfLogitsWeights',
        bias='af2DistogramHalfLogitsBias', bins=64, firstBreak=2.3125, lastBreak=21.6875)
    manifest['alphalens_contact_head_sha256'] = HEAD_SHA256
    manifest['alphalens_upstream_manifest_sha256'] = MANIFEST_SHA256
    (target / '.source').write_text(BUNDLE_URL)
    temporary = target / 'manifest.json.part'
    temporary.write_text(json.dumps(manifest) + '\n')
    temporary.replace(manifest_file)
    return root


if __name__ == '__main__':
    import sys
    try:
        root = prepare(sys.argv[1])
        args = [f'--weights-dir={root}' if a.startswith('--weights-dir=') else a for a in sys.argv[2:]]
        os.execv(args[0], args)
    except Exception as error:
        print(f'Contact-head setup failed: {error}', file=sys.stderr, flush=True)
        sys.exit(1)
