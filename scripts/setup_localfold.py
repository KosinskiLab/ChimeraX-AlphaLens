#!/usr/bin/env python3
"""Build a pinned native Metal engine outside ChimeraX's Python environment."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess

REVISION = '68c59196fa7d32c6edfc061ed369889c1cdb680b'
REPOSITORY = 'https://github.com/sokrypton/localfold.git'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix', type=Path, default=Path.home() / '.local/share/alphalens')
    p.add_argument('--source', type=Path, help='Use an existing clean checkout at the pinned revision')
    args = p.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        p.error('This prototype requires Apple Silicon macOS and Xcode Command Line Tools.')
    prefix = args.prefix.expanduser().resolve()
    source = args.source or prefix / 'engine' / REVISION
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout', REPOSITORY, str(source)], check=True)
        subprocess.run(['git', '-C', str(source), 'checkout', '--detach', REVISION], check=True)
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    changed = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if head != REVISION or changed:
        p.error('The source must be clean and at the pinned revision; existing files were preserved.')
    subprocess.run(['bash', 'metal/build.sh', 'af2'], cwd=source, check=True)
    target = prefix / 'bin' / 'localfold-af2'
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.part')
    shutil.copy2(source / 'metal/af2/localfold-af2', temporary)
    temporary.replace(target)
    shutil.copy2(source / 'LICENSE', target.parent / 'LocalFold-LICENSE.txt')
    target.with_suffix('.json').write_text(json.dumps(dict(repository=REPOSITORY, revision=head,
                                   sha256=hashlib.sha256(target.read_bytes()).hexdigest()), indent=2) + '\n')
    print(f'Installed {target}. AF2 weights download automatically on the first prediction.')


if __name__ == '__main__':
    main()
