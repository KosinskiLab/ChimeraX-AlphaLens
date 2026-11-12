"""Regression 1: run in a fresh ChimeraX GUI with the candidate bundle installed.

    ChimeraX --cmd "open tests/chimerax_seed_provenance_check.py"

Creates synthetic predictions and closes this test process (exit 0 on success).
Do not run in a session containing user work. No external prediction data required.
"""
import csv, json, os, pickle, tempfile, time, traceback, threading, sys
from pathlib import Path
import numpy as np
from Qt.QtCore import QTimer
from chimerax.core.commands import run
from chimerax.atomic import AtomicStructure
from chimerax.alphalens import cmd, contacts, data, agreement


root = Path(tempfile.mkdtemp(prefix='alphalens-regression-'))

def pump(predicate=lambda: False, seconds=0.3):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        session.ui.processEvents()
        if predicate(): return
        time.sleep(.01)
    if predicate is not None and seconds > 1:
        assert predicate(), 'timed out waiting for GUI load'

def fixture(name, n=4, external=False):
    d=root/name; d.mkdir()
    lines=[]
    for i in range(n):
        chain='A' if i<n//2 else 'B'; r=i%(n//2)+1
        for k, atom in enumerate(('N','CA','C','O','CB')):
            x=(r-1)*3.8 + (k-1)*.3; y=0. if chain=='A' else 5.
            lines.append(f'ATOM  {i*5+k+1:5d} {atom:^4s} ALA {chain}{r:4d}    {x:8.3f}{y:8.3f}{0.:8.3f}{1.:6.2f}{90.:6.2f}          {atom[0]:>2s}  \n')
        if i==n//2-1: lines.append('TER\n')
    (d/'ranking_debug.json').write_text(json.dumps({'order':['model_1','model_2']}))
    p=np.full((n,n), .9, np.float32); p[n//2-1,n-1]=p[n-1,n//2-1]=.6
    for rank in (0,1):
        (d/f'ranked_{rank}.pdb').write_text(''.join(lines)+'END\n')
        (d/f'pae_model_{rank+1}.json').write_text(json.dumps({'pae':np.full((n,n),5.).tolist()}))
        logits=np.full((n,n,64), -30., np.float32)
        logits[...,0]=np.log(p); logits[...,-1]=np.log(1-p)
        (d/f'result_model_{rank+1}.pkl').write_bytes(pickle.dumps({'distogram':{'logits':logits}}))
    ext=None
    if external:
        ext=root/f'{name}-external.pkl'; (d/'result_model_1.pkl').rename(ext)
    return d,ext

def opened(name,n=4,external=False):
    d,ext=fixture(name,n,external)
    pae=cmd.alphalens_open(session,str(d),wait=True,distogram=str(ext) if ext else None)
    return d,ext,pae,pae.structure._alphalens_plot

def issue1():
    d,_,pae,p=opened('seed')
    run(session,f'save {d}/test_model.cif models #{pae.structure.id_string}')
    (d/'test_confidences.json').write_text(json.dumps({'pae':[[5.]*4]*4,'contact_probs':[[.1]*4]*4}))
    for seed in (1,2):
        (d/f'seed-{seed}_sample-0').mkdir()
    (d/'seed-1_distogram').mkdir()
    np.savez(d/'seed-1_distogram/d.npz',distogram=np.ones((4,4,64))/64)
    files=data.find_prediction_files(str(d/'test_model.cif'))
    assert files.distogram_path is None, f'inferred {files.model_name}: {files.distogram_path}'

def main():
    try:
        issue1()
    except Exception:
        traceback.print_exc(file=sys.__stderr__)
        result = 1
    else:
        print('AlphaLens regression 1 PASSED', file=sys.__stdout__, flush=True)
        result = 0
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(result)

QTimer.singleShot(1000, main)
