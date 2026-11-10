"""Regression 4: run in a fresh ChimeraX GUI with the candidate bundle installed.

    ChimeraX --cmd "open tests/chimerax_csv_region_check.py"

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

def issue4():
    d,_,pae,p=opened('csv')
    p._struct_region=(0,0,2,2)
    p.save_contacts_csv(str(d/'button.csv'))
    cmd.alphalens(session,pae.structure,save_contacts=str(d/'command.csv'),wait=True)
    a=list(csv.DictReader((d/'button.csv').open())); b=list(csv.DictReader((d/'command.csv').open()))
    assert a==b, f'button {len(a)} rows; command {len(b)} rows'

def main():
    try:
        issue4()
    except Exception:
        traceback.print_exc(file=sys.__stderr__)
        result = 1
    else:
        print('AlphaLens regression 4 PASSED', file=sys.__stdout__, flush=True)
        result = 0
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(result)

QTimer.singleShot(1000, main)
