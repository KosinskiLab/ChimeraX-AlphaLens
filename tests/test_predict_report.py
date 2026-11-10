import json
from alphalens.predict_report import report_text, save_report


def test_report_survives_dialog_and_includes_comparison(tmp_path):
    job = dict(model='af3', state='complete', elapsed_seconds=12,
               options=dict(seed=123, recycles=10, steps=200, samples=5),
               result=dict(iptm=.5, ptm=.6, mean_plddt=80),
               runtime_environment=dict(LOCALFOLD_GEMM_TENSOR='0'))
    (tmp_path/'job.json').write_text(json.dumps(job))
    comparison = dict(source=dict(core_mean_pae=15,core_mean_plddt=70),
                      predicted_core_mean_pae=12,predicted_core_mean_plddt=80)
    (tmp_path/'comparison.json').write_text(json.dumps(comparison))
    text = save_report(tmp_path)
    assert 'mean PAE 15.00 → 12.00 Å' in text and 'pLDDT 70.0 → 80.0' in text
    assert '200 steps · 5 samples' in text
    assert (tmp_path/'prediction_report.txt').read_bytes() == (text+'\n').encode('utf-8')
    assert report_text(tmp_path) == text
    assert 'earlier tensor fast path' not in text
    job.pop('runtime_environment')
    (tmp_path/'job.json').write_text(json.dumps(job))
    assert 'earlier tensor fast path' in report_text(tmp_path)
