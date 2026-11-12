"""Persistent, human-readable prediction results, reconstructed from saved job files."""
import json
from pathlib import Path
from .predict_contacts import summary as contact_summary
from .predict_compare import summary_lines as interface_lines


def report_text(directory):
    directory = Path(directory)
    job = json.loads((directory / 'job.json').read_text())
    result = job.get('result') or {}
    options = job.get('options') or {}
    lines = [f'Local prediction · {job.get("model", "unknown model")}',
             f'Status: {job.get("state", "unknown")} · {job.get("elapsed_seconds", 0):.1f} s',
             f'Seed: {options.get("seed", "unknown")} · Recycles: {options.get("recycles", "unknown")}']
    if job.get('model') == 'af3':
        lines.append(f'Diffusion: {options.get("steps", "unknown")} steps · {options.get("samples", 1)} samples')
    for label, key, digits in [('ipTM', 'iptm', 3), ('pTM', 'ptm', 3), ('Mean pLDDT', 'mean_plddt', 1)]:
        if result.get(key) is not None:
            lines.append(f'{label}: {result[key]:.{digits}f}')
    comparison_path = directory / 'comparison.json'
    if comparison_path.exists():
        comparison = json.loads(comparison_path.read_text())
        source = comparison['source']
        lines.append(f'Same selected pairs: mean PAE {source["core_mean_pae"]:.2f} → '
                     f'{comparison["predicted_core_mean_pae"]:.2f} Å')
        if source.get('core_mean_plddt') is not None:
            lines.append(f'Same residues: pLDDT {source["core_mean_plddt"]:.1f} → '
                         f'{comparison["predicted_core_mean_plddt"]:.1f}')
        interface = comparison.get('interface') or {}
        if interface.get('error'):
            lines.append(interface['error'])
        else:
            lines.extend(interface_lines(interface))
        if comparison.get('aligned_chain_ca_rmsd') is not None:
            lines.append(f'Superposed on source chain {comparison.get("aligned_source_chain")} '
                         f'(CA RMSD {comparison["aligned_chain_ca_rmsd"]:.2f} Å over its selected residues)')
    contacts = directory / 'contact_report.json'
    if contacts.exists():
        lines.extend(['', contact_summary(json.loads(contacts.read_text()))])
    if job.get('error'):
        lines.extend(['', job['error']])
    if job.get('model') == 'af3' and job.get('runtime_environment', {}).get('LOCALFOLD_GEMM_TENSOR') != '0':
        lines.extend(['', 'This AF3 job used the earlier tensor fast path, which produced collapsed '
                      'coordinates on M5 hardware. Re-predict with AlphaLens 1.1.2 or newer.'])
    lines.extend(['', f'Results: {directory}'])
    return '\n'.join(lines)


def save_report(directory):
    text = report_text(directory)
    (Path(directory) / 'prediction_report.txt').write_text(text + '\n', encoding='utf-8')
    return text
