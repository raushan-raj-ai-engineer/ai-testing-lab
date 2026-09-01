from __future__ import annotations
import html
from pathlib import Path

def write_html_dashboard(*,summary,records,path):
    rows=''.join(f'<tr><td>{html.escape(r.scenario_id)}</td><td>{html.escape(r.category)}</td><td>{"PASS" if r.passed else "FAIL"}</td><td>{r.score:.3f}</td></tr>' for r in records)
    cats=''.join(f'<li>{html.escape(c)}: {rate:.1%}</li>' for c,rate in summary.category_pass_rates.items())
    page=f"""<!doctype html><html><head><meta charset='utf-8'><title>AI Evaluation Dashboard</title><style>body{{font-family:system-ui;max-width:1000px;margin:40px auto}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}</style></head><body><h1>AI Evaluation Dashboard</h1><p>Release: <b>{'PASS' if summary.release_passed else 'FAIL'}</b></p><p>Pass rate: {summary.pass_rate:.1%} | Average score: {summary.average_score:.3f}</p><ul>{cats}</ul><table><tr><th>ID</th><th>Category</th><th>Status</th><th>Score</th></tr>{rows}</table></body></html>"""
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(page,encoding='utf-8')
