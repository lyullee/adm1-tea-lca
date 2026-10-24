"""Write self-describing run records and hashes without changing old results."""
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import platform


def write_json(path: Path, value) -> None:
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')


def record_run(directory: Path, evaluation: dict, *, analyses: dict | None=None, command: list | None=None) -> dict:
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    write_json(directory/'evaluation.json',evaluation)
    if analyses is not None:
        write_json(directory/'analyses.json',analyses)
        rows=analyses.get('grid',{}).get('accepted',[])
        if rows:
            with (directory/'grid.csv').open('x',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    package=Path(__file__).resolve().parent
    inputs=[]
    for source in sorted(package.rglob('*')):
        if source.suffix in ('.py','.csv') and '__pycache__' not in source.parts:
            inputs.append({'path':source.relative_to(package).as_posix(),
                           'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
    import numpy, scipy
    manifest={'schema':'adm1-tea-lca/run-record/v1','recorded_at_utc':datetime.now(timezone.utc).isoformat(),
              'python':platform.python_version(),'numpy':numpy.__version__,'scipy':scipy.__version__,
              'command':command,'source_and_input_hashes':inputs,
              'result_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()},
              'scope':'software execution record; no experimental, global-optimum or full-LCA validation claim'}
    write_json(directory/'run_manifest.json',manifest)
    return manifest
