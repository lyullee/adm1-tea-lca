"""Archive code, provenance and the small executed demo; exclude build caches."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SKIP={'.git','__pycache__','.pytest_cache','build','dist','artifacts','.venv','runs'}


def main():
    payloads={}
    for p in sorted(ROOT.rglob('*')):
        rel=p.relative_to(ROOT)
        if p.is_file() and not any(x in SKIP or x.endswith('.egg-info') for x in rel.parts):
            payloads[rel.as_posix()]=p.read_bytes()
    preserved=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for row in preserved['records']:
        assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
    manifest={'schema':'adm1-tea-lca/code-record/v1','version':'0.1.0','date':'2026-10-24',
              'purpose':'historical research code record and light execution checks',
              'members':[{'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
                         for name,data in payloads.items()]}
    payloads['CODE_RECORD_MANIFEST.json']=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    out=ROOT/'artifacts';out.mkdir(exist_ok=True)
    target=out/'ADM1-TEA-LCA_0.1.0_record_20261024.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,data in sorted(payloads.items()):
            info=zipfile.ZipInfo('ADM1-TEA-LCA/'+name,(2026,10,24,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;info.create_system=3;info.external_attr=0o100644<<16
            z.writestr(info,data)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for row in manifest['members']:
            assert hashlib.sha256(z.read('ADM1-TEA-LCA/'+row['path'])).hexdigest()==row['sha256']
    result={'archive':target.name,'bytes':target.stat().st_size,
            'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'members':len(payloads),'preserved_sources_verified':len(preserved['records'])}
    (out/'archive_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
