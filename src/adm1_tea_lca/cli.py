"""Small command-line workflow for preserving an integrated calculation run."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from . import ADM1Config, ADM1Simulator, BoundaryParams, LCAParams, PlantEvaluator, TEAParams, __version__
from .analysis import compact, grid_search, oat_sensitivity, pareto_front, robust_grid_screen, sampled_sensitivity
from .records import record_run


def main(argv=None):
    parser=argparse.ArgumentParser(prog='adm1-tea-lca')
    parser.add_argument('--version',action='version',version=__version__)
    parser.add_argument('command',choices=('run','demo'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--silage-vs-fraction',type=float,default=.8)
    parser.add_argument('--hrt-days',type=float,default=30.)
    parser.add_argument('--olr',type=float,default=2.)
    parser.add_argument('--historical-boundaries',action='store_true',help='Enable reconstructed HRT/rheology postprocessing')
    args=parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f'Output already exists; choose a new run folder: {args.output}')
        raw=json.loads(args.config.read_text(encoding='utf-8-sig')) if args.config else {}
        if set(raw)-{'adm1','tea','lca','boundary'}:
            raise ValueError('Unknown configuration section')
        boundary=BoundaryParams(**raw.get('boundary',{}))
        if args.historical_boundaries:
            boundary=replace(boundary,apply_hrt_loss=True,apply_rheology=True)
        evaluator=PlantEvaluator(ADM1Simulator(ADM1Config(**raw.get('adm1',{}))),
            TEAParams(**raw.get('tea',{})),LCAParams(**raw.get('lca',{})),boundary)
        result=evaluator.evaluate(args.silage_vs_fraction,args.hrt_days,args.olr)
        analyses=None
        if args.command=='demo':
            grid=grid_search(evaluator)
            analyses={'grid':grid,'pareto_screen':pareto_front(grid['accepted']),
                      'robust_grid':robust_grid_screen(grid['accepted'],.12),
                      'oat':oat_sensitivity(result,evaluator),
                      'sampled_sensitivity':sampled_sensitivity(result,evaluator,n=64)}
        record_run(args.output,result,analyses=analyses,command=list(sys.argv[1:] if argv is None else argv))
        print(json.dumps({'status':'completed','summary':compact(result),'output':str(args.output),
                          'purpose':'recorded execution of reconstructed ADM1 biomethane workflow'},indent=2,ensure_ascii=False))
        return 0
    except (ValueError,RuntimeError,OSError,TypeError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
