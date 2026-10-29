"""Run the retained-candidate numerical discovery policy without comparison arms."""
from __future__ import annotations
import argparse
from .benchmark import main as run_benchmark


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[200,201,202])
    p.add_argument('--corners', type=int, choices=(3,4,5), nargs='+', default=[3,4,5])
    p.add_argument('--briefs', choices=('base','wide','shifted'), nargs='+', default=['base','wide'])
    p.add_argument('--search-calls', type=int, default=400)
    p.add_argument('--workers', type=int, default=6)
    a=p.parse_args(argv)
    # Reuse the exact benchmark search/audit path; no production-only ranking or
    # hidden budget changes. "development" makes no fresh confirmation claim.
    run_benchmark(['--out',a.out,'--stage','development','--candidate','portfolio',
                   '--arms','portfolio','--seeds',*map(str,a.seeds),
                   '--corners',*map(str,a.corners),'--briefs',*a.briefs,
                   '--search-calls',str(a.search_calls),'--workers',str(a.workers)])


if __name__=='__main__':
    main()
