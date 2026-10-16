"""Plot measured archive coverage and add the comparison to the local demo."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(); p.add_argument('out'); args=p.parse_args(); out=Path(args.out)
    report=json.loads((out/'report.json').read_text())
    arms=['random','portfolio','archive']; colors=['#94a3b8','#f59e0b','#38bdf8']
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    rows=[]
    for arm,color in zip(arms,colors):
        runs=[r for r in report['runs'] if r['arm']==arm]
        curves=np.array([[x['cells'] for x in r['curve']] for r in runs]); x=[v['calls'] for v in runs[0]['curve']]
        axes[0].plot(x,curves.mean(axis=0),label=arm,color=color)
        axes[0].fill_between(x,curves.min(axis=0),curves.max(axis=0),color=color,alpha=.15)
        vals=np.array([[r['summary'][k] for k in ['coarse_cells','dense_cells','heldout_cells']] for r in runs])
        j=arms.index(arm); xpos=np.arange(3)+.23*(j-1)
        axes[1].bar(xpos,vals.mean(axis=0),width=.22,label=arm,color=color)
        for v in vals: axes[1].scatter(xpos,v,color='black',s=10,alpha=.5)
        rows.append((arm,*vals.mean(axis=0)))
    axes[0].set(xlabel='Candidate evaluations per seed',ylabel='Occupied appearance cells',title='Search coverage: mean and seed range'); axes[0].legend()
    axes[1].set(xticks=np.arange(3),xticklabels=['Coarse training','Dense training','Held-out'],ylabel='Occupied appearance cells',title='Re-evaluated coverage: mean and individual seeds')
    for ax in axes: ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    fig.savefig(out/'coverage.png',dpi=160); fig.savefig(out/'coverage.svg'); plt.close(fig)
    table='<h2>Measured comparison</h2><p>Mean occupied bins over seeds. These are descriptor-space counts, not a beauty rating. Black dots show individual seeds.</p><img src="coverage.svg"><table><tr><th>Method</th><th>Coarse</th><th>Dense training</th><th>Held-out</th></tr>'
    for arm,a,b,c in rows: table+=f'<tr><td>{arm}</td><td>{a:.1f}</td><td>{b:.1f}</td><td>{c:.1f}</td></tr>'
    table+='</table>'
    h=(out/'index.html').read_text(); pos=h.index('<h2>archive</h2>'); h=h[:pos]+table+h[pos:]; (out/'index.html').write_text(h)
    print(rows)

if __name__=='__main__': main()
