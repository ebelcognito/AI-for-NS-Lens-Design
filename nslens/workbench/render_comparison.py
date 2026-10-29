"""Plot every held-out case, including regressions, for a paired model comparison."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    rows=[[json.loads(path.read_text()) for path in sorted(root.glob('episode*.json'))] for root in [a.baseline,a.candidate]]
    assert [e['task']['task_id'] for e in rows[0]]==[e['task']['task_id'] for e in rows[1]]
    count=len(rows[0]);x=np.arange(count)
    fig,axes=plt.subplots(2,1,figsize=(14,8),layout='constrained',sharex=True)
    for i,(episodes,color,label) in enumerate(zip(rows,['#277da8','#d58b32'],['Frozen Qwen3-8B','Qwen3-8B + LoRA'])):
        positions=x+(i-.5)*.36
        rms=[e['outcome']['final_audit']['rms']*1000 for e in episodes]
        axes[0].bar(positions,rms,width=.34,color=color,label=label)
        axes[1].bar(positions,[sum(e['usage'].values())/1000 for e in episodes],width=.34,color=color)
        for j,e in enumerate(episodes):
            if not e['outcome']['success']:
                axes[0].scatter(positions[j],rms[j]*1.15,marker='x',color='#b2182b',s=65,zorder=4)
    thresholds={e['task']['spec']['max_rms']*1000 for row in rows for e in row}
    if len(thresholds)==1:
        threshold=thresholds.pop()
        axes[0].axhline(threshold,color='#b2182b',ls='--',lw=1,label=f'Aggregate RMS limit: {threshold:g} µm')
    axes[0].set(yscale='log',ylabel='Final audit RMS (µm; logarithmic scale)',title='All 12 held-out tasks: optical quality')
    all_rms=[e['outcome']['final_audit']['rms']*1000 for row in rows for e in row]
    axes[0].set_ylim(min(all_rms)*.7,max(all_rms)*3)
    axes[0].legend(frameon=False,ncol=3,loc='upper left')
    axes[1].set(ylabel='Input + output tokens (thousands)',title='Language-model cost under the same 64k ceiling',xlabel='Test family / case')
    axes[1].set_xticks(x,[f"F{int(e['task']['family_id'].split('/')[-1])}/C{e['task']['task_id'][-1]}" for e in rows[0]])
    for ax in axes:
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        for edge in [2.5,5.5,8.5]:ax.axvline(edge,color='#ccc',lw=.7)
    successes=[sum(e['outcome']['success'] for e in row) for row in rows]
    fig.suptitle(f'LoRA learning check: frozen {successes[0]}/{count} passed → trained {successes[1]}/{count} passed',fontsize=17,fontweight='bold')
    fig.supxlabel('Identical tasks and 24-candidate tracing allowance. Red × = failed final acceptance. All cases shown; no selection of successes.',fontsize=10)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.out,dpi=180,facecolor='white');plt.close(fig)
    print(a.out)


if __name__=='__main__':main()
