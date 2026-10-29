"""Report all curriculum arms, including failures and zero or negative learning gains."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .pencil_loop import write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('corpus',type=Path);p.add_argument('run',type=Path)
    a=p.parse_args();tasks=json.loads((a.corpus/'tasks.json').read_text());split={t['task_id']:t['split'] for t in tasks}
    base=json.loads((a.run/'baseline_development.json').read_text())
    base_test=json.loads((a.run/'test_baseline_0.json').read_text())
    rows=[];costs=[]
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,part in zip(axes,['train','validation','test']):
        for arm in ['uniform','fixed','adaptive']:
            points=[]
            for round_index in range(3):
                if round_index==0:episodes=base_test if part=='test' else base
                else:
                    file=a.run/f'test_{arm}_{round_index}.json' if part=='test' else a.run/arm/f'round{round_index}'/'development.json'
                    episodes=json.loads(file.read_text())
                selected=[e for e in episodes if split[e['task_id']]==part]
                wins=sum(e['success'] for e in selected);points.append(wins/len(selected))
                rows.append(dict(arm=arm,round=round_index,split=part,successes=wins,tasks=len(selected)))
            ax.plot(range(3),points,marker={'uniform':'o','fixed':'s','adaptive':'^'}[arm],
                    linestyle={'uniform':'-','fixed':'--','adaptive':':'}[arm],fillstyle='none',label=arm)
        ax.set(title=part,xlabel='Training round',ylabel='Audited success fraction',xticks=[0,1,2],ylim=(-.05,1.05));ax.grid(alpha=.2)
    axes[2].text(.04,.86,'Fixed and adaptive overlap',transform=axes[2].transAxes,fontsize=9)
    axes[0].legend();fig.suptitle('Qwen pencil-beam planner curriculum • one seed • 8 train / 4 validation / 4 test tasks')
    fig.savefig(a.run/'learning_curves.png',dpi=170);plt.close(fig)
    for arm in ['uniform','fixed','adaptive']:
        training=[json.loads((a.run/arm/f'round{i}'/'training.json').read_text()) for i in [1,2]]
        policies=[json.loads((a.run/arm/f'round{i}'/'development.json').read_text()) for i in [1,2]]
        costs.append(dict(arm=arm,padded_training_tokens=sum(r['padded_training_tokens'] for r in training),
            supervised_tokens=sum(r['supervised_tokens'] for r in training),
            training_seconds=sum(r['elapsed_seconds'] for r in training),
            development_language_tokens=sum(sum(e['response']['usage'].values()) for batch in policies for e in batch),
            development_optical_rays=sum(e['charged_optical_rays'] for batch in policies for e in batch)))
    assert len({c['padded_training_tokens'] for c in costs})==1
    write(a.run/'learning_curves.json',dict(rows=rows,costs=costs,
        caveats=['One seed and four test tasks; not a statistical demonstration.',
                 'Finite-pool adaptive selection, not learned environment synthesis.',
                 'Tasks grouped by prescription; archival evolutionary ancestry is unresolved.',
                 'Numerical plan outcomes are cached, with optical replay verification.']))
    print(json.dumps(dict(rows=rows,costs=costs),indent=2))

if __name__=='__main__':main()
