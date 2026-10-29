"""Offline, escaped HTML gallery plus machine-readable exports."""
import csv
import io
import json
from collections import defaultdict
from html import escape as esc
from .storage import write,write_json,now

STYLE='''
:root{font-family:system-ui,sans-serif;color:#203346;background:#f0f3f6}body{margin:0}
header{padding:30px 4vw 20px;background:#15374b;color:white}h1{margin:0 0 10px;font-size:28px}
header p{max-width:1100px;line-height:1.5;color:#d7e7ee}.controls{padding:18px 4vw;background:white;position:sticky;top:0;z-index:2;display:flex;gap:10px;flex-wrap:wrap;border-bottom:1px solid #cbd5dc}
input,select{padding:10px;border:1px solid #bccbd5;border-radius:5px;font:inherit}#query{flex:1;min-width:240px}
main{padding:20px 4vw}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:18px}
article{background:white;border:1px solid #dce2e7;border-radius:10px;overflow:hidden;padding:18px}
article img{height:175px;width:100%;object-fit:contain;background:white}h2{font-size:18px;line-height:1.35;margin:12px 0}a{color:#116985}header a{color:#afe7f3}
.badge{display:inline-block;background:#eaf1f5;border-radius:4px;padding:3px 7px;margin:2px;font-size:12px}.muted{color:#607383;font-size:13px}.abstract{font-size:14px;line-height:1.5}.score{font-weight:700;color:#12624e}summary{cursor:pointer;margin:10px 0}details p,details li{font-size:13px;line-height:1.5}#count{margin:0 0 15px}.links{display:flex;gap:12px;flex-wrap:wrap;font-size:13px} .empty{height:90px;padding-top:50px;text-align:center;background:#f3f5f7;color:#758590}
'''
SCRIPT='''
const cards=[...document.querySelectorAll('article')];
function filter(){let q=document.querySelector('#query').value.toLowerCase(),category=document.querySelector('#category').value,level=document.querySelector('#level').value,min=+document.querySelector('#minimum').value,group=document.querySelector('#group').checked;let seen=new Set(),count=0;
for(const c of cards){let ok=c.dataset.search.includes(q)&&(!category||JSON.parse(c.dataset.categories).includes(category))&&(!level||c.dataset.level===level)&&(+c.dataset.score>=min);if(ok&&group){if(seen.has(c.dataset.family))ok=false;else seen.add(c.dataset.family)}c.hidden=!ok;if(ok)count++}document.querySelector('#count').textContent=count+' publications shown'+(group?' · highest-ranked matching member per provider family':'');}
document.querySelectorAll('input,select').forEach(x=>x.addEventListener('input',filter));filter();
'''


def render(store):
    records=store.records();stats=store.stats();families=defaultdict(list);embodiments=[]
    for r in records:
        families[r['family_id']].append(r['publication_id']);embodiments.extend(r['embodiment_references'])
    write(store.root/'catalog.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    write_json(store.root/'families.json',dict(families));write_json(store.root/'embodiment_references.json',embodiments)
    stats['unreviewed_figure_embodiment_references']=len(embodiments)
    write_json(store.root/'summary.json',dict(stats,generated_at=now()))
    stream=io.StringIO();writer=csv.writer(stream)
    writer.writerow(['publication_id','family_id','title','assignees','priority_date','triage_score','triage_level','category_mentions','verification','source_url'])
    for r in records:writer.writerow([r['publication_id'],r['family_id'],r['title'],'; '.join(r['assignees']),r['dates'].get('priorityDate',''),r['triage_score'],r['triage_level'],'; '.join(r['category_mentions']),r['verification'],r['source_url']])
    write(store.root/'catalog.csv',stream.getvalue())
    queue_stream=io.StringIO();queue_writer=csv.writer(queue_stream)
    queue_writer.writerow(['publication_id','title','collection_status','discovery_depth','last_error','source_url'])
    for row in store.db.execute('SELECT * FROM candidates ORDER BY priority DESC,publication_id'):
        queue_writer.writerow([row['publication_id'],row['title'],row['status'],row['depth'],row['error'] or '',
                               'https://patents.google.com/patent/'+row['publication_id']+'/en'])
    write(store.root/'discovered_candidates.csv',queue_stream.getvalue())
    cards=[]
    for r in records:
        pid=r['publication_id'];base='documents/'+pid
        image=f'<img loading="lazy" src="{esc(r["local_thumbnail"])}" alt="Patent drawing preview; figure mapping unverified">' if r.get('local_thumbnail') else '<div class="empty">No local drawing preview</div>'
        evidence=[]
        for key,e in r['evidence'].items():
            snippet=e.get('excerpt',', '.join(e.get('labels',[])))
            href=r['source_url']+('#'+e['anchor'] if e.get('anchor') else '')
            evidence.append(f'<li><b>{esc(key.replace("_"," "))}</b>: {esc(snippet)} <a href="{esc(href)}">source</a></li>')
        figures=''.join(f'<li>Figure {esc(e["figure"])}: {esc(e["excerpt"])} <a href="{esc(r["source_url"]+"#"+e["source_anchor"])}">source</a></li>' for e in r['embodiment_references'])
        discoveries=store.db.execute('SELECT kind,source,category FROM discoveries WHERE publication_id=?',(pid,)).fetchall()
        origins=''.join(f'<li>{esc(d[0])}: {esc(d[1])} {esc(d[2])}</li>' for d in discoveries)
        search=' '.join([pid,r['title'],r['abstract'],*r['assignees'],*r['cpc'],*r['category_mentions']]).lower()
        links=f'<a href="{esc(r["source_url"])}">Patent</a><a href="{base}/record.json">Record</a><a href="{base}/tables.json">Tables</a><a href="{base}/description.txt">Text</a>'
        if r.get('local_pdf'):links+=f'<a href="{esc(r["local_pdf"])}">Local PDF</a>'
        elif r.get('pdf_url'):links+=f'<a href="{esc(r["pdf_url"])}">Source PDF</a>'
        badges=''.join(f'<span class="badge">{esc(t)}</span>' for t in r['category_mentions'])
        cards.append(f'''<article data-search="{esc(search,quote=True)}" data-family="{esc(r['family_id'])}" data-categories="{esc(json.dumps(r['category_mentions']),quote=True)}" data-level="{r['triage_level']}" data-score="{r['triage_score']}">
{image}<h2>{esc(r['title'])}</h2><p class="muted">{pid} · {esc('; '.join(r['assignees']))}<br>{esc(r['family_id'])} · {len(families[r['family_id']])} collected publication(s)</p>
<p><span class="score">Evidence score {r['triage_score']}/100</span> · {r['triage_level'].replace('_',' ')}<br><span class="muted">Unreviewed · no optical validation</span></p>
{badges}<p class="abstract">{esc(r['abstract'][:500])}{'…' if len(r['abstract'])>500 else ''}</p><div class="links">{links}</div>
<details><summary>Evidence · {r['numeric_table_count']} numerical tables</summary><ul>{''.join(evidence)}</ul><p>Missing text evidence: {esc(', '.join(r['missing_evidence']) or 'none detected; completeness still unverified')}</p></details>
<details><summary>Physics mentions</summary><p>{esc(', '.join(r['physics_mentions']))}. Mentions may describe alternatives or prior art, not required physics of every embodiment.</p></details>
<details><summary>{len(r['embodiment_references'])} figure/embodiment references</summary><p>Automatically detected references; distinct designs and table associations are unverified.</p><ul>{figures}</ul></details>
<details><summary>Discovery provenance</summary><ul>{origins}</ul></details></article>''')
    categories=sorted({c for r in records for c in r['category_mentions']})
    options=''.join(f'<option value="{c}">{c.replace("_"," ")}</option>' for c in categories)
    html=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Folded optics patent catalog</title><style>{STYLE}</style>
<header><h1>Folded optics · patent design catalog</h1><p>{stats['documents']} publications · {stats['families']} provider families · {len(embodiments)} unreviewed figure references · {stats['candidates']} discovered candidates</p><p>Search, filter, and inspect source evidence. Scores prioritize reading; they do not certify complete prescriptions or successful ray traces. Category labels are text mentions. Family grouping uses Google Patents' supplied identifier.</p><p><a href="catalog.csv">Collected CSV</a> · <a href="discovered_candidates.csv">All discovered candidates</a> · <a href="catalog.jsonl">JSONL</a> · <a href="families.json">Families</a> · <a href="summary.json">Counts</a> · <a href="last_run.json">Last collection status</a></p></header>
<div class="controls"><input id="query" type="search" placeholder="Search title, assignee, publication, CPC…" aria-label="Search catalog"><select id="category" aria-label="Category"><option value="">All category mentions</option>{options}</select><select id="level" aria-label="Evidence level"><option value="">All evidence levels</option><option>prescription_candidate</option><option>numerical_layout_candidate</option><option>layout_candidate</option></select><select id="minimum" aria-label="Minimum evidence score"><option value="0">Any score</option><option value="50">Score ≥ 50</option><option value="75">Score ≥ 75</option></select><label><input id="group" type="checkbox" checked> Group families</label></div>
<main><p id="count"></p><div class="grid">{''.join(cards)}</div></main><script>{SCRIPT}</script></html>'''
    write(store.root/'index.html',html)
    return stats
