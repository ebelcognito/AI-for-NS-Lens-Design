"""Automated keyword/CPC search, bounded graph expansion, local audit gallery."""
import argparse
import csv
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from .extract import extract_page,search_records,publication_id,CATEGORY_PATTERNS
from .storage import Fetcher,Store,AccessStopped,ServiceUnavailable,write,write_json,now
from .render import render
import re


def search_url(query,page,size):
    terms={'q':query['q'],'num':size,'page':page,'language':'ENGLISH','type':'PATENT'}
    if query.get('cpc'):terms['cpc']=query['cpc']
    return 'https://patents.google.com/xhr/query?'+urlencode({'url':urlencode(terms),'exp':''})


def discover(store,fetcher,config,pages,size):
    for pid in config.get('seeds',[]):
        pid=publication_id(pid);store.discover(pid,'seed','searches.json',priority=1000)
    for query in config['queries']:
        for page in range(pages):
            url=search_url(query,page,size)
            rows=search_records(json.loads(fetcher.get(url)))
            for rank,row in enumerate(rows):
                # Round-robin across query ranks instead of consuming one category first.
                store.discover(row['publication_id'],'query',url,query['category'],row['title'],priority=100-rank-page*size)
            print(f"Search [{query['category']}] page {page+1}: {len(rows)} results",flush=True)
            if len(rows)<size:break


def expand(store,record,depth,max_depth,per_kind):
    if depth>=max_depth:return
    for kind in ('cites','cited_by','family_member'):
        related=[r for r in record['relations'] if r['kind']==kind]
        def relevance(r):
            return sum(bool(re.search(pattern,r['title'],re.I)) for pattern in CATEGORY_PATTERNS.values())
        related.sort(key=lambda r:(-relevance(r),not r['target'].startswith('US'),r['target']))
        for rel in related[:per_kind]:
            store.discover(rel['target'],kind,record['publication_id'],title=rel['title'],depth=depth+1,
                           priority=95-depth*10+relevance(rel))


def assets(store,fetcher,record,pdf=False):
    pid=record['publication_id'];folder=store.root/'documents'/pid
    errors=[]
    denied=record.setdefault('unavailable_assets',[])
    if record['thumbnail_urls'] and not record.get('local_thumbnail') and record['thumbnail_urls'][0] not in denied:
        try:
            blob=fetcher.get(record['thumbnail_urls'][0],max_bytes=4_000_000)
            if not (blob.startswith(b'\x89PNG') or blob.startswith(b'\xff\xd8')):raise ValueError('Thumbnail response is not PNG/JPEG')
            ext='png' if blob.startswith(b'\x89PNG') else 'jpg'
            rel=f'documents/{pid}/preview.{ext}';write(store.root/rel,blob);record['local_thumbnail']=rel
        except AccessStopped as e:
            if e.code==429:raise
            denied.append(e.url);errors.append('thumbnail unavailable: '+str(e))
        except Exception as e:errors.append('thumbnail: '+str(e))
    if pdf and record['pdf_url'] and not record.get('local_pdf') and record['pdf_url'] not in denied:
        try:
            blob=fetcher.get(record['pdf_url'],max_bytes=30_000_000)
            if not blob.startswith(b'%PDF-'):raise ValueError('PDF response is not a PDF')
            rel=f'documents/{pid}/source.pdf';write(store.root/rel,blob);record['local_pdf']=rel
        except AccessStopped as e:
            if e.code==429:raise
            denied.append(e.url);errors.append('PDF unavailable: '+str(e))
        except Exception as e:errors.append('PDF: '+str(e))
    record['asset_errors']=list(dict.fromkeys(record.get('asset_errors',[])+errors));store.update_record(record)


def collect(store,fetcher,max_new,max_depth,per_kind):
    completed=0;attempts=0;service_failures=0
    # Existing records can expand further when max-depth is increased on resume.
    for record in store.records():
        row=store.db.execute('SELECT depth FROM candidates WHERE publication_id=?',(record['publication_id'],)).fetchone()
        expand(store,record,row[0],max_depth,per_kind)
    while attempts<max_new:
        candidate=store.next()
        if candidate is None:break
        pid=candidate['publication_id'];attempts+=1
        try:
            url=f'https://patents.google.com/patent/{pid}/en'
            raw=fetcher.get(url)
            record,text,tables=extract_page(raw,pid)
            record['collected_at']=now();record['source_html']=f'documents/{pid}/source.html'
            write(store.root/record['source_html'],raw)
            store.save(record,text,tables);completed+=1
            service_failures=0
            expand(store,record,candidate['depth'],max_depth,per_kind)
            assets(store,fetcher,record)
            print(f"Collect {attempts}/{max_new}: {pid} · score {record['triage_score']} · {record['title'][:75]}",flush=True)
        except AccessStopped:raise
        except Exception as e:
            # A thumbnail failure must never downgrade an already persisted document.
            exists=store.db.execute('SELECT 1 FROM documents WHERE publication_id=?',(pid,)).fetchone()
            if not exists:store.failed(pid,e)
            print(f'Failed {pid}: {type(e).__name__}: {e}',flush=True)
            unavailable=(isinstance(e,HTTPError) and e.code>=500) or isinstance(e,(URLError,TimeoutError))
            service_failures=service_failures+1 if unavailable else 0
            if service_failures>=3:
                raise ServiceUnavailable('Three consecutive upstream failures; collection paused. Retry later with --retry-errors.') from e
    return {'new_documents':completed,'attempts':attempts}


def import_ids(store,path):
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            pid=publication_id(line);store.discover(pid,'import',str(path),priority=500)


def import_csv(store,path):
    """Import our CSV or a Google Patents export without another search request."""
    lines=Path(path).read_text(encoding='utf-8-sig').splitlines()
    keys=('publication_id','publication number','publication_number','id','result link','url')
    header=None
    for i,line in enumerate(lines[:10]):
        fields=[v.strip().lower() for v in next(csv.reader([line]))]
        if any(k in fields for k in keys):header=i;break
    if header is None:raise ValueError('CSV has no recognized publication ID or URL column')
    added=0;skipped=0
    for row in csv.DictReader(lines[header:]):
        row={str(k).strip().lower():v for k,v in row.items() if k is not None}
        pid=None
        for key in keys:
            if row.get(key):
                try:pid=publication_id(row[key]);break
                except ValueError:continue
        if pid is None:skipped+=1;continue
        store.discover(pid,'csv_import',str(path),title=row.get('title',''),priority=500);added+=1
    print(f'CSV import: {added} valid rows, {skipped} skipped',flush=True)
    if not added:raise ValueError('No valid patent publications in CSV')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['run','discover','collect','render','status'])
    parser.add_argument('--output',default='results/patent_catalog')
    parser.add_argument('--config',default=str(Path(__file__).with_name('searches.json')))
    parser.add_argument('--max-new',type=int,default=60,help='Maximum document fetch attempts this invocation')
    parser.add_argument('--pages',type=int,default=1)
    parser.add_argument('--page-size',type=int,choices=[10,20,50,100],default=10)
    parser.add_argument('--max-depth',type=int,default=1)
    parser.add_argument('--expand-per-kind',type=int,default=2)
    parser.add_argument('--pdf-limit',type=int,default=0,help='Ensure local PDFs for this many highest-ranked collected documents')
    parser.add_argument('--delay',type=float,default=1.,help='Minimum interval between network request starts')
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--refresh-search',action='store_true',help='Refresh search responses; document cache remains reusable')
    parser.add_argument('--retry-errors',action='store_true')
    parser.add_argument('--import-ids',help='Text file: one publication number or patent URL per line')
    parser.add_argument('--import-csv',help='Google Patents export or catalog CSV; supports a metadata preamble')
    args=parser.parse_args()
    if any(v<0 for v in [args.max_new,args.max_depth,args.expand_per_kind,args.pdf_limit,args.delay]) or args.pages<1:
        parser.error('Budgets must be nonnegative and pages must be positive')
    store=Store(args.output)
    if args.command=='status':print(json.dumps(store.stats(),indent=2));return
    if args.command=='render':print(json.dumps(render(store),indent=2));return
    fetcher=Fetcher(args.output,delay=args.delay,offline=args.offline)
    run={'started_at':now(),'arguments':vars(args),'status':'running','before':store.stats()}
    stopped=False;fatal=False
    try:
        if args.retry_errors:store.retry_errors()
        if args.import_ids:import_ids(store,args.import_ids)
        if args.import_csv:import_csv(store,args.import_csv)
        if args.command in ('run','discover'):
            config=json.loads(Path(args.config).read_text())
            write_json(store.root/'search_config.json',config)
            fetcher.refresh=args.refresh_search
            discover(store,fetcher,config,args.pages,args.page_size)
            fetcher.refresh=False
        if args.command in ('run','collect'):
            run.update(collect(store,fetcher,args.max_new,args.max_depth,args.expand_per_kind))
            for record in store.records()[:args.pdf_limit]:assets(store,fetcher,record,pdf=True)
        run['status']='complete'
    except AccessStopped as e:
        run['status']='access_stopped';run['error']=str(e);stopped=True
        print(str(e),flush=True)
    except ServiceUnavailable as e:
        run['status']='service_unavailable';run['error']=str(e);stopped=True
        print(str(e),flush=True)
    except KeyboardInterrupt:
        run['status']='interrupted';stopped=True;print('Interrupted; saving catalog',flush=True)
    except Exception as e:
        run['status']='failed';run['error']=f'{type(e).__name__}: {e}';fatal=True
        print(run['error'],flush=True)
    finally:
        stats=render(store);run.update(finished_at=now(),after=stats,network_requests=fetcher.requests,cache_hits=fetcher.cache_hits)
        run_id=run['started_at'];store.db.execute('INSERT INTO runs VALUES(?,?)',(run_id,json.dumps(run)));store.db.commit()
        write_json(store.root/'last_run.json',run)
        print(json.dumps(stats,indent=2));print('Gallery:',store.root/'index.html')
    if stopped or fatal:raise SystemExit(2)

if __name__=='__main__':main()
