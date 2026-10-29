"""Resumable queue, provenance, atomic local files, and a bounded HTTP cache."""
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
import hashlib
import json
import sqlite3
import time


def now():return datetime.now(timezone.utc).isoformat()


def write(path, content):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    if isinstance(content,bytes):tmp.write_bytes(content)
    else:tmp.write_text(content,encoding='utf-8')
    tmp.replace(path)


def write_json(path, content):write(path,json.dumps(content,ensure_ascii=False,indent=2)+'\n')


class AccessStopped(RuntimeError):
    def __init__(self, code, url):
        self.code, self.url = code, url
        super().__init__(f'HTTP {code} at {url}; access stopped, cached progress retained')


class ServiceUnavailable(RuntimeError):
    """Repeated upstream failures; stop the batch rather than exhaust its queue."""


class Fetcher:
    def __init__(self,root,delay=1.0,offline=False,refresh=False):
        self.cache=Path(root)/'cache';self.cache.mkdir(parents=True,exist_ok=True)
        self.delay=max(0.,delay);self.offline=offline;self.refresh=refresh
        self.last_request=0.;self.requests=0;self.cache_hits=0
    def get(self,url,max_bytes=12_000_000):
        parsed=urlparse(url)
        if parsed.scheme!='https' or parsed.hostname not in ('patents.google.com','patentimages.storage.googleapis.com'):
            raise ValueError('Unexpected source URL: '+url)
        key=hashlib.sha256(url.encode()).hexdigest();path=self.cache/(key+'.bin')
        if path.exists() and (self.offline or not self.refresh):
            data=path.read_bytes()
            if len(data)>max_bytes:raise ValueError('Cached response exceeds size limit')
            self.cache_hits+=1;return data
        if self.offline:raise FileNotFoundError('Uncached URL in offline mode: '+url)
        for attempt in range(3):
            time.sleep(max(0.,self.delay-(time.monotonic()-self.last_request)))
            self.requests+=1;self.last_request=time.monotonic()
            try:
                req=Request(url,headers={'User-Agent':'NSLens-PatentCatalog/0.1 (bounded optical research catalog)'})
                with urlopen(req,timeout=25) as response:
                    data=response.read(max_bytes+1)
                    if len(data)>max_bytes:raise ValueError('Response exceeds size limit')
                    content_type=response.headers.get('Content-Type','')
                write(path,data);write_json(self.cache/(key+'.json'),{'url':url,'retrieved_at':now(),'sha256':hashlib.sha256(data).hexdigest(),'content_type':content_type})
                return data
            except HTTPError as e:
                if e.code in (401,403,429):raise AccessStopped(e.code,url) from e
                if e.code<500 or attempt==2:raise
            except (URLError,TimeoutError):
                if attempt==2:raise
            time.sleep(2**attempt)
        raise RuntimeError('unreachable')


class Store:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.root/'catalog.sqlite')
        self.db.row_factory=sqlite3.Row
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS candidates (
          publication_id TEXT PRIMARY KEY,title TEXT NOT NULL DEFAULT '',depth INTEGER NOT NULL,
          priority REAL NOT NULL,status TEXT NOT NULL DEFAULT 'pending',error TEXT);
        CREATE TABLE IF NOT EXISTS discoveries (
          publication_id TEXT NOT NULL,kind TEXT NOT NULL,source TEXT NOT NULL,
          category TEXT NOT NULL DEFAULT '', UNIQUE(publication_id,kind,source,category));
        CREATE TABLE IF NOT EXISTS documents (
          publication_id TEXT PRIMARY KEY,family_id TEXT NOT NULL,score INTEGER NOT NULL,
          record_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS edges (
          source TEXT NOT NULL,target TEXT NOT NULL,kind TEXT NOT NULL,UNIQUE(source,target,kind));
        CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY,record_json TEXT NOT NULL);
        ''')
        self.db.commit()
    def discover(self,pid,kind,source,category='',title='',depth=0,priority=0.):
        self.db.execute('''INSERT INTO candidates(publication_id,title,depth,priority) VALUES(?,?,?,?)
          ON CONFLICT(publication_id) DO UPDATE SET depth=min(depth,excluded.depth),
          priority=max(priority,excluded.priority),title=CASE WHEN title='' THEN excluded.title ELSE title END''',
          (pid,title,depth,priority))
        self.db.execute('INSERT OR IGNORE INTO discoveries VALUES(?,?,?,?)',(pid,kind,source,category))
        self.db.commit()
    def save(self,record,text,tables):
        pid=record['publication_id'];folder=self.root/'documents'/pid
        write_json(folder/'record.json',record);write(folder/'description.txt',text+'\n');write_json(folder/'tables.json',tables)
        self.db.execute('INSERT OR REPLACE INTO documents VALUES(?,?,?,?)',
                        (pid,record['family_id'],record['triage_score'],json.dumps(record)))
        self.db.execute("UPDATE candidates SET status='fetched',error=NULL WHERE publication_id=?",(pid,))
        for r in record['relations']:
            self.db.execute('INSERT OR IGNORE INTO edges VALUES(?,?,?)',(pid,r['target'],r['kind']))
        self.db.commit()
    def update_record(self,record):
        write_json(self.root/'documents'/record['publication_id']/'record.json',record)
        self.db.execute('UPDATE documents SET record_json=? WHERE publication_id=?',(json.dumps(record),record['publication_id']))
        self.db.commit()
    def next(self):
        return self.db.execute("SELECT * FROM candidates WHERE status='pending' ORDER BY priority DESC,depth,publication_id LIMIT 1").fetchone()
    def failed(self,pid,error):
        self.db.execute("UPDATE candidates SET status='error',error=? WHERE publication_id=?",(str(error),pid));self.db.commit()
    def retry_errors(self):
        self.db.execute("UPDATE candidates SET status='pending',error=NULL WHERE status='error'");self.db.commit()
    def records(self):
        return [json.loads(r[0]) for r in self.db.execute('SELECT record_json FROM documents ORDER BY score DESC,publication_id')]
    def stats(self):
        return {'documents':self.db.execute('SELECT count(*) FROM documents').fetchone()[0],
                'families':self.db.execute('SELECT count(DISTINCT family_id) FROM documents').fetchone()[0],
                'candidates':self.db.execute('SELECT count(*) FROM candidates').fetchone()[0],
                'queue':dict(self.db.execute('SELECT status,count(*) FROM candidates GROUP BY status')),
                'citation_and_family_edges':self.db.execute('SELECT count(*) FROM edges').fetchone()[0]}
