"""Offline tests of data provenance, deduplication, resumability, and output safety."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from nslens.patent_catalog.extract import extract_page,publication_id,search_records
from nslens.patent_catalog.storage import Store,Fetcher,AccessStopped,ServiceUnavailable,write
from nslens.patent_catalog.render import render
from nslens.patent_catalog.__main__ import collect,expand,search_url,assets,import_csv

HTML='''<html><head><meta name="DC.title" content="Folded &lt;script&gt; optics">
<meta name="citation_patent_publication_number" content="US:20210096338:A1">
<meta name="DC.description" content="Freeform prism with total internal reflection."></head>
<body><dd itemprop="assigneeOriginal">Optics Company</dd>
<span itemprop="Code">G02B13/0065</span>
<section itemprop="description"><div class="description-line" id="p-1">FIG. 3A illustrates a second example embodiment with local coordinates and refractive index. The radius of curvature, aperture stop, and Zernike coefficients are in Table 3.</div>
<patent-tables><table><tr><td>TABLE 3</td></tr><tr><td>1.2 2.3 3.4 4.5 5.6 6.7 7.8 8.9 9.1 10.1 11.1 12.1</td></tr></table></patent-tables></section>
<section itemprop="family"><h2>ID=12345</h2><a href="/patent/US11933949B2/en">grant</a></section>
<tr itemprop="backwardReferences"><td itemprop="publicationNumber">US9239453B2</td><td itemprop="title">Freeform prism</td></tr>
<tr itemprop="forwardReferences"><td itemprop="publicationNumber">US10739578B2</td><td itemprop="title">Folded camera</td></tr></body></html>'''


class PatentCatalogTest(unittest.TestCase):
    def test_extraction_is_evidence_not_verification(self):
        r,text,tables=extract_page(HTML,'US20210096338A1')
        self.assertEqual(r['family_id'],'google:12345')
        self.assertEqual(r['triage_level'],'prescription_candidate')
        self.assertEqual(r['verification'],'unreviewed')
        self.assertEqual(r['embodiment_references'][0]['figure'],'3A')
        self.assertFalse(r['embodiment_references'][0]['prescription_link_verified'])
        self.assertEqual(r['evidence']['coefficients']['anchor'],'p-1')
        self.assertEqual(r['table_count'],1)
        self.assertIn('tir',r['category_mentions'])
        self.assertEqual(len(r['relations']),3)
        self.assertEqual(r['cpc'],['G02B13/0065'])
        with self.assertRaisesRegex(ValueError,'Unexpected document'):
            extract_page(HTML,'US11933949B2')
        with self.assertRaisesRegex(ValueError,'Missing patent'):
            extract_page('<html>Challenge</html>','US11933949B2')

    def test_search_schema_and_publication_ids(self):
        pid='US20210096338A1'
        self.assertEqual(publication_id('https://patents.google.com/patent/'+pid+'/en'),pid)
        self.assertEqual(publication_id('US:20210096338:A1'),pid)
        for bad in ['../../secret','not a patent','https://example.com/not-patent']:
            with self.assertRaises(ValueError):publication_id(bad)
        payload={'results':{'total_num_results':1,'cluster':[{'result':[{'patent':{'publication_number':pid,'title':'<b>Prism</b> &amp; lens'}}]}]}}
        self.assertEqual(search_records(payload)[0]['title'],'Prism & lens')
        with self.assertRaises(ValueError):search_records({'results':{'total_num_results':5}})
        self.assertEqual(search_records({'results':{'total_num_results':0}}),[])
        self.assertIn('cpc%3DG02B',search_url({'q':'prism','cpc':'G02B17/08'},0,10))

    def test_queue_dedup_resume_and_gallery(self):
        with tempfile.TemporaryDirectory() as root:
            store=Store(root);pid='US20210096338A1'
            store.discover(pid,'seed','config',priority=100)
            store.discover(pid,'query','query-url',priority=10)
            self.assertEqual(store.stats()['candidates'],1)
            self.assertEqual(store.next()['priority'],100)
            class FakeFetcher:
                def get(self,url):return HTML.encode()
            result=collect(store,FakeFetcher(),1,1,1)
            self.assertEqual(result['new_documents'],1)
            self.assertEqual(store.stats()['documents'],1)
            self.assertEqual(store.stats()['queue']['pending'],3)
            self.assertNotEqual(store.next()['publication_id'],pid)
            record=store.records()[0]
            store.discover('US11933949B2','seed','config')
            grant=dict(record,publication_id='US11933949B2')
            store.save(grant,'text',[])
            self.assertEqual(store.stats()['families'],1)
            render(store)
            html=Path(root,'index.html').read_text()
            self.assertIn('&lt;script&gt;',html)
            self.assertNotIn('Folded <script> optics',html)
            self.assertEqual(len(Path(root,'catalog.jsonl').read_text().splitlines()),2)
            self.assertIn('query-url',html)
            reopened=Store(root)
            self.assertEqual(reopened.stats()['documents'],2)
            self.assertEqual(reopened.db.execute('SELECT count(*) FROM edges').fetchone()[0],6)
            store.db.close();reopened.db.close()

    def test_stops_on_rate_limit_and_offline_uses_cache(self):
        with tempfile.TemporaryDirectory() as root:
            fetcher=Fetcher(root,delay=0)
            url='https://patents.google.com/patent/US20210096338A1/en'
            with patch('nslens.patent_catalog.storage.urlopen',side_effect=HTTPError(url,429,'limited',{},None)) as mocked:
                with self.assertRaises(AccessStopped):fetcher.get(url)
                self.assertEqual(mocked.call_count,1)
            fetcher.offline=True
            with self.assertRaises(FileNotFoundError):fetcher.get(url)
            import hashlib
            write(Path(root,'cache',hashlib.sha256(url.encode()).hexdigest()+'.bin'),b'cached')
            with patch('nslens.patent_catalog.storage.urlopen',side_effect=AssertionError('network in offline mode')):
                self.assertEqual(fetcher.get(url),b'cached')
            with self.assertRaises(ValueError):fetcher.get('https://unexpected.example/file')

    def test_three_service_failures_pause_queue(self):
        with tempfile.TemporaryDirectory() as root:
            store=Store(root)
            for i in range(4):store.discover(f'US2021009633{i}A1','seed','test')
            class Unavailable:
                def get(self,url):raise HTTPError(url,503,'unavailable',{},None)
            with self.assertRaises(ServiceUnavailable):collect(store,Unavailable(),10,0,0)
            self.assertEqual(store.stats()['queue'],{'error':3,'pending':1})
            store.retry_errors()
            self.assertEqual(store.stats()['queue'],{'pending':4})
            store.db.close()

    def test_denied_thumbnail_is_recorded_without_retry(self):
        with tempfile.TemporaryDirectory() as root:
            store=Store(root);record,text,tables=extract_page(HTML,'US20210096338A1')
            record['thumbnail_urls']=['https://patentimages.storage.googleapis.com/denied.png']
            store.discover(record['publication_id'],'seed','test');store.save(record,text,tables)
            class Denied:
                calls=0
                def get(self,url,**kwargs):
                    self.calls+=1
                    raise AccessStopped(403,url)
            f=Denied();assets(store,f,record);assets(store,f,record)
            self.assertEqual(f.calls,1)
            self.assertEqual(store.stats()['queue'],{'fetched':1})
            self.assertEqual(len(record['unavailable_assets']),1)
            self.assertIn('403',record['asset_errors'][0])
            store.db.close()

    def test_csv_import_and_offline_render(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root,'input.csv')
            path.write_text('Search URL: example\nid,title\nUS-20210096338-A1,Folded optics\nUS-20210096338-A1,Duplicate\nnot-a-patent,Skip\n')
            store=Store(root);import_csv(store,path)
            self.assertEqual(store.stats()['candidates'],1)
            with patch('nslens.patent_catalog.storage.urlopen',side_effect=AssertionError('network')):
                render(store)
            self.assertTrue(Path(root,'index.html').exists())
            store.db.close()

if __name__=='__main__':unittest.main()
