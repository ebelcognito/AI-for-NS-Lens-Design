"""Deterministic extraction. Scores are triage hints, never optical verification."""
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup

SCHEMA_VERSION = 1
ID_RE = re.compile(r'^(?:[A-Z]{2}\d{5,14}|JP[SH]\d{5,12})[A-Z]\d{0,2}$')


def publication_id(value):
    value = str(value).strip()
    if '://' in value:
        match = re.search(r'/patent/([^/]+)', urlparse(value).path)
        if not match:
            raise ValueError(f'Not a patent URL: {value}')
        value = match.group(1)
    value = re.sub(r'[\s:,-]', '', value).upper()
    if not ID_RE.fullmatch(value):
        raise ValueError(f'Invalid publication number: {value}')
    return value


def clean(value):
    return ' '.join(str(value).split())


def plain(value):
    return clean(BeautifulSoup(value or '', 'html.parser').get_text(' ', strip=True))


CATEGORY_PATTERNS = {
    'folded_camera': r'folded (?:camera|optical)|periscope|telephoto',
    'powered_prism': r'free[ -]?form.{0,30}prism|prism.{0,30}optical power|curved reflecting surface',
    'near_eye': r'near[ -]?(?:to[ -]?)?eye|eyepiece|head[ -]mounted',
    'tir': r'total internal reflection|\bTIR\b',
    'catadioptric': r'catadioptric|mangin',
    'reflective_relay': r'optical relay|relay (?:optical|system)|off[ -]axis.{0,30}mirror',
    'polarization_folded': r'pancake|reflective polari[sz]er|polari[sz]ation.{0,25}fold',
    'nonimaging': r'collimator|illumination|non[ -]?imaging',
    'lightguide': r'light[ -]?guide|waveguide',
}
SIGNALS = {
    'coefficients': (r'coefficient|zernike|XY polynomial|aspheric.{0,15}parameter', 20),
    'positions_and_tilts': (r'local coordinates|global coordinates|decent[er]+|surface.{0,20}tilt|angle of each surface', 20),
    'refractive_index': (r'refractive ind(?:ex|ices)|Abbe|\bNd[123]?\b', 10),
    'radii': (r'radii|radius of curvature|curvature radius', 10),
    'aperture_or_pupil': (r'aperture stop|entrance pupil|exit pupil|F-number|F/no', 10),
    'performance': (r'modulation transfer|\bMTF\b|spot diagram|distortion', 5),
}


def extract_page(html, requested_id):
    soup = BeautifulSoup(html, 'html.parser')
    meta = lambda key: soup.find('meta', attrs={'name': key})
    def content(key):
        e = meta(key)
        return clean(e.get('content', '')) if e else ''
    actual = publication_id(content('citation_patent_publication_number') or requested_id)
    if actual != requested_id:
        raise ValueError(f'Unexpected document: requested {requested_id}, got {actual}')
    title = content('DC.title')
    description = soup.select_one('section[itemprop="description"]')
    if not title or description is None:
        raise ValueError('Missing patent title/description; response is not a patent document')
    text = description.get_text(' ', strip=True)
    paras = description.select('.description-line')
    if not paras:
        paras = description.select('p')
    paragraphs = [{'anchor': e.get('id', ''), 'text': clean(e.get_text(' ', strip=True))} for e in paras]
    if not paragraphs:
        paragraphs = [{'anchor': '', 'text': clean(text)}]
    abstract_node = soup.select_one('section[itemprop="abstract"]')
    abstract = clean(abstract_node.get_text(' ', strip=True)) if abstract_node else content('DC.description')
    family = soup.select_one('section[itemprop="family"]')
    family_match = re.search(r'\bID\s*=\s*(\d+)', family.get_text(' ', strip=True)) if family else None
    # Provider family IDs are retained without claiming a DOCDB/INPADOC definition.
    family_id = 'google:' + family_match[1] if family_match else 'publication:' + actual
    tables = []
    nodes = description.select('patent-tables') or description.select('table')
    for i, t in enumerate(nodes):
        value = clean(t.get_text(' ', strip=True))
        if value:
            label = re.search(r'\bTABLE\s+[\w.-]+', value, re.I)
            tables.append({'label': label[0] if label else f'table-{i+1}',
                           'text': value, 'numeric_tokens': len(re.findall(r'[-+−]?\d+(?:\.\d+)?(?:[Ee]\s*[-+−]?\s*\d+)?', value))})
    evidence = {}
    for key, (pattern, weight) in SIGNALS.items():
        hits = [p for p in paragraphs if re.search(pattern, p['text'], re.I)]
        if hits:
            p = hits[0]; match = re.search(pattern, p['text'], re.I)
            evidence[key] = {'weight': weight, 'anchor': p['anchor'],
                             'excerpt': p['text'][max(0, match.start()-100):match.end()+180]}
    numeric_tables = sum(t['numeric_tokens'] >= 12 for t in tables)
    if numeric_tables:
        evidence['numerical_tables'] = {'weight':25, 'count':numeric_tables,
            'labels':[t['label'] for t in tables if t['numeric_tokens'] >= 12]}
    score = sum(e['weight'] for e in evidence.values())
    combined = title + ' ' + abstract + ' ' + text
    categories = [k for k,p in CATEGORY_PATTERNS.items() if re.search(p, combined, re.I)]
    required_physics = ['geometric_refraction_reflection']
    for tag,pattern in [('polarization',r'polari[sz]|wave[ -]?plate'),('diffraction',r'diffracti|holograph'),('gradient_index',r'gradient[ -]index|\bGRIN\b')]:
        if re.search(pattern, combined, re.I):required_physics.append(tag)
    level = ('prescription_candidate' if numeric_tables and 'coefficients' in evidence else
             'numerical_layout_candidate' if numeric_tables else 'layout_candidate')
    # Only explicit figure/embodiment associations; references remain unreviewed.
    embodiments = {}
    for p in paragraphs:
        for sentence in re.split(r'(?<=[.!?])\s+(?=[A-Z])', p['text']):
            if re.search(r'(?:example|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+(?:example\s+)?embodiment', sentence, re.I):
                figures = re.findall(r'FIG(?:S|URES?)?\.?\s*(\d+[A-Z]?)',sentence,re.I)
                for figure in figures:
                    key = 'figure-' + figure.upper()
                    embodiments.setdefault(key, {'id': actual+':'+key, 'figure':figure.upper(),
                        'source_anchor':p['anchor'],'excerpt':sentence[:700],
                        'status':'unreviewed_reference','prescription_link_verified':False})
    relations = []
    for prop,kind in [('backwardReferences','cites'),('forwardReferences','cited_by'),
                      ('backwardReferencesFamily','family_cites'),('forwardReferencesFamily','family_cited_by')]:
        for row in soup.select(f'tr[itemprop="{prop}"]'):
            e = row.select_one('[itemprop="publicationNumber"]')
            if not e:continue
            try: target = publication_id(e.get_text())
            except ValueError:continue
            label = row.select_one('[itemprop="title"]')
            relations.append({'target':target,'kind':kind,'title':clean(label.get_text()) if label else ''})
    if family:
        for a in family.select('a[href*="/patent/"]'):
            try:target=publication_id('https://patents.google.com'+a['href'])
            except ValueError:continue
            if target!=actual:relations.append({'target':target,'kind':'family_member','title':''})
    relations = list({(r['kind'],r['target']):r for r in relations}.values())
    images = list(dict.fromkeys(e.get('src','') for e in soup.select('img[itemprop="thumbnail"]') if e.get('src')))
    cpc = sorted(set(clean(e.get_text()) for e in soup.select('[itemprop="Code"]') if re.match(r'^[A-HY]\d\d[A-Z]\d',clean(e.get_text()))))
    dates={}
    for key in ('priorityDate','filingDate','publicationDate'):
        e=soup.select_one(f'time[itemprop="{key}"]') or soup.select_one(f'dd[itemprop="{key}"]')
        dates[key]=e.get('datetime',clean(e.get_text())) if e else ''
    assignees=[clean(e.get_text()) for e in soup.select('dd[itemprop="assigneeOriginal"]')]
    record = {'schema_version':SCHEMA_VERSION,'publication_id':actual,'title':title,'abstract':abstract,
        'source_url':f'https://patents.google.com/patent/{actual}/en','pdf_url':content('citation_pdf_url'),
        'family_id':family_id,'family_id_provider':'Google Patents','dates':dates,'assignees':assignees,
        'cpc':cpc,'category_mentions':categories,'physics_mentions':required_physics,
        'triage_score':score,'triage_level':level,'verification':'unreviewed',
        'evidence':evidence,'numeric_table_count':numeric_tables,'table_count':len(tables),
        'embodiment_references':list(embodiments.values()),'relations':relations,
        'thumbnail_urls':images,'missing_evidence':[k for k in SIGNALS if k not in evidence]}
    return record, clean(text), tables


def search_records(payload):
    results=payload.get('results')
    if not isinstance(results,dict):raise ValueError('Search response has no results object')
    rows=[]
    for cluster in results.get('cluster',[]):
        for r in cluster.get('result',[]):
            p=r.get('patent')
            if not p:continue # non-patent literature is not a patent ID
            try:pid=publication_id(p.get('publication_number',''))
            except ValueError:continue
            rows.append({'publication_id':pid,'title':plain(p.get('title','')),
                         'snippet':plain(p.get('snippet',''))})
    # Missing cluster with nonzero results is a schema error, not successful emptiness.
    if not rows and int(results.get('total_num_results',0) or 0)>0:
        raise ValueError('Nonempty search response contained no parseable patent results')
    return rows
