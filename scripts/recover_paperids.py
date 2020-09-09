from s2base import elastic
import os
import glob
import datetime


es = elastic.default_es_client(ES_URL=elastic.paths.ES_URL_DEV) 
in_dir = 'raw_text_files/out_without_DONE'
out_dir = 'raw_text_files/paper_ids'
for in_filename in sorted(glob.glob(f'{in_dir}/*.out')):

    filename = in_filename.split('/')[-1]
    out_filename = f'{out_dir}/{filename}'
   
    if os.path.exists(out_filename):
        print(f'{str(datetime.datetime.now())} == SKIPPING: {in_filename}. Output: {out_filename}')
        continue
    else:
        print(f'{str(datetime.datetime.now())} == Processing: {in_filename}. Output: {out_filename}')
        
    with open(in_filename) as f:
        with open(out_filename, 'w') as fout:
            line_index = -1
            paper_count = 0
            found_count = 0
            found = False
            for line in f:
                line = line.strip()

                if (line == ""):
                    fout.write(f'{paper_id}\n')
                    if found:
                        found_count  += 1
                    paper_count += 1

                    if paper_count % 500 == 0:
                        print(f'{str(datetime.datetime.now())} == {found_count}/{paper_count}')

                    line_index = -1
                    found = False
                    continue

                line_index += 1

                if line_index >= 2:
                    continue

                if not found:
                    try:
                        r = es.search(index='paper', doc_type='paper', body={"query": {"match": { "paperAbstract": {"query": line, "operator" : "and"}}}}, _source_include='_id')
                        hits_count = r['hits']['total']
                        paper_id = 'no paper found'
                        if hits_count > 0:
                            paper_id = r['hits']['hits'][0]['_id']
                            found = True
                    except Exception as e:
                        print(e)

                if not found:
                    try:
                        r = es.search(index='paper', doc_type='paper', body={"query": {"match": { "bodyText": {"query": line, "operator" : "and"}}}}, _source_include='_id')
                        hits_count = r['hits']['total']
                        if hits_count > 0:
                            paper_id = r['hits']['hits'][0]['_id']
                            found = True
                    except Exception as e:
                        print(e)
