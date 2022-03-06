# From https://gitlab.com/ignitionrobotics/web/app/-/issues/4
import json

import requests

collection_name = 'Google%20Scanned%20Objects'
owner_name = 'GoogleResearch'

# The server URL
base_url ='https://fuel.ignitionrobotics.org'

# Path to get the models in the collection
next_url = '/1.0/models?per_page=100&q=collections:{}'.format(collection_name)

# Path to download a single model in the collection
download_url = 'https://fuel.ignitionrobotics.org/1.0/{}/models/'.format(owner_name)

count = 0
total_count = 0

# Iterate over the pages
while next_url:
    # Get the contents of the current page.
    r = requests.get(base_url + next_url)
    print(r.headers)
    # Convert to JSON
    models = json.loads(r.text)

    # Get the next page's URL
    next_url = ''
    if 'Link' in r.headers:
        links = r.headers['Link'].split(',')
        for link in links:
            parts = link.split(';')
            if 'next' in parts[1]:
                next_url = parts[0].replace('<','').replace('>','')

    # Get the total number of models to download
    if total_count <= 0 and 'X-Total-Count' in r.headers:
        total_count = int(r.headers['X-Total-Count'])

    # Download each model
    for model in models:
        count+=1
        model_name = model['name']
        print ('Downloading (%d/%d) %s' % (count, total_count, model_name))
        download = requests.get(download_url+model_name+'.zip', stream=True)
        with open(model_name+'.zip', 'wb') as fd:
            for chunk in download.iter_content(chunk_size=1024*1024):
                fd.write(chunk)

print('Done.')
