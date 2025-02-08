import json
with open('NYT10/dev.json', 'r', encoding='utf-8') as f:
    with open('NYT10/new_valid.json', 'w', encoding='utf-8') as f1:
        lines = json.loads(f.read())
        for line in lines:
            res = {}
            res['sentText'] = ' '.join(line['tokens'])
            rel = []
            for item in line['spo_list']:
                rel.append({
                    'em1Text': item[0],
                    'em2Text': item[2],
                    'label': item[1]
                })
                #print(rel['em1Text'])
                assert item[0] in res['sentText'] and item[2] in res['sentText']
            res['relationMentions'] = rel
            f1.write(json.dumps(res)+'\n')