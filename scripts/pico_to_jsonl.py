"""

Input:
    ###9793223:
    OBJECTIVE|A|To compare the results of recording enamel opacities using the TF and modified DDE indices .
    DESIGN|M|Enamel opacities on the maxillary central incisors were recorded two weeks apart .
    DESIGN|M|On the first occasion scoring was according to the criteria of the TF index and on the second occasion the modified DDE index was used .
    DESIGN|M|An intraoral colour slide photograph was taken of the teeth of each subject and these were scored on two occasions in random order , first using the TF index and secondly using the modified DDE index .
    SUBJECTS|P|Three hundred and twenty-five @ children who were life time residents in an area where the domestic water was fluoridated at one part per million .
    RESULTS|R|Agreement between clinical and photograph scoring was good for both TF and modified DDE indices , Spearman 's Rank Correlation Coefficients being @ and @ respectively .
    RESULTS|R|There was also good agreement between the distribution of scores for the two indices as indicated by Coefficients of @ for clinical scores and @ for photographic scores .
    CONCLUSIONS|C|There was good agreement between the TF and modified DDE indices when recording the distribution of milder types of enamel opacities in a population of @ children .

    ###
    ...

Output:
    Each line contains a single JSON:
    {
        'text': <str>,
        'label': <str>,
        'metadata': {}
    }
"""

import json

for file in ['train', 'dev', 'test']:
    with open(f'data/text_classification/pico/raw/{file}.txt', 'r') as f_in:
        with open(f'data/text_classification/pico/{file}.txt', 'w') as f_out:
            for line in f_in:
                if '###' in line or line.strip() == '':
                    continue
                _, label, sent = line.split('|', maxsplit=2)
                instance = {
                    'text': sent.strip(),
                    'label': label,
                    'metadata': {}
                }
                json.dump(instance, f_out)
                f_out.write('\n')