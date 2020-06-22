# semeval2017-ScienceIE

Scripts for SemEval 2017 ScienceIE task (Task 10).
Please contact the task organisers on the ScienceIE mailing list (scienceie@googlegroups.com) if there are any problems with using the scripts.

Scripts contained are eval.py, for evaluating performance on the task, and util.py, for reading parsing ScienceDirect .xml files

Update (20 October 2016): the eval.py script now performs micro averaging, not macro averaging as before, and does not print metrics for "none" anymore. Thanks to Matthew Peters (AI2) for spotting this and improving the script!

Update (12 December 2016): there is now an additional eval_py27.py which is the same as eval.py, but for Python 2.7 instead of Python 3.

Update (18 January 2017): synonym-of relations are now evaluated as undirected relations, i.e. the order of the arguments is not taken into account. Thanks to Makoto Miwa (Toyota Technological Institute) for pointing this out!

Update (21 February 2017): the eval.py script now has the additional option "rels" for "remove_anno", which leads to only displaying performance for relation extraction.


##Requirements:
* Python 3 or Python 2.7
* sklearn
* xml.sax

## Script usage:
* eval.py: evaluation script. Usage: ```python eval.py <gold folder> <pred folder> <remove anno>```
    * gold folder (optional): (default: "data/dev/") folder containing the gold standard files distributed by the SemEval 2017 Task 10 organisers, in .ann format.
    * pred folder (optional): (default: "data_pred/dev/") folder containing the prediction files, which should be in the same format as the gold files. Note that the evaluation script ignores IDs and surface forms and only judges based on the provided character offsets.
    * remove anno (optional): "rel", "types", "keys" or "" (default). This is for removing relation annotations if you want to test performance for keyphrase boundary identification and keyphrase classification only ("rel"), for removing relation and keyphrase type annotations if you want to test performance for keyphrase boundary identification only ("types"), or for removing keyphrase annotations if you want to test performance for relation extraction only ("rels").
* util.py: script containing utilities for parsing the original ScienceDirect .xml files to obtain text only and for parsing .ann files and looking up spans in corresponding .txt files
    
## References:
* SemEval task: https://scienceie.github.io/
* .ann format: http://brat.nlplab.org/standoff.html
* sklearn: http://scikit-learn.org/
* ScienceDirect: http://www.sciencedirect.com/