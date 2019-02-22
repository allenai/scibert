#!/usr/bin/python

import os
import sys
import copy

def calculateMeasures(folder_gold="data/dev/", folder_pred="data_pred/dev/", remove_anno = ""):
    '''
    Calculate P, R, F1, Macro F
    :param folder_gold: folder containing gold standard .ann files
    :param folder_pred: folder containing prediction .ann files
    :param remove_anno: if set if "rel", relations will be ignored. Use this setting to only evaluate
    keyphrase boundary recognition and keyphrase classification. If set to "types", only keyphrase boundary recognition is evaluated.
    Note that for the later, false positive
    :return:
    '''

    flist_gold = os.listdir(folder_gold)
    res_all_gold = []
    res_all_pred = []
    targets = []

    for f in flist_gold:
        # ignoring non-.ann files, should there be any
        if not str(f).endswith(".ann"):
            continue
        f_gold = open(os.path.join(folder_gold, f), "r")
        try:
            f_pred = open(os.path.join(folder_pred, f), "r")
            res_full_pred, res_pred, spans_pred, rels_pred = normaliseAnnotations(f_pred, remove_anno)
        except IOError:
            print(f + " file missing in " + folder_pred + ". Assuming no predictions are available for this file.")
            res_full_pred, res_pred, spans_pred, rels_pred = [], [], [], []

        res_full_gold, res_gold, spans_gold, rels_gold = normaliseAnnotations(f_gold, remove_anno)

        spans_all = set(spans_gold + spans_pred)

        for i, r in enumerate(spans_all):
            if r in spans_gold:
                target = res_gold[spans_gold.index(r)].split(" ")[0]
                res_all_gold.append(target)
                if not target in targets:
                    targets.append(target)
            else:
                # those are the false positives, contained in pred but not gold
                res_all_gold.append("NONE")

            if r in spans_pred:
                target_pred = res_pred[spans_pred.index(r)].split(" ")[0]
                res_all_pred.append(target_pred)
            else:
                # those are the false negatives, contained in gold but not pred
                res_all_pred.append("NONE")

    p_map, r_map, f1_map = precision_recall_fscore(res_all_gold, res_all_pred, targets)

    return p_map, r_map, f1_map


def precision_recall_fscore(y_true, y_pred, labels=None, average=None):

    p_map, r_map, f1_map = {}, {}, {}
    tp_map, fp_map, fn_map = {}, {}, {}
    for l in set(labels):
        tp_map[l], fp_map[l], fn_map[l] = 0.0, 0.0, 0.0

    for i, gold in enumerate(y_true):
        pred = y_pred[i]
        if gold == pred:
            tp_map[gold] += 1
        elif gold == "NONE" and pred != "NONE":
            fp_map[pred] += 1
        else:
            fn_map[gold] += 1

    tp_all = sum(tp_map[l] for l in set(labels))
    fp_all = sum(fp_map[l] for l in set(labels))
    fn_all = sum(fn_map[l] for l in set(labels))

    print "\nlabel\t\tprecision\trecall\tf1"
    for l in set(labels):
        r_map[l] = tp_map[l] / (tp_map[l] + fn_map[l])
        p_map[l] = tp_map[l] / (tp_map[l] + fp_map[l])
        f1_map[l] = (2 * p_map[l] * r_map[l] / (p_map[l] + r_map[l]))

        print l, p_map[l], r_map[l], f1_map[l]

    r_map["all_micro"] = tp_all / (tp_all + fn_all)
    p_map["all_micro"] = tp_all / (tp_all + fp_all)
    f1_map["all_micro"] = (2 * p_map["all_micro"] * r_map["all_micro"] / (p_map["all_micro"] + r_map["all_micro"]))

    print "all_micro", p_map["all_micro"], r_map["all_micro"], f1_map["all_micro"]

    return p_map, r_map, f1_map



def normaliseAnnotations(file_anno, remove_anno):
    '''
    Parse annotations from the annotation files: remove relations (if requested), convert rel IDs to entity spans
    :param file_anno:
    :param remove_anno:
    :return:
    '''
    res_full_anno = []
    res_anno = []
    spans_anno = []
    rels_anno = []

    for l in file_anno:
        r_g = l.strip().split("\t")
        r_g_offs = r_g[1].split(" ")

        # remove relation instances if specified
        if remove_anno != "" and r_g_offs[0].endswith("-of"):
            continue

        res_full_anno.append(l.strip())
        # normalise relation instances by looking up entity spans for relation IDs
        if r_g_offs[0].endswith("-of"):
            arg1 = r_g_offs[1].replace("Arg1:", "")
            arg2 = r_g_offs[2].replace("Arg2:", "")
            for l in res_full_anno:
                r_g_tmp = l.strip().split("\t")
                if r_g_tmp[0] == arg1:
                    ent1 = r_g_tmp[1].replace(" ", "_")
                if r_g_tmp[0] == arg2:
                    ent2 = r_g_tmp[1].replace(" ", "_")

            if r_g_offs[0] == "Synonym-of":
                ent1_spl = ent1.split("_")
                ent2_spl = ent2.split("_")
                if ent1_spl[1] > ent2_spl[1]:
                    ent1_old = copy.copy(ent1)
                    ent1 = copy.copy(ent2)
                    ent2 = ent1_old

            spans_anno.append(" ".join([ent1, ent2]))
            res_anno.append(" ".join([r_g_offs[0], ent1, ent2]))
            rels_anno.append(" ".join([r_g_offs[0], ent1, ent2]))

        else:
            spans_anno.append(" ".join([r_g_offs[1], r_g_offs[2]]))
            keytype = r_g[1]
            if remove_anno == "types":
                keytype = "KEYPHRASE-NOTYPES"
            res_anno.append(keytype)



    for r in rels_anno:
        r_offs = r.split(" ")
        # reorder hyponyms to start with smallest index
        if r_offs[0] == "Synonym-of" and r_offs[2].split("_")[1] < r_offs[1].split("_")[1]:  # 1, 2
            r = " ".join([r_offs[0], r_offs[2], r_offs[1]])

        # Check, in all other hyponym relations, if the synonymous entity with smallest index is used for them.
        # If not, change it so it is.
        if r_offs[0] == "Synonym-of":
            for r2 in rels_anno:
                r2_offs = r2.split(" ")
                if r2_offs[0] == "Hyponym-of" and r_offs[1] == r2_offs[1]:
                    r_new = " ".join([r2_offs[0], r_offs[2], r2_offs[2]])
                    rels_anno[rels_anno.index(r2)] = r_new

                if r2_offs[0] == "Hyponym-of" and r_offs[1] == r2_offs[2]:
                    r_new = " ".join([r2_offs[0], r2_offs[1], r_offs[2]])
                    rels_anno[rels_anno.index(r2)] = r_new

    rels_anno = list(set(rels_anno))

    res_full_anno_new = []
    res_anno_new = []
    spans_anno_new = []

    for r in res_full_anno:
        r_g = r.strip().split("\t")
        if r_g[0].startswith("R") or r_g[0] == "*":
            continue
        ind = res_full_anno.index(r)
        res_full_anno_new.append(r)
        res_anno_new.append(res_anno[ind])
        spans_anno_new.append(spans_anno[ind])

    for r in rels_anno:
        res_full_anno_new.append("R\t" + r)
        res_anno_new.append(r)
        spans_anno_new.append(" ".join([r.split(" ")[1], r.split(" ")[2]]))

    return res_full_anno_new, res_anno_new, spans_anno_new, rels_anno


if __name__ == '__main__':
    folder_gold = "data/dev/"
    folder_pred = "data_pred/dev/"
    remove_anno = ""  # "", "rel" or "types"
    if len(sys.argv) >= 2:
        folder_gold = sys.argv[1]
    if len(sys.argv) >= 3:
        folder_pred = sys.argv[2]
    if len(sys.argv) == 4:
        remove_anno = sys.argv[3]

    calculateMeasures(folder_gold, folder_pred, remove_anno)
