"""
Most code: credits to Anguelos Nicolaou, https://github.com/anguelos/wi19_evaluate
"""
import numpy as np
import os
import argparse
import pickle
import lzma
import gzip
import bz2

def _get_sorted_retrievals(D, classes, remove_self_column=True, apply_e=False):
    correct_retrievals = classes[None, :] == classes[:, None]
    if apply_e:
        D = D + _get_d_plus_e(D) * correct_retrievals
    sorted_indexes = np.argsort(D, axis=1)
    if remove_self_column:
        sorted_indexes = sorted_indexes[:, 1:]  # removing self
    sorted_retrievals = correct_retrievals[np.arange(
        sorted_indexes.shape[0], dtype="int64")[:, None], sorted_indexes]
    return sorted_retrievals


def _get_precision_recall_matrices(D, classes, remove_self_column=True):
    sorted_retrievals = _get_sorted_retrievals(
        D, classes, remove_self_column=remove_self_column)
    relevant_count = sorted_retrievals.sum(axis=1).reshape(-1, 1)
    precision_at = np.cumsum(sorted_retrievals, axis=1).astype(
        "float") / np.cumsum(np.ones_like(sorted_retrievals), axis=1)
    recall_at = np.cumsum(sorted_retrievals, axis=1).astype(
        "float") / np.maximum(relevant_count, 1)
    recall_at[relevant_count.reshape(-1) == 0, :] = 1
    return precision_at, recall_at, sorted_retrievals


def _compute_map(precision_at, sorted_retrievals):
    # Removing singleton queries from mAP computation
    valid_entries = sorted_retrievals.sum(axis=1) > 0
    precision_at = precision_at[valid_entries, :]
    sorted_retrievals = sorted_retrievals[valid_entries, :]
    AP = (precision_at * sorted_retrievals).sum(axis=1) / \
        sorted_retrievals.sum(axis=1)
    return AP.mean()

def get_top1_mAP(D, query_classes, remove_self_column=True):
    """
    D: distance matrix, typically square, distance with itself should be 0
    query_classes: list of labels, each entry denotes one query row in D
    remove_self_column: removes the first column of the sorted distance matrix.
        Thus, distances between same entries should be 0
    """
    precision_at, recall_at, sorted_retrievals = _get_precision_recall_matrices(
        D, query_classes, remove_self_column=remove_self_column)
    non_singleton_idx = sorted_retrievals.sum(axis=1)>0
    del D
    accuracy = precision_at[non_singleton_idx,0].mean()
    mAP = _compute_map(precision_at[non_singleton_idx,:], sorted_retrievals[non_singleton_idx,:])
    return accuracy, mAP

# TODO: This code could be cleaned-up quite a lot using pandas...
def evaluate(dm_fname, gt_fname, outdir=None):
    """
    dm_fname: submission csv file; first column = query filename, must be present in
        gt_fname, rest contains the square distance matrix (distance w. itself
        must be 0, should not contain negative values)
    gt_fname: ground trut csv: first column = query filename, 2nd = label
    outdir: output dir where 'scores.txt' will end up containing the mAP
    """

    assert(os.path.exists(dm_fname))
    assert(os.path.exists(gt_fname))
    fname2sample=lambda x: os.path.basename(x.strip()).split(".")[0]

    id_class_tuples=[tuple(reversed(l.split(","))) for l in open(gt_fname).read().strip().split("\n")]
    id2class_dict = {fname2sample(k):int(v) for v,k in id_class_tuples}

    dm=[]
    fnames=[]

    if dm_fname.lower().endswith(".csv"):
        for e,line in enumerate(open(dm_fname).readlines()):
            if(len(line.strip())>0):
                line=line.split(",")
                fnames.append(fname2sample(line[0]))
                dm.append([float(col) for col in line[1:]])
    elif dm_fname.lower().endswith(".tsv"):
        for line in open(dm_fname).readlines():
            if(len(line.strip())>0):
                line=line.split("\t")
                fnames.append(fname2sample(line[0]))
                dm.append([float(col) for col in line[1:]])
    elif dm_fname.lower().endswith(".json"):
        mat=json.load(open(dm_fname))
        dm = [[float(col) for col in row[1:]] for row in mat]
        fnames=[fname2sample(row[0]) for row in mat]
    elif dm_fname.lower().endswith(".pkl.bz2"):
        with bz2.open(dm_fname, 'rb') as f:
            obj = pickle.load(f)
            fnames = [fname2sample(row) for row in (obj[0])]
            dm = obj[1]
    elif dm_fname.lower().endswith(".pkl.gz"):
        with gzip.open(dm_fname, 'rb') as f:
            obj = pickle.load(f)
            fnames = [fname2sample(row) for row in (obj[0])]
            dm = obj[1]

    dm=np.array(dm,"float")
    assert(dm.shape[0]==dm.shape[1])
    assert(dm.min() >= 0.0)
    
    fnames=np.array(fnames)

    if len(id2class_dict) != fnames.shape[0]:
        print('len(id2class_dict) {} != fnames.shape[0]'
             ' {}'.format(len(id2class_dict),fnames.shape[0]))

        keep_idx = []
        for n in range(fnames.shape[0]):
            keep_idx.append(fnames[n] in id2class_dict.keys())
        keep_idx = np.array(keep_idx)>0
        fnames = fnames[keep_idx]
        dm = dm[:,keep_idx][keep_idx,:]

    classes = np.array([id2class_dict[n] for n in fnames.tolist()])
   
    top1, mAP = get_top1_mAP(dm, classes, remove_self_column=True)
    print('top1: {}, mAP: {}'.format(top1,mAP))

    if outdir:
        output_filename = os.path.join(outdir, 'scores.txt')
        with open(output_filename, 'wt') as out:
            out.write('mAP: %f\n' % mAP)

if __name__ == '__main__':  
    parser = argparse.ArgumentParser('eval distance map')
    parser.add_argument('dm_fname', 
                        help='distance map csv file')
    parser.add_argument('gt_fname',
                        help='ground truth csv file')
    parser.add_argument('--outdir',
                        help='if given, a scores.txt file is written')
    args = parser.parse_args()

    evaluate(args.dm_fname, args.gt_fname, args.outdir)

