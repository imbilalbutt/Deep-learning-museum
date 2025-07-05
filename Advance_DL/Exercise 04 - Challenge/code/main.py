import argparse
import os
import numpy as np
from tqdm import tqdm
import pickle
import gzip
import bz2

import torch
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import v2

from sklearn.preprocessing import normalize, LabelEncoder

from hisfrag20dataset import Hisfrag20Dataset
from eval_map import get_top1_mAP

def parseArgs(parser):
    parser.add_argument('--tmp', default='tmp', 
                        help='default temporary folder')
    parser.add_argument('--labels_test',
                        help='contains test images + labels')
    parser.add_argument('--dataset', default='hisfrag20',
                        choices=['hisfrag20'])
    parser.add_argument('--in_test',
                        help='the input folder of the test images')
    parser.add_argument('--model', default='resnet18',
                        help='model name')
    parser.add_argument('--weights', default='ResNet18_Weights.IMAGENET1K_V1', type=str, 
                        help='the weights enum name to load')
    parser.add_argument('--write_dm', action='store_true',
                        help='write distance matrix')
    parser.add_argument('--dm_format', default='gz',
                        choices=['bz2', 'csv', 'gz'],
                        help='format for the distance matrix, gz is fastest')

    return parser

def distances(encs):
    """
    compute pairwise distances
    parameters:
        encs:  TxD encoding matrix
        returns: TxT distance matrix
    """

    # compute cosine distance = 1 - dot product between l2-normalized
    # encodings
    dists = 1.0 - encs.dot(encs.T)   
    # sanity check, this should actually already be 0 
    np.fill_diagonal(dists,0)
    return dists

if __name__ == '__main__':
    parser = argparse.ArgumentParser('retrieval')
    parser = parseArgs(parser)
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.tmp):
        os.mkdir(args.tmp)

    # imageNet mean + std
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)

    data_transforms = {
        'val': v2.Compose([
            v2.Resize((216,216),antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean,std=std)
        ])
    }

    if args.dataset != 'hisfrag20':
        raise ValueError('unknown dataset')

    # we are not using the labels for testing
    # --> no target_transform needed 
    test_data = Hisfrag20Dataset(args.labels_test,
                                 args.in_test,
                                 transform=data_transforms['val'])

    test_dataloader = DataLoader(test_data, batch_size=128, 
                                 num_workers=8,
                                 shuffle=False,
                                 pin_memory=True)

    model = models.get_model(args.model, weights=args.weights \
                             if args.weights != 'None' else None)

    # Detect if we have a GPU available                                                                                                 
    device = torch.device("cuda:0" if torch.backends.cuda.is_built() else "cpu") 
    model.to(device)    

    # let's try to use the penultimate layer, typically this is the avgpool
    # layer 
    model_fe = create_feature_extractor(model, {'avgpool':'feat'})

    # inference
    all_outputs = []
    for e, (inputs, labels) in enumerate(tqdm(test_dataloader)):
        inputs = inputs.to(device)
        outputs = model_fe(inputs)['feat'].flatten(1).cpu().data
        all_outputs.append(outputs)

    all_outputs = np.concatenate(all_outputs, axis=0)

    # l2 normalize outputs (needed for cosine distance
    all_outputs = normalize(all_outputs, norm='l2')
    dists = distances(all_outputs)

    # evaluate, here we use the labels
    # comment this out if you just want to write the distance map
    labels = test_data.img_labels.to_numpy()[:,1]
    le = LabelEncoder()                                                                                                             
    labels = le.fit(labels).transform(labels)
    top1, mAP = get_top1_mAP(dists, labels)
    print('top1: {}, mAP: {}'.format(top1,mAP))

    if args.write_dm:
        print('write distance map')
        # not recommended but potentially useful as sanity check
        if args.dm_format == 'csv':
            with open(os.path.join(args.tmp, 'dm.csv'), 'w') as f:
                for i in tqdm(range(len(test_data))):
                    f.write('{},{}\n'.format(test_data.img_labels.iloc[i,0],
                                         ','.join(map('{:.5f}'.format,dists[i]))))            
        # better compression but quite slow
        elif args.dm_format == 'bz2':
            obj = [test_data.img_labels.iloc[:,0].to_numpy(), dists]
            print(obj[0])
            with bz2.BZ2File(os.path.join(args.tmp, 'dm.pkl.bz2'), 'wb') as f:
                pickle.dump(obj, f)
        # preferred fast solution
        elif args.dm_format == 'gz':
            obj = [test_data.img_labels.iloc[:,0].to_numpy(), dists]
            with gzip.open(os.path.join(args.tmp, 'dm.pkl.gz'), 'wb') as f:
                pickle.dump(obj, f)
        else:
            print('wrong fileformat given')


