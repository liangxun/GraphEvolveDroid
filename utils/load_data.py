"""
load dataset
"""
import os
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import json
from collections import defaultdict


# all_views = ['app_permission_app', 'app_url_app', 'app_component_app', 'app_tpl_app']
# all_views = ['app_author_app', 'app_component_app', 'app_mamadroid_app']

def load_data(data_dir, keyword="drebin", view=None):

    adj_lists = []
    if view is not None:
        with open(os.path.join(data_dir, 'adj_list_{}.pkl'.format(view)), 'rb') as f:
            adj_lists.append(pkl.load(f))
    else:
        all_views = ['app_author_app', 'app_component_app', 'app_{}_app'.format(keyword)]
        for view in all_views:
            with open(os.path.join(data_dir, 'adj_list_{}.pkl'.format(view)), 'rb') as f:
                adj_lists.append(pkl.load(f))

    with open(os.path.join(data_dir, 'label_info','label_info.json'), 'r') as f:
        label_info = json.load(f)
    labels = []
    for _, _, label in label_info:
        labels.append(label)
    labels = np.array(labels)

    feats = sp.load_npz(os.path.join(data_dir, '{}_feat.npz'.format(keyword)))

    return adj_lists, feats, labels
