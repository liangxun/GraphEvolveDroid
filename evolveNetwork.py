import os
from collections import defaultdict
import json
import pickle as pkl
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import euclidean_distances
from rpforest import RPForest


class KnnGraph:
    def __init__(self, feat_mat, n_neighs=10, n_embed=64, leaf_size=10, no_trees=20):
        self.n_neighs = n_neighs
        self.feat = feat_mat

        self.pca = self._get_pca_tranformer(n_embed)
        print("pca done")
        self.rpf = self._get_random_projection_forest(leaf_size=leaf_size, no_trees=no_trees)
        print("construct rpf, done")

        self.adj_list = self.get_knn_graph()

    def _get_pca_tranformer(self, n):
        pca = TruncatedSVD(n_components=n)
        pca.fit(self.feat)
        return pca

    def _get_random_projection_forest(self, leaf_size=20, no_trees=10):
        self.embed_feat = self.pca.transform(self.feat)
        rpf = RPForest(leaf_size=leaf_size, no_trees=no_trees)
        rpf.fit(self.embed_feat)
        return rpf
    
    def get_knn_graph(self):
        """
        有向图，边的权重为节点的相似度（欧式距离越小越相似）
        """
        print("func: get_knn_graph")
        # adj_list_candidate = defaultdict(list)
        adj_list_candidate = dict()
        no_candidate = self.n_neighs * 2
        for i in range(self.feat.shape[0]):
            # tmp = self.rpf.query(self.embed_feat[i], no_candidate).tolist()
            # tmp = list(filter(lambda x: x!=i, tmp))
            # adj_list_candidate[i] = tmp
            adj_list_candidate[i] = list(filter(lambda x: x!=i, self.rpf.query(self.embed_feat[i], no_candidate).tolist()))
        return adj_list_candidate
        # # 从candidate中选出neighs
        # adj_list = dict()
        # for i in range(self.feat.shape[0]):
        #     candidates = adj_list_candidate[i]
        #     values = euclidean_distances(self.feat[candidates], self.feat[i]).squeeze()
        #     items = list(zip(candidates, values))
        #     items.sort(key=lambda item: item[1])
        #     neighs, values = zip(*items[:self.n_neighs])
        #     values = np.array(values)
        #     # 将欧式距离转换为相似度，并归一化
        #     weights = np.exp(-values)/np.sum(np.exp(-values)) # (softmax归一化)边的权值, 将欧式距离转换为相似度
        #     # adj_list[i] = dict(zip(neighs, values))
        #     adj_list[i] = neighs
        # return adj_list
    
    def get_symmetic_knn_graph(self):
        """
        无向图，边的权值修改过
        """
        print("func: get_symmetic_knn_graph")
        if not self.adj_list:
            self.adj_list = self.get_knn_graph()

        symmetic_adj_list = defaultdict(dict)
        for i in range(self.feat.shape[0]):
            for j in self.adj_list[i].keys():
                w_ij = self.adj_list[i][j]
                w_ji = self.adj_list[j].get(i, 0)
                symmetic_adj_list[i][j] = (w_ij + w_ji) / 2
        return symmetic_adj_list
    
    def get_k_neighs(self, x, n_neighs=20):
        embed_x = pca.transform(x)
        candidates = self.rpf.query(embed_x, n_neighs*2).tolist()
        values = euclidean_distances(self.feat[candidates], x).squeeze()
        items = list(zip(candidates, values))
        items.sort(key=lambda item: item[1])
        neighs, values = zip(*items[:n_neighs])
        weights = np.exp(-values)/np.sum(np.exp(-values))
        
        out_edges = dict(zip(neighs, values))
        normal_out_edges = dict(zip(neighs, weights))

        neighs = dict()
        for n in out_edges.keys():
            w_ij = normal_out_edges[n]

            hop_2 = self.adj_list[n]
            values = list(hop_2.values())
            values.append(out_edges[n])
            w_ji = np.exp(-values)/np.sum(np.exp(-values))[-1]
            neighs[n] = (w_ij + w_ji) / 2

        return neighs


if __name__ == '__main__':
    data_dir = "/data/android/exp_data/tesseract/for_graphevolvedroid"
    feat_mat = sp.load_npz(os.path.join(data_dir, "mamadroid_feat.npz"))
    graph = KnnGraph(feat_mat, n_neighs=20)
    candicate_adj = graph.get_knn_graph()
    with open(os.path.join(data_dir, "adj_knn_candidate.pkl"), 'wb') as f:
        pkl.dump(candicate_adj, f)
    # graph_adj = graph.symmetic_adj_list
    # with open(os.path.join(data_dir, 'adj_list_app_drebincluster_app.pkl'), 'wb') as f:
    #     pkl.dump(graph.adj_list, f)
    # with open(os.path.join(data_dir, 'symmetic_adj_list_mamadroid.json'), 'w') as f:
    #     json.dump(graph_adj, f, indent=4)