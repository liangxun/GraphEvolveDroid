"""构造演化网络

每个app有k个祖先节点，构造出包含时序信息的演化网络出来
"""
import os
import pickle as pkl
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD


# with open("./mystat/map_cur_mostoldancestor.pkl", 'rb') as f:
#     map_id_oldancestor = pkl.load(f)


def get_pca_tranformer(feat, n=64):
    pca = TruncatedSVD(n_components=n)
    pca.fit(feat)
    return pca


class EvolutionGraph:
    def __init__(self, feat, k, batchsize):
        self.k = k # 邻居个数
        
        # pca降维
        self.pca = get_pca_tranformer(feat)
        self.feat = self.pca.transform(feat)

        self.get_adj(batchsize)
    
    def _get_pca_tranformer(self, n=64):
        pca = TruncatedSVD(n_components=n)
        pca.fit(feat)
        return pca
    
    def get_adj(self, batch_size=5000):
        feat_t = self.feat.T
        slides = []
        # step1：第一批
        mat = self.feat[:batch_size]
        mat = mat.dot(feat_t)
        mat = self._get_neighs_init(mat, batch_size)
        slides.append(mat)
        # step2: 逐批添加
        k = batch_size
        while (k+batch_size) < self.feat.shape[0]:
            mat = self.feat[k: k+batch_size]
            mat = mat.dot(feat_t)
            mat = self._get_neighs(mat, k)
            slides.append(mat)
            k += batch_size
            print("{} app handled".format(k))
        mat = self.feat[k:]
        mat = mat.dot(feat_t)
        mat = self._get_neighs(mat, k)
        slides.append(mat)

        self.adj = sp.vstack(slides)

    def _get_neighs(self, mat, start_id):
        print(mat.shape)
        data = []
        row = []
        col = []
        for iloc, d in enumerate(mat):
            # print(iloc)
            d = d.squeeze()
            d[iloc+start_id:] = 0 # 在当前app后面才出现的app不可能是祖先节点
            # d[:map_id_oldancestor[iloc+start_id]] = 0 # 在当前app之前超过六个月的app不算作祖先节点
            inds = np.argpartition(d, -self.k)[-self.k:]  # 只需要最相似的topk作为祖先节点

            for ind in inds:
                if d[ind]: # 过滤掉那些是0的
                    data.append(d[ind])
                    row.append(iloc)
                    col.append(ind)
        coo_mat = sp.coo_matrix((data, (row, col)), shape=mat.shape)
        return coo_mat.tocsr()

    def _get_neighs_init(self, mat, threshold):
        """给最开始的那批app确定“祖先节点”。这些app是最先出现的，其实没有祖先，所以在第一批里面找topk个节点作为邻居
        """
        print(mat.shape)
        data = []
        row = []
        col = []
        for iloc, d in enumerate(mat):
            d = d.squeeze()
            d[threshold:] = 0 # 只能在给定时间之前中找最相似的作为邻居
            inds = np.argpartition(d, -self.k)[-self.k:]  # 只需要最相似的topk作为祖先节点

            for ind in inds:
                if d[ind]:
                    data.append(d[ind])
                    row.append(iloc)
                    col.append(ind)
        coo_mat = sp.coo_matrix((data, (row, col)), shape=mat.shape)
        return coo_mat.tocsr()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="construct knn.")
    parser.add_argument('--keyword', type=str, default='drebin')
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()

    data_dir = "/data/android/exp_data/apigraph-2012_2013/knngraph"
    feat = sp.load_npz(os.path.join(data_dir, "{}_feat.npz".format(args.keyword)))
    graph = EvolutionGraph(feat, args.k, batchsize=10000)
    print(args.keyword, args.k, graph.adj)
    sp.save_npz(os.path.join(data_dir, "{}_knn_{}.npz".format(args.keyword, args.k)), graph.adj)
