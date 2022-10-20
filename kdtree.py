from collections import deque
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import make_classification
from sklearn import datasets
import matplotlib.pyplot as plt


class KdTree:

    def __init__(self) -> None:
        # 根节点
        self.root = None
        # 遍历列表
        self.visit = deque()
        self.visit.append(self.root)

    def divide(self, node, k: int):
        # 获取节点数据
        if node == None:
            node = self.root = self.Node(data=self.data, depth=1)
        data = node.get_data()
        # 提取第k维数据
        data_k = [data[i][k] for i in range(len(data))]
        # 计算特征中位数
        m = sorted(data_k)[len(data_k) // 2]
        # 构造节点间父子关系
        child_depth = node.get_depth() + 1
        node_lchild, node_rchild = self.Node(parent=node, depth=child_depth, state=-1), self.Node(parent=node, depth=child_depth, state=1)
        node_data, lchild_data, rchild_data = None, [], []
        # 数据分割
        for i in range(len(data)):
            # 小数据放在左叶子节点
            if data_k[i] < m:
                lchild_data.append(data[i])
            # 大数据放在右叶子节点
            elif data_k[i] > m:
                rchild_data.append(data[i])
            # 中位数放在自身节点
            elif data_k[i] == m:
                if node_data == None:
                    node_data = data[i][:-1]
                    node.set_label(data[i][-1])
                else:
                    lchild_data.append(data[i])
        # 存储数据到节点
        node_lchild.set_data(lchild_data)
        node_rchild.set_data(rchild_data)
        node.set_data(node_data)
        # 将叶子节点压进队列，广度优先遍历
        if len(lchild_data) != 0:
            self.visit.append(node_lchild)
            node.set_lchild(node_lchild)
        if len(rchild_data) != 0:
            self.visit.append(node_rchild)
            node.set_rchild(node_rchild)

    def train(self, data, label):
        # num为样本数，dim为样本特征维度
        self.num, self.dim = len(data), len(data[0])
        # 样本数据及标签
        self.data = [data[i] + [label[i]] for i in range(self.num)]
        depth = 0
        # 轮流在各个维度进行划分
        while True:
            if len(self.visit) == 0:
                break
            self.divide(self.visit.popleft(), depth % self.dim)
            depth += 1

    # def mahalanobis_distance(self, a:list, b:list):
    #     m = [(a[i]+b[i])/2 for i in range(a,b)]
    #     c=[[0 for i in range(len(a))] for j in range(len(a))]
    #     for i in range(len(a)):
    #         for j in range(len(b)):
    #             c[i][j]=(a[i]-m[i])*(a[j]-m[j])+(b[i]-m[i])*(b[j]-m[j])
    #     return c

    def euclidean_distance(self, a: list, b: list):
        d = 0
        for i in range(len(a)):
            d += (b[i] - a[i])**2
        return math.sqrt(d)

    def classify(self, data: list, k: int = 5, distance_method='e'):
        node = self.root
        if distance_method == 'e':
            distance = self.euclidean_distance
        # 寻找起始最近点
        depth, dim = 0, len(data)
        while True:
            if node.get_lchild() == None and node.get_rchild() == None:
                break
            index = depth % dim
            if data[index] > node.get_data()[index] and node.get_rchild() != None:
                node = node.get_rchild()
            else:
                node = node.get_lchild()
        # knn点集
        knn_list = []
        while True:
            if node.get_parent() == None:
                break
            lchild, rchild = node.get_parent().get_lchild(), node.get_parent().get_rchild()
            # 计算距离并加入knn点集
            dl = dr = 0
            if lchild != None:
                dl = distance(data, lchild.get_data())
                knn_list.append((dl, lchild))
            if rchild != None:
                dr = distance(data, rchild.get_data())
                knn_list.append((dr, rchild))
            knn_list.sort(key=lambda x: x[0])
            knn_list = knn_list[:k]
            node = node.get_parent()
        # 多数表决
        c_dict, max_num, predict = {}, 0, 0
        for item in knn_list:
            label = item[1].get_label()
            if label not in c_dict:
                c_dict[label] = 1
            else:
                c_dict[label] += 1
        for label in c_dict:
            if c_dict[label] > max_num:
                max_num = c_dict[label]
                predict = label
        return predict

    def test(self, data: list, label: list, k: int = 3, distance_method='e'):
        predict = []
        for item in data:
            predict.append(self.classify(item, k=k, distance_method=distance_method))
        cm = confusion_matrix(label, predict)
        print(classification_report(label, predict))
        print(cm)

    def print(self):
        visit = deque()
        visit.append(self.root)
        while True:
            if len(visit) == 0:
                return
            node = visit.popleft()
            print('{}, label: {}, depth: {}'.format(node.get_data(), node.get_label(), node.get_depth()))
            if node.get_lchild() != None:
                visit.append(node.get_lchild())
            if node.get_rchild() != None:
                visit.append(node.get_rchild())

    class Node:
        def __init__(self, data: list = None, parent: list = None, lchild: list = None, rchild: list = None, depth: int = 0, label: int = 0, state: int = 0) -> None:
            self.data = data
            self.state = state
            self.label = label
            self.parent = parent
            self.lchild = lchild
            self.rchild = rchild
            self.depth = depth

        def get_data(self):
            return self.data

        def get_lchild(self):
            return self.lchild

        def get_rchild(self):
            return self.rchild

        def get_parent(self):
            return self.parent

        def get_depth(self):
            return self.depth

        def get_state(self):
            return self.state

        def get_label(self):
            return self.label

        def set_data(self, data):
            self.data = data

        def set_lchild(self, lchild):
            self.lchild = lchild

        def set_rchild(self, rchild):
            self.rchild = rchild

        def set_parent(self, parent):
            self.parent = parent

        def set_depth(self, depth):
            self.depth = depth

        def set_state(self, state):
            self.state = state

        def set_label(self, label):
            self.label = label


def sklearn_dataset_test():
    X, Y = make_classification(n_samples=10000, n_features=2, n_clusters_per_class=1, n_classes=2, n_informative=1, n_redundant=0)
    # data = datasets.load_iris()
    # X, Y = data['data'], data['target']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 4)
    X_train, X_test, Y_train, Y_test = X_train.tolist(), X_test.tolist(), Y_train.tolist(), Y_test.tolist()
    kdtree = KdTree()
    print('Training...')
    kdtree.train(X_train, Y_train)
    print('Testing...')
    kdtree.test(X_test, Y_test, k=3)
    if len(X_train[0]) == 2:
        scatter_dict, colors = {}, ['red', 'blue', 'green', 'grey', 'brown']
        for i in range(len(X_train)):
            if Y_train[i] not in scatter_dict:
                scatter_dict[Y_train[i]] = [[], []]
            scatter_dict[Y_train[i]][0].append(X_train[i][0])
            scatter_dict[Y_train[i]][1].append(X_train[i][1])
        for label in range(len(scatter_dict)):
            plt.scatter(*(scatter_dict[label]), c=colors[list(scatter_dict.keys()).index(label)], s=0.5)
        plt.show()


if __name__ == '__main__':
    # data = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    # label = [1, 2, 3, 4, 5, 6]
    # kdtree = KdTree()
    # kdtree.train(data, label)
    # kdtree.print()
    sklearn_dataset_test()
