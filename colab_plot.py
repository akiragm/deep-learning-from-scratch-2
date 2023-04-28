import pickle
import numpy as np
from google.colab import drive
drive.mount('/content/drive')

# pickleファイルを読み込む
with (open('/content/deep-learning-from-scratch-2/ch04/cbow_params.pkl', "rb")) as f:
    params = pickle.load(f)

# CBOWで学習した分散表現を取得
vectors = params['word_vecs']

from sklearn.decomposition import PCA

# PCAを用いて2次元に次元削減
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

import matplotlib.pyplot as plt

# 2次元にプロット
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])
plt.show()
