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

# PCAを用いて3次元に次元削減
pca = PCA(n_components=3)
vectors_3d = pca.fit_transform(vectors)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3次元にプロット
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(vectors_3d[:, 0], vectors_3d[:, 1], vectors_3d[:, 2])

# 視点を変更
ax.view_init(elev=50, azim=50)

plt.savefig('/content/drive/MyDrive/cbow_plot_3_1d.png')
plt.show()
