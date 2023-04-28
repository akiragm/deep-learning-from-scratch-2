import pickle
import numpy as np
from google.colab import drive
drive.mount('/content/drive')

# pickle�t�@�C����ǂݍ���
with (open('/content/deep-learning-from-scratch-2/ch04/cbow_params.pkl', "rb")) as f:
    params = pickle.load(f)

# CBOW�Ŋw�K�������U�\�����擾
vectors = params['word_vecs']

from sklearn.decomposition import PCA

# PCA��p����2�����Ɏ����팸
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

import matplotlib.pyplot as plt

# 2�����Ƀv���b�g
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])
plt.show()
