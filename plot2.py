import pickle
import numpy as np

# pickleファイルを読み込む
with (open('D:\GItHub\deep-learning-from-scratch-2\ch04\cbow_params.pkl', "rb")) as f:
    params = pickle.load(f)
    
#CBOWで学習した分散表現を取得
    
vectors = params['word_vecs']


from sklearn.decomposition import PCA

# PCAを用いて2次元に次元削減
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)


import matplotlib.pyplot as plt

# 2次元にプロット
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])
plt.show()


# 特定の単語の分散表現を取得する
word = 'apple'
word_index = cbow.wv.vocab[word].index
word_vector = vectors[word_index]

# 全ての単語の分散表現と特定の単語の分散表現のcosine similarityを計算する
cos_similarities = np.dot(vectors, word_vector) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(word_vector))

# cosine similarityが最も高い単語を取得する
most_similar_word_index = np.argsort(cos_similarities)[-1]
most_similar_word = cbow.wv.index2word[most_similar_word_index]

# 2次元空間上のどの位置にプロットされたかを調べる
word_position = vectors_2d[word_index]
most_similar_word_position = vectors_2d[most_similar_word_index]
