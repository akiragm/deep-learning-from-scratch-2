import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

with (open('C:\Git\DeepLearnng\ch04\cbow_params.pkl', "rb")) as f:
    params = pickle.load(f)
#CBOWで学習した分散表現を取得
word_vecs = params['word_vecs']
 

#プロットしたい単語を設定する
#FIreなどの災害の意味で使われそうなものを設定
words = []
words.append(["fire","r"])
words.append(["earthquake","r"])
words.append(["accident","r"])
words.append(["involving","r"])
words.append(["Hawaii","r"])
words.append(["Africa","c"])
words.append(["San","c"])
words.append(["Francisco","c"])
words.append(["maskusername","c"])

#頻出単語もセットする
print(model.wv.index2word[:50])
for s in model.wv.index2word[:50]:
    words.append([s,"b"])


length = len(words)
data = []
 
j = 0
while j < length:
    data.append(model[words[j][0]])
    j += 1
    

#主成分分析により２次元に圧縮する
pca = PCA(n_components=2)
pca.fit(data)
data_pca= pca.transform(data)
 
length_data = len(data_pca)

#プロットの設定
#fig=plt.figure(figsize=(10,6),facecolor='w')
fig=plt.figure(figsize=(20,12),facecolor='w')

plt.rcParams["font.size"] = 10
i = 0
while i < length_data:
    #点プロット
    plt.plot(data_pca[i][0], data_pca[i][1], ms=5.0, zorder=2, marker="x", color=words[i][1])
 
    #文字プロット
    plt.annotate(words[i][0], (data_pca[i][0], data_pca[i][1]), size=12)
 
    i += 1

plt.show()


# ## プロットの結果
# プロットの結果、なんとなく近い単語が集まっている。
# 例えば、「was」「or」「a」「is」などのstopwordに設定されやすい単語が集まっているのがわかるかと思います。
