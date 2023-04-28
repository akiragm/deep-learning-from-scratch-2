# coding: utf-8
import sys
sys.path.append('..')
from common import config
# GPUで実行する場合は、下記のコメントアウトを消去（要cupy）
# ===============================================
#config.GPU = True
# ===============================================
from common.np import *
import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from cbow import CBOW
from skip_gram import SkipGram
from common.util import create_contexts_target, to_cpu, to_gpu
from dataset import ptb
# Bayesian Optimizationの実行
from bayes_opt import BayesianOptimization

# ハイパーパラメータの範囲
pbounds = {
    'window_size': (2, 5),
    'hidden_size': (50, 150),
    'learning_rate': (-6, -2),
    'batch_size': (50, 150)
}

# データの読み込み
corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
window_size = 2  # 追加

contexts, target = create_contexts_target(corpus, window_size)
if config.GPU:
    contexts, target = to_gpu(contexts), to_gpu(target)

# モデルなどの生成
def train_cbow(window_size, hidden_size, learning_rate, batch_size):
    model = CBOW(vocab_size, hidden_size, window_size, corpus)
    optimizer = Adam(lr=10**learning_rate)
    trainer = Trainer(model, optimizer)
    trainer.fit(contexts, target, max_epoch=10, batch_size=batch_size)
    return -trainer.get_loss()


bo = BayesianOptimization(
    f=train_cbow,
    pbounds=pbounds,
    random_state=42
)
bo.maximize(init_points=5, n_iter=20)

# 最適化したパラメータを取得
params = bo.max['params']
window_size = int(params['window_size'])
hidden_size = int(params['hidden_size'])
learning_rate = 10**params['learning_rate']
batch_size = int(params['batch_size'])

# 学習開始
model = CBOW(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam(lr=learning_rate)
trainer = Trainer(model, optimizer)
contexts, target = create_contexts_target(corpus, window_size)  # 変更
if config.GPU:
    contexts, target = to_gpu(contexts), to_gpu(target)
trainer.fit(contexts, target, max_epoch=10, batch_size=batch_size)
trainer.plot()

# 後ほど利用できるように、必要なデータを保存
word_vecs = model.word_vecs
if config.GPU:
    word_vecs = to_cpu(word_vecs)
params = {}
params['word_vecs'] = word_vecs.astype(np.float16)
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word

pkl_file = 'cbow_params.pkl'  # or 'skipgram_params.pkl'
with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)
