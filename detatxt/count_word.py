with open('ptb.train.txt', 'r') as f:
    text = f.read()

# 空白文字で分割し、単語数を数える
word_count = len(text.split())

print("単語数：", word_count)