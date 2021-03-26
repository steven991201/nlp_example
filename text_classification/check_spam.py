from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data = pd.read_csv('spam.csv', encoding='latin1')

del data['Unnamed: 2']
del data['Unnamed: 3']
del data['Unnamed: 4']
data['v1'] = data['v1'].replace(['ham', 'spam'], [0, 1])
data.drop_duplicates(subset=['v2'], inplace=True)
print('총 샘플의 수 :', len(data))


X_data = data['v2']
y_data = data['v1']

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_data)  # 5169개의 행을 가진 X의 각 행에 토큰화를 수행
sequences = tokenizer.texts_to_sequences(X_data)  # 단어를 숫자값, 인덱스로 변환하여 저장
print(sequences[:5])

vocab_size = len(tokenizer.word_index)+1
n_of_train = int(len(data) * 0.8)
n_of_test = int(len(data) - n_of_train)

X_data = sequences
max_len = 189
data = pad_sequences(X_data, maxlen=max_len)

X_test = data[n_of_train:]  # X_data 데이터 중에서 뒤의 1034개의 데이터만 저장
y_test = np.array(y_data[n_of_train:])  # y_data 데이터 중에서 뒤의 1034개의 데이터만 저장
X_train = data[:n_of_train]  # X_data 데이터 중에서 앞의 4135개의 데이터만 저장
y_train = np.array(y_data[:n_of_train])  # y_data 데이터 중에서 앞의 4135개의 데이터만 저장


model = Sequential()
model.add(Embedding(vocab_size, 32))  # 임베딩 벡터의 차원은 32
model.add(SimpleRNN(32))  # RNN 셀의 hidden_size는 32
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=4,
                    batch_size=64, validation_split=0.2)
