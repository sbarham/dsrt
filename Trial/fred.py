'''FRED with categorical_crossentropy, last layer one hot encoded, without masking working, 
reads input as format (input \t target \n)'''
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Input, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from collections import Counter
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

np.random.seed(42)

BATCH_SIZE = 32
NUM_EPOCHS = 1
HIDDEN_UNITS = 64
DATA_PATH = 'dialogues_1000.txt'
TESTING_DATASET = 'dlgs_trial5.txt'
SIZE_OF_DATASET = sum(1 for line in open(DATA_PATH))
print('Number of adjacency pairs used for training:',SIZE_OF_DATASET)
MAX_INPUT_SEQ_LENGTH = 60
MAX_TARGET_SEQ_LENGTH = 60
MAX_VOCAB_SIZE = 10000

WEIGHT_FILE_DIR = os.path.splitext(DATA_PATH)[0]
WEIGHT_FILE_PATH = os.path.join(WEIGHT_FILE_DIR, 'encdec_final.h5')
if not os.path.exists(WEIGHT_FILE_DIR):
    os.makedirs(WEIGHT_FILE_DIR)

input_counter = Counter()
target_counter = Counter()

input_texts = []
target_texts = []

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(SIZE_OF_DATASET,len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    
    next_words = [w.lower() for w in nltk.word_tokenize(input_text)]
    if len(next_words) > MAX_TARGET_SEQ_LENGTH:
        next_words = next_words[0:MAX_TARGET_SEQ_LENGTH]

    if len(next_words) > 0:
        input_texts.append(next_words)
        for w in next_words:
            input_counter[w] += 1

        
    next_words1 = [w.lower() for w in nltk.word_tokenize(target_text)]
    if len(next_words1) > MAX_TARGET_SEQ_LENGTH:
        next_words1 = next_words1[0:MAX_TARGET_SEQ_LENGTH]
    target_words = next_words1[:]
    target_words.insert(0, 'START')
    target_words.append('END')
    for w in target_words:
        target_counter[w] += 1
    target_texts.append(target_words)
            

input_word2idx = dict()
target_word2idx = dict()
for idx, word in enumerate(input_counter.most_common(MAX_VOCAB_SIZE)):
    input_word2idx[word[0]] = idx + 2
for idx, word in enumerate(target_counter.most_common(MAX_VOCAB_SIZE)):
    target_word2idx[word[0]] = idx + 1

input_word2idx['PAD'] = 0
input_word2idx['UNK'] = 1
target_word2idx['UNK'] = 0

input_idx2word = dict([(idx, word) for word, idx in input_word2idx.items()])
target_idx2word = dict([(idx, word) for word, idx in target_word2idx.items()])
# print('INPUT:',input_texts)
# print('TARGET:',target_texts)

num_encoder_tokens = len(input_idx2word)
num_decoder_tokens = len(target_idx2word)

np.save('input-word2idx.npy', input_word2idx)
np.save('input-idx2word.npy', input_idx2word)
np.save('target-word2idx.npy', target_word2idx)
np.save('target-idx2word.npy', target_idx2word)

encoder_input_data = []

encoder_max_seq_length = 0
decoder_max_seq_length = 0

for input_words, target_words in zip(input_texts, target_texts):
    encoder_input_wids = []
    for w in input_words:
        w2idx = 1  # default [UNK]
        if w in input_word2idx:
            w2idx = input_word2idx[w]
        encoder_input_wids.append(w2idx)

    encoder_input_data.append(encoder_input_wids)
    encoder_max_seq_length = max(len(encoder_input_wids), encoder_max_seq_length)
    decoder_max_seq_length = max(len(target_words), decoder_max_seq_length)

context = dict()
context['num_encoder_tokens'] = num_encoder_tokens
context['num_decoder_tokens'] = num_decoder_tokens
context['encoder_max_seq_length'] = encoder_max_seq_length
context['decoder_max_seq_length'] = decoder_max_seq_length

# print(context)
np.save('word-context.npy', context)


def generate_batch(input_data, output_text_data):
    num_batches = len(input_data) // BATCH_SIZE
#     print(num_batches)
    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * BATCH_SIZE
            end = (batchIdx + 1) * BATCH_SIZE
            encoder_input_data_batch = pad_sequences(input_data[start:end], encoder_max_seq_length)
            decoder_target_data_batch = np.zeros(shape=(BATCH_SIZE, decoder_max_seq_length, num_decoder_tokens))
            decoder_input_data_batch = np.zeros(shape=(BATCH_SIZE, decoder_max_seq_length, num_decoder_tokens))
            for lineIdx, target_words in enumerate(output_text_data[start:end]):
                for idx, w in enumerate(target_words):
                    w2idx = 0  # default [UNK]
                    if w in target_word2idx:
                        w2idx = target_word2idx[w]
                    decoder_input_data_batch[lineIdx, idx, w2idx] = 1
                    if idx > 0:
                        decoder_target_data_batch[lineIdx, idx - 1, w2idx] = 1

            yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch


encoder_inputs = Input(shape=(None,), name='encoder_inputs')
encoder_embedding = Embedding(input_dim=num_encoder_tokens, output_dim=HIDDEN_UNITS,
                              input_length=encoder_max_seq_length, name='encoder_embedding')
encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm')
encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedding(encoder_inputs))
encoder_states = [encoder_state_h, encoder_state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens), name='decoder_inputs')
decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm')
decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                 initial_state=encoder_states)
decoder_dense = Dense(units=num_decoder_tokens, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


Xtrain, Xtest, Ytrain, Ytest = train_test_split(encoder_input_data, target_texts, test_size=0.2, random_state=42)

print(len(Xtrain))
print(len(Xtest))

train_gen = generate_batch(Xtrain, Ytrain)
test_gen = generate_batch(Xtest, Ytest)

train_num_batches = len(Xtrain) // BATCH_SIZE
test_num_batches = len(Xtest) // BATCH_SIZE

weight_file_path = os.path.join(WEIGHT_FILE_DIR, "encdec_model_epoch_{epoch:02d}.h5")
checkpoint = ModelCheckpoint(filepath=weight_file_path, save_best_only=False, period=25)
history=model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                    epochs=NUM_EPOCHS,
                    verbose=1, validation_data=test_gen, validation_steps=test_num_batches, callbacks=[checkpoint])

model.save_weights(WEIGHT_FILE_PATH)
model.save(WEIGHT_FILE_PATH)


# ========================================= #
# ========================================= #
# ========================================= #


# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)
decoder_state_inputs = [Input(shape=(HIDDEN_UNITS,)), Input(shape=(HIDDEN_UNITS,))]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

#-# Graph plotting
fig1 = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
fig1.savefig('loss1.png')

fig2 = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
fig2.savefig('accuracy1.png')


def decode_sequence(input_text):
    input_seq = []
    input_wids = []
    for word in nltk.word_tokenize(input_text):
        idx = 1  # default [UNK]
        if word in input_word2idx:
            idx = input_word2idx[word]
        input_wids.append(idx)
    input_seq.append(input_wids)
    input_seq = pad_sequences(input_seq, encoder_max_seq_length)

    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_word2idx['START']] = 1

    target_text = ''
    target_text_len = 0
    terminated = False
    while not terminated:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sample_token_idx = np.argmax(output_tokens[0, -1, :])

        sample_word = target_idx2word[sample_token_idx]

        target_text_len += 1

        if sample_word != 'START' and sample_word != 'END':
            target_text += ' ' + sample_word

        if sample_word == 'END' or target_text_len >= decoder_max_seq_length:
            terminated = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sample_token_idx] = 1

        states_value = [h, c]
        
    return target_text.strip()

with open(TESTING_DATASET, 'r', encoding='utf-8') as f:
    lines1 = f.read().split('\n')
#     [: min(5,len(lines1) - 1)]
for line in lines1:
    input_text, target_text = line.split('\t')
    decoded_sentence = decode_sequence(input_text)
    print('-')
    print('Input sentence:', input_text)
    print('Decoded sentence:', decoded_sentence)
