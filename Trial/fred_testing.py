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
import glob

np.random.seed(42)

## Change the directory addresses, hidden units and test dataset ####
HIDDEN_UNITS = 50
TESTING_DATASET = 'dlgs_trial5.txt'
WEIGHTS_DIRECTORY_NAME = 'dlgs_trial5'
FINAL_RESULTS_FILE = WEIGHTS_DIRECTORY_NAME+'_all_weights.txt'

c= np.load('word-context.npy')
context = c.item()

num_encoder_tokens = context['num_encoder_tokens']
num_decoder_tokens = context['num_decoder_tokens']
encoder_max_seq_length = context['encoder_max_seq_length']
decoder_max_seq_length = context['decoder_max_seq_length']


input_word2idx1 = np.load('input-word2idx.npy')
input_word2idx = input_word2idx1.item()
input_idx2word1 = np.load('input-idx2word.npy')
input_idx2word = input_idx2word1.item()
target_word2idx1 = np.load('target-word2idx.npy')
target_word2idx = target_word2idx1.item()
target_idx2word1 = np.load('target-idx2word.npy')
target_idx2word = target_idx2word1.item()

##========NETWORK ARCHITECTURE===================================

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

## =============== DECODER FUNCTION ===========================
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

##========== Load Weight File ========= ####
os.chdir(WEIGHTS_DIRECTORY_NAME)

if os.path.isfile(FINAL_RESULTS_FILE):
    os.remove(FINAL_RESULTS_FILE)

output_file = open(FINAL_RESULTS_FILE,'a') 
weight_files = glob.glob('*.h5')
print(weight_files)
for weight_file in weight_files:
    model.load_weights(weight_file)
    output_file.write("\n==============================")
    output_file.write("\n"+weight_file)
    output_file.write("\n==============================")

    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_inputs = [Input(shape=(HIDDEN_UNITS,)), Input(shape=(HIDDEN_UNITS,))]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)


    with open(TESTING_DATASET, 'r', encoding='utf-8') as f:
        lines1 = f.read().split('\n')

    length_of_file= sum(1 for line in open(TESTING_DATASET))
    for line in lines1[: min(length_of_file,len(lines1) - 1)]:
        input_text, target_text = line.split('\t')
        decoded_sentence = decode_sequence(input_text)
        user_input = 'Input sentence:'+input_text
        bot_output = 'Decoded sentence:'+decoded_sentence
        #print('-')
        #print(user_input)
        #print(bot_output)
        output_file.write("\n-")
        output_file.write('\n'+user_input)
        output_file.write('\n'+bot_output)

output_file.close()
