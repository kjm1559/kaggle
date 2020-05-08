import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df, batch_size=35, shuffle=True):
        'Initialization'
        # self.dim = dim
        # self.batch_size = batch_size
        # self.labels = labels
        # self.list_IDs = list_IDs
        # self.n_channels = n_channels
        # self.n_classes = n_classes
        self.shuffle = shuffle
        self.df = df
        self.list_IDs = np.arange(len(self.df))
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # tmp_df = self.df[self.df.d == self.list_IDs[index]]
        # tmp_df = self.group_df.get_group(self.list_IDs[index])
        # X = make_X(tmp_df)
        # y = tmp_df[['sales', 'sales_diff_year', 'sales_price_86']].to_numpy()
        X = {'input_word_ids':[], 'input_mask':[], 'segment_ids':[]}
        for idata in self.df.loc[index].itertuples():
            X['input_word_ids'].append(list(idata.input_word_ids))
            X['input_mask'].append(list(idata.input_mask))
            X['segment_ids'].append(list(idata.all_segment_id))
        X['input_word_ids'] = np.array(X['input_word_ids'])
        X['input_mask'] = np.array(X['input_mask'])
        X['segment_ids'] = np.array(X['segment_ids'])
        y = self.df.loc[index]['toxic'].values
        return X, y


max_seq_length = 128  # Your choice here.
input_word_ids = Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_word_ids")
input_mask = Input(shape=(max_seq_length,), dtype=tf.int32,
                                   name="input_mask")
segment_ids = Input(shape=(max_seq_length,), dtype=tf.int32,
                                    name="segment_ids")
# bert_layer = hub.KerasLayer("https://tfhub.dev/google/bert_cased_L-12_H-768_A-12/2", trainable=True)
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/2", trainable=True)
# pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
x, y = bert_layer([input_word_ids, input_mask, segment_ids])
x = Dense(32, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)


model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# input_word_ids = []
# input_word_ids = []
# for 

train_examples = pd.read_pickle('./outputs/jigsaw-unintended-bias-train-processed-seqlen128.pickle')

train_gen = DataGenerator(train_examples)

model.fit(train_gen)


# s = "This is a nice sentence."
# stokens = tokenizer.tokenize(s)
# stokens = ["[CLS]"] + stokens + ["[SEP]"]

# input_ids = get_ids(stokens, tokenizer, max_seq_length)
# input_masks = get_masks(stokens, max_seq_length)
# input_segments = get_segments(stokens, max_seq_length)

# pool_embs, all_embs = model.predict([[input_ids],[input_masks],[input_segments]])