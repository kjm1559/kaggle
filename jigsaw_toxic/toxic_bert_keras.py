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
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Flatten, Embedding
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df, batch_size=256, shuffle=True):
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
            # X['segment_ids'].append(list(idata.all_segment_id))
        X['input_word_ids'] = np.array(X['input_word_ids'])
        X['input_mask'] = np.array(X['input_mask'])
        # X['segment_ids'] = np.array(X['segment_ids'])
        y = self.df.loc[index]['toxic'].values
        return X, y

def make_X(df):
    dict_df = {}
    # for columns in df.columns:
    #     dict_df[columns] = df[columns].values
    dict_df['input_word_ids'] = np.array([list(dd) for dd in df['input_word_ids'].values])
    dict_df['input_mask'] = np.array([list(dd) for dd in df['input_mask'].values])
    dict_df['segment_ids'] = np.array([list(dd) for dd in df['all_segment_id'].values])
    # print(df['input_word_ids'].values.shape, df['input_word_ids'].values[0])
    return dict_df
    

max_seq_length = 128  # Your choice here.
input_word_ids = Input(shape=(max_seq_length,), dtype=tf.int32,
                                       name="input_word_ids")
input_mask = Input(shape=(max_seq_length,), dtype=tf.int32,
                                   name="input_mask")
segment_ids = Input(shape=(max_seq_length,), dtype=tf.int32,
                                    name="segment_ids")
# bert_layer = hub.KerasLayer("https://tfhub.dev/google/bert_cased_L-12_H-768_A-12/2", trainable=True)
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/2", trainable=False)
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
# x, y = bert_layer([input_word_ids, input_mask, segment_ids])

bert_model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])

def attention_mechanism(days, input_):
    x = Dense(256, activation='sigmoid')(input_)
    x = Dense(days, activation='softmax')(x)
    return x

x = Embedding(120000, 128, input_length=128)(input_word_ids)

x = attention_mechanism(128, x)
x = Flatten()(x)
# x = Dense(128, activation='relu')(x)
# x = BatchNormalization()(x)
# x = Dropout(0.25)(x)
x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
x = Dense(32, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)


# model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=outputs)
model = Model(inputs=[input_word_ids, input_mask], outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])



# input_word_ids = []
# input_word_ids = []
# for 

# train_examples = pd.read_pickle('./outputs/jigsaw-unintended-bias-train-processed-seqlen128.pickle')

try:
    train_examples = pd.read_pickle('./outputs/jigsaw-toxic-comment-train-processed-seqlen128.pickle')
except:
    train_examples = pd.read_csv('jigsaw-toxic-comment-train-processed-seqlen128.csv', usecols=['input_word_ids', 'input_mask', 'toxic'])
    train_examples['input_word_ids'] = [eval(dd) for dd in train_examples['input_word_ids'].values]
    train_examples['input_mask'] = [eval(dd) for dd in train_examples['input_mask'].values]
    # train_examples['all_segment_id'] = [eval(dd) for dd in train_examples['all_segment_id'].values]
    train_examples.to_pickle('./outputs/jigsaw-toxic-comment-train-processed-seqlen128.pickle')

train_gen = DataGenerator(train_examples)

model_path = './toxic_classification.h5'  # '{epoch:02d}-{val_loss:.4f}.h5'
cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(patience=10)
# model.fit(train_gen, validation_data=train_gen, epochs=20, callback=[cb_checkpoint, early_stopping])

try:
    test_examples = pd.read_pickle('./outputs/test-processed-seqlen128.pickle')
except:
    test_examples = pd.read_csv('test-processed-seqlen128.csv', usecols=['input_word_ids', 'input_mask', 'all_segment_id'])
    test_examples['input_word_ids'] = [eval(dd) for dd in test_examples['input_word_ids'].values]
    test_examples['input_mask'] = [eval(dd) for dd in test_examples['input_mask'].values]
    test_examples['all_segment_id'] = [eval(dd) for dd in test_examples['all_segment_id'].values]
    test_examples.to_pickle('./outputs/test-processed-seqlen128.pickle')

print('translate start')
trans = bert_model.predict(make_X(test_examples[:10]))
print(np.argmax(trans[1][0][0]))
print(np.argmax(trans[0][0]))
print('translate end')


model.load_weights('./toxic_classification.h5')
toxic = model.predict({'input_mask':test_examples['input_mask'], 'input_word_ids':trans})


test_examples = pd.read_csv('test-processed-seqlen128.csv')
test_examples['toxic'] = toxic
test_examples.to_csv('test-processed-seqlen128.csv')