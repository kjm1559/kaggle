from tensorflow.keras.utils import Sequence
from tqdm import tqdm
from datetime import datetime
from datetime import timedelta
import tensorflow.keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, GRU, Masking, Permute, Concatenate, \
                                    LSTM, BatchNormalization, Flatten, TimeDistributed, \
                                    Concatenate, Cropping1D, RepeatVector
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tqdm import tqdm
import gzip
import pickle
import pandas as pd
import numpy as np


class DataGenerator(Sequence):
    def __init__(self, data_df, calendar_df, sell_price_df, seq_len=30 * 4, predict_day=30 * 2, skip_day=30 * 2, batch_size=32, shuffle=True):
        self.data_df = data_df
        self.calendar_df = calendar_df
        self.sell_price_df = sell_price_df
        self.seq_len = seq_len
        self.predict_day = predict_day
        self.batch_size = batch_size
        self.event1_label = self.calendar_df['event_name_1'].unique().tolist()
        self.event_type_label = self.calendar_df['event_type_1'].unique().tolist()
        self.event2_label = self.calendar_df['event_name_2'].unique().tolist()
        self.item_id_label = self.sell_price_df['item_id'].unique().tolist()
        self.store_id_label = self.sell_price_df['store_id'].unique().tolist()
        self.dept_id_label = self.data_df['dept_id'].unique().tolist()
        self.price_dict = self.__make_price_dict()
        self.date_dict = self.__make_date_dict()
        self.sell_count_dict = self.__make_sell_count_dict()        
        self.date_list = self.data_df.columns[6:].tolist()
        self.list_IDs = self.__make_id_list()
        
        
        
        self.shuffle=shuffle
        self.on_epoch_end()
        
    
    def __make_id_list(self):
        target_date = self.date_list[:-(self.predict_day + self.seq_len)]
        data_id = self.data_df['id'].unique().tolist()
        list_IDs = []
        print('Create data id')        
        for id_ in tqdm(data_id, total=len(data_id), position=0):
            for i in range(0, len(target_date), 7):
                start_date = target_date[i]                
                if start_date in self.sell_count_dict[id_]:
                    # print(id_)
                    list_IDs.append([id_, start_date])
        return list_IDs
    
    def __make_price_dict(self):
        price_dict = {}
        print('Create price dict')
        for idata in tqdm(self.sell_price_df.itertuples(), total=len(self.sell_price_df), position=0):
            if not idata.store_id in price_dict:
                price_dict[idata.store_id] = {}
            if not idata.item_id in price_dict[idata.store_id]:
                price_dict[idata.store_id][idata.item_id] = {}
            if not idata.wm_yr_wk in price_dict[idata.store_id][idata.item_id]:
                price_dict[idata.store_id][idata.item_id][idata.wm_yr_wk] = idata.sell_price
        return price_dict
    
    def __make_date_dict(self):
        date_dict = {}#[np.zeros(len(calendar_df)).tolist(), np.zeros(len(calendar_df)).tolist()]
        print('Create date dict')
        for idata in tqdm(self.calendar_df.itertuples(), total=len(self.calendar_df), position=0):
            #date = int(idata.d[2:])
            date_dict[idata.d] = {}
            date_dict[idata.d]['encoded_data'] = [self.event1_label.index(idata.event_name_1) / 31]
            date_dict[idata.d]['encoded_data'] += [self.event_type_label.index(idata.event_type_1) / 5]
            date_dict[idata.d]['encoded_data'] += [self.event2_label.index(idata.event_name_2) / 5]
            date_dict[idata.d]['encoded_data'] += [self.event_type_label.index(idata.event_type_2) / 5]
            date_dict[idata.d]['encoded_data'] += np.eye(2)[idata.snap_CA].tolist()
            date_dict[idata.d]['encoded_data'] += np.eye(2)[idata.snap_TX].tolist()
            date_dict[idata.d]['encoded_data'] += np.eye(2)[idata.snap_WI].tolist()
            date_dict[idata.d]['encoded_data'] += [idata.wday]#np.eye(7)[idata.wday - 1].tolist()
            date_dict[idata.d]['encoded_data'] += [idata.month]
            date_dict[idata.d]['wm_yr_wk'] = idata.wm_yr_wk
        return date_dict
    
    def __make_sell_count_dict(self):
        sell_count_dict = {}
        print('Create sell count dict')
        for idata in tqdm(self.data_df.itertuples(), total=len(self.data_df), position=0):
#             print(idata)
            sell_count_dict[idata[1]] = {}
            for i, index in enumerate(self.data_df.columns[6:]):
                store_id = '_'.join(idata[1].split('_')[3:5])
                item_id = '_'.join(idata[1].split('_')[:3])
                if self.date_dict[index]['wm_yr_wk'] in self.price_dict[store_id][item_id]:
                    sell_count_dict[idata[1]][index] = idata[i + 7] / 800
        return sell_count_dict
    
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        return self.__data_generation(list_IDs_temp)
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def get_train_data(self):
        train_data = [] 
        for c in tqdm(self.data_df.columns[6:]):
            date = self.date_dict[c]['encoded_data']
            data = self.data_df[c].values.tolist()
            price = []
            wm = self.date_dict[c]['wm_yr_wk']
            for id_ in self.data_df.id.values:
                store_id = '_'.join(id_.split('_')[3:5])
                tiem_id = '_'.join(id_.split('_')[:3])
                try:
                    price.append(self.price_dict[store_id][item_id][wm])
                except:
                    price.append(0)
            train_data.append(date + data + price)
        return train_data
            
    
    def __data_generation(self, list_IDs_temp):
        # [item_id(3049), store_id(10), dept_id(7), target_price(1, 107.32)]
        # [sell_count(sequence length, 763), price(sell_price(sequence length, 107.32)]
        # [event1(31), event_type1(5), event2(5), event_type2(5), snap_CA(2), snap_TX(2), snap_WI(2), weak(7)] 
#         train_data = {'sequence_data': [], 'item_info': [], 'predict_info': []}
#         label_data = []
        
        train_data = {}
        train_data['sequence_data'] = np.zeros((len(list_IDs_temp), self.seq_len, 19))
        train_data['item_info'] = np.zeros((len(list_IDs_temp), 3))
        train_data['predict_info'] = np.zeros((len(list_IDs_temp), self.predict_day, 18))
        label_data = np.zeros((len(list_IDs_temp), self.predict_day, 1))

        # loop of train_data
        for batch_num, IDs in enumerate(list_IDs_temp):
            id_, start_date = IDs
            predict_start_d = int(start_date[2:]) + self.seq_len
            train_start_d = int(start_date[2:])
            item_id = '_'.join(id_.split('_')[:3])
            store_id = '_'.join(id_.split('_')[3:5])
            dept_id = '_'.join(id_.split('_')[:2])
#             tmp_sequence_data = []
#             tmp_predict_info = []
#             tmp_sequence_label = []

            # get item information
            train_data['item_info'][batch_num, 0] = self.item_id_label.index(item_id) / 3049
            train_data['item_info'][batch_num, 1] = self.store_id_label.index(store_id) / 10
            train_data['item_info'][batch_num, 2] = self.dept_id_label.index(dept_id) / 7
#             item_info = [self.item_id_label.index(item_id) / 3049]
#             item_info += [self.store_id_label.index(store_id) / 10]
#             item_info += [self.dept_id_label.index(dept_id) / 7]

            # make sequence data
            for i in range(self.seq_len):
                current_wm_yr_wk = self.date_dict['d_' + str(train_start_d + i)]['wm_yr_wk']
                date_info = self.date_dict['d_' + str(train_start_d + i)]['encoded_data']
                if current_wm_yr_wk in self.price_dict[store_id][item_id]:
                    price = self.price_dict[store_id][item_id][current_wm_yr_wk] / 108 # price normalization
                else:
                    price = 0
#                 sell_count = self.data_df[self.data_df.id == id_]['d_' + str(train_start_d + i)].values[0] / 800 # sell counter normalization
                sell_count = self.sell_count_dict[id_]['d_' + str(train_start_d + i)]
#                 tmp_sequence_data.append(date_info + [price] + [sell_count])
                for j, data in enumerate(date_info + [price] + [sell_count]):
                    train_data['sequence_data'][batch_num, i, j] = data

            # make prediction information data
            for i in range(self.predict_day):
                current_wm_yr_wk = self.date_dict['d_' + str(predict_start_d + i)]['wm_yr_wk']     
                date_info = self.date_dict['d_' + str(predict_start_d + i)]['encoded_data']
                if current_wm_yr_wk in self.price_dict[store_id][item_id]:
                    price = self.price_dict[store_id][item_id][current_wm_yr_wk] / 108 # price normalization
                else:
                    price = 0
#                 sell_count = self.data_df[self.data_df.id == id_]['d_' + str(predict_start_d + i)].values[0] / 800 # sell counter normalization
                sell_count = self.sell_count_dict[id_]['d_' + str(predict_start_d + i)]
                
                for j, data in enumerate(date_info + [price]):
                    train_data['predict_info'][batch_num, i, j] = data
                label_data[batch_num, i, 0] = sell_count
#                 tmp_predict_info.append(date_info + [price])            
#                 tmp_sequence_label.append(sell_count)

#             train_data['predict_info'].append(tmp_predict_info)
#             train_data['sequence_data'].append(tmp_sequence_data)
#             train_data['item_info'].append(item_info)
#             label_data.append([tmp_sequence_label])
            
#         train_data['predict_info'] = np.array(train_data['predict_info'])
#         train_data['sequence_data'] = np.array(train_data['sequence_data'])
#         train_data['item_info'] = np.array(train_data['item_info'])
#         label_data = np.array(label_data)
        
        return train_data, label_data


TRAIN_TIMESEQUENCE = 30 * 4 # 1 quarter
PREDICT_TIMESEQUENCE = 30 * 2 # 2 month

def rmsse(pred,true):
    assert pred.shape[0]==true.shape[0]
#     pred = pred[:, :PREDICT_TIMESEQUENCE,:]
    return K.sqrt(K.sum(K.square((true - pred))) / \
                  (((K.sum(K.square((true[1:] - true[:-1])))) / (TRAIN_TIMESEQUENCE - 1)) * PREDICT_TIMESEQUENCE + 1))

def attention_mechanism(days, input_):
    x = Dense(256, activation='sigmoid')(input_)
    x = Dense(days, activation='softmax')(x)
    return x

def attention_model(lr=1e-3):
    predict_info = Input(shape=(None, 18), name='predict_info')
    sequence_data = Input(shape=(None, 19), name='sequence_data')    
    item_info = Input(shape=(3,), name='item_info')
    
    x = Masking(mask_value=0, input_shape=(None, 19))(sequence_data)
    x = GRU(128, name='GRU_layer1', return_sequences=True)(x)

    # past sequence understanding
    attention_x = attention_mechanism(TRAIN_TIMESEQUENCE, x)
    gru_out = Permute((2, 1))(x)
    attention_mul = K.batch_dot(gru_out, attention_x)
    attention_mul = Permute((2, 1))(attention_mul)       

    x = GRU(128, name='GRU_layer2', return_sequences=True)(attention_mul)
    
    # item feature added
    v = RepeatVector(TRAIN_TIMESEQUENCE)(item_info) # past sequence
    x = Concatenate(axis=-1)([v, x])
    x = TimeDistributed(Dense(32, activation='relu'))(x)    
    x = TimeDistributed(Dense(128, activation='relu'))(x)
    
    # get extraction of item feature
    x = Flatten()(x)
    v = RepeatVector(PREDICT_TIMESEQUENCE)(x) # predict sequence
    x = Concatenate(axis=-1)([predict_info, v])
    
    x = GRU(128, name='GRU_predict_layer1', return_sequences=True)(x)

    # past sequence understanding
    attention_x = attention_mechanism(PREDICT_TIMESEQUENCE, x)
    gru_out = Permute((2, 1))(x)
    attention_mul = K.batch_dot(gru_out, attention_x)
    attention_mul = Permute((2, 1))(attention_mul)       

    x = GRU(128, name='GRU_predict_layer2', return_sequences=True)(attention_mul)
    
    x = TimeDistributed(Dense(32, activation='relu'))(x)    
    x = TimeDistributed(Dense(128, activation='relu'))(x)
    outputs = TimeDistributed(Dense(1, activation='sigmoid'))(x)

    print(outputs.shape)    
    
    optimizer = Adam(lr=lr, name='adam')
    model = Model([sequence_data, item_info, predict_info], outputs, name='gru_network')
    model.compile(optimizer=optimizer, loss=rmsse)
    return model

class rnn_network():
    def __init__(self, data_df, calendar_df, sell_price_df, epochs=200):
        date_list = data_df.columns[6:]
        self.train_date_list = date_list[:int(len(date_list) * 0.8)]
        self.validation_date_list = date_list[int(len(date_list) * 0.8):]
        print(len(self.train_date_list), len(self.validation_date_list))
        self.train_generator = DataGenerator(data_df[data_df.columns[:6].tolist() + self.train_date_list.tolist()], calendar_df, sell_price_df, seq_len=TRAIN_TIMESEQUENCE, predict_day=PREDICT_TIMESEQUENCE)
        self.validation_generator = DataGenerator(data_df[data_df.columns[:6].tolist() + self.validation_date_list.tolist()], calendar_df, sell_price_df, seq_len=TRAIN_TIMESEQUENCE, predict_day=PREDICT_TIMESEQUENCE)
        self.seq_len = self.train_generator.seq_len
        self.predict_day = self.train_generator.predict_day
        self.model = attention_model()
        self.epochs=epochs
        self.batch_size=32
    
    def train(self):        
        tb_hist = tensorflow.keras.callbacks.TensorBoard(log_dir='./graph_gru_attention', histogram_freq=1, write_graph=True, write_images=True)
        model_path = './predict_gru_attention.h5'  # '{epoch:02d}-{val_loss:.4f}.h5'
        cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(patience=10)

        history = self.model.fit_generator(generator=self.train_generator, \
                                           validation_data=self.validation_generator, \
                                           epochs=self.epochs, \
#                                            use_multiprocessing=True, workers=6,
                                           callbacks=[tb_hist, cb_checkpoint, early_stopping])  # , class_weight=class_weights) 
#         y_predict = self.model.predict({'encoder_input':x_train[0], 'country_onehot':x_train[1], 'target_input': X_train[0][:, -1, 0].reshape(list(X_train[0].shape[:-2]) + [1]), 'flag_input': np.zeros(len(x_train[0]))})
#         return y_predict, y_train


if __name__ == '__main__':
    import tensorflow as tf

    # tf.config.gpu.set_per_process_memory_fraction(0.10)
    # tf.config.gpu.set_per_process_memory_growth(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
            print(e)
    sell_prices = pd.read_csv('sell_prices.csv')
    calendar = pd.read_csv('calendar.csv')
    sales_train_validation = pd.read_csv('sales_train_validation.csv')
    # test = DataGenerator(sales_train_validation, calendar, sell_prices)
    # train_data = test.__getitem__(0)
    # print(train_data)
    train = rnn_network(sales_train_validation, calendar, sell_prices)
    train.train()