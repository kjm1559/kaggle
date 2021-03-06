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
    def __init__(self, data_df, calendar_df, sell_price_df, max_count, seq_len=30 * 4, predict_day=30 * 2, skip_day=30, batch_size=32, shuffle=True):
        self.data_df = data_df
        self.calendar_df = calendar_df
        self.sell_price_df = sell_price_df
        self.seq_len = seq_len
        self.predict_day = predict_day
        self.batch_size = batch_size
        self.max_count = max_count
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
            for i in range(0, len(target_date), self.predict_day):
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
            date_dict[idata.d]['encoded_data'] += [idata.wday / 7]#np.eye(7)[idata.wday - 1].tolist()
            date_dict[idata.d]['encoded_data'] += [idata.month / 12]
            date_dict[idata.d]['wm_yr_wk'] = idata.wm_yr_wk
        return date_dict
    
    def __make_sell_count_dict(self):
        sell_count_dict = {}
        print('Create sell count dict')
        for j, idata in tqdm(enumerate(self.data_df.itertuples()), total=len(self.data_df), position=0):
#             print(idata)
            sell_count_dict[idata[1]] = {}
            for i, index in enumerate(self.data_df.columns[6:]):
                store_id = '_'.join(idata[1].split('_')[3:5])
                item_id = '_'.join(idata[1].split('_')[:3])
                if self.date_dict[index]['wm_yr_wk'] in self.price_dict[store_id][item_id]:                    
                    sell_count_dict[idata[1]][index] = idata[i + 7] / self.max_count[j]#800
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
            
    def get_test_data(self):
        train_data = {}
        train_data['sequence_data'] = np.zeros((30490, self.seq_len, 14))
        train_data['item_info'] = np.zeros((30490, 3))
        train_data['predict_info'] = np.zeros((30490, self.predict_day, 13))     

        
        data_id = self.data_df['id'].unique().tolist()
        list_IDs = []
        for id_ in tqdm(data_id, total=len(data_id), position=0):
            list_IDs.append([id_, self.date_list[0]])  
            print(self.date_list[0]) 

        # loop of train_data
        for batch_num, IDs in tqdm(enumerate(list_IDs), position=0, total=len(list_IDs)):
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
                try:
                    sell_count = self.sell_count_dict[id_]['d_' + str(train_start_d + i)]
                except:
                    sell_count = 0
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
                # sell_count = self.sell_count_dict[id_]['d_' + str(predict_start_d + i)]
                
                for j, data in enumerate(date_info + [price]):
                    train_data['predict_info'][batch_num, i, j] = data
                
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
        
        return train_data

    def __data_generation(self, list_IDs_temp):
        # [item_id(3049), store_id(10), dept_id(7), target_price(1, 107.32)]
        # [sell_count(sequence length, 763), price(sell_price(sequence length, 107.32)]
        # [event1(31), event_type1(5), event2(5), event_type2(5), snap_CA(2), snap_TX(2), snap_WI(2), weak(7)] 
#         train_data = {'sequence_data': [], 'item_info': [], 'predict_info': []}
#         label_data = []
        
        train_data = {}
        train_data['sequence_data'] = np.zeros((len(list_IDs_temp), self.seq_len, 14))
        train_data['item_info'] = np.zeros((len(list_IDs_temp), 3))
        train_data['predict_info'] = np.zeros((len(list_IDs_temp), self.predict_day, 13))
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

def attention_model(lr=1e-5):
    predict_info = Input(shape=(None, 13), name='predict_info')
    sequence_data = Input(shape=(None, 14), name='sequence_data')    
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
    model.compile(optimizer=optimizer, loss='mse')
    return model

class rnn_network():
    def __init__(self, data_df, calendar_df, sell_price_df, epochs=200):                
        self.model = attention_model()
        self.epochs=epochs
        self.batch_size=32
        self.data_df = data_df
        self.calendar_df = calendar_df
        self.sell_price_df = sell_price_df
        self.max_count = np.max(data_df[data_df.columns[6:]].values, axis=1)        
    
    def train(self):   
        date_list = self.data_df.columns[6:]
        self.train_date_list = date_list[:int(len(date_list) * 0.8)]
        self.validation_date_list = date_list[int(len(date_list) * 0.8):] 
        self.train_generator = DataGenerator(self.data_df[self.data_df.columns[:6].tolist() + self.train_date_list.tolist()], self.calendar_df, self.sell_price_df, self.max_count, seq_len=TRAIN_TIMESEQUENCE, predict_day=PREDICT_TIMESEQUENCE)
        self.validation_generator = DataGenerator(self.data_df[self.data_df.columns[:6].tolist() + self.validation_date_list.tolist()], self.calendar_df, self.sell_price_df, self.max_count, seq_len=TRAIN_TIMESEQUENCE, predict_day=PREDICT_TIMESEQUENCE)
        self.seq_len = self.train_generator.seq_len
        self.predict_day = self.train_generator.predict_day
        print(len(self.train_date_list), len(self.validation_date_list))    
        tb_hist = tensorflow.keras.callbacks.TensorBoard(log_dir='./graph_gru_attention3', histogram_freq=1, write_graph=True, write_images=True)
        model_path = './predict_gru_attention2.h5'  # '{epoch:02d}-{val_loss:.4f}.h5'
        cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(patience=10)

        history = self.model.fit(self.train_generator, \
                                           validation_data=self.validation_generator, \
                                           epochs=self.epochs, \
#                                            use_multiprocessing=True, workers=6,
                                           callbacks=[tb_hist, cb_checkpoint, early_stopping])  # , class_weight=class_weights) 
#         y_predict = self.model.predict({'encoder_input':x_train[0], 'country_onehot':x_train[1], 'target_input': X_train[0][:, -1, 0].reshape(list(X_train[0].shape[:-2]) + [1]), 'flag_input': np.zeros(len(x_train[0]))})
#         return y_predict, y_train

    def test(self):
        self.model.load_weights('./predict_gru_attention2.h5')
        self.test_generator = DataGenerator(self.data_df[self.data_df.columns[:6].tolist() + self.data_df.columns[-TRAIN_TIMESEQUENCE - 1 - 7:].tolist()], self.calendar_df, self.sell_price_df, self.max_count, seq_len=TRAIN_TIMESEQUENCE, predict_day=PREDICT_TIMESEQUENCE, shuffle=False, batch_size=len(self.data_df))
        # print(len(self.test_data_list))
        print(self.test_generator.data_df.columns)
        test = self.test_generator.get_test_data()
        print(test['sequence_data'].shape, test['item_info'].shape, test['predict_info'].shape, len(self.data_df))
        print(len(self.test_generator.list_IDs))
        result_list = []
        result_list_e = []
        for i in tqdm(range(int(30490 / 32) + 1)):
            IDs = self.data_df.loc[i * 32:(i + 1) * 32 - 1].id
            tmp_dict = {}
            tmp_dict['sequence_data'] = test['sequence_data'][i * 32:(i + 1) * 32]
            tmp_dict['item_info'] = test['item_info'][i * 32:(i + 1) * 32]
            tmp_dict['predict_info'] = test['predict_info'][i * 32:(i + 1) * 32]
            max_count = self.max_count[i * 32:(i + 1) * 32]
            # print(tmp_dict['sequence_data'][0][-120:].tolist())
            predict_data = self.model.predict_on_batch(tmp_dict)
            # print(len(IDs), IDs)
            for j, id_ in enumerate(IDs):
                if id_ == 'FOODS_1_018_TX_2_validation':
                    print(id_, predict_data[j].numpy().tolist(), max_count[j])
                if id_ == 'FOODS_1_017_TX_2_validation':
                    print(id_, predict_data[j].numpy().tolist(), max_count[j])
                result_list.append([id_] + (predict_data.numpy()[j, -28 * 2 : -28] * max_count[j]).reshape(28).tolist())
                result_list_e.append(['_'.join(id_.split('_')[:-1] + ['evaluation'])] + (predict_data.numpy()[j, -28 : ] * max_count[j]).reshape(28).tolist())
        df = pd.DataFrame(result_list, columns=['id'] + ['F' + str(i + 1) for i in range(28)])
        df2 = pd.DataFrame(result_list_e, columns=['id'] + ['F' + str(i + 1) for i in range(28)])
        df = pd.concat([df, df2])
        df.to_csv('submission.csv', index=False)


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
    # train.train()
    train.test()