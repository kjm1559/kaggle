from  datetime import datetime, timedelta
import gc
import numpy as np, pandas as pd
import tensorflow.keras as keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "category", "year": "category", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }

h = 28 
max_lags = 57
tr_last = 1913
fday = datetime(2016,4, 25) 
# fday = datetime(2016,4, 24) 

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, df, shuffle=True):
        'Initialization'
        # self.dim = dim
        # self.batch_size = batch_size
        # self.labels = labels
        # self.list_IDs = list_IDs
        # self.n_channels = n_channels
        # self.n_classes = n_classes
        self.shuffle = shuffle
        self.df = df
        self.group_df = df.groupby(by='d')
        self.list_IDs = self.df.d.unique().tolist()
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.list_IDs))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # tmp_df = self.df[self.df.d == self.list_IDs[index]]
        tmp_df = self.group_df.get_group(self.list_IDs[index])
        X = make_X(tmp_df)
        y = tmp_df[['sales', 'sales_diff_year', 'sales_price_86']].to_numpy()
       
        return X, y

def create_dt(is_train = True, nrows = None, first_day = 1200):
    prices = pd.read_csv("sell_prices.csv", dtype = PRICE_DTYPES)
    for col, col_dtype in PRICE_DTYPES.items():
        if col_dtype == "category":
            prices[col] = prices[col].cat.codes.astype("int16")
            prices[col] -= prices[col].min()
            
    cal = pd.read_csv("calendar.csv", dtype = CAL_DTYPES)
    cal["date"] = pd.to_datetime(cal["date"])
    for col, col_dtype in CAL_DTYPES.items():
        if col_dtype == "category":
            cal[col] = cal[col].cat.codes.astype("int16")
            cal[col] -= cal[col].min()
    
    start_day = max(1 if is_train  else tr_last-max_lags, first_day)
    numcols = [f"d_{day}" for day in range(start_day,tr_last+1)]
    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
    dtype = {numcol:"float32" for numcol in numcols} 
    dtype.update({col: "category" for col in catcols if col != "id"})
    dt = pd.read_csv("sales_train_validation.csv", 
                     nrows = nrows, usecols = catcols + numcols, dtype = dtype)
    
    for col in catcols:
        if col != "id":
            dt[col] = dt[col].cat.codes.astype("int16")
            dt[col] -= dt[col].min()
    
    if not is_train:
        # for day in range(tr_last+1, tr_last+ 28 +1):
        for day in range(tr_last+1, tr_last+ 28 + 28 +1):
            dt[f"d_{day}"] = np.nan
    
    dt = pd.melt(dt,
                id_vars = catcols,
                value_vars = [col for col in dt.columns if col.startswith("d_")],
                var_name = "d",
                value_name = "sales")    
    
    dt = dt.merge(cal, on= "d", copy = False)
    dt = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
    dt.drop(index=dt[dt['sell_price'] == 0].index, inplace=True)
    if is_train:
        dt['sales_price'] = dt['sales'] * dt['sell_price']
        dt['sales_price_86'] = dt.groupby(by='id')['sales_price'].rolling(28).sum().reset_index(0, drop=True)

        # dt['sales_diff'] = dt.groupby(by='id')['sales'].diff().pow(2).shift(periods=1).rolling(28).sum() / 28
        dt['sales_diff'] = dt.groupby(by='id')['sales'].diff()
        dt['sales_diff'] = dt.groupby(by='id')['sales_diff'].shift(periods=1).pow(2)
        # dt['sales_diff_year'] = dt.groupby(by='id')['sales_diff'].rolling(28).mean().reset_index(0, drop=True)
        dt['sales_diff_comsum'] = dt.groupby(by='id')['sales_diff'].cumsum().reset_index(0, drop=True)
        dt['sales_diff_group_count'] = dt.groupby(by='id').cumcount()
        dt['sales_diff_year'] = dt['sales_diff_comsum'] / dt['sales_diff_group_count']
        dt['sales_diff_year'] = np.sqrt(dt['sales_diff_year'])
        # dt.drop(columns=['sales_diff_group_count'], inplace=True)
        dt.drop(index=dt[dt['sales_diff_year'] == 0].index, inplace=True)    
    return dt

def create_fea(dt):
    lags = [7, 14, 21, 28]
    lag_cols = [f"lag_{lag}" for lag in lags ]
    for lag, lag_col in zip(lags, lag_cols):
        dt[lag_col] = dt[["id","sales"]].groupby("id")["sales"].shift(lag)

    wins = [7, 14, 21, 28]
    for win in wins :
        for lag,lag_col in zip(lags, lag_cols):
            dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x : x.rolling(win).mean())

    for i in range(6):
        dt['sales_before_' + str(i+1)] = dt.groupby(by='id')['sales'].shift(periods=(i+1)) 
    
    date_features = {
        
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
#         "ime": "is_month_end",
#         "ims": "is_month_start",
    }
    
#     dt.drop(["d", "wm_yr_wk", "weekday"], axis=1, inplace = True)
    
    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(dt["date"].dt, date_feat_func).astype("int16")


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

from tensorflow.keras.layers import Dense, Input, BatchNormalization, Concatenate, Embedding, Flatten, Dropout, Activation, Conv1D, Reshape, add, AveragePooling1D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2

def rmse(true, pred):
    assert pred.shape[0]==true.shape[0]
    loss_1 = K.sqrt(K.mean(K.square(true[:, 0:1] - pred) + 1e-18))
    return loss_1

def rmsse(true, pred):
    assert pred.shape[0]==true.shape[0]
    # min : 0.03571428571428571
    loss_1 = K.sqrt(K.sum(K.square(true[:, 0:1] - pred)) / K.sum(true[:, 1:2]) + 1e-18)
    return loss_1    

def wrmsse(true, pred):
    assert pred.shape[0]==true.shape[0]
    msse = K.sqrt(K.square(true[:, 0:1] - pred) / (true[:, 1:2] + 1e-18) + 1e-18)
    sales_sum = K.sum(true[:, 2:3]) + 1e-19 # 같은 제품일 경우 고려, 기간이 같아야함
    loss = K.sum((msse * true[:, 2:3])) / sales_sum
    # msse_2 = K.sum(K.square(true[:, 0:1] - pred)) / (K.sum(true[:, 1:2] + 1e-18))
    # rmsse_total = K.sqrt(msse_2 + 1e-18) # rmsse
    rmsse_total = K.sqrt(K.sum(msse) + 1e-18)
    return (loss + rmsse_total) / 2 # aggregation level = 2

# mse, rmsse, wrmsse ensemble model make ->

def predict_model(input_size, epochs=200, lr=1e-3):    
    inputs = Input(shape=input_size, name='inputs')

    # Embedding input
    wday_input = Input(shape=(1,), name='wday')
    month_input = Input(shape=(1,), name='month')
    year_input = Input(shape=(1,), name='year')
    mday_input = Input(shape=(1,), name='mday')
    quarter_input = Input(shape=(1,), name='quarter')
    event_name_1_input = Input(shape=(1,), name='event_name_1')
    event_type_1_input = Input(shape=(1,), name='event_type_1')
    event_name_2_input = Input(shape=(1,), name='event_name_2')
    event_type_2_input = Input(shape=(1,), name='event_type_2')
    item_id_input = Input(shape=(1,), name='item_id')
    dept_id_input = Input(shape=(1,), name='dept_id')
    store_id_input = Input(shape=(1,), name='store_id')
    cat_id_input = Input(shape=(1,), name='cat_id')
    state_id_input = Input(shape=(1,), name='state_id')
    snap_CA_input = Input(shape=(1,), name='snap_CA')
    snap_TX_input = Input(shape=(1,), name='snap_TX')
    snap_WI_input = Input(shape=(1,), name='snap_WI')


    wday_emb = Flatten()(Embedding(7, 1)(wday_input))
    month_emb = Flatten()(Embedding(12, 2)(month_input))
    year_emb = Flatten()(Embedding(6, 1)(year_input))
    mday_emb = Flatten()(Embedding(31, 2)(mday_input))
    quarter_emb = Flatten()(Embedding(4, 1)(quarter_input))
    event_name_1_emb = Flatten()(Embedding(31, 2)(event_name_1_input))
    event_type_1_emb = Flatten()(Embedding(5, 1)(event_type_1_input))
    event_name_2_emb = Flatten()(Embedding(5, 1)(event_name_2_input))
    event_type_2_emb = Flatten()(Embedding(5, 1)(event_type_2_input))

    item_id_emb = Flatten()(Embedding(3049, 4)(item_id_input))
    dept_id_emb = Flatten()(Embedding(7, 1)(dept_id_input))
    store_id_emb = Flatten()(Embedding(10, 1)(store_id_input))
    cat_id_emb = Flatten()(Embedding(6, 1)(cat_id_input))
    state_id_emb = Flatten()(Embedding(3, 1)(state_id_input))
    

    x = Concatenate(-1)([inputs, wday_emb, month_emb, month_emb, year_emb, mday_emb, \
                        quarter_emb, event_name_1_emb, event_type_1_emb, event_name_2_emb, \
                        event_type_2_emb, item_id_emb, dept_id_emb, store_id_emb, cat_id_emb, \
                        state_id_emb])

    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)        
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)

    x_deep = Dense(64, activation='relu')(x)
    x_deep = BatchNormalization()(x_deep)
    x_deep = Dense(128, activation='relu')(x_deep)
    x_deep = BatchNormalization()(x_deep)
    x_deep = Dense(256, activation='relu')(x_deep)

    x = Concatenate(-1)([inputs, x])
    x_res = Dense(64, activation='relu')(x)
    x_res = BatchNormalization()(x_res)
    x_res = Dense(128, activation='relu')(x_res)
    x_res = BatchNormalization()(x_res)
    x_res = Dense(256, activation='relu')(x_res)

    x = Concatenate(-1)([x_deep, x_res])

    outputs = Dense(1, activation='sigmoid')(x)
    # outputs = resnet_v2(x)
    
    # optimizer = Adam(lr=lr)#Adam(lr=lr)
    input_dic = {
        'inputs': inputs, 'wday': wday_input, 'month': month_input, 'year': year_input,
        'mday': mday_input, 'quarter': quarter_input, 'event_name_1': event_name_1_input,
        'event_type_1': event_type_1_input, 'event_name_2': event_name_2_input,
        'event_type_2': event_type_2_input, 'item_id': item_id_input, 'dept_id': dept_id_input,
        'store_id': store_id_input, 'cat_id': cat_id_input, 'state_id': state_id_input,

    }
    model = Model(input_dic, outputs)#, name='predict_model')
    
    return model

class M5_predict_ensemble_model:
    def __init__(self, input_size, batch_size=2**14, epochs=200, lr=1e-3):
        self.model = define_ensemble_model(input_size, lr)
        self.epochs = epochs  
        self.batch_size = batch_size
    
    def train(self, X_train, y_train):
        model_path = './m5_predict5_ensemble.h5'  # '{epoch:02d}-{val_loss:.4f}.h5'
        cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(patience=10)
        # print('lenght :', X_train['inputs'].shape)
        history = self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1, shuffle=True,
                                validation_split=0.2,
                                callbacks=[cb_checkpoint, early_stopping])  # , class_weight=class_weights) 
    # def train(self, train_gen, val_gen):
    #     model_path = './m5_predict5_ensemble.h5'  # '{epoch:02d}-{val_loss:.4f}.h5'
    #     cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
    #     early_stopping = EarlyStopping(patience=10)
    #     # print('lenght :', X_train['inputs'].shape)
    #     history = self.model.fit(train_gen, epochs=self.epochs, verbose=1, #shuffle=True,
    #                             validation_data = val_gen, #validation_split=0.2,
    #                            callbacks=[cb_checkpoint, early_stopping])  # , class_weight=class_weights) 

class M5_predict:
    def __init__(self, input_size, batch_size=3049, epochs=200, lr=1e-3):
        self.batch_size = batch_size
        self.epochs = epochs        
        self.model = predict_model(input_size)              
        self.optimizer = Adam(lr=lr)
        
    def train_mse(self, X_train, y_train):
        self.model.compile(optimizer=self.optimizer, loss=rmse)
        model_path = './m5_predict5_mse.h5'  # '{epoch:02d}-{val_loss:.4f}.h5'
        cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(patience=10)
        print('lenght :', X_train['inputs'].shape)
        history = self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1, shuffle=True,
                                validation_split=0.2,
                                callbacks=[cb_checkpoint, early_stopping])  # , class_weight=class_weights) 

    def train_rmsse(self, X_train, y_train):
        self.model.compile(optimizer=self.optimizer, loss=rmsse)
        model_path = './m5_predict5_rmsse.h5'  # '{epoch:02d}-{val_loss:.4f}.h5'
        cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(patience=10)
        print('lenght :', X_train['inputs'].shape)
        history = self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1, shuffle=True,
                                validation_split=0.2,
                                callbacks=[cb_checkpoint, early_stopping])  # , class_weight=class_weights) 

    def train_wrmsse(self, train_gen, val_gen):
        self.model.compile(optimizer=self.optimizer, loss=wrmsse)
        model_path = './m5_predict5_wrmsse.h5'  # '{epoch:02d}-{val_loss:.4f}.h5'
        cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(patience=10)
        # print('lenght :', X_train['inputs'].shape)
        history = self.model.fit(train_gen, epochs=self.epochs, verbose=1, #shuffle=True,
                                validation_data = val_gen, #validation_split=0.2,
                               callbacks=[cb_checkpoint, early_stopping])  # , class_weight=class_weights) 




import pickle
import gzip
train_columns = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id', 'wday', 'month', 'year', \
                'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', \
                'snap_WI', 'sell_price', 'lag_7', 'lag_14', 'lag_21', 'lag_28', \
                'rmean_7_7', 'rmean_14_7', 'rmean_21_7', 'rmean_28_7', \
                'rmean_7_14', 'rmean_14_14', 'rmean_21_14', 'rmean_28_14', \
                'rmean_7_21', 'rmean_14_21', 'rmean_21_21', 'rmean_28_21', \
                'rmean_7_28', 'rmean_14_28', 'rmean_21_28', 'rmean_28_28', 'week', \
                'quarter', 'mday', 'sales_before_2', 'sales_before_3', \
                'sales_before_4', 'sales_before_5', 'sales_before_6']

def create_lag_features_for_test(dt, day):
    # create lag feaures just for single day (faster)
    lags = [7, 14, 21, 28]
    lag_cols = [f"lag_{lag}" for lag in lags]
    for lag, lag_col in zip(lags, lag_cols):        
        dt.loc[dt.date == day, lag_col] = \
            dt.loc[dt.date == day-timedelta(days=lag), 'sales'].values  # !!! main

    windows = [7, 14, 21, 28]
    for window in windows:
        for lag in lags:
            df_window = dt[(dt.date <= day-timedelta(days=lag)) & (dt.date > day-timedelta(days=lag+window))]
            df_window_grouped = df_window.groupby("id").agg({'sales':'mean'}).reindex(dt.loc[dt.date==day,'id'])
            dt.loc[dt.date == day,f"rmean_{lag}_{window}"] = df_window_grouped.sales.values   
    for i in range(6):
        dt['sales_before_' + str(i+1)] = dt.groupby(by='id')['sales'].shift(periods=i+1)

def create_date_features_for_test(dt):
    # copy of the code from `create_dt()` above
    date_features = {
        "wday": "weekday",
        "week": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
    }

    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in dt.columns:
            dt[date_feat_name] = dt[date_feat_name].astype("int16")
        else:
            dt[date_feat_name] = getattr(
                dt["date"].dt, date_feat_func).astype("int16")

input_dense = [ 'snap_CA', 'snap_TX', \
                'snap_WI', 'sell_price', 'lag_7', 'lag_14', 'lag_21', 'lag_28', \
                'rmean_7_7', 'rmean_14_7', 'rmean_21_7', 'rmean_28_7', \
                'rmean_7_14', 'rmean_14_14', 'rmean_21_14', 'rmean_28_14', \
                'rmean_7_21', 'rmean_14_21', 'rmean_21_21', 'rmean_28_21', \
                'rmean_7_28', 'rmean_14_28', 'rmean_21_28', 'rmean_28_28',
                'week', 'sales_before_2', 'sales_before_3', \
                'sales_before_4', 'sales_before_5', 'sales_before_6']
cat_cols = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id', 'wday', 'year', 'month',\
            'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'quarter', 'mday']                
                
def make_X_class(df):
    X = {'inputs': df[input_dense].to_numpy()}
    for i, v in enumerate(cat_cols):
        if v in ['wday', 'mday', 'quarter']:
            X[v] = df[[v]].to_numpy() - 1
        else:
            X[v] = df[[v]].to_numpy()
    return X

def make_X(df):
    X = {'inputs': df[input_dense].to_numpy()}
    for i, v in enumerate(cat_cols):
        if v in ['wday', 'mday', 'quarter']:
            X[v] = df[[v]].to_numpy() - 1
        else:
            X[v] = df[[v]].to_numpy()
    return X

def train(loss='mse'):
    try:
        df = pd.read_pickle('train_data_df.pkl')
    except:
        FIRST_DAY =  350# If you want to load all the data set it to '1' -->  Great  memory overflow  risk !

        df = create_dt(is_train=True, first_day= FIRST_DAY)
        print(df.info())
        create_fea(df)
        print(df.info())
        df.dropna(inplace = True)
        df.to_pickle('train_data_df.pkl')
        # del df; gc.collect()
    
    try:
        min_max_scaler = joblib.load('min_max_data.pkl')
    except:
        min_max_scaler = MinMaxScaler(copy=False)
        fitted = min_max_scaler.fit(df[input_dense + ['sales']])
        print(fitted.data_max_)
        joblib.dump(min_max_scaler, 'min_max_data.pkl')
    # normalization
    print('normalization', min_max_scaler.data_min_, min_max_scaler.data_max_)
    
    # df[input_dense + ['sales', 'sales_diff_year', 'sales_price_86']] = min_max_scaler.transform(df[input_dense + ['sales', 'sales_diff_year', 'sales_price_86']])
    df[input_dense + ['sales']] = (df[input_dense + ['sales']] - min_max_scaler.data_min_) /(min_max_scaler.data_max_-min_max_scaler.data_min_)
        
    if loss == 'mse':
        test = M5_predict(len(input_dense), 2**14)
        test.train_mse(make_X(df), df[['sales']].to_numpy())
    elif loss == 'rmsse':
        test = M5_predict(len(input_dense), 2**14)
        test.train_rmsse(make_X(df), df[['sales', 'sales_diff_year']].to_numpy())
    elif loss == 'wrmsse':
        test = M5_predict(len(input_dense))
        train_gen = DataGenerator(df)
        val_gen = DataGenerator(df[df.d > 'd_1870'])
        test.train_wrmsse(train_gen, val_gen)
    else:
        test = M5_predict_ensemble_model(len(input_dense))
        # train_gen = DataGenerator(df)
        # val_gen = DataGenerator(df[df.d > 'd_1870'])
        # test.train(train_gen, val_gen)
        test.train(make_X(df), df[['sales']].to_numpy())

def define_ensemble_model(input_size, lr=1e-3):
    
    inputs = Input(shape=(input_size, ), name='inputs')

    # Embedding input
    wday_input = Input(shape=(1,), name='wday')
    month_input = Input(shape=(1,), name='month')
    year_input = Input(shape=(1,), name='year')
    mday_input = Input(shape=(1,), name='mday')
    quarter_input = Input(shape=(1,), name='quarter')
    event_name_1_input = Input(shape=(1,), name='event_name_1')
    event_type_1_input = Input(shape=(1,), name='event_type_1')
    event_name_2_input = Input(shape=(1,), name='event_name_2')
    event_type_2_input = Input(shape=(1,), name='event_type_2')
    item_id_input = Input(shape=(1,), name='item_id')
    dept_id_input = Input(shape=(1,), name='dept_id')
    store_id_input = Input(shape=(1,), name='store_id')
    cat_id_input = Input(shape=(1,), name='cat_id')
    state_id_input = Input(shape=(1,), name='state_id')
    snap_CA_input = Input(shape=(1,), name='snap_CA')
    snap_TX_input = Input(shape=(1,), name='snap_TX')
    snap_WI_input = Input(shape=(1,), name='snap_WI')


    wday_emb = Flatten()(Embedding(7, 1)(wday_input))
    month_emb = Flatten()(Embedding(12, 2)(month_input))
    year_emb = Flatten()(Embedding(6, 1)(year_input))
    mday_emb = Flatten()(Embedding(31, 2)(mday_input))
    quarter_emb = Flatten()(Embedding(4, 1)(quarter_input))
    event_name_1_emb = Flatten()(Embedding(31, 2)(event_name_1_input))
    event_type_1_emb = Flatten()(Embedding(5, 1)(event_type_1_input))
    event_name_2_emb = Flatten()(Embedding(5, 1)(event_name_2_input))
    event_type_2_emb = Flatten()(Embedding(5, 1)(event_type_2_input))

    item_id_emb = Flatten()(Embedding(3049, 4)(item_id_input))
    dept_id_emb = Flatten()(Embedding(7, 1)(dept_id_input))
    store_id_emb = Flatten()(Embedding(10, 1)(store_id_input))
    cat_id_emb = Flatten()(Embedding(6, 1)(cat_id_input))
    state_id_emb = Flatten()(Embedding(3, 1)(state_id_input))

    x = Concatenate(-1)([inputs, wday_emb, month_emb, month_emb, year_emb, mday_emb, \
                        quarter_emb, event_name_1_emb, event_type_1_emb, event_name_2_emb, \
                        event_type_2_emb, item_id_emb, dept_id_emb, store_id_emb, cat_id_emb, \
                        state_id_emb])

    input_dic = {
        'inputs': inputs, 'wday': wday_input, 'month': month_input, 'year': year_input,
        'mday': mday_input, 'quarter': quarter_input, 'event_name_1': event_name_1_input,
        'event_type_1': event_type_1_input, 'event_name_2': event_name_2_input,
        'event_type_2': event_type_2_input, 'item_id': item_id_input, 'dept_id': dept_id_input,
        'store_id': store_id_input, 'cat_id': cat_id_input, 'state_id': state_id_input,

    }

    models = [predict_model(len(input_dense)), predict_model(len(input_dense)), predict_model(len(input_dense))]
    models[0].load_weights('./m5_predict5_mse.h5')
    models[1].load_weights('./m5_predict5_rmsse.h5')
    models[2].load_weights('./m5_predict5_wrmsse.h5')

    i = 0
    for model in models:
        for layer in model.layers:
            layer.trainable = False
            # layer.name = 'ensemble_' + str(i) + layer.name
        i += 1
    
    ensemble_outputs = [model(input_dic) for model in models]
    merge = Concatenate(-1)(ensemble_outputs)
    merge = Concatenate(-1)([x, merge])
    x = Dense(512, activation='relu')(merge)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(1, activation='relu')(x)
    x = Concatenate(-1)([x, merge])
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_dic, outputs=outputs)
    model.compile(optimizer=Adam(lr=lr), loss=rmse)
    return model

def test():    
    try:
        min_max_scaler = joblib.load('min_max_data.pkl')
    except:
        min_max_scaler = MinMaxScaler(copy=False)
        fitted = min_max_scaler.fit(df[input_dense + ['sales', 'sales_diff_year', 'sales_price_86']])
        print(fitted.data_max_)
        joblib.dump(min_max_scaler, 'min_max_data.pkl')

    test_0 = M5_predict_ensemble_model(len(input_dense), 0)
    test_0.model.load_weights('./m5_predict5_ensemble.h5')
    # test_0 = M5_predict_ensemble_model(len(input_dense))
    # test_0.model.load_weights('./m5_predict5_ensemble.h5')
    # test_1 = M5_predict(len(input_dense) + 1, 1)
    # test_1.model.load_weights('./m5_predict51.h5')
    # test_2 = M5_predict(len(input_dense) + 1, 2)
    # test_2.model.load_weights('./m5_predict52.h5')
    # class_ = M5_classifier(len(input_dense), 55)
    # class_.model.load_weights('./m5_predict555.h5')

    te0 = create_dt(False)  # create master copy of `te`

    create_date_features_for_test(te0)

    # for icount, (alpha, weight) in enumerate(zip(alphas, weights)):
    te = te0.copy()  # just copy
    # te['sales_diff_year'] = np.zeros(len(te))
    # te['sales_price_86'] = np.zeros(len(te))
    cols = [f"F{i}" for i in range(1, 29)]
    print(te[['date', 'd']])

    for tdelta in range(0, 28 + 28):
        day = fday + timedelta(days=tdelta)
        print(tdelta, day.date())
        tst = te[(te.date >= day - timedelta(days=max_lags))
                    & (te.date <= day)].copy()
        # create_fea(tst)  # correct, but takes much time
        create_lag_features_for_test(tst, day)  # faster  
        # tst = tst.loc[tst.date == day, train_columns]
        # tst['sales_before'] = tst['sales_before'] * 800        
        # print(test_0.model.predict(make_X(tst[(tst.date == day) & (tst.cat_id == 0)])).shape, len(te.loc[(te.date == day, "sales") & (te.cat_id == 0)]))
        
        # tst['class'] = np.zeros(len(tst))
        # tmp_np = class_.model.predict_on_batch(make_X_class(tst[tst.date == day])).numpy()
        # tmp_np[tmp_np < 0.5] = 0
        # tmp_np[tmp_np >= 0.5] = 1
        # tst.loc[tst[tst.date == day].index, 'class'] = tmp_np
        tmp_test_data = tst[(tst.date == day)]
        tmp_test_data.loc[tmp_test_data.index, input_dense + ['sales']] = min_max_scaler.transform(tmp_test_data[input_dense + ['sales']])
        tmp_test_data.loc[tmp_test_data.index, 'sales'] = test_0.model.predict(make_X(tmp_test_data))
        tmp_test_data.loc[tmp_test_data.index, input_dense + ['sales']] = min_max_scaler.inverse_transform(tmp_test_data[input_dense + ['sales']])
        te.loc[((te.date == day), "sales")] = tmp_test_data['sales']
        # te.loc[((te.date == day), "sales")] = test_0.model.predict(make_X(tst[(tst.date == day)])) * 800
        # te.loc[((te.date == day) & (te.cat_id == 0), "sales")] = test_0.model.predict(make_X(tst[(tst.date == day) & (tst.cat_id == 0)])) * 10
        # te.loc[((te.date == day) & (te.cat_id == 1), "sales")] = test_1.model.predict(make_X(tst[(tst.date == day) & (tst.cat_id == 1)])) * 10
        # te.loc[((te.date == day) & (te.cat_id == 2), "sales")] = test_2.model.predict(make_X(tst[(tst.date == day) & (tst.cat_id == 2)])) * 10
        # print(test.model.predict_on_batch(tst))

    print(te[['d', 'date']])
    #d_1914 ~ d_1941, d_1942 ~ d_1969
    id_list = te.id.unique().tolist()
    result1 = pd.DataFrame(id_list, columns=['id'])
    id_e_list = ['_'.join(dd.split('_')[:-1]) + '_evaluation' for dd in id_list]
    result2 = pd.DataFrame(id_e_list, columns=['id'])


    # for i in range(28):
    #     day_data = np.zeros(len(id_list))
    #     for idata in te[te.d == 'd_' + str(1914 + i)].itertuples():
    #         day_data[id_list.index(idata.id)] = idata.sales
    #     result1['F' + str(i + 1)] = day_data

    # for i in range(28):
    #     day_data = np.zeros(len(id_list))
    #     for idata in te[te.d == 'd_' + str(1942 + i)].itertuples():
    #         day_data[id_list.index(idata.id)] = idata.sales
    #     result2['F' + str(i + 1)] = day_data


    result1 = te.pivot(index='id', columns='d')['sales']
    result1['id'] = result1.index.values

    result_v = result1[['id'] + ['d_' + str(dd) for dd in range(1914, 1942)]]
    result_e = result1[['id'] + ['d_' + str(dd) for dd in range(1942, 1970)]]
    print(result_v)
    result_e["id"] = result_e["id"].str.replace("validation$", "evaluation")
    result_v = result_v.rename(columns={result_v.columns[1:][i]:'F'+str(i+1) for i in range(28)})
    result_e = result_e.rename(columns={result_e.columns[1:][i]:'F'+str(i+1) for i in range(28)})



    result = pd.concat([result_v, result_e], axis=0, sort=False)
    result.to_csv('submission.csv', index=False)




# train('mse')
# train('rmsse')
# train('wrmsse')
train('e')
# test()