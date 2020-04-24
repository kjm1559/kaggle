from  datetime import datetime, timedelta
import gc
import numpy as np, pandas as pd

CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "category", "year": "category", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }

h = 28 
max_lags = 57
tr_last = 1913
fday = datetime(2016,4, 25) 
# fday = datetime(2016,4, 24) 

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
    if is_train:
        # dt['sales_diff'] = dt.groupby(by='id')['sales'].diff().pow(2).shift(periods=1).rolling(28).sum() / 28
        dt['sales_diff'] = dt.groupby(by='id')['sales'].diff()
        dt['sales_diff'] = dt.groupby(by='id')['sales_diff'].shift(periods=1).pow(2)
        dt['sales_diff_year'] = dt.groupby(by='id')['sales_diff'].rolling(84).mean().reset_index(0, drop=True)
        # dt['sales_diff_56'] = dt.groupby(by='id')['sales_diff'].rolling(56).mean().reset_index(0, drop=True)
        # dt['sales_diff_28'] = dt.groupby(by='id')['sales_diff'].rolling(28).mean().reset_index(0, drop=True)
        # dt['sales_diff_7'] = dt.groupby(by='id')['sales_diff'].rolling(7).mean().reset_index(0, drop=True)
    # dt['sales_before'] = dt.groupby(by='id')['sales'].shift(periods=1) / 800
        # dt.loc[dt.sales_diff == 0, 'sales_diff'] = 1
    
    dt = dt.merge(cal, on= "d", copy = False)
    dt = dt.merge(prices, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)
    
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

    dt['sales_before'] = dt.groupby(by='id')['sales'].shift(periods=1) / 800
    
    
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


def rmsle(pred, true):
    assert pred.shape[0]==true.shape[0]
    return K.sqrt(K.mean(K.square(K.log(pred+1) - K.log(true + 1))))

def rmsse(true, pred):
    assert pred.shape[0]==true.shape[0]
    print(pred.shape, true[:,0:1].shape)
    loss_1 = K.mean(K.square(true[:, 0:1] - pred) / K.clip(true[:, 1:2], 1, 36000))
    # loss_7 = K.mean(K.square(true[:, 0:1] - pred) / K.clip(true[:, 2:3], 1, 36000))
    # loss_28 = K.mean(K.square(true[:, 0:1] - pred) / K.clip(true[:, 3:4], 1, 36000)) 
    # loss_56 = K.mean(K.square(true[:, 0:1] - pred) / K.clip(true[:, 4:5], 1, 36000)) 
    # return 0.1*loss_1 + 0.3*loss_7 + 0.6*loss_28
    return loss_1
    # return K.mean(K.square(true[:, 0:1] - pred))

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv1D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v2(inputs, depth=11, num_classes=1):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16      55, 16
    stage 0: 32x32,  64      55, 64
    stage 1: 16x16, 128      27, 128
    stage 2:  8x8,  256      13, 256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Reshape((inputs.shape[1], 1))(inputs)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling1D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='linear',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    # model = Model(inputs=inputs, outputs=outputs)
    # return model
    return outputs


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
    x_deep = Dense(32, activation='relu')(x_deep)
    x_deep = BatchNormalization()(x_deep)
    x_deep = Dense(16, activation='relu')(x_deep)

    x = Concatenate(-1)([inputs, x])
    x_res = Dense(64, activation='relu')(x)
    x_res = BatchNormalization()(x_res)
    x_res = Dense(32, activation='relu')(x_res)
    x_res = BatchNormalization()(x_res)
    x_res = Dense(16, activation='relu')(x_res)

    x = Concatenate(-1)([x_deep, x_res])

    x = BatchNormalization()(x)    
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)    
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)    
    x = Dense(16, activation='relu')(x)
    x = Concatenate(-1)([inputs, x])
    outputs = Dense(1, activation='linear')(x)
    # outputs = resnet_v2(x)
    
    optimizer = Adam(lr=lr)#Adam(lr=lr)
    input_dic = {
        'inputs': inputs, 'wday': wday_input, 'month': month_input, 'year': year_input,
        'mday': mday_input, 'quarter': quarter_input, 'event_name_1': event_name_1_input,
        'event_type_1': event_type_1_input, 'event_name_2': event_name_2_input,
        'event_type_2': event_type_2_input, 'item_id': item_id_input, 'dept_id': dept_id_input,
        'store_id': store_id_input, 'cat_id': cat_id_input, 'state_id': state_id_input,

    }
    model = Model(input_dic, outputs, name='predict_model')
    model.compile(optimizer=optimizer, loss=rmsse)
    return model    

class M5_predict:
    def __init__(self, input_size, batch_size=2**14, epochs=200):
        self.batch_size = batch_size
        self.epochs = epochs        
        self.model = predict_model(input_size)
        
    def train(self, X_train, y_train):
        model_path = './m5_predict5.h5'  # '{epoch:02d}-{val_loss:.4f}.h5'
        cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(patience=10)
        
        history = self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1, shuffle=True,
                              validation_split=0.2,
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
                'quarter', 'mday', 'sales_before']

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
    dt['sales_before'] = dt.groupby(by='id')['sales'].shift(periods=1) / 800

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
                'rmean_7_28', 'rmean_14_28', 'rmean_21_28', 'rmean_28_28', \
                'week', 'sales_before']
cat_cols = ['item_id', 'dept_id', 'store_id', 'cat_id', 'state_id', 'wday', 'month', 'year',\
            'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'quarter', 'mday']                
                
def make_X(df):
    X = {'inputs': df[input_dense].to_numpy()}
    for i, v in enumerate(cat_cols):
        if v in ['wday', 'mday', 'quarter']:
            X[v] = df[[v]].to_numpy() - 1
        else:
            X[v] = df[[v]].to_numpy()
    return X

def train():
    try:
        # with gzip.open('train_data.pkl', 'rb') as f:
        #     X_train, y_train, max_data, max_label = pickle.load(f)
        df = pd.read_pickle('train_data_df.pkl')

    except:
        FIRST_DAY =  1# If you want to load all the data set it to '1' -->  Great  memory overflow  risk !

        df = create_dt(is_train=True, first_day= FIRST_DAY)
        create_fea(df)
        df.dropna(inplace = True)

        # X_train = df[train_columns].values
        # y_train = df[['sales', 'sales_diff']].values
        # max_data = df[train_columns].max().tolist()
        # max_label = df[['sales']].max().tolist()
        # X_train = X_train / max_data
        # y_train = y_train / max_label
        # with gzip.open('train_data.pkl', 'wb') as f:
        #     pickle.dump([X_train, y_train, max_data, max_label], f, protocol=4)
        # del df; gc.collect()
        df.to_pickle('train_data_df.pkl')

    # print(y_train.shape)

    test = M5_predict(len(input_dense))
    # test.model.load_weights('./m5_predict4.h5')
    # df['sell_price'] = df['sell_price'] / 110
    # df['rmean_7_7'] = df['rmean_7_7'] / 603
    # df['rmean_7_7'] = df['rmean_28_7'] / 603
    # df['rmean_7_7'] = df['rmean_7_28'] / 443
    # df['rmean_7_7'] = df['rmean_28_28'] / 443
    # df['sales_before'] = df['sales_before'] * 800
    test.train(make_X(df), df[['sales', 'sales_diff_year']].to_numpy())

def test():    
    with gzip.open('train_data.pkl', 'rb') as f:
        X_train, y_train, max_data, max_label = pickle.load(f)

    test = M5_predict(len(input_dense))
    test.model.load_weights('./m5_predict5.h5')

    te0 = create_dt(False)  # create master copy of `te`

    create_date_features_for_test(te0)

    # for icount, (alpha, weight) in enumerate(zip(alphas, weights)):
    te = te0.copy()  # just copy
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
        te.loc[te.date == day, "sales"] = test.model.predict(make_X(tst[tst.date == day]))
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


test()
# train()