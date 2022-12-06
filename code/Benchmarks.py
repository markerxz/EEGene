import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,f1_score
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, BatchNormalization, Activation, MaxPooling2D, DepthwiseConv2D, AveragePooling2D, SeparableConv2D
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import numpy as np
import os
import time
from sklearn.metrics import classification_report, f1_score,accuracy_score
from min2net.utils import TimeHistory, compute_class_weight
class EEGNet:
    def __init__(self,
                input_shape=(1,20,400),
                num_class=2,
                loss='sparse_categorical_crossentropy',
                epochs=200,
                batch_size=100,
                optimizer = Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                lr=0.01,
                min_lr=0.01,
                factor=0.25,
                patience=10,
                es_patience=20,
                verbose=1,
                log_path='log',
                model_name='EEGNet',
                **kwargs):
        self.input_shape = input_shape
        self.num_class = num_class
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.optimizer.lr = lr
        self.lr = lr
        self.min_lr = min_lr
        self.factor = factor
        self.patience = patience
        self.es_patience = es_patience
        self.verbose = verbose
        self.log_path = log_path
        self.model_name = model_name
        self.weights_dir = log_path+'/'+model_name+'_out_weights.tf'
        self.csv_dir = log_path+'/'+model_name+'_out_log.log'
        self.time_log = log_path+'/'+model_name+'_time_log.csv'

        # use **kwargs to set the new value of below args.
        self.kernLength = 200
        self.F1 = 8
        self.D = 2
        self.F2 = int(self.F1*self.D)
        self.norm_rate = 0.25
        self.dropout_rate = 0.5
        self.f1_average = 'binary' if self.num_class == 2 else 'macro'
        self.data_format = 'channels_first'
        self.shuffle = False
        self.metrics = 'accuracy'
        self.monitor = 'val_loss'
        self.mode = 'min'
        self.save_best_only = True
        self.save_weight_only = True
        self.seed = 1234
        self.class_balancing = False
        self.class_weight = None

        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])
            
        if self.data_format == 'channels_first':
            self.Chans = self.input_shape[1]
            self.Samples = self.input_shape[2]
        else:
            self.Chans = self.input_shape[0]
            self.Samples = self.input_shape[1]

        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        K.set_image_data_format(self.data_format)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def build(self):
        input1       = Input(shape=self.input_shape)

        ##################################################################
        block1       = Conv2D(self.F1, (1, self.kernLength), padding='same',
                              input_shape=self.input_shape,
                              use_bias=False)(input1)
        block1       = BatchNormalization()(block1)
        block1       = DepthwiseConv2D((self.Chans, 1), use_bias=False,
                                       depth_multiplier=self.D,
                                       depthwise_constraint=max_norm(1.))(block1)
        block1       = BatchNormalization()(block1)
        block1       = Activation('elu')(block1)
        block1       = AveragePooling2D((1, 4))(block1)
        block1       = Dropout(self.dropout_rate)(block1)

        block2       = SeparableConv2D(self.F2, (1, self.kernLength//4),
                                       use_bias=False, padding='same')(block1)
        block2       = BatchNormalization()(block2)
        block2       = Activation('elu')(block2)
        block2       = AveragePooling2D((1, 8))(block2)
        block2       = Dropout(self.dropout_rate)(block2)

        flatten      = Flatten(name='flatten')(block2)

        dense        = Dense(self.num_class, name='dense',
                             kernel_constraint=max_norm(self.norm_rate))(flatten)
        softmax      = Activation('softmax', name='softmax')(dense)

        return Model(inputs=input1, outputs=softmax)

    def fit(self, X_train, y_train, X_val, y_val):

        if X_train.ndim != 4:
            raise Exception('ValueError: `X_train` is incompatible: expected ndim=4, found ndim='+str(X_train.ndim))
        elif X_val.ndim != 4:
            raise Exception('ValueError: `X_val` is incompatible: expected ndim=4, found ndim='+str(X_val.ndim))

        self.input_shape = X_train.shape[1:]
        if self.data_format == 'channels_first':
            self.Chans = self.input_shape[1]
            self.Samples = self.input_shape[2]
        else:
            self.Chans = self.input_shape[0]
            self.Samples = self.input_shape[1]
        
        csv_logger = CSVLogger(self.csv_dir)
        time_callback = TimeHistory(self.time_log)
        checkpointer = ModelCheckpoint(monitor=self.monitor, filepath=self.weights_dir, verbose=self.verbose, 
                                       save_best_only=self.save_best_only, save_weight_only=self.save_weight_only)
        reduce_lr = ReduceLROnPlateau(monitor=self.monitor, patience=self.patience, factor=self.factor, 
                                      mode=self.mode, verbose=self.verbose, min_lr=self.min_lr)
        es = EarlyStopping(monitor=self.monitor, mode=self.mode, verbose=self.verbose, patience=self.es_patience)

        model = self.build()
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        model.summary()
        print("The first kernel size is (1, {})".format(self.kernLength))
        
        if self.class_balancing: # compute_class_weight if class_balancing is True
            self.class_weight = compute_class_weight(y_train)
        else:
            self.class_weight = None         
            
        model.fit(X_train, y_train,
                  batch_size=self.batch_size, shuffle=self.shuffle,
                  epochs=self.epochs, validation_data=(X_val, y_val), class_weight=self.class_weight,
                  callbacks=[checkpointer,csv_logger,reduce_lr,es, time_callback])

    def predict(self, X_test, y_test):

        if X_test.ndim != 4:
            raise Exception('ValueError: `X_test` is incompatible: expected ndim=4, found ndim='+str(X_test.ndim))

        model = self.build()
        model.load_weights(self.weights_dir)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        start = time.time()
        y_pred = model.predict(X_test)
        end = time.time()
        loss, accuracy = model.evaluate(x=X_test, y=y_test, batch_size=self.batch_size, verbose=self.verbose)
        y_pred_argm = np.argmax(y_pred, axis=1)
        print(classification_report(y_test, y_pred_argm))
        print("F1-score is computed based on {}".format(self.f1_average))
        f1 = f1_score(y_test, y_pred_argm, average=self.f1_average)
        evaluation = {'loss': loss, 
                      'accuracy': accuracy, 
                      'f1-score': f1, 
                      'prediction_time': end-start}
        Y = {'y_true': y_test, 
             'y_pred': y_pred_argm}
        return Y, evaluation

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import numpy as np
import os
import time
from sklearn.metrics import classification_report, f1_score
from min2net.utils import TimeHistory, compute_class_weight

class DeepConvNet:
    def __init__(self,
                input_shape=(1,20,400),
                num_class=2,
                loss='sparse_categorical_crossentropy',
                epochs=200,
                batch_size=100,
                optimizer = Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                lr=0.01,
                min_lr=0.01,
                factor=0.25,
                patience=10,
                es_patience=20,
                verbose=1,
                log_path='log',
                model_name='DeepConvNet',
                **kwargs):
        self.input_shape = input_shape
        self.num_class = num_class
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.optimizer.lr = lr
        self.lr = lr
        self.min_lr = min_lr
        self.factor = factor
        self.patience = patience
        self.es_patience = es_patience
        self.verbose = verbose
        self.log_path = log_path
        self.model_name = model_name
        self.weights_dir = log_path+'/'+model_name+'_out_weights.tf'
        self.csv_dir = log_path+'/'+model_name+'_out_log.log'
        self.time_log = log_path+'/'+model_name+'_time_log.csv'

        # use **kwargs to set the new value of below args.
        self.kernLength = 125
        self.F1 = 8
        self.D = 2
        self.F2 = int(self.F1*self.D)
        self.norm_rate = 0.25
        self.dropout_rate = 0.5
        self.f1_average = 'binary' if self.num_class == 2 else 'macro'
        self.data_format = 'channels_first'
        self.shuffle = False
        self.metrics = 'accuracy'
        self.monitor = 'val_loss'
        self.mode = 'min'
        self.save_best_only = True
        self.save_weight_only = True
        self.seed = 1234
        self.class_balancing = False
        self.class_weight = None

        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])
            
        if self.data_format == 'channels_first':
            self.Chans = self.input_shape[1]
            self.Samples = self.input_shape[2]
        else:
            self.Chans = self.input_shape[0]
            self.Samples = self.input_shape[1]

        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        K.set_image_data_format(self.data_format)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def build(self):
        """ Keras implementation of the Deep Convolutional Network as described in
        Schirrmeister et. al. (2017), Human Brain Mapping.
        This implementation assumes the input is a 2-second EEG signal sampled at
        128Hz, as opposed to signals sampled at 250Hz as described in the original
        paper. We also perform temporal convolutions of length (1, 5) as opposed
        to (1, 10) due to this sampling rate difference.
        Note that we use the max_norm constraint on all convolutional layers, as
        well as the classification layer. We also change the defaults for the
        BatchNormalization layer. We used this based on a personal communication
        with the original authors.
                          ours        original paper
        pool_size        1, 2        1, 3
        strides          1, 2        1, 3
        conv filters     1, 5        1, 10
        Note that this implementation has not been verified by the original
        authors.
        """

        # start the model
        input_main   = Input(self.input_shape)
        block1       = Conv2D(25, (1, 5),
                              input_shape=(self.input_shape),
                              kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
        block1       = Conv2D(25, (self.Chans, 1),
                              kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
        block1       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
        block1       = Activation('elu')(block1)
        block1       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
        block1       = Dropout(self.dropout_rate)(block1)

        block2       = Conv2D(50, (1, 5),
                              kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
        block2       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
        block2       = Activation('elu')(block2)
        block2       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
        block2       = Dropout(self.dropout_rate)(block2)

        block3       = Conv2D(100, (1, 5),
                              kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
        block3       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
        block3       = Activation('elu')(block3)
        block3       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
        block3       = Dropout(self.dropout_rate)(block3)

        block4       = Conv2D(200, (1, 5),
                              kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
        block4       = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
        block4       = Activation('elu')(block4)
        block4       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
        block4       = Dropout(self.dropout_rate)(block4)

        flatten      = Flatten()(block4)

        dense        = Dense(self.num_class, kernel_constraint = max_norm(0.5))(flatten)
        softmax      = Activation('softmax')(dense)

        return Model(inputs=input_main, outputs=softmax)

    def fit(self, X_train, y_train, X_val, y_val):

        if X_train.ndim != 4:
            raise Exception('ValueError: `X_train` is incompatible: expected ndim=4, found ndim='+str(X_train.ndim))
        elif X_val.ndim != 4:
            raise Exception('ValueError: `X_val` is incompatible: expected ndim=4, found ndim='+str(X_val.ndim))

        self.input_shape = X_train.shape[1:]
        if self.data_format == 'channels_first':
            self.Chans = self.input_shape[1]
            self.Samples = self.input_shape[2]
        else:
            self.Chans = self.input_shape[0]
            self.Samples = self.input_shape[1]
        
        csv_logger = CSVLogger(self.csv_dir)
        time_callback = TimeHistory(self.time_log)
        checkpointer = ModelCheckpoint(monitor=self.monitor, filepath=self.weights_dir, verbose=self.verbose, 
                                       save_best_only=self.save_best_only, save_weight_only=self.save_weight_only)
        reduce_lr = ReduceLROnPlateau(monitor=self.monitor, patience=self.patience, factor=self.factor, mode=self.mode, 
                                      verbose=self.verbose, min_lr=self.min_lr)
        es = EarlyStopping(monitor=self.monitor, mode=self.mode, verbose=self.verbose, patience=self.es_patience)

        model = self.build()
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        model.summary()
        
        if self.class_balancing: # compute_class_weight if class_balancing is True
            self.class_weight = compute_class_weight(y_train)
        else:
            self.class_weight = None
            
        model.fit(X_train, y_train,
                  batch_size=self.batch_size, shuffle=self.shuffle,
                  epochs=self.epochs, validation_data=(X_val, y_val), class_weight=self.class_weight,
                  callbacks=[checkpointer,csv_logger,reduce_lr,es, time_callback])

    def predict(self, X_test, y_test):

        if X_test.ndim != 4:
            raise Exception('ValueError: `X_test` is incompatible: expected ndim=4, found ndim='+str(X_test.ndim))

        model = self.build()
        model.load_weights(self.weights_dir)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        start = time.time()
        y_pred = model.predict(X_test)
        end = time.time()
        loss, accuracy = model.evaluate(x=X_test, y=y_test, batch_size=self.batch_size, verbose=self.verbose)
        y_pred_argm = np.argmax(y_pred, axis=1)
        print(classification_report(y_test, y_pred_argm))
        print("F1-score is computed based on {}".format(self.f1_average))
        f1 = f1_score(y_test, y_pred_argm, average=self.f1_average)
        evaluation = {'loss': loss, 
                      'accuracy': accuracy, 
                      'f1-score': f1, 
                      'prediction_time': end-start}
        Y = {'y_true': y_test, 
             'y_pred': y_pred_argm}
        return Y, evaluation
from tensorflow.keras.layers import Concatenate, AveragePooling2D, BatchNormalization, Conv2D, Conv2DTranspose, Dense, Input, Reshape, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import backend as K

import numpy as np
import os
import time
from sklearn.metrics import classification_report, f1_score
from min2net.loss import mean_squared_error, triplet_loss, SparseCategoricalCrossentropy
from min2net.utils import TimeHistory, compute_class_weight

class MIN2Net:
    def __init__(self,
                input_shape=(1,400,20), 
                num_class=2, 
                loss=[mean_squared_error, triplet_loss(margin=1.0), 'sparse_categorical_crossentropy'],
                loss_weights=[1., 1., 1.], 
                latent_dim = None,
                epochs=200,
                batch_size=100,
                optimizer=Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                lr=1e-2,
                min_lr=1e-3,
                factor=0.5,
                patience=5, 
                es_patience=20,
                verbose=1,
                log_path='log',
                model_name='MIN2Net', 
                **kwargs):
        D, T, C = input_shape
        self.latent_dim = latent_dim if latent_dim is not None else C if num_class==2 else 64
        self.num_class = num_class
        self.input_shape = input_shape
        self.loss = loss
        self.loss_weights = loss_weights
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.optimizer.lr = lr
        self.lr = lr
        self.min_lr = min_lr
        self.factor = factor
        self.patience = patience
        self.es_patience = es_patience
        self.verbose = verbose
        self.log_path = log_path
        self.model_name = model_name
        self.weights_dir = log_path+'/'+model_name+'_out_weights.tf'
        self.csv_dir = log_path+'/'+model_name+'_out_log.log'
        self.time_log = log_path+'/'+model_name+'_time_log.csv'

        # use **kwargs to set the new value of below args.
        self.f1_average = 'binary' if self.num_class == 2 else 'macro'
        self.data_format = 'channels_last'
        self.shuffle = False
        self.metrics = 'accuracy'
        self.monitor = 'val_loss'
        self.mode = 'min'
        self.save_best_only = True
        self.save_weight_only = True
        self.seed = 1234
        self.class_balancing = False
        # 'set params'
        self.subsampling_size = 100
        self.pool_size_1 = (1,T//self.subsampling_size)
        self.pool_size_2 = (1,4)
        self.filter_1 = C
        self.filter_2 = 10
        
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])
        
        self.flatten_size = T//self.pool_size_1[1]//self.pool_size_2[1]
        
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        K.set_image_data_format(self.data_format)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def build(self):
        'encoder'
        encoder_input  = Input(self.input_shape)
        en_conv        = Conv2D(self.filter_1, (1, 64), activation='elu', padding="same", 
                                kernel_constraint=max_norm(2., axis=(0, 1, 2)))(encoder_input)
        en_conv        = BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1)(en_conv)
        en_conv        = AveragePooling2D(pool_size=self.pool_size_1)(en_conv)  
        en_conv        = Conv2D(self.filter_2, (1, 32), activation='elu', padding="same", 
                                kernel_constraint=max_norm(2., axis=(0, 1, 2)))(en_conv)
        en_conv        = BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1)(en_conv)
        en_conv        = AveragePooling2D(pool_size=self.pool_size_2)(en_conv)
        en_conv        = Flatten()(en_conv)
        encoder_output = Dense(self.latent_dim, kernel_constraint=max_norm(0.5))(en_conv)
        encoder        = Model(inputs=encoder_input, outputs=encoder_output, name='encoder')
        encoder.summary()
        
        'decoder'
        decoder_input  = Input(shape=(self.latent_dim,), name='decoder_input')
        de_conv        = Dense(1*self.flatten_size*self.filter_2, activation='elu', 
                               kernel_constraint=max_norm(0.5))(decoder_input)
        de_conv        = Reshape((1, self.flatten_size, self.filter_2))(de_conv)
        de_conv        = Conv2DTranspose(filters=self.filter_2, kernel_size=(1, 64), 
                                         activation='elu', padding='same', strides=self.pool_size_2, 
                                         kernel_constraint=max_norm(2., axis=(0, 1, 2)))(de_conv)
        decoder_output = Conv2DTranspose(filters=self.filter_1, kernel_size=(1, 32), 
                                         activation='elu', padding='same', strides=self.pool_size_1, 
                                         kernel_constraint=max_norm(2., axis=(0, 1, 2)))(de_conv)
        decoder        = Model(inputs=decoder_input, outputs=decoder_output, name='decoder')
        decoder.summary()

        'Build the computation graph for training'
        latent         = encoder(encoder_input)
        train_xr       = decoder(latent)
        z              = Dense(self.num_class, activation='softmax', kernel_constraint=max_norm(0.5), 
                               name='classifier')(latent)

        return Model(inputs=encoder_input, outputs=[train_xr, latent, z], 
                            name=self.model_name)
    
    def fit(self, X_train, y_train, X_val, y_val):
            
        if X_train.ndim != 4:
            raise Exception('ValueError: `X_train` is incompatible: expected ndim=4, found ndim='+str(X_train.ndim))
        elif X_val.ndim != 4:
            raise Exception('ValueError: `X_val` is incompatible: expected ndim=4, found ndim='+str(X_val.ndim))

        csv_logger    = CSVLogger(self.csv_dir)
        time_callback = TimeHistory(self.time_log)
        checkpointer  = ModelCheckpoint(monitor=self.monitor, filepath=self.weights_dir, 
                                        verbose=self.verbose, save_best_only=self.save_best_only, 
                                        save_weight_only=self.save_weight_only)
        reduce_lr     = ReduceLROnPlateau(monitor=self.monitor, patience=self.patience, 
                                          factor=self.factor, mode=self.mode, verbose=self.verbose, 
                                          min_lr=self.min_lr)
        es            = EarlyStopping(monitor=self.monitor, mode=self.mode, verbose=self.verbose, 
                                      patience=self.es_patience)
        model = self.build()     
        model.summary()
        
        if self.class_balancing: # compute_class_weight if class_balancing is True
            class_weight  = compute_class_weight(y_train)
            self.loss[-1] = SparseCategoricalCrossentropy(class_weight=class_weight)
        
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics, loss_weights=self.loss_weights)

        model.fit(x=X_train, y=[X_train,y_train,y_train],
                          batch_size=self.batch_size, shuffle=self.shuffle,
                          epochs=self.epochs, validation_data=(X_val, [X_val,y_val,y_val]),
                          callbacks=[checkpointer,csv_logger,reduce_lr,es, time_callback])
        
    def predict(self, X_test, y_test):

        if X_test.ndim != 4:
            raise Exception('ValueError: `X_test` is incompatible: expected ndim=4, found ndim='+str(X_test.ndim))

        model = self.build()
        model.summary()
        model.load_weights(self.weights_dir)
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics, loss_weights=self.loss_weights)

        start = time.time()
        y_pred_decoder, y_pred_trip, y_pred_clf = model.predict(X_test)
        end = time.time()
        loss, decoder_loss, trip_loss, classifier_loss, decoder_acc, trip_acc, classifier_acc  = model.evaluate(x=X_test,
                                                                                                                y=[X_test,y_test,y_test],
                                                                                                                batch_size=self.batch_size, 
                                                                                                                verbose=self.verbose)
        y_pred_argm = np.argmax(y_pred_clf, axis=1)
        print("F1-score is computed based on {}".format(self.f1_average))
        f1 = f1_score(y_test, y_pred_argm, average=self.f1_average)
        print('(loss: {}, accuracy: {})'.format(loss, classifier_acc))
        print(classification_report(y_test, y_pred_argm))
        evaluation = {'loss': loss, 
                      'decoder_loss': decoder_loss, 
                      'triplet_loss':trip_loss, 
                      'classifier_loss': classifier_loss, 
                      'accuracy': classifier_acc,
                      'f1-score': f1 ,
                      'prediction_time': end-start}
        Y = {'y_true': y_test,
             'y_pred': y_pred_argm,
             'y_pred_decoder': y_pred_decoder}

        return Y, evaluation