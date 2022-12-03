class Data_Loader:
    # def __init__(self,):
    def load_muse(self,sub=-1):
        import numpy as np
        import scipy
        import scipy.signal

        def matrix_from_csv_file(file_path):
            csv_data = np.genfromtxt(file_path, delimiter = ',')
            full_matrix = csv_data[1:]
            return full_matrix
        def get_time_slice(full_matrix, start = 0., period = 1.):
            rstart  = full_matrix[0, 0] + start
            index_0 = np.max(np.where(full_matrix[:, 0] <= rstart))
            index_1 = np.max(np.where(full_matrix[:, 0] <= rstart + period))
            duration = full_matrix[index_1, 0] - full_matrix[index_0, 0]
            return full_matrix[index_0:index_1, :], duration
        def generate_feature_vectors_from_samples(file_path, nsamples, period, 
                                                  state = None, 
                                                  remove_redundant = True,
                                                  cols_to_ignore = None):
            matrix = matrix_from_csv_file(file_path)
            t = 0.
            previous_vector = None
            ret = None
            arr = []
            while True:
                try:
                    s, dur = get_time_slice(matrix, start = t, period = period)
                    if cols_to_ignore is not None:
                        s = np.delete(s, cols_to_ignore, axis = 1)
                except IndexError:
                    break
                if len(s) == 0:
                    break
                if dur < 0.9 * period:
                    break
                ry, rx = scipy.signal.resample(s[:, 1:], num = nsamples, t = s[:, 0], axis = 0)
                arr.append(ry.T)
                t += 0.5 * period
            return np.array(arr)
        def gen_training_matrix(directory_path, cols_to_ignore,sub='a'):
            import os, sys
            import numpy as np
            # Initialise return matrix
            X = None
            y = None
            for x in os.listdir(directory_path):
                #print('X = ',x)
                # Ignore non-CSV files
                if not x.lower().endswith('.csv'):
                    continue
                if x[7] != sub:
                    continue
                # For safety we'll ignore files containing the substring "test". 
                # [Test files should not be in the dataset directory in the first place]
                if 'test' in x.lower():
                    continue
                try:
                    name, state, _ = x[:-4].split('-')
                except:
                    print ('Wrong file name', x)
                    sys.exit(-1)
                if state.lower() == 'concentrating':
                    state = 1.
                elif state.lower() == 'neutral':
                    state = 2.
                elif state.lower() == 'relaxed':
                    state = 0.
                else:
                    print ('Wrong file name', x)
                    sys.exit(-1)

                #print ('Using file', x)
                full_file_path = directory_path  +   '/'   + x
                arr = generate_feature_vectors_from_samples(file_path = full_file_path, 
                                                                        nsamples = 150, 
                                                                        period = 1.,
                                                                        state = state,
                                                                        remove_redundant = True,
                                                                        cols_to_ignore = cols_to_ignore)

                if state != 2:
                    if type(X) == np.ndarray:
                        X = np.concatenate((X,arr))
                        #print('Xshape = ',X.shape)
                        y = np.concatenate((y,np.array([state for i in range(len(arr))])))
                        #print('yshape = ',y.shape)
                        #print()

                    else:
                        X = arr
                        y = np.array([state for i in range(len(arr))])

            return X,y
        if sub == -1:
            X = []
            y = []
            for insub in 'abcd':
                Xs,ys = gen_training_matrix('kaggle/dataset/muse', cols_to_ignore = -1,sub=insub)
                Xs = np.array(Xs)[:,:,:148]
                X.append(Xs)
                y.append(ys)
        else:
            X,y = gen_training_matrix('kaggle/dataset/muse', cols_to_ignore = -1,sub=sub)
            X = np.array(X)[:,:,:148]
        
        return X,y

        def gen_training_matrix(directory_path, cols_to_ignore,sub='a'):
            import os, sys
            import numpy as np
            # Initialise return matrix
            X = None
            y = None
            for x in os.listdir(directory_path):
                #print('X = ',x)
                # Ignore non-CSV files
                if not x.lower().endswith('.csv'):
                    continue
                if x[7] != sub:
                    continue
                # For safety we'll ignore files containing the substring "test". 
                # [Test files should not be in the dataset directory in the first place]
                if 'test' in x.lower():
                    continue
                try:
                    name, state, _ = x[:-4].split('-')
                except:
                    print ('Wrong file name', x)
                    sys.exit(-1)
                if state.lower() == 'concentrating':
                    state = 1.
                elif state.lower() == 'neutral':
                    state = 2.
                elif state.lower() == 'relaxed':
                    state = 0.
                else:
                    print ('Wrong file name', x)
                    sys.exit(-1)

                #print ('Using file', x)
                full_file_path = directory_path  +   '/'   + x
                arr = generate_feature_vectors_from_samples(file_path = full_file_path, 
                                                                        nsamples = 150, 
                                                                        period = 1.,
                                                                        state = state,
                                                                        remove_redundant = True,
                                                                        cols_to_ignore = cols_to_ignore)
                arr = np.array(arr)[:,:,:148]
                if state != 2:
                    if type(X) == np.ndarray:
                        X = np.concatenate((X,arr))
                        #print('Xshape = ',X.shape)
                        y = np.concatenate((y,np.array([state for i in range(len(arr))])))
                        #print('yshape = ',y.shape)
                        #print()

                    else:
                        X = arr
                        y = np.array([state for i in range(len(arr))])

            return X,y


    def load_seizure(self,size = 100):
        import numpy as np
        import pandas as pd 

        npz_file = np.load('kaggle/dataset/seizure/eeg-seizure_train.npz', allow_pickle=True)
        signals_train = npz_file['train_signals']
        labels_train = npz_file['train_labels']
        def checkZero(X):
            for i in range(X.shape[0]):
                if X[i][0] == X[i][-1]:
                    return False
            return True
        cc = [0,0]
        X = None
        y = None
        i = 0
        cap = int(size/2)
        while cc[0] < cap or cc[1] < cap:
            if cc[labels_train[i]] + 1 <= cap and checkZero(signals_train[i]):
                cc[labels_train[i]] = cc[labels_train[i]] + 1
                if type(X) == np.ndarray:
                    X = np.concatenate((X,[signals_train[i]]))
                    y = np.concatenate((y,[labels_train[i]]))
                else:
                    X = np.array([signals_train[i]])
                    y = np.array([labels_train[i]])
            i+=1    
        return X,y


    def load_alpha(self,fmin=0.5,fmax=50,sfreq=100):
        from scipy.io import loadmat
        import mne
        import numpy as np
        from scipy.io import loadmat
        def _get_single_subject_data(subject):
            """return data for a single subject"""
            if subject < 10:
                sub = '0' + str(subject)
            else:
                sub = str(subject)
            filepath = 'moabb/alpha/dataset/subject_'+sub+'.mat'
            data = loadmat(filepath)

            S = data['SIGNAL'][:, 1:17]
            stim_close = data['SIGNAL'][:, 17]
            stim_open = data['SIGNAL'][:, 18]
            stim = 1 * stim_close + 2 * stim_open

            chnames = [
                'Fp1',
                'Fp2',
                'Fc5',
                'Fz',
                'Fc6',
                'T7',
                'Cz',
                'T8',
                'P7',
                'P3',
                'Pz',
                'P4',
                'P8',
                'O1',
                'Oz',
                'O2',
                'stim']
            chtypes = ['eeg'] * 16 + ['stim']
            X = np.concatenate([S, stim[:, None]], axis=1).T

            info = mne.create_info(ch_names=chnames, sfreq=512,
                                   ch_types=chtypes,
                                   verbose=False)
            raw = mne.io.RawArray(data=X, info=info, verbose=False)

            return raw
        Subject_list = [1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19]
        XX = np.zeros((18,10,16,601))
        y = np.zeros((18,10))
        for i,s in enumerate(Subject_list):
            raw = _get_single_subject_data(s)
            raw.filter(fmin, fmax, verbose=False)
            raw.resample(sfreq=sfreq, verbose=False)

            # detect the events and cut the signal into epochs
            events = mne.find_events(raw=raw, shortest_event=1, verbose=False)
            event_id = {'closed': 1, 'open': 2}
            epochs = mne.Epochs(raw, events, event_id, tmin=2.0, tmax=8.0, baseline=None,
                                verbose=False, preload=True)
            epochs.pick_types(eeg=True)

            # get trials and labels
            XX[i] = epochs.get_data()
            y[i] = events[:, -1]
        for sub in range(18):
            for i in range(10):
                y[sub][i] = float(y[sub][i] == 1)
        X = np.zeros((18,10,16,600))
        for sub in range(18):
            for trial in range(10):
                for ch in range(16):
                    X[sub][trial][ch] = XX[sub][trial][ch][:-1]
        return X,y
    
    def load_bcic(self,sub = -1,fmin=8,fmax=13,fsamp = 100):
        import moabb
        import numpy as np
        from moabb.paradigms import LeftRightImagery
        from moabb.datasets import BNCI2014001
        BNCI001 = BNCI2014001()
        paradigm = LeftRightImagery(resample = fsamp, fmin = fmin, fmax = fmax)
        Subject_list = BNCI001.subject_list
        
        if sub == -1:
            Xall = []
            yall = []
            for i in range(len(Subject_list)):
                X,ytemp,metadata = paradigm.get_data(dataset=BNCI001,subjects=[Subject_list[i]])
                y = np.zeros(X.shape[0])
                for k in range(X.shape[0]):
                    y[k] = float(ytemp[k] == 'left_hand')
                Xall.append(X)
                yall.append(y)
            return Xall,yall
        else:
            X,ytemp,metadata = paradigm.get_data(dataset=BNCI001,subjects=[sub])
            y = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                y[i] = float(ytemp[i] == 'left_hand')
            return X,y
        
    def load_smr(self,sub = -1,fmin=8,fmax=13,fsamp = 100):
        import moabb
        import numpy as np
        from moabb.paradigms import MotorImagery
        from moabb.datasets import BNCI2014002
        BNCI002 = BNCI2014002()
        paradigm = MotorImagery(resample = fsamp, fmin = fmin, fmax = fmax)
        Subject_list = BNCI002.subject_list
               
        if sub == -1:
            Xall = []
            yall = []
            for i in range(len(Subject_list)):
                X,ytemp,metadata = paradigm.get_data(dataset=BNCI002,subjects=[Subject_list[i]])
                y = np.zeros(X.shape[0])
                for k in range(X.shape[0]):
                    y[k] = float(ytemp[k] == 'right_hand')
                Xall.append(X)
                yall.append(y)
            return Xall,yall
        else:
            X,ytemp,metadata = paradigm.get_data(dataset=BNCI002,subjects=[sub])
            y = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                y[i] = float(ytemp[i] == 'right_hand')
            return X,y
    def load_zhou(self,sub = -1,fmin=8,fmax=13,fsamp = 100):
        import moabb
        import numpy as np
        from moabb.paradigms import LeftRightImagery
        from moabb.datasets import Zhou2016
        dataset = Zhou2016()
        subject_list = dataset.subject_list
        paradigm = LeftRightImagery(resample = fsamp, fmin = fmin, fmax = fmax)
        if sub == -1:
            Xall = []
            yall = []
            for i in range(len(subject_list)):
                X,ytemp,metadata = paradigm.get_data(dataset=dataset,subjects=[subject_list[i]])
                y = np.zeros(X.shape[0])
                for k in range(X.shape[0]):
                    y[k] = float(ytemp[k] == 'left_hand')
                Xall.append(X)
                yall.append(y)
            return Xall,yall
        else:
            X,ytemp,metadata = paradigm.get_data(dataset=dataset,subjects=[subject_list[sub]])
            y = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                y[i] = float(ytemp[i] == 'left_hand')
            return X,y
    
    
    
    def load_weibo(self,sub = -1,fmin=8,fmax=13,fsamp = 100):
        import moabb
        import numpy as np
        from moabb.paradigms import LeftRightImagery
        from moabb.datasets import Weibo2014
        dataset = Weibo2014()
        subject_list = dataset.subject_list
        paradigm = LeftRightImagery(resample = fsamp, fmin = fmin, fmax = fmax)
        if sub == -1:
            Xall = []
            yall = []
            for i in range(len(subject_list)):
                X,ytemp,metadata = paradigm.get_data(dataset=dataset,subjects=[subject_list[i]])
                y = np.zeros(X.shape[0])
                for k in range(X.shape[0]):
                    y[k] = float(ytemp[k] == 'left_hand')
                Xall.append(X)
                yall.append(y)
            return Xall,yall
        else:
            X,ytemp,metadata = paradigm.get_data(dataset=dataset,subjects=[subject_list[sub]])
            y = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                y[i] = float(ytemp[i] == 'left_hand')
            return X,y
        
class Baseline_model:
    def __init__(self,method='covariance',seed=0):
        import EEGene
        from EEGene import Processing
        self.method = method
        self.model = None
        self.P = Processing()
        self.seed = seed
    
    def fit(self,X_train,y_train):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, random_state=self.seed)
        X_train_scaled = self.P.scaler(X_train)
        if self.method == 'all':
            post_X = self.P.pre_processing(X_train_scaled)
            selected_X = self.P.selected_X(post_X,[1,1,1,1,1])
            model.fit(selected_X, y_train)
        else:
            post_X = self.P.find_pre_processing(X_train_scaled,self.method)
            model.fit(post_X, y_train)

        self.model = model
        return model
    
    def predict(self,X_test):
        X_test_scaled = self.P.scaler(X_test)
        if self.method == 'all':
            post_X = self.P.pre_processing(X_test_scaled)
            selected_X = self.P.selected_X(post_X,[1,1,1,1,1])
            y_pred = self.model.predict(selected_X)
        else:
            post_X = self.P.find_pre_processing(X_test_scaled,self.method)
            y_pred = self.model.predict(post_X)
            
        return y_pred
    
    def score(self,X_test,y_test):
        from sklearn.metrics import accuracy_score, f1_score
        import numpy as np
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test,y_pred)
        return {'accuracy':acc, 'f1-score':f1}
    
class Experiment:
    
    def subject_independent_cv(X,y,baselines =['covariance','raw','fft2','fft','cross_correlate','maxmin','meanstd','variance','spectral_entropy','all']
                               ,benchmarks=['EEGNet','DeepConvNet','MIN2Net','EEGene']
                               ,verbose = 0,n_splits=5,seed = 0,n_gens = 10,k=5,n_pops=18,n_save = 5,model_name='model_name'):
        import numpy as np
        from sklearn.model_selection import StratifiedKFold,train_test_split
        import EEGene
        from EEGene import Processing
        P = Processing()
        scores = {meth:[] for meth in baselines+benchmarks}
        subjects = len(X)
        gene_reports = []
        for test_sub in range(subjects):
            print('subject : ',test_sub)
            X_train = None
            y_trian = None
            X_test = X[test_sub]
            y_test = y[test_sub]
            for sub in range(subjects):
                if sub!=test_sub:
                    if type(X_train) == type(None):
                        X_train = X[sub]
                        y_train = y[sub]
                    else:
                        X_train = np.concatenate((X_train,X[sub]))
                        y_train = np.concatenate((y_train,y[sub]))
            X_test = P.scaler(X_test)
            X_train = P.scaler(X_train)
            for method in baselines:

                bm = Baseline_model(method = method)
                bm.fit(X_train,y_train)
                score = bm.score(X_test,y_test)
                scores[method].append(score) 

            for bench in benchmarks:
                if bench == 'EEGene':
                    gene = EEGene(methods = [i for i in baselines if i!='all'],n_pops = n_pops,n_gens = n_gens,n_save=n_save,verbose=verbose,seed = seed,k=k)
                    gene.fit(X_train,y_train)
                    score = gene.score(X_test,y_test)
                    scores['EEGene'].append(score)


                X_train_NN, X_val_NN, y_train_NN, y_val_NN = train_test_split(X_train, y_train, test_size=0.25, random_state=seed)
                trials,chs,samples = X_train_NN.shape
                if bench == 'EEGNet':
                    X_EEGNet_train = np.expand_dims(X_train_NN,-1)
                    X_EEGNet_val = np.expand_dims(X_val_NN,-1)
                    X_EEGNet_test = np.expand_dims(X_test,-1)
                    model = EEGNet(input_shape=(chs,samples,1), num_class=2, dropout_rate=0.25, shuffle=True, data_format='channels_last',batch_size=trials//4,verbose=0,model_name=model_name)
                    model.fit(X_EEGNet_train, y_train_NN, X_EEGNet_val, y_val_NN)
                    Y, evaluation1 = model.predict(X_EEGNet_test, y_test)
                    scores['EEGNet'].append(evaluation1)

                if bench == 'DeepConvNet':
                    X_DeepConvNet_train = np.expand_dims(X_train_NN,-1)
                    X_DeepConvNet_val = np.expand_dims(X_val_NN,-1)
                    X_DeepConvNet_test = np.expand_dims(X_test,-1)
                    model = DeepConvNet(input_shape=(chs,samples,1), num_class=2, dropout_rate=0.25, shuffle=True, data_format='channels_last',batch_size=trials//4,verbose=0,model_name=model_name)
                    model.fit(X_DeepConvNet_train, y_train_NN, X_DeepConvNet_val, y_val_NN)
                    Y, evaluation2 = model.predict(X_DeepConvNet_test, y_test)
                    scores['DeepConvNet'].append(evaluation2)

                if bench == 'MIN2Net':
                    X_MIN2Net_train = np.expand_dims(X_train_NN,-1)
                    X_MIN2Net_train = np.transpose(X_MIN2Net_train,(0,3,2,1))
                    X_MIN2Net_val = np.expand_dims(X_val_NN,-1)
                    X_MIN2Net_val = np.transpose(X_MIN2Net_val,(0,3,2,1))
                    X_MIN2Net_test = np.expand_dims(X_test,-1)
                    X_MIN2Net_test = np.transpose(X_MIN2Net_test,(0,3,2,1))
                    model = MIN2Net(input_shape=(1, samples, chs), num_class=2, monitor='val_loss', shuffle=True,batch_size=trials//4,verbose=0,model_name=model_name)
                    model.fit(X_MIN2Net_train, y_train_NN, X_MIN2Net_val, y_val_NN)
                    Y, evaluation3 = model.predict(X_MIN2Net_test, y_test)
                    scores['MIN2Net'].append(evaluation3)
            if verbose>0:
                print('update score after ',test_sub)
                print(make_df(scores))
            np.save('final_report/'+model_name+'_indep.npy',scores)
        return scores

    def multi_subjects_dependent_cv(Xall,yall,baselines=['covariance','raw','fft2','fft','cross_correlate','maxmin','meanstd','variance','spectral_entropy','all'],
                             benchmarks=['EEGNet','DeepConvNet','MIN2Net','EEGene'],verbose = 0,
                             n_splits=5,seed = 0,n_gens = 10,k=5,n_pops=18,n_save = 5,model_name='model_name'):
        import numpy as np
        from sklearn.model_selection import StratifiedKFold,train_test_split
        import EEGene
        import Benchmarks
        from EEGene import Processing
        from Benchmarks import EEGNet,MIN2Net,DeepConvNet
        scores = {meth:[] for meth in baselines+benchmarks}
        for sub in range(len(Xall)):
            X = Xall[sub]
            y = yall[sub]
            skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
            skf.get_n_splits(X, y)
            P = Processing()
            batch = 0
            for train_index, test_index in skf.split(X, y):
                if verbose > 0:
                    print('batch : {}'.format(batch))
                batch+=1
                #print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                print(X_train.shape,X_test.shape)
                X_test_scaled = P.scaler(X_test)
                X_train_scaled = P.scaler(X_train)

                for method in baselines:
                    print('method = ',method)
                    bm = Baseline_model(method = method)
                    bm.fit(X_train_scaled,y_train)
                    score = bm.score(X_test_scaled,y_test)
                    scores[method].append(score) 

                for bench in benchmarks:
                    if bench == 'EEGene':
                        gene = EEGene(methods = [i for i in baselines if i!='all'],n_pops = n_pops,n_gens = n_gens,n_save=n_save,verbose=verbose,seed = seed,k=k)
                        gene.fit(X_train_scaled,y_train)
                        score = gene.score(X_test_scaled,y_test)
                        scores['EEGene'].append(score)

                    inner_skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
                    inner_skf.get_n_splits(X_train, y_train)
                    for inner_train, inner_val in skf.split(X_train, y_train):
                        X_inner_train, X_inner_val = X_train[inner_train], X_train[inner_val]
                        y_inner_train, y_inner_val = y_train[inner_train], y_train[inner_val]
                        X_inner_train_scaled = P.scaler(X_inner_train)
                        X_inner_val_scaled = P.scaler(X_inner_val)
                        trials,chs,samples = X_inner_train_scaled.shape
                        if bench == 'EEGNet':
                            X_EEGNet_train = np.expand_dims(X_inner_train_scaled,-1)
                            X_EEGNet_val = np.expand_dims(X_inner_val_scaled,-1)
                            X_EEGNet_test = np.expand_dims(X_test_scaled,-1)
                            model = EEGNet(input_shape=(chs,samples,1), num_class=2, dropout_rate=0.25, shuffle=True, data_format='channels_last',batch_size=trials//4,verbose=0,model_name=bench+model_name)
                            model.fit(X_EEGNet_train, y_inner_train, X_EEGNet_val, y_inner_val)
                            _, evaluation1 = model.predict(X_EEGNet_test, y_test)
                            scores['EEGNet'].append(evaluation1)

                        if bench == 'DeepConvNet':
                            X_DeepConvNet_train = np.expand_dims(X_inner_train_scaled,-1)
                            X_DeepConvNet_val = np.expand_dims(X_inner_val_scaled,-1)
                            X_DeepConvNet_test = np.expand_dims(X_test_scaled,-1)
                            model = DeepConvNet(input_shape=(chs,samples,1), num_class=2, dropout_rate=0.25, shuffle=True, data_format='channels_last',batch_size=trials//4,verbose=0,model_name=bench+model_name)
                            model.fit(X_DeepConvNet_train, y_inner_train, X_DeepConvNet_val, y_inner_val)
                            _, evaluation2 = model.predict(X_DeepConvNet_test, y_test)
                            scores['DeepConvNet'].append(evaluation2)

                        if bench == 'MIN2Net':
                            X_MIN2Net_train = np.expand_dims(X_inner_train_scaled,-1)
                            X_MIN2Net_train = np.transpose(X_MIN2Net_train,(0,3,2,1))
                            X_MIN2Net_val = np.expand_dims(X_inner_val_scaled,-1)
                            X_MIN2Net_val = np.transpose(X_MIN2Net_val,(0,3,2,1))
                            X_MIN2Net_test = np.expand_dims(X_test_scaled,-1)
                            X_MIN2Net_test = np.transpose(X_MIN2Net_test,(0,3,2,1))
                            model = MIN2Net(input_shape=(1, samples, chs), num_class=2, monitor='val_loss', shuffle=True,batch_size=trials//4,verbose=0,model_name=bench+model_name)
                            model.fit(X_MIN2Net_train, y_inner_train, X_MIN2Net_val, y_inner_val)
                            _, evaluation3 = model.predict(X_MIN2Net_test, y_test)
                            scores['MIN2Net'].append(evaluation3)
        np.save('final_report/'+model_name+'_all_subjects_dep.npy',scores)
        return scores
    
    def single_subject_dependent_cv(X,y,baselines=['covariance','raw','fft2','fft','cross_correlate','maxmin','meanstd','variance','spectral_entropy','all'],
                             benchmarks=['EEGNet','DeepConvNet','MIN2Net','EEGene'],verbose = 0,
                             n_splits=5,seed = 0,n_gens = 10,k=5,n_pops=18,n_save = 5,model_name='model_name'):
        import numpy as np
        from sklearn.model_selection import StratifiedKFold,train_test_split
        import EEGene
        from EEGene import Processing
        skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
        skf.get_n_splits(X, y)
        P = Processing()
        scores = {meth:[] for meth in baselines+benchmarks}
        batch = 0
        for train_index, test_index in skf.split(X, y):
            if verbose > 0:
                print('batch : {}'.format(batch))
            batch+=1
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            print(X_train.shape,X_test.shape)
            X_test_scaled = P.scaler(X_test)
            X_train_scaled = P.scaler(X_train)

            for method in baselines:
                print('method = ',method)
                bm = Baseline_model(method = method)
                bm.fit(X_train_scaled,y_train)
                score = bm.score(X_test_scaled,y_test)
                scores[method].append(score) 

            for bench in benchmarks:
                if bench == 'EEGene':
                    gene = EEGene(methods = [i for i in baselines if i!='all'],n_pops = n_pops,n_gens = n_gens,n_save=n_save,verbose=verbose,seed = seed,k=k)
                    gene.fit(X_train_scaled,y_train)
                    score = gene.score(X_test_scaled,y_test)
                    scores['EEGene'].append(score)

                inner_skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
                inner_skf.get_n_splits(X_train, y_train)
                for inner_train, inner_val in skf.split(X_train, y_train):
                    X_inner_train, X_inner_val = X_train[inner_train], X_train[inner_val]
                    y_inner_train, y_inner_val = y_train[inner_train], y_train[inner_val]
                    X_inner_train_scaled = P.scaler(X_inner_train)
                    X_inner_val_scaled = P.scaler(X_inner_val)
                    trials,chs,samples = X_inner_train_scaled.shape
                    if bench == 'EEGNet':
                        X_EEGNet_train = np.expand_dims(X_inner_train_scaled,-1)
                        X_EEGNet_val = np.expand_dims(X_inner_val_scaled,-1)
                        X_EEGNet_test = np.expand_dims(X_test_scaled,-1)
                        model = EEGNet(input_shape=(chs,samples,1), num_class=2, dropout_rate=0.25, shuffle=True, data_format='channels_last',batch_size=trials//4,verbose=0,model_name=model_name)
                        model.fit(X_EEGNet_train, y_inner_train, X_EEGNet_val, y_inner_val)
                        _, evaluation1 = model.predict(X_EEGNet_test, y_test)
                        scores['EEGNet'].append(evaluation1)

                    if bench == 'DeepConvNet':
                        X_DeepConvNet_train = np.expand_dims(X_inner_train_scaled,-1)
                        X_DeepConvNet_val = np.expand_dims(X_inner_val_scaled,-1)
                        X_DeepConvNet_test = np.expand_dims(X_test_scaled,-1)
                        model = DeepConvNet(input_shape=(chs,samples,1), num_class=2, dropout_rate=0.25, shuffle=True, data_format='channels_last',batch_size=trials//4,verbose=0,model_name=model_name)
                        model.fit(X_DeepConvNet_train, y_inner_train, X_DeepConvNet_val, y_inner_val)
                        _, evaluation2 = model.predict(X_DeepConvNet_test, y_test)
                        scores['DeepConvNet'].append(evaluation2)

                    if bench == 'MIN2Net':
                        X_MIN2Net_train = np.expand_dims(X_inner_train_scaled,-1)
                        X_MIN2Net_train = np.transpose(X_MIN2Net_train,(0,3,2,1))
                        X_MIN2Net_val = np.expand_dims(X_inner_val_scaled,-1)
                        X_MIN2Net_val = np.transpose(X_MIN2Net_val,(0,3,2,1))
                        X_MIN2Net_test = np.expand_dims(X_test_scaled,-1)
                        X_MIN2Net_test = np.transpose(X_MIN2Net_test,(0,3,2,1))
                        model = MIN2Net(input_shape=(1, samples, chs), num_class=2, monitor='val_loss', shuffle=True,batch_size=trials//4,verbose=0,model_name=model_name)
                        model.fit(X_MIN2Net_train, y_inner_train, X_MIN2Net_val, y_inner_val)
                        _, evaluation3 = model.predict(X_MIN2Net_test, y_test)
                        scores['MIN2Net'].append(evaluation3)
        np.save('final_report/'+model_name+'_dep.npy',scores)
        return scores

    def make_df (scores):
        import numpy as np
        d = {'method':[],'accuracy':[],'+-acc':[],'f1-score':[],'+-f1':[]}
        for method in scores:
            accs = np.array([i['accuracy'] for i in scores[method]])
            f1s = np.array([i['f1-score'] for i in scores[method]])
            #print('{} : {:.2f}({:.2f}) , {:.2f}({:.2f})'.format(method,accs.mean(),accs.std(),f1s.mean(),f1s.std()))
            d['method'].append(method)
            d['accuracy'].append(100*accs.mean())
            d['+-acc'].append(100*accs.std())
            d['f1-score'].append(100*f1s.mean())
            d['+-f1'].append(100*f1s.std())
        import pandas as pd
        pd.options.display.float_format = "{:,.2f}".format
        df = pd.DataFrame.from_dict(d)
        return df.sort_values(by=['accuracy','f1-score'], ascending=False)

    def make_final_report (scores):
        import numpy as np
        dx = {}
        for score in scores:
            for method in score:
                if method not in dx:
                    dx[method] = {'accuracy':np.array([i['accuracy'] for i in score[method]]),'f1-score':np.array([i['f1-score'] for i in score[method]])}
                else:
                    dx[method]['accuracy'] = np.concatenate((dx[method]['accuracy'],np.array([i['accuracy'] for i in score[method]])))
                    dx[method]['f1-score'] = np.concatenate((dx[method]['f1-score'],np.array([i['f1-score'] for i in score[method]])))
        d = {'method':[],'accuracy':[],'+-acc':[],'f1-score':[],'+-f1':[]}
        for method in dx:
            d['method'].append(method)
            d['accuracy'].append(100*dx[method]['accuracy'].mean())
            d['+-acc'].append(100*dx[method]['accuracy'].std())
            d['f1-score'].append(100*dx[method]['f1-score'].mean())
            d['+-f1'].append(100*dx[method]['f1-score'].std())
        import pandas as pd
        pd.options.display.float_format = "{:,.2f}".format
        df = pd.DataFrame.from_dict(d)
        return df.sort_values(by=['accuracy','f1-score'], ascending=False)

