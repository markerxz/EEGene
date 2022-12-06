class Processing:
    def __init__(self,methods =['covariance','raw','fft2','fft','cross_correlate','maxmin','meanstd','variance','spectral_entropy']):
        import numpy
        self.methods = methods
        
    def scaler (self,X):
        import numpy as np
        import sklearn
        from sklearn.preprocessing import StandardScaler,MinMaxScaler
        trials,_,_ = X.shape
        X_scaled = np.zeros(X.shape)
        for trial in range(trials):
            trans1 = StandardScaler()
            trans2 = MinMaxScaler()
            X_scaled[trial] = trans1.fit_transform(X[trial])
            #X_scaled[trial] = trans2.fit_transform(trans1.fit_transform(X[trial]))
        return X_scaled
    
    def maxmin (self,X):
        import numpy as np
        trials,chs,samples = X.shape
        Xm = np.zeros((trials,chs,2))
        for trial in range(trials):
            Xm[trial] = np.array([[np.max(a),np.min(a)] for a in X[trial]])
        return Xm.reshape((trials,chs*2))
    
    def meanstd(self,X):
        import numpy as np
        trials,chs,samples = X.shape
        Xm = np.zeros((trials,chs,2))
        for trial in range(trials):
            Xm[trial] = np.array([[np.mean(a),np.std(a)] for a in X[trial]])
        return Xm.reshape((trials,chs*2))
    
    def variance(self,X):
        import numpy as np
        trials,chs,samples = X.shape
        Xv = np.zeros((trials,chs))
        for trial in range(trials):
            Xv[trial] = np.array([np.var(a) for a in X[trial]])
        return Xv
    
    def spectral_entropy(self,X):
        import numpy as np
        import antropy as ant
        trials,chs,samples = X.shape
        Xse = np.zeros((trials,chs))
        for trial in range(trials):
            for ch in range(chs):
                Xse[trial][ch] = ant.spectral_entropy(X[trial][ch], sf=100, method='fft',normalize=True)
                if np.isnan(Xse[trial][ch]):
                    Xse[trial][ch] = 0
        return Xse
                
    
    def covariance (self,X):
        import numpy as np
        X_temp = self.scaler(X)
        trials,chs,samples = X_temp.shape
        Xcov = np.zeros((trials,chs,chs))
        for trial in range(trials):
            Xcov[trial] = np.cov(X_temp[trial])
        return Xcov.reshape((trials,chs*chs))
    
    def raw (self,X):
        import numpy as np
        trials,chs,samples = X.shape
        return X.reshape((trials,chs*samples))
    
    def fft2 (self,X):
        import numpy as np
        import warnings
        warnings.simplefilter("ignore", np.ComplexWarning)
        trials,chs,samples = X.shape
        X_fft2 = np.zeros((trials,chs,samples))
        for trial in range(trials):
            X_fft2[trial] = np.fft.fft2(X[trial])
        return X_fft2.reshape((trials,chs*samples))
    
    def fft (self,X):
        import numpy as np
        import warnings
        warnings.simplefilter("ignore", np.ComplexWarning)
        trials,chs,samples = X.shape
        X_fft = np.zeros((trials,chs,samples))
        for trial in range(trials):
            X_fft[trial] = np.fft.fft(X[trial])
        return X_fft.reshape((trials,chs*samples))
    
    def cross_correlate(self,X):
        from scipy import signal
        import numpy as np
        trials,chs,samples = X.shape
        X_cc = np.zeros((trials,int((chs-1)*chs/2),samples))
        for trial in range(trials):
            cc = 0
            for ch1 in range(chs):
                for ch2 in range(chs): 
                    if ch1 < ch2:
                        corr = signal.correlate(X[trial][ch1], X[trial][ch2], mode='same',method='fft')
                        X_cc[trial][cc] = corr
                        cc += 1
        return X_cc.reshape((trials,int((chs-1)*chs/2)*(samples)))
    
    def cross_approximate_entropy(self,X):
        import numpy as np
        import EntropyHub as EH
        trials,chs,samples = X.shape
        m = 3
        X_ApEn = np.zeros((trials,int((chs-1)*chs/2),((m*2)+3)))
        for trial in range(trials):
            # if trial%2 == 0:
                #print('trial = ',trial)
            cc = 0
            for ch1 in range(chs):
                for ch2 in range(chs): 
                    if ch1 < ch2:
                        Sig = [X[trial][ch1],X[trial][ch2]]
                        XAp,Phi = EH.XApEn(Sig,m=m)
                        X_ApEn[trial][cc] = np.concatenate([XAp,Phi])
                        cc += 1
        return X_ApEn.reshape((trials,int((chs-1)*chs/2)*((m*2)+3)))
    
    
    def find_pre_processing (self,X,method):
        if method == 'covariance':
            return self.covariance(X)
        if method == 'raw':
            return self.raw(X)
        if method == 'fft2':
            return self.fft2(X)
        if method == 'fft':
            return self.fft(X)
        if method == 'cross_approximate_entropy':
            return self.cross_approximate_entropy(X)
        if method == 'cross_correlate':
            return self.cross_correlate(X)
        if method == 'maxmin':
            return self.maxmin(X)
        if method == 'meanstd':
            return self.meanstd(X)
        if method == 'variance':
            return self.variance(X)
        if method == 'spectral_entropy':
            return self.spectral_entropy(X)
                           
    def pre_processing(self,X):
        import numpy as np
        trials,chs,samples = X.shape
        n_method = len(self.methods)
        post_X = []
        for i,method in enumerate(self.methods):
            post_X.append(self.find_pre_processing(X,method))
        return post_X
    
    def selected_X(self,post_X,pop):
        total_len = 0
        for i in range(len(pop)):
            if pop[i]:
                total_len += post_X[i].shape[1]
        
        selected_X = np.zeros((post_X[0].shape[0],total_len))
        for trial in range(post_X[0].shape[0]):
            this_X = None
            for i in range(len(pop)):
                if pop[i]:
                    if type(this_X) != np.ndarray:
                        this_X = post_X[i][trial]
                    else:
                        this_X = np.concatenate((this_X,post_X[i][trial]))
            selected_X[trial] = this_X
        return selected_X
   
    
    def selected_X(self,post_X,pop):
        import numpy as np
        total_len = 0
        for i in range(len(pop)):
            if pop[i]:
                total_len += post_X[i].shape[1]
        
        selected_X = np.zeros((post_X[0].shape[0],total_len))
        for trial in range(post_X[0].shape[0]):
            this_X = None
            for i in range(len(pop)):
                if pop[i]:
                    if type(this_X) != np.ndarray:
                        this_X = post_X[i][trial]
                    else:
                        this_X = np.concatenate((this_X,post_X[i][trial]))
            selected_X[trial] = this_X
        return selected_X
    
    def RF (self,X_train,y_train,X_val,y_val,seed = 0):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score
        clf = RandomForestClassifier(n_estimators=50, random_state=seed)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val,y_pred)
        return {'accuracy':acc, 'f1-score':f1}
        
    def skf_test(self,selected_X,y_rest,k = 5,seed = 0,subject_independent = False):
        if selected_X.shape[0] <= 20 or k == 1:
            from sklearn.model_selection import train_test_split
            reports = []
            X_train, X_val, y_train, y_val = train_test_split(selected_X, y_rest, test_size=0.25, random_state=seed,stratify=y_rest )
            report = self.RF(X_train,y_train,X_val,y_val)
            reports.append(report)
            return reports

        else:
            from sklearn.model_selection import StratifiedKFold      
            skf = StratifiedKFold(n_splits=k, random_state=1, shuffle=True)
            skf.get_n_splits(selected_X, y_rest)
            reports = []
            for train_index, val_index in skf.split(selected_X, y_rest):
                #print("TRAIN:", train_index, "VAL:", val_index)
                X_train, X_val = selected_X[train_index], selected_X[val_index]
                y_train, y_val = y_rest[train_index], y_rest[val_index]
                report = self.RF(X_train,y_train,X_val,y_val)
                reports.append(report)
            return reports
        
class EEGene:
    def __init__ (self,methods = ['covariance','raw','fft2','fft','cross_correlate','maxmin','meanstd','variance','spectral_entropy'],n_gens=5,n_pops=18,n_save=5,verbose=0,seed = 0,k = 5):
        import numpy as np
        self.methods = methods
        self.n_methods = len(methods)
        self.n_pops = n_pops
        self.n_save = n_save
        self.n_gens = n_gens
        self.P = Processing(methods = methods)
        self.best_gene = []
        self.best_model = []
        self.verbose = verbose
        self.seed = seed
        self.model_report = {method:0 for method in methods}
        self.k = k
        self.dp = set()
    
    def create_model(self,post_X,y_rest,pop):
        from sklearn.ensemble import RandomForestClassifier
        selected_X = self.P.selected_X(post_X,pop['gene'])
        model = RandomForestClassifier(n_estimators=50, random_state=self.seed)
        model.fit(selected_X, y_rest)
        return model
        
    def select(self,pops,post_X,y_rest):
        import numpy as np
        pops_not_dupe = []
        parents = []
        
        for p1 in pops:
            if sum(p1['gene']) == 0:
                continue
            b = True
            for p2 in pops_not_dupe:
                if  (p1['gene'] == p2['gene']).all():
                    b = False
                    break
            if b:
                pops_not_dupe.append(p1)
    
        for pop in pops_not_dupe:
            if pop['accuracy'] == -1:
                self.dp.add(tuple(pop['gene']))
                selected_X = self.P.selected_X(post_X,pop['gene'])
                caled = self.P.skf_test(selected_X,y_rest,k = self.k)
                accs = np.array([i['accuracy'] for i in caled])
                f1s = np.array([i['f1-score'] for i in caled])
                pop['accuracy'] = accs.mean()
                pop['f1-score'] = f1s.mean()
                parents.append(pop)
            else:
                parents.append(pop)      
        parents = sorted(parents, key=lambda x: (x['accuracy']+x['f1-score']))[::-1]
        if len(parents) > self.n_pops//2:
            return parents[:self.n_pops//2]
        else:
            return parents
    
    def crossover(self,parents, num_offsprings):
        import numpy as np
        temp_seed = self.seed
        offsprings = np.empty((num_offsprings, parents[0]['gene'].shape[0])).astype(int)
        i = 0
        pre_i = 0
        cc = 0
        while (i < num_offsprings) and (cc<10):
            temp_seed = (temp_seed + np.random.randint(1,1000000))%np.random.randint(1,1000000)
            np.random.seed(temp_seed)
            # print('temp_seed = ',temp_seed)
            #print('i = ',i)
            #print('dp now (',len(self.dp),') : ',self.dp)
            parent1_index = np.random.randint(0,len(parents))
            parent2_index = np.random.randint(0,len(parents))
            #print('parents : ',[parent1_index,parent2_index])
            crossover_point = np.random.randint(1,len(parents[0]['gene'])-1)
            # print('p1p2 : ',[parent1_index,parent2_index])
            # print('cp : ',crossover_point)
            offsprings[i,0:crossover_point] = parents[parent1_index]['gene'][0:crossover_point]
            offsprings[i,crossover_point:] = parents[parent2_index]['gene'][crossover_point:]
            offsprings[i] = self.mutation(offsprings[i],temp_seed).astype(int)
            #print(tuple(offsprings[i]))
            if tuple(offsprings[i]) not in self.dp and sum(offsprings[i]) != 0:
                # print('yes : ',i)
                self.dp.add(tuple(offsprings[i]))
                i = i+1
            if pre_i != i:
                pre_i = i
                cc = 0
            else:
                cc += 1
        return offsprings.astype(int)    
    
    def mutation(self,offspring,temp_seed):
        import random as rd
        rd.seed(temp_seed)
        import numpy as np
        np.random.seed(temp_seed)
        mutation_rate = 0.9
        i = 0
        random_value = rd.random()
        if random_value > mutation_rate:
            return offspring.astype(int)  
        int_random_value = np.random.randint(0,offspring.shape[0]-1)    
        if offspring[int_random_value] == 0 :
            offspring[int_random_value] = 1
        else :
            offspring[int_random_value] = 0
        return offspring.astype(int)    
    
    def fit (self,X_rest,y_rest):
        import numpy as np
        X_rest_scaled = self.P.scaler(X_rest)
        post_X = self.P.pre_processing(X_rest_scaled)
        
        import numpy as np
        pop_size = (self.n_methods, self.n_methods)
        initial_population = np.zeros((pop_size))
        for i in range(self.n_methods):
            initial_population[i][i] = 1
        pops = initial_population.astype(int)
        pops = [{'accuracy':-1,'f1-score':-1,'gene':pop,'birth':0} for pop in pops]
        parents = self.select(pops,post_X,y_rest)
        for pop in pops:
            self.dp.add(tuple(pop['gene']))
        
        for gen in range(self.n_gens):
            #print(pops)
            if self.verbose > 1:
                print('gen = ',gen)
            parents = self.select(pops,post_X,y_rest)
            
            if self.verbose > 2:
                print('This gen gene')
                for pop in parents:
                    print('{}. {:.2f} ; {:.2f} ; {}'.format(pop['birth'],pop['accuracy'],pop['f1-score'],self.gene_to_meths(pop)))
                
                print('need = ',self.n_pops - len(parents))
                print()
            mutants = self.crossover(parents,self.n_pops - len(parents))
            pops = np.zeros((self.n_pops,parents[0]['gene'].shape[0]))
            pops = parents
            mutants = [{'accuracy':-1,'f1-score':-1,'gene':pop,'birth':gen+1} for pop in mutants]
            if self.verbose > 20:
                print('Mutant')
            for mu in mutants:
                pops.append(mu)
                if self.verbose > 20:
                    print('{}. {:.2f} ; {:.2f} ; {}'.format(mu['birth'],mu['accuracy'],mu['f1-score'],self.gene_to_meths(mu)))
            if self.verbose > 20:
                print()
            parents = self.select(pops,post_X,y_rest)
        
        if self.verbose > 2:
            print()
            print('Final Gene')
            for pop in parents:
                print('{}. {:.2f} ; {:.2f} ; {}'.format(pop['birth'],pop['accuracy'],pop['f1-score'],self.gene_to_meths(pop)))
            print()
        for top in range(self.n_save):
            pop = parents[top]
            self.best_gene.append(pop)
            self.best_model.append(self.create_model(post_X,y_rest,pop))
            self.update_model_report(pop)
        for method in self.methods:
            self.model_report[method]/=(2*self.n_save)
        
    def update_model_report(self,pop):
        for i in range(self.n_methods):
            if pop['gene'][i]:
                self.model_report[self.methods[i]] += pop['accuracy']+pop['f1-score']
    
    def show_model_report(self):
        return self.model_report
    
    def predict(self,X_test):
        import numpy as np
        X_test_scaled = self.P.scaler(X_test)
        post_X = self.P.pre_processing(X_test_scaled)
        y_pred_dict = None
        d = np.zeros((X_test.shape[0],2))
        for top in range(self.n_save):
            model = self.best_model[top]
            pop = self.best_gene[top]
            selected_X = self.P.selected_X(post_X,pop['gene'])
            if y_pred_dict == None:
                y_pred = model.predict(selected_X).astype(int)
                for i in range(X_test.shape[0]):
                    d[i][y_pred[i]] += pop['accuracy']+pop['f1-score']
                
        y_pred = np.zeros((X_test.shape[0]))
        for i in range(X_test.shape[0]):
            y_pred[i] = int(d[i][1]>d[i][0])
        return y_pred,d
    
    def score(self,X_test,y_test):
        from sklearn.metrics import accuracy_score, f1_score
        import numpy as np
        X_test_scaled = self.P.scaler(X_test)
        post_X = self.P.pre_processing(X_test_scaled)
        y_pred_dict = None
        d = np.zeros((X_test.shape[0],2))
        for top in range(self.n_save):
            model = self.best_model[top]
            pop = self.best_gene[top]
            selected_X = self.P.selected_X(post_X,pop['gene'])
            if y_pred_dict == None:
                y_pred = model.predict(selected_X).astype(int)
                for i in range(X_test.shape[0]):
                    d[i][y_pred[i]] += pop['accuracy']+pop['f1-score']
                
        y_pred = np.zeros((X_test.shape[0]))
        for i in range(X_test.shape[0]):
            y_pred[i] = int(d[i][1]>d[i][0])
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test,y_pred)
        return {'accuracy':acc, 'f1-score':f1}
    
    def gene_to_meths(self,pop):
        meths = []
        for i in range(self.n_methods):
            if pop['gene'][i]:
                meths.append(self.methods[i])
        return meths
                
