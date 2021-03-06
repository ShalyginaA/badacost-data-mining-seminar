import numpy as np
import scipy.optimize

class BAdaCost:
    def __init__(self,optimizer, n_iters, learning_rate, C, eps):
        self.optimizer = optimizer #weak learners
        self.n_iters = n_iters #number of iterations
        self.lr = learning_rate #learning rate
        self.weights = np.zeros(n_iters) #weights of weak learners
        self.weak_learners = [0]*n_iters #weak learners
        self.Cprime = C - np.diag(np.array(np.sum(C,axis=1).flatten())[0]) #modified cost matrix
        self.n_classes = C.shape[1] #number of classes
        self.margin = ((-1/(self.n_classes-1))*np.ones([self.n_classes,self.n_classes]) 
                 + (self.n_classes/(self.n_classes-1))*np.eye(self.n_classes)) #margin vector
        self.eps = eps 
        
    def translate_to_cost_matrix(self,C2,beta):
        K = C2.shape[0]
        Cexp = np.exp(beta*C2)
        for j in range(K):
            Cexp[j,:] -= Cexp[j,j]*np.ones(K)
        return Cexp
    
    def compute_weak_learner_weight(self,C_star, W, pred, y):
    #Computes  Weak Learner weight (\alpha) in order to minimize the 
    #cost sensitive loss function.
        K = C_star.shape[0]
        WeightsSum = np.zeros([K,K])
        for i in range(K):
            for j in range(K):
                y1 = []
                pred1 = []
                for k,z in zip(y,pred):
                    if k == i:     
                        y1.append(1)
                    else:
                        y1.append(0)
                    if z == j:
                        pred1.append(1)
                    else:
                        pred1.append(0)
                predicted_as_j_being_i = y1 and pred1
                WeightsSum[i,j] = np.dot(W,predicted_as_j_being_i)
        alpha0 = 1.0
        alpha = scipy.optimize.fmin(lambda x: self.cost_sensitive_loss_function(x,C_star,WeightsSum),
                                    x0=alpha0,disp=0)
        return alpha 
    
    def cost_sensitive_loss_function(self,alpha,C_star,WeightsSum):
        #Loss function computation for a given
        #alpha 
        K = C_star.shape[0]
        func_value = 0
        for i in range(K):
            for j in range(K):
                func_value += WeightsSum[i,j]*np.exp(alpha*C_star[i,j])
        return func_value 
    
    def compute_weak_learner_cost(self,pred_wl, y, C2, beta, W):
        cost = 0.0
        for i in range(len(pred_wl)):
            cost += W[i]*np.exp(beta*C2[y[i],pred_wl[i]])
            
        return cost
        
    def fit(self,X,y):
        N = X.shape[0]
        sample_weights = (1/N)*np.ones(N)
        C2 = self.Cprime * self.margin
        for i in range(self.n_iters):
            beta = 1
            c = 2**63-1 #inf
            delta_c = 2**63-1 #inf
            while delta_c >= self.eps:
                C_wl = self.translate_to_cost_matrix(C2,beta)
                G = self.train_multiclass_cost_sensitive_WL(X,y,sample_weights,C_wl)
                #wl_preds = G.predict(X)
                wl_preds = self.weak_learner_prediction(G,X,C_wl)
                beta = self.compute_weak_learner_weight(C2,sample_weights,wl_preds,y)[0]                
                c_new = self.compute_weak_learner_cost(wl_preds,y,C2,beta,sample_weights)
                delta_c = c-c_new
                c = c_new
            self.weak_learners[i] = G
            self.weights[i] = beta
            beta = self.lr*beta
            
            # update sample weights: only the right classified samples changes the
            #weight
            for j in range(X.shape[1]):
                exp_j = C2[y[j],wl_preds[j]]
                sample_weights[j] *= np.exp(beta*exp_j)
            #renormalize sample weights
            sample_weights /= np.sum(sample_weights)
            
    def train_multiclass_cost_sensitive_WL(self,X,y,w,C_wl):
        wl = self.optimizer
        wl.fit(X,y)
        return wl
            
    def weak_learner_prediction(self,weak_learner,X,C):
        probs = weak_learner.predict_proba(X)
        prediction = []
        n = probs.shape[0]
        for i in range(n):
            a = np.dot(C,probs[i])
            a_min = np.argmin(a,axis=1)
            prediction.append(a_min)
        return np.array(prediction).reshape(1,n)[0]
            
                   
    def predict(self,X):
        n = X.shape[0]
        M = len(self.weak_learners)
        margin_vec = np.zeros([self.n_classes,n])
        for i in range(len(self.weak_learners)):
            #row vector with the labels
            z = self.weak_learner_prediction(self.weak_learners[i],X,self.Cprime)
            for j in range(n):
                margin_vec[:,j] += self.weights[i]*self.margin[:,z[j]]
          
        predicted = -self.Cprime*margin_vec
        predicted = np.argmin(predicted.transpose(),axis=1)
        return np.array(predicted.reshape(1,n))[0]