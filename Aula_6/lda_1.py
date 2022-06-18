import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

class lda():
    def __init__(self):
        self.k = -1
        self.eps = 1.1e-16
    
    def Treina(self,features,labels,k=2,alfa=.05):
        m,n = features.shape
        
        if m > n:
            x = np.array(features)
            y = np.array(labels, dtype=int)
        else:
            x = np.array(np.transpose(features))
            y = np.array(np.transpose(labels))
        
        N,p = features.shape
        self.k = k #Variável global número de classes
        M = 0 #Teste de Box
        S = np.zeros([p,p])
        mu = np.zeros([n,k])
        pik = np.zeros(k)
        ni = np.zeros(k)
        
        for i in range(k):
            aux = x[y==i,:]
            nk = aux.shape[0]
            pik = ni/N
            Sk = np.cov(np.transpose(aux))
            S += (nk-1)*Sk
            mu[i,:] = np.mean(aux,axis=0)
            M -= (nk-1)*np.log(np.linalg.det(Sk))
            ni[i] = nk
            
        self.mu = np.transpose(mu)
        self.S = S/(N-k)
        self.pik = pik
        M += (N-k)*np.log(np.linalg.det(S)) #Estatística do teste M de Box
        
        #Estatística M de Box
        a1 = 2*p**2 + 3*p - 1
        a2 = 6*(p+1)*(k-1)
        a3 = np.sum(1/(ni-1+self.eps)) - 1/(N-k+self.eps)
        A1 = a1*a3/(a2 + self.eps)
        
        v1 = p*(p-1)*(k-1)/2
        a1 = (p-1)*(p-2)
        a2 = 6*(k-1) 
        a3 = np.sum(1/(ni-1+self.eps)**2) - 1/(N-k+self.eps)**2
        A2 = a1*a3/a2
        
        if (A2-A1**2>0):
            v2 = (v1+2)/(A2-A1**2+self.eps)
            b = v1/(1-A1-(v1/v2) + self.eps)
            F = M/b
            Fc = st.f.ppf(1-alfa/2,v1,v2)
        else:
            v2 = (v1+2)/(A1**2-A2 + self.eps)
            b = v2/(1-A1+2/(v2+self.eps))
            F = v2*M/(v1*(b-M)+self.eps)
            Fc = st.f.ppf(1-alfa/2,v1,v2)
            
        if Fc>F:
            return False
        else:
            return True
        
    def Avalia(self,features):
        N,p = features.shape
        y = np.zeros([self.k,N])
        inv_S = np.linalg.inv(self.S)
        
        for j in range (self.k):
            t_1 = np.transpose(inv_S@self.mu[:,j])
            t_2 = -0.5*np.transpose(self.mu[:,j])@inv_S@self.mu[:,j]
            t_3 = np.log(self.pik[j])
            for i in range (N):
                y[j,i] = features[i,:]@t_1+t_2+t_3
        d = np.argmax(y,axis=0)
        return d
            



if __name__ == '__main__':
    features = np.loadtxt('features.txt')
    labels = np.loadtxt('labels.txt')
    plt.scatter(features[:,0], features[:,1], s=40, c = labels)
    plt.grid(True)
    plt.show()
    lda = lda()
    lda.Treina(features, labels,k=2,alfa=0.1)
    
    x = np.loadtxt('teste.txt')
    features_t = x[:,0:2]
    labels_t = x[:,2]
    
    d = lda.Avalia(features)
    y = labels
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    for i in range(len(y)):
        if y[i] == 0:
            if d[i] == 0:
                TN +=1
            else:
                FP +=1
        else:
            if d[i] == 1:
                TP +=1
            else:
                FN +=1
    
    #Sensibilidade
    sen = 100*TP/(TP+FN)
    
    #Especificidade
    esp = 100*TN/(TN+FP)
    
    #Precisão
    pre = 100*(TN + TP)/len(y)
    
    txt = 'Sen. = %5.1f  -  Esp = %5.1f  -  Prec = %5.1f'%(sen,esp,pre)
    print(txt)
    
    i = 1