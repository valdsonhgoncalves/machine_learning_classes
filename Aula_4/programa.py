import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as py
import tkintertable as tk

class regressaolinear() :
    def __init__(self):
        pass
    def Ajusta(self, x1, x2, y) :
        # Dados os vetores de features x1 e x2 e o vetor de label y
        # ajusta f(x) = a0 + a1*x1 + a2*x2 + a3*x1*x2 via minimos quadrados
        # a = inv(X'*X)* X'*y X=[1 x1 x2 x1*x2]
        npto=len(x1)
        um = np.ones(npto)
        Xt=np.array([um,x1,x2,x1*x2])
        X=np.transpose(Xt)
        Y=np.transpose(y)
        a = np.linalg.inv(Xt@X) @ (Xt @ Y)
        return a
        
    def AvaliaReta(self, x1, x2, a) :
        y = a[0] + a[1]*x1 + a[2]*x2 + a[3]*x1*x2
        return y

class knn() :
    def __init__(self):
        self.k = 2
    
    def Treinamento(self, features, label) :
        self.features = np.array(features)
        self.label = np.array(label)

    def Classifica(self, x, k = -1) :
        # Dado o vetor x com as características, esta rotina retorna o rótulo
        # dos k vizinhos mais próximos
        if k == -1 :
            k = self.k
        dist = self.features - x
        dist2 = dist*dist
        dist_euc = np.sum (dist2, axis = 1)
        posic = np.zeros(k, dtype = int)
        for i in range(k) :
            posic [k] = np.argmin(dist_euc)
            li=np.argmin(dist_euc)
            posic[k]=ii
            dist_euc[ii]=1e16
        aux=self.label(posic)
        meio=int(k/2)
        if np.sum(aux) <=meio:
            return 0
        else:
            return 1


# programa principal
if __name__=="__main__" :

    #mat=np.loadtxt('treino_1.txt')
    filename = tk.filedialog.askopenfilename(title = "Esolha o Arquivo de Dados")
    filetypes = [("Arquivos extensão ASCII ","*.txt"),("Arquivos extensão CSV","*.csv")]
    if filename :
        mat = np.loadtxt (filename)
    else :
        exit ()
    i=1

