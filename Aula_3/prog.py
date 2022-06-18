import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
import tkinter as tk
import tkinter.filedialog

class regressaolinear():
    def __init__(self):
        pass

    def Ajuste(self,x1,x2,y):
        #Dados os vetores de features x1 e x2 e o vetor de label y
        # Ajusta f(x) = a0 + a1*x1 + a2*x2 + a3*x1*x2
        # a = inv(X'*X)*X'*y  X = [1 x1 x2 x1*x2]
        npto = len(x1)
        um = np.ones(npto)
        Xt = np.concatenate((um,x1,x2,x1*x2),axis=0)
        Xt = Xt.reshape((4,-1),order='C')
        X = np.transpose(Xt)
        #Y = np.transpose(y)
        a = np.linalg.inv(Xt@X)@(Xt@y)
        return a

    def AvaliaReta(self,x1,x2,a):
        y = a[0] + a[1]*x1 + a[2]*x2 + a[3]*x1*x2
        return y


class knn():
    def __init__(self):
        self.k = 2

    def Treinamento(self,features,label):
        self.features = np.array(features)
        self.label = np.array(label)

    def Classifica(self,x,k=-1):
        #Dado vetor x com as características, esta rotina retorna o rótulo dos k vizinhos mais próximos
        if k == -1:
            k = self.k
        
        dist = self.features - x
        dist2 = dist*dist
        dist_euc = np.sum(dist2,axis=1)
        posic  = np.zeros(k,dtype=int)
        
        for i in range(k):
            ii = np.argmin(dist_euc)
            posic[k] = ii
            dist_euc[ii] = 1e16
        
        aux = self.label(posic)
        meio = int(k/2)
        
        if np.sum(aux)<=meio:
            return 0
        else:
            return 1


if __name__ == "__main__":
    
    #filename = tk.filedialog.askopenfilename(title = "Escolha o Arquivo de Dados", filetypes = [("Arquivos de extensão ASCII","*.txt"),("Arquivos extensão CSV","*.csv")])
    
    #if filename:
    mat = np.loadtxt('treino_2.txt')
    #else:
    #    exit()
    fig1 = plt.figure()
    ax1 = plt.subplot(3,1,1)
    ax1.plot(mat[:,0])
    ax2 = plt.subplot(3,1,2)
    ax2.plot(mat[:,1],'r')
    ax3 = plt.subplot(3,1,3)
    ax3.plot(mat[:,2],'g')
    plt.show()
    
    x1 = mat[:,0]
    x2 = mat[:,1]
    y = mat[:,2]
    fig2 = plt.figure()
    ax4 = plt.scatter(x1,x2,s=30,c=y)
    plt.show()
    
    dados = x2.reshape((-1,2),order='F')
    fig3 = plt.figure()
    plt.boxplot(dados)
    plt.show()
    
    #regressão linear
    a = regressaolinear().Ajuste(x1,x2,y)
    # cria um grid
    x1v = np.linspace(np.min(x1),np.max(x1),num=80)
    x2v = np.linspace(np.min(x2),np.max(x2),num=80)
    x1m,x2m = np.meshgrid(x1v,x2v)
    ym = regressaolinear().AvaliaReta(x1m,x2m,a)
    cor = np.array(ym)
    cor[cor<=.5] = 0
    cor[cor>.5] = 1
    fig4 = plt.figure()
    plt.scatter(x1,x2,s=50,c=y)
    plt.scatter(x1m,x2m,s=0.5,c=cor)
    plt.show()