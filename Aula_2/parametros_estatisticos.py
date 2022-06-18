import numpy as np
from scipy import signal as sig

#------------------Parâmetros Estatísticos-----------------
class calcula_estatistica():
    def __init__(self):
        pass

#-----------------Regularizar Matriz-----------------------    
    def RegularizaMatriz(self,dados):
        #Dado uma array dados, esta função retorna uma matriz dados_reg organizada por linhas
        
        dados_reg = np.array(dados)
        dim = dados_reg.ndim
        if dim == 1: #dados é vetor
            dados_reg = dados_reg.reshape([1,len(dados)])
        else: #Veriicar se a matriz está organizada por linhas
            n_lin, n_col = dados_reg.shape
            if n_lin > n_col: #Transpor a Matriz
                dados_reg = np.transpose(dados_reg)
        return dados_reg
    
#----------------Momento Estatístico de Primeira Ordem - Média--
    def Media(self, dados):
        #retorna a média amostral de dados
        dados_reg = self.RegularizaMatriz(dados)
        media = np.sum(dados_reg,axis=1)/dados_reg.shape[1]
        return media
    
#----------------Momento Estatístico de Segunda Ordem - Valor Médio Quadrático
    def M2(self,dados,rms=False):
        #retorna os momentos de segunda ordem da array dados 
        #se rms=True: retorna o valor médio quadrático
        dados_reg = self.RegularizaMatriz(dados)
        npto = dados_reg.shape[1]
        m2 = np.sum(dados_reg*dados_reg,axis=1)/npto
        if rms == True:
            m2 = np.sqrt(m2)
        return m2
        

