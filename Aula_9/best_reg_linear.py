import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
import tkinter as tk
import scipy.stats as sci

class bestreglinear() :
    def __init__(self , num_par = -1 , alfa = .1 , mat_posic = [] , n_components = 2 ) :
        self.num_par = num_par
        self.alfa = alfa
        self.coef = [] # Vetoe com os coeficientes da Regressão
        self.mat_posic = mat_posic # Vetor com as posiçoes para montar um modelo linear completo
        self.n_components = n_components

    def fit( self , x , y  ) :
        n_lin = x.shape[0]
        um = np.ones(n_lin).reshape(-1,1)
        aux = np.concatenate( ( um , x ) , axis = 1 )
        if self.mat_posic != [] : # Posições para montar a matriz de sensibilidade foi passada
            X = aux[:,self.mat_posic[:,0]] * aux[:,self.mat_posic[:,1]]
        else : 
            X = aux       
        Xt = np.transpose( X )
        A = Xt @ X 
        if np.linalg.det( A ) < 1e-15 : # Matriz SIngular
            self.coef = [] 
            return
        self.coef = np.linalg.inv( A ) @ Xt @ y

    # Classificação
    def predict( self , x  ) : 
        n_lin = x.shape[0]
        um = np.ones(n_lin).reshape(-1,1)
        aux = np.concatenate( ( um , x ) , axis = 1 )
        if self.mat_posic != [] : # Posições para montar a matriz de sensibilidade foi passada
            X = aux[:,self.mat_posic[:,0]] * aux[:,self.mat_posic[:,1]]
        else : 
            X = aux       
        yt = X @ self.coef
        yt[ yt < .5 ] = 0
        yt[ yt >= .5 ] = 1
        return yt

    def get_params(self):
        return self.coef

    
    # Apoio a Otimizaçcão 

    def fit_otim( self , x , y  ) :
        X = np.copy( x )       
        Xt = np.transpose( X )
        A = Xt @ X 
        if np.linalg.det( A ) < 1e-15 : # Matriz SIngular
            self.coef = [] 
            return
        self.coef = np.linalg.inv( A ) @ Xt @ y
 
      
    def predit_otim( self , x  ) : 
        n_lin = x.shape[0]
        if self.coef == [] : 
            return np.zeros( n_lin )
        X = np.copy( x )
        yt = X @ self.coef
        return yt



    # Processo Forward para escolha dos melhores coeficientes     

    def TesteF( self , e12 , e22 , ngdl_num , ngdl_den ) :
        f_critico = sci.f.ppf( 1 - self.alfa , ngdl_num , ngdl_den )
        f = ( np.abs( e12 - e22 ) / ngdl_num ) / ( e22 / ngdl_den ) 
        return f / f_critico

    def MelhoresCoeficientes( self , x , y ) : 
        n_lin , n_features = x.shape
        um = np.ones(n_lin).reshape(-1,1)
        X = np.concatenate( ( um , x ) , axis = 1 )
        n_col = n_features + 1
        ii = 1
        for i in range( n_col - 1 ) : 
            for j in range ( i , n_col ) :
                ii = ii + 1
        mat_posic = np.zeros([ ii , 2 ] , dtype = int )
        S = np.zeros( [ n_lin , ii ] )
        ii = int(0)  
        for i in range(n_col-1) :
            for j in range(i,n_col) :
                S[:,ii] = X[:,i]*X[:,j]    
                mat_posic[ii,:] = [ i , j ]     
                ii = ii + 1
        Y = np.array( y ).reshape(-1,1) 
        n_lin , n_col = S.shape    
        if self.num_par < 1 :
            self.num_par = n_col 
        posic = np.zeros( self.num_par , dtype = int ) # guardar a posição de X
        i_cont = 0
        # intercepto 
        a0 = np.sum( Y ) / n_lin 
        e12 = np.sum( ( Y - a0 ) ** 2 )
        X = S 
        X0 = np.zeros( ( n_lin , self.num_par ) ) 
        posic[i_cont] = 0
        X0[:,0] = X[:,0]
        X[:,0] = 0 # zera coluna para nao computar novamente
        ngdl1 = 2
        i_cont += 1
        for i in range( 1 , self.num_par  ) :
            f_max = 0 
            j_max = 0
            ngdl2 = ngdl1 + 1 
            for j in range( 1 , n_col ) :
                if np.abs( np.sum( X[:,j] ) ) > 1e-12 :
                    X0[:,i] = X[:,j]  
                    self.fit_otim( X0[:,0:i+1] , Y )
                    yt = self.predit_otim( X0[:,0:i+1] )
                    e22 = np.sum( ( Y - yt ) ** 2 ) 
                    f = self.TesteF( e12 , e22 , 1 , n_lin - ngdl2 )
                    if f > f_max : 
                        f_max = f 
                        j_max = j
                        e22_max = e22 
            if f_max < 1 : # acabou 
                i_cont -= 1 # O último não passou no teste F
                break
            print( [e22_max , f_max ] )    
            posic[i_cont] = j_max
            X0[:,i] = X[:,j_max]
            X[:,j_max] = 0 # zera coluna para nao computar novamente
            e12 = e22_max
            ngdl1 = ngdl2
            i_cont += 1
        posic = posic[0:i_cont]    
        return mat_posic[posic,:] 








          
        