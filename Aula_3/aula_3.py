import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt

class regressaolinear() :
    def __init__( self ) : # Inicialização da classe
        pass
    def ajusta( self, x , y ) :
        # Dado os vetores de entrada x e saída y, esta funcão ajusta
        # f(x) = a0 + x*a1 via mínimos quadrados e retorna os coeficientes no vetor a
        npto = len(x)
        um = np.ones( npto )
        X = np.array( [ um , x ] )
        a = np.linalg.inv( X @ np.transpose(X) ) @ ( X @ np.transpose( y ) )
        return a   
    
    def AvaliarReta(self, x , a ) : 
        y = x*a[1] + a[0]
        return y      

   
# programa principal
if __name__ == "__main__" :

    mat=np.loadtxt('treino_1.txt')

    # Vamos dar uma olhadinha nos dados
    ax1=plt.subplot(2, 1 ,  1)
    ax1.plot( mat[:,0] )
    ax2=plt.subplot(2, 1 , 2 )
    ax2.plot( mat[:,1] )
    plt.show()

    #vVamos ver um grafico de dispersão 
    x = np.linspace( 0 , 1 , num = 40 )
    x = np.append( x , x )
    c = mat[:,1] 

    plt.scatter( x , mat[:,0] , s = 50 , c = c )
    plt.show() 

    # Vamos ver os boxplot
    dados = mat[:,0].reshape( (-1,2) , order = 'F') 
    plt.boxplot(dados)
    plt.show()

    # Regressão Linear
    feat = mat[:,0]
    targ = mat[:,1]
    a = regressaolinear().ajusta( feat , targ )
    # Monta uma matriz de features teóricas
    feat_min = np.min( feat )
    feat_max = np.max( feat ) 

    y = (x[0:40] - a[0] ) / a[1] 
    plt.plot( x[0:40] , y  )  
    # Dados do treinamento
    plt.scatter( x , mat[:,0] , s = 50 , c = c )
    plt.show()        
    i=1
    #for i in range()
    