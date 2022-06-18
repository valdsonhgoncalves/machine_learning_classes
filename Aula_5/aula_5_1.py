import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
import tkintertable as tk
import scipy.stats as sci

class analisegrafica() :
    def __init__(self , master ) :
        self.master = master
        self.features = []
        self.labels = []
        self.media = []
        self.st = []
        self.num_classes = 2
        self.alfa = 0.05 
        self.IniciaGrafico()

    def IniciaGrafico( self ) :
        # Menu principal
        menubar = tk.Menu( self.master )
        self.master.config(menu=menubar)
        # Carregar Arquivos
        arquivo_menu= tk.Menu(menubar)
        arquivo_menu.add_command(label="Carregar Features ...",command=self.CarregarFeatures)
        arquivo_menu.add_command(label="Carregar Labels",command=self.CarregarLabels)
        arquivo_menu.add_separator()
        menubar.add_cascade(label="Arquivo", menu=arquivo_menu)
        menubar.update()

        # Opções gráficas
        graficos_menu= tk.Menu(menubar)
        graficos_menu.add_command(label="Dispersão" , command = self.GraficoDisper)
        graficos_menu.add_command(label="Boxplot" , command = self.GraficoBoxPlot)
        graficos_menu.add_command(label="Teste t" , command = self.GraficoT)
        graficos_menu.add_command(label="Teste U" , command = self.GraficoU)
        graficos_menu.add_command(label="Valores Singulares" , command = self.SingVal)
        graficos_menu.add_command(label="Sensibilidade Linear" , command = self.LinSen)
        menubar.add_cascade(label="Gráficos", menu = graficos_menu )
        menubar.update()


# Funções command

    def CarregarFeatures( self ) :
        filename = tk.filedialog.askopenfilename(title = "Escolha o Arquivo com os Features",
        filetypes = [("Arquivos extensão ASCII ","*.txt")])
        if filename :
            self.features = np.loadtxt(filename) 
        
    def CarregarLabels( self ) :
        filename = tk.filedialog.askopenfilename(title = "Escolha o Arquivo com os Labels",
        filetypes = [("Arquivos extensão ASCII ","*.txt")])
        if filename :
            self.labels = np.loadtxt(filename) 

    def GraficoDisper(self) : 
        nptos , n_features = self.features.shape
        if self.media == [] : 
            self.Normaliza_z()
        # criar eixo x 
        eixo_x = np.linspace(0 , 20 , num = nptos )
        # montar a matriz de graficos
        if n_features < 3 :
            graf_x = 3
            graf_y = 1
        else : 
            graf_y = int( np.sqrt( n_features / 1.5 ) ) + 1
            graf_x = n_features // graf_y
        if graf_x * graf_y < n_features : 
            graf_x += 1               
        for i in range( n_features ) : 
            ax = plt.subplot( graf_x , graf_y , i+1 )
            ax.scatter( eixo_x , self.features[:,i] , s = 20 , c = self.labels)
            ax.grid( True )
        plt.show()    

    def GraficoBoxPlot(self) : 
        nptos , n_features = self.features.shape
        if self.media == [] : 
            self.Normaliza_z()
        # montar a matriz de graficos
        if n_features < 3 :
            graf_x = 3
            graf_y = 1
        else : 
            graf_y = int( np.sqrt( n_features / 1.5 ) ) + 1
            graf_x = n_features // graf_y
        if graf_x * graf_y < n_features : 
            graf_x += 1    
        fig=plt.figure()    
        for i in range( n_features ) : 
            # Separar os labels
            aux = self.features[ self.labels == 0 , i ] 
            data = aux.tolist()  
            for k in range( 1 , self.num_classes ) :
                aux = self.features[ self.labels == k , i ] 
                data = [ data , aux ]   
            ax = fig.add_subplot( graf_x , graf_y , i+1 )
            ax.boxplot( data  )
            ax.grid( True )
        plt.show()    

    def GraficoT( self ) : 
        nptos , n_features = self.features.shape
        if self.num_classes > 2 :
            tk.messagebox.showerror( 'ATENÇÃO' , "Teste t não pode ser usada para número de classes > 2 ") 
            return
        # Valor crítico do teste    
        t_critico = sci.t.ppf( 1 - self.alfa , nptos - 2 )
        teste_result = np.zeros( n_features ) 
        for i in range( n_features ) :
            amostra_1 = self.features[ self.labels == 0 , i ]
            amostra_2 = self.features[ self.labels == 1 , i ] 
            # Teste de Bartelett para igualdade de variâncias
            stat, p_value = sci.bartlett( amostra_1 , amostra_2)
            if p_value < self.alfa :
                equal_var = True
            else :
                equal_var = False
            stat , p_value = sci.ttest_ind( amostra_1 , amostra_2 , equal_var = equal_var )
            teste_result[ i ] = stat / t_critico   
        fig = plt.figure()      
        ax = fig.add_subplot(1, 1, 1)     
        ax.plot( teste_result ) 
        ax.grid( True )
        plt.show() 

    def GraficoU( self ) : 
        nptos , n_features = self.features.shape
        if self.num_classes > 2 :
            tk.messagebox.showerror( 'ATENÇÃO' , "Teste U não pode ser usada para número de classes > 2 ") 
            return
        if nptos < 20 :
            tk.messagebox.showerror( 'ATENÇÃO' , "Teste U não pode ser usada para amostras menores do que 20 ") 
            return    
        teste_result = np.zeros( n_features ) 
        for i in range( n_features ) :
            amostra_1 = self.features[ self.labels == 0 , i ]
            amostra_2 = self.features[ self.labels == 1 , i ] 
            # Teste de Bartelett para igualdade de variâncias
            stat, p_value = sci.mannwhitneyu( amostra_1 , amostra_2)
            n1 = len( amostra_1 )
            n2 = len( amostra_2 )
            xb = n1 * n2 / 2
            st = np.sqrt( n1 * n2 * ( n1 + n2 + 1 ) / 12 )
            n_critico = sci.norm.ppf( 1 - self.alfa , loc = xb , scale = st )
            teste_result[ i ] = stat / n_critico    
        fig = plt.figure() 
        ax = fig.add_subplot(1, 1, 1)           
        ax.plot( teste_result ) 
        ax.grid( True )
        plt.show()

    def SingVal( self ) : 
        nptos , n_features = self.features.shape
        X = np.transpose( self.features ) @ self.features
        u, s , vt = np.linalg.svd( X ) 
        fig = plt.figure()
        ax = fig.add_subplot( 1 , 1 , 1 )
        ax.plot( s , '*-')
        ax.grid( True )
        plt.show() 

    def LinSen( self ) :
        nptos , n_features = self.features.shape
        X = np.concatenate( (np.ones( nptos ).reshape(-1,1) , self.features ) , axis = 1 ) 
        Xt = np.transpose( X )
        y = self.labels.reshape(-1,1)
        det = np.linalg.det( Xt @ X )
        if det < 1e-12 : # norma mínima
            u , s , vt = np.linalg.svd( X , full_matrices = False )
            s_min = s[ 1 ] / 1e8 
            s[ s < s_min ] = 1e12 
            # estimar coeficientes
            a = vt @ np.transpose( u ) @ y / s.rehape(-1,1) 
            fig = plt.figure()
            ax = fig.add_subplot( 1 , 1 , 1 )
            ax.plot( a[1:] , '*-')
            ax.grid( True )
            plt.show()
        else : # mínimos quadrados
            a = np.linalg.inv( Xt @ X ) @ Xt @ y
            yt = X @ a
            e = y - yt
            e2 = np.sum( e*e )
            s2e = e2 / ( nptos - n_features -1 ) # estimativa da variância experimental 
            aux = np.linalg.inv(Xt @ X ) 
            s2a = np.diag( aux ) * s2e 
            sa = np.sqrt( s2a )
            t = sci.t.ppf( 1 - self.alfa / 2 , nptos - n_features - 1 )
            IC = t * sa 
            fig = plt.figure()
            ax = fig.add_subplot( 1 , 1 , 1 )
            ax.errorbar( np.linspace(1 , n_features , num = n_features ), a[1:] , yerr = IC[1:] )
            ax.grid( True )
            plt.show()        



# ---------------------------------------------------- rotinas --------------------------------------------------------------
    def Normaliza_z ( self ) : 
        self.media = np.mean( self.features , axis = 0 )
        self.st = np.std( self.features , axis = 0 )
        self.features = ( self.features - self.media ) / self.st        

# programa principal
if __name__ == "__main__" :
    root = tk.Tk()
    root.title("Análise Gráfica")
    root.geometry('800x600+50+50')
    analisegrafica(root)
    root.mainloop()

 