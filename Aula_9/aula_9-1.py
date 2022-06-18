import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
import tkinter as tk
import tkinter.messagebox as msg
import scipy.stats as sci
import best_reg_linear as best
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn import mixture
from sklearn.naive_bayes import GaussianNB


class analisegrafica() :
    def __init__(self , master ) :
        self.master = master


        self.num_classes = 2
        self.alfa = 0.05 
        self.ml_procedure = []
        self.IniciaGrafico()

    def IniciaGrafico( self ) :
        # Menu principal
        menubar = tk.Menu( self.master )
        self.master.config(menu=menubar)
        # Carregar Arquivos
        arquivo_menu= tk.Menu(menubar)
        arquivo_menu.add_command(label="Carregar Features Treino ...",command=self.CarregarFeaturesTreino)
        arquivo_menu.add_command(label="Carregar Labels Treino",command=self.CarregarLabelsTreino)
        arquivo_menu.add_separator()
        arquivo_menu.add_command(label="Carregar Features Valida ...",command=self.CarregarFeaturesValida)
        arquivo_menu.add_command(label="Carregar Labels Valida",command=self.CarregarLabelsValida)
        arquivo_menu.add_separator()
        arquivo_menu.add_command(label="Carregar Features Teste ...",command=self.CarregarFeaturesTeste)
        arquivo_menu.add_command(label="Carregar Labels Teste",command=self.CarregarLabelsTeste)
        menubar.add_cascade(label="Arquivo", menu=arquivo_menu)
#        menubar.update()

        # Carregar os Dados para ficar mais rápido
        self.features = np.loadtxt("features_treino.txt")
        self.Normaliza_z()
        self.labels = np.loadtxt("labels_treino.txt")
        self.x_teste = np.loadtxt("features_teste.txt")
        self.x_teste = self.Normaliza( self.x_teste )
        self.y_teste = np.loadtxt("labels_teste.txt")


        # Opções gráficas
        graficos_menu= tk.Menu(menubar)
        graficos_menu.add_command(label="Dispersão" , command = self.GraficoDisper)
        graficos_menu.add_command(label="Boxplot" , command = self.GraficoBoxPlot)
        graficos_menu.add_command(label="Teste t" , command = self.GraficoT)
        graficos_menu.add_command(label="Teste U" , command = self.GraficoU)
        graficos_menu.add_command(label="Valores Singulares" , command = self.SingVal)
        menubar.add_cascade(label="Gráficos", menu = graficos_menu )
#        menubar.update()

        ajustes_menu= tk.Menu(menubar)
        ajustes_menu.add_command(label="Regressão Linear Ótima " , command = self.Otimizar)    
        ajustes_menu.add_command(labe='Naive Bayes', command=self.NaiveBG)
        ajustes_menu.add_command(label='Kmeans', command=self.Kmeans)
        ajustes_menu.add_command(label='Logística',command=self.Logistica)
        ajustes_menu.add_command(label='Mistura Gaussiana',command=self.MisturaGaussiana)
        menubar.add_cascade(label="Testar Modelos", menu = ajustes_menu )

#Opções Ferramentas

        ferramenta_menu = tk.Menu(menubar)
        ferramenta_menu.add_command(label="Bootstrap Parametrico",command=self.Bootstrap)
        ferramenta_menu.add_command(label="AIC",command=self.AIC)
        menubar.add_cascade(label="Ferramentas", menu=ferramenta_menu)
        menubar.update()




# Funções arquivo

    def CarregarFeaturesTreino( self ) :
        filename = tk.filedialog.askopenfilename(title = "Escolha o Arquivo Features Treino",
        filetypes = [("Arquivos extensão ASCII ","*.txt")])
        if filename :
            self.features = np.loadtxt(filename)
            self.Normaliza_z() 
        
    def CarregarLabelsTreino( self ) :
        filename = tk.filedialog.askopenfilename(title = "Escolha o Arquivo Labels Treino",
        filetypes = [("Arquivos extensão ASCII ","*.txt")] )
        if filename :
            self.labels = np.loadtxt(filename) 

    def CarregarFeaturesValida( self ) :
        filename = tk.filedialog.askopenfilename(title = "Escolha o Arquivo Features Valida",
        filetypes = [("Arquivos extensão ASCII ","*.txt")])
        if filename :
            self.features_valida = np.loadtxt(filename)
            self.Normaliza( self.features_valida ) 
        
    def CarregarLabelsValida( self ) :
        filename = tk.filedialog.askopenfilename(title = "Escolha o Arquivo Labels Valida",
        filetypes = [("Arquivos extensão ASCII ","*.txt")] )
        if filename :
            self.labels_valida = np.loadtxt(filename) 

    def CarregarFeaturesTeste( self ) :
        filename = tk.filedialog.askopenfilename(title = "Escolha o Arquivo Features Teste",
        filetypes = [("Arquivos extensão ASCII ","*.txt")])
        if filename :
            self.features_teste = np.loadtxt(filename)
            self.Normaliza( self.features_teste ) 
        
    def CarregarLabelsTeste( self ) :
        filename = tk.filedialog.askopenfilename(title = "Escolha o Arquivo Labels Valida",
        filetypes = [("Arquivos extensão ASCII ","*.txt")] )
        if filename :
            self.labels_teste = np.loadtxt(filename) 



# Funções Gráficos
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
        n_features = self.features.shape[1]

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
            # Teste de Mann Whitney 
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



# Funções testa modelos
    def Otimizar( self ) : 
        otim_rl = best.bestreglinear( num_par = 100 , alfa = .05 )
        mat_posic = otim_rl.MelhoresCoeficientes( self.features , self.labels )
        np.savetxt( 'posic.txt' , mat_posic )
        # Construir o modelo
        otim_rl.mat_posic = mat_posic
        otim_rl.fit( self.features , self.labels )
        d = otim_rl.predict( self.features ) 
        sen, esp, prec = self.Score( self.labels , d )
        txt = ' Sensibilidade = %5.1f \n Especificidade = %5.1f  \n Precisão = %5.1f ' %(sen , esp , prec )
        msg.showinfo(title='Métricas do Grupo Treinamento', message = txt )  
    
        # grupo de teste  
        Xn = np.copy( self.x_teste ) 
        y = np.copy( self.y_teste )
        d = otim_rl.predict( Xn )
        sen, esp, prec = self.Score( y , d )
        txt = ' Sensibilidade = %5.1f \n Especificidade = %5.1f  \n Precisão = %5.1f ' %(sen , esp , prec )
        msg.showinfo(title='Métricas do Grupo Teste', message = txt )  
        self.ml_procedure = otim_rl

    def NaiveBG( self ) : 
        NBG = GaussianNB(priors = None) 
        # Construir o modelo
        NBG.fit( self.features , self.labels )
        d = NBG.predict( self.features ) 
        sen, esp, prec = self.Score( self.labels , d )
        txt = ' Sensibilidade = %5.1f \n Especificidade = %5.1f  \n Precisão = %5.1f ' %(sen , esp , prec )
        msg.showinfo(title='Métricas do Grupo Treinamento', message = txt )  
    
        # grupo de teste  
        Xn = np.copy( self.x_teste ) 
        y = np.copy( self.y_teste )
        d = NBG.predict( Xn )
        sen, esp, prec = self.Score( y , d )
        txt = ' Sensibilidade = %5.1f \n Especificidade = %5.1f  \n Precisão = %5.1f ' %(sen , esp , prec )
        msg.showinfo(title='Métricas do Grupo Teste', message = txt )  
        self.ml_procedure = NBG

    def Kmeans( self ) : 
        km = KMeans(n_clusters=4,init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='auto') 
        # Construir o modelo
        km.fit( self.features , self.labels )
        d = km.predict( self.features ) 
        sen, esp, prec = self.Score( self.labels , d )
        txt = ' Sensibilidade = %5.1f \n Especificidade = %5.1f  \n Precisão = %5.1f ' %(sen , esp , prec )
        msg.showinfo(title='Métricas do Grupo Treinamento', message = txt )  
    
        # grupo de teste  
        Xn = np.copy( self.x_teste ) 
        y = np.copy( self.y_teste )
        d = km.predict( Xn )
        sen, esp, prec = self.Score( y , d )
        txt = ' Sensibilidade = %5.1f \n Especificidade = %5.1f  \n Precisão = %5.1f ' %(sen , esp , prec )
        msg.showinfo(title='Métricas do Grupo Teste', message = txt )  
        self.ml_procedure = km
    
    def Logistica( self ) : 
        log = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=None)
        # Construir o modelo
        log.fit( self.features , self.labels )
        d = log.predict( self.features ) 
        sen, esp, prec = self.Score( self.labels , d )
        txt = ' Sensibilidade = %5.1f \n Especificidade = %5.1f  \n Precisão = %5.1f ' %(sen , esp , prec )
        msg.showinfo(title='Métricas do Grupo Treinamento', message = txt )  
    
        # grupo de teste  
        Xn = np.copy( self.x_teste ) 
        y = np.copy( self.y_teste )
        d = log.predict( Xn )
        sen, esp, prec = self.Score( y , d )
        txt = ' Sensibilidade = %5.1f \n Especificidade = %5.1f  \n Precisão = %5.1f ' %(sen , esp , prec )
        msg.showinfo(title='Métricas do Grupo Teste', message = txt )  
        self.ml_procedure = log

    def MisturaGaussiana( self ) : 
        mg = mixture.GaussianMixture(n_components=2, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10)
        # Construir o modelo
        mg.fit( self.features , self.labels )
        d = mg.predict( self.features ) 
        sen, esp, prec = self.Score( self.labels , d )
        txt = ' Sensibilidade = %5.1f \n Especificidade = %5.1f  \n Precisão = %5.1f ' %(sen , esp , prec )
        msg.showinfo(title='Métricas do Grupo Treinamento', message = txt )  
    
        # grupo de teste  
        Xn = np.copy( self.x_teste ) 
        y = np.copy( self.y_teste )
        d = mg.predict( Xn )
        sen, esp, prec = self.Score( y , d )
        txt = ' Sensibilidade = %5.1f \n Especificidade = %5.1f  \n Precisão = %5.1f ' %(sen , esp , prec )
        msg.showinfo(title='Métricas do Grupo Teste', message = txt )  
        self.ml_procedure = mg

    #    self.LinSen( features = features )

#Funções Ferramentas
    def Bootstrap(self):
        if self.ml_procedure == []: #Não tem modelo ajustado
            msg.showerror('ERRO ', 'Precisa Ajustar um Modelo Primeiro')
            return
        N = len(self.labels)
        indice = np.arange(N)
        boot = np.zeros(500)
        for i in range (500):
            indice_boot = np.random.choice(indice, size=N, replace=True, p=None)
            x = self.features[indice_boot, :]
            y = self.labels[indice_boot]
            self.ml_procedure.fit(x,y)
            d = self.ml_procedure.predict(self.x_teste)
            sen,esp,prec = self.Score(self.y_teste, d)
            boot[i] = prec
            if i%20==0:
                print(i)
        alfa = .05
        IC_inf = np.quantile(boot,alfa/2)
        IC_sup = np.quantile(boot, 1-alfa/2)
        txt = "[%6.2f %6.2f]" %(IC_inf,IC_sup)
        msg.showinfo("IC (95%) = ", txt)

    def AIC(self):
        if self.ml_procedure == []: #Não tem modelo ajustado
            msg.showerror('ERRO ', 'Precisa Ajustar um Modelo Primeiro')
            return
        N = len(self.labels)
        p = len(self.ml_procedure.get_params())
        if p >= N:
            p = N-1e-12
        x_treinamento = self.features
        y_treinamento = self.labels
        y_fit = self.ml_procedure.predict(self.features)
        SSE = np.sum((y_fit-y_treinamento)**2) + 1e-6
        loglik = -N*(np.log(2*np.pi)+np.log((SSE)/(N-p))+(N-p)/N)/2
        aic = -(2/N)*loglik+2*p/N
        txt = "%7.1f"%aic
        msg.showinfo('AIC = ', txt)






# ---------------------------------------------------- rotinas de Apoio --------------------------------------------------------------

    def Score( self , y , d ) : 
    # Sensibilidade = 100 * TP / (TP + FN )
        TP = 0
        TN = 0 
        FN = 0
        FP = 0 
        N = len(d)
        for i in range( N ) : 
            if y[i] == 0 : 
                if d[i] == 0 :
                    TN +=1  
                else : 
                    FP +=1
            else : 
                if d[i] == 1 : 
                    TP += 1
                else : 
                    FN += 1
        # Sensibilidade = 100 * TP / (TP + FN )
        sen = 100 * TP / (TP + FN )
        # Especificidade = 100 * TN /(TN+FP )                            
        esp = 100 * TN / (TN + FP)
        # Precisão 100*(TP+FN)/ N 
        prec = 100 * (TP+TN) / N
        if sen<50:
            TP = 0
            TN = 0 
            FN = 0
            FP = 0 
            N = len(d)
            for i in range( N ) : 
                if y[i] == 1 : 
                    if d[i] == 0 :
                        TN +=1  
                    else : 
                        FP +=1
                else : 
                    if d[i] == 0 : 
                        TP += 1
                    else : 
                        FN += 1
        # Sensibilidade = 100 * TP / (TP + FN )
            sen = 100 * TP / (TP + FN )
        # Especificidade = 100 * TN /(TN+FP )                            
            esp = 100 * TN / (TN + FP)
        # Precisão 100*(TP+FN)/ N 
            prec = 100 * (TP+TN) / N
        return sen , esp , prec
    
    def Normaliza_z ( self ) : 
        self.media = np.mean( self.features , axis = 0 )
        self.st = np.std( self.features , axis = 0 )
        self.features = ( self.features - self.media ) / (3*self.st)     
        m,n = self.features.shape
        for i in range( n ) : # limitar a features
            self.features[self.features[:,i]>1,i] = 1
            self.features[self.features[:,i]<-1,i] = -1

    def Normaliza ( self , X ) : 
        features = ( X - self.media ) / (3*self.st)     
        m,n = features.shape
        for i in range( n ) : # limitar a features
            features[features[:,i]>1,i] = 1
            features[features[:,i]<-1,i] = -1        
        return features



   

# programa principal
if __name__ == "__main__" :
    root = tk.Tk()
    root.title("Análise Gráfica")
    root.geometry('800x600+50+50')
    analisegrafica(root)
    root.mainloop()

 