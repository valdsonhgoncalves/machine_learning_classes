import numpy as np
import scipy
import matplotlib.pyplot as plt 

np.random.seed(42)
#Eixos e nós
x_min, x_knot_1, x_knot_2, x_max = -1, 1.5, 4.5, 7

# Gerar o modelo
x_true = np.linspace(x_min,x_max,50)
y_true = np.cos(x_true)
y_obs = y_true + np.random.normal(scale=0.5,size=len(x_true))

# Obter x-y das regiões
x_region_1 = x_true[x_true <= x_knot_1]
x_region_2 = x_true[(x_true > x_knot_1) & (x_true < x_knot_2)]
x_region_3 = x_true[x_true >= x_knot_2]
y_region_1 = y_true[x_true <= x_knot_1]
y_region_2 = y_true[(x_true > x_knot_1) & (x_true < x_knot_2)]
y_region_3 = y_true[x_true >= x_knot_2]

#Ajuste pelo intercepto
plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
plt.plot(x_true, y_true, linewidth = 3, c = 'darkorange')
plt.scatter(x_true,y_obs)

#plotar região de separação
plt.axvline(x=x_knot_1, c='gray',ls='--')
plt.axvline(x=x_knot_2,c='gray',ls='--')

#plotar os interceptos
plt.axhline(y=y_region_1.mean(), c='green',xmin=0,xmax=.33)
plt.axhline(y=y_region_2.mean(), c='green',xmin=.33,xmax=.66)
plt.axhline(y=y_region_1.mean(), c='green',xmin=.66,xmax=1.0)

# Regressão Linear a1 = x1*(y-a2)/x1*x1

beta_region_1 = ((y_region_1-y_region_1.mean()).dot(x_region_1)/(x_region_1**2).sum())
beta_region_2 = ((y_region_2-y_region_2.mean()).dot(x_region_2)/(x_region_2**2).sum())
beta_region_3 = ((y_region_3-y_region_3.mean()).dot(x_region_3)/(x_region_3**2).sum())

#y = a0+a1*x
y_hat_region_1 = beta_region_1*x_region_1 + np.mean(y_region_1)
y_hat_region_2 = beta_region_2*x_region_2 + np.mean(y_region_2)
y_hat_region_3 = beta_region_3*x_region_3 + np.mean(y_region_3)

#Plotar
plt.subplot(2,2,2)
plt.plot(x_true, y_true, linewidth = 3, c = 'darkorange')
plt.scatter(x_true,y_obs)

#plotar região de separação
plt.axvline(x=x_knot_1, c='gray',ls='--')
plt.axvline(x=x_knot_2,c='gray',ls='--')

#plotar a curva teorica
plt.plot(x_region_1, y_hat_region_1, c='green')
plt.plot(x_region_2, y_hat_region_2, c='green')
plt.plot(x_region_3, y_hat_region_3, c='green')

#Obrigar a continuidade das retas
h1 = np.ones_like(x_true)
h2 = np.copy(x_true)
h3 = np.where(x_true<x_knot_1,0,x_true-x_knot_1)
h4 = np.where(x_true<x_knot_2,0,x_true-x_knot_2)
H = np.vstack((h1,h2,h3,h4)).T
HH = H.T @ H 
beta = np.linalg.solve(HH,H.T @ y_obs)
y_hat = H @ beta

plt.subplot(2,2,3)
plt.plot(x_true, y_true, linewidth = 3, c = 'darkorange')
plt.scatter(x_true,y_obs)

#plotar região de separação
plt.axvline(x=x_knot_1, c='gray',ls='--')
plt.axvline(x=x_knot_2,c='gray',ls='--')

#plotar a curva teorica
plt.plot(x_true, y_hat, c='green')

#Obrigar a continuidade das retas
h1 = np.ones_like(x_true)
h2 = np.copy(x_true)
h3 = h2**2
h4 = h2**3
h5 = np.where(x_true<x_knot_1,0,(x_true-x_knot_1)**3)
h6 = np.where(x_true<x_knot_2,0,(x_true-x_knot_2)**3)
H = np.vstack((h1,h2,h3,h4,h5,h6)).T
HH = H.T @ H 
beta = np.linalg.solve(HH,H.T @ y_obs)
y_hat = H @ beta

plt.subplot(2,2,4)
plt.plot(x_true, y_true, linewidth = 3, c = 'darkorange')
plt.scatter(x_true,y_obs)

#plotar região de separação
plt.axvline(x=x_knot_1, c='gray',ls='--')
plt.axvline(x=x_knot_2,c='gray',ls='--')

#plotar a curva teorica
plt.plot(x_true, y_hat, c='green')

plt.show()
