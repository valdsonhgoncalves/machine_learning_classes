import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig



# Construindo um sinal estacionário 

dt = 1 / 256
t = np.linspace(0,1-dt,256)
x = 2*np.sin(2*np.pi*10*t) + np.sin(2*np.pi*25*t) + .5*np.sin(2*np.pi*50*t) +.25*np.sin(2*np.pi*100*t)
# Aplicando a fft
X = np.fft.fft(x/256)
plt.figure(figsize=(12, 10))
plt.subplot(2,1,1)
plt.plot(t,x , c = 'red' , linewidth = 1 )
plt.grid(True)
plt.xlabel('tempo [s]')
plt.ylabel('x(t) [mm/s]')
plt.subplot(2,1,2)
plt.plot(2*np.abs(X[0:128]) , c = 'red' , linewidth = 3 )
plt.grid(True)
plt.xlabel('frequência [Hz]')
plt.ylabel('|x(f)| [mm/s]')
plt.show()
# costruindo o sinal não estacionário na janela de amostragem
t1,t2,t3,t4=np.copy(t),np.copy(t),np.copy(t),np.copy(t)
t1[t>.2] = 0
t2[t<.2] = 0
t2[t>.4] = 0
t3[t<.4 ] = 0
t3[t>.8] = 0
t4[t<.8] = 0 
x = np.sin(2*np.pi*10*t1) + np.sin(2*np.pi*25*t2) + np.sin(2*np.pi*50*t3) +np.sin(2*np.pi*100*t4)
X = np.fft.fft(x/256)
plt.figure(figsize=(12, 10))
plt.subplot(2,1,1)
plt.plot(t,x , c = 'red' , linewidth = 1 )
plt.grid(True)
plt.xlabel('tempo [s]')
plt.ylabel('x(t) [mm/s]')
plt.subplot(2,1,2)
plt.plot(2*np.abs(X[0:128]) , c = 'red' , linewidth = 3 )
plt.grid(True)
plt.xlabel('frequência [Hz]')
plt.ylabel('|x(f)| [mm/s]')
plt.show()
# Aplicando a transformada curta de Fourier do scipy
f_stft, t_stft, Zxx = sig.stft(x, fs=256, window='hann', nperseg=32, noverlap=16, nfft=32) 
plt.figure(figsize=(12, 10))
plt.subplot(2,1,1)
plt.plot(t,x , c = 'red' , linewidth = 1 )
plt.grid(True)
plt.xlabel('tempo [s]')
plt.ylabel('x(t) [mm/s]')
plt.subplot(2,1,2)

plt.pcolormesh(t_stft, f_stft, 2*np.abs(Zxx) , vmin=0)
plt.colorbar()
plt.grid(True)
plt.xlabel('tempo [s]')
plt.ylabel('frequência [Hz]')
plt.show()
# Aumentar resolução em frequência
f_stft_1, t_stft_1, Zxx_1 = sig.stft(x, fs=256, window='hann', nperseg=64, noverlap=32, nfft=64) 
plt.figure(figsize=(12, 10))
plt.subplot(2,1,1)
plt.pcolormesh(t_stft, f_stft, 2*np.abs(Zxx) , vmin=0)
plt.colorbar()
plt.grid(True)
plt.xlabel('tempo [s]')
plt.ylabel('frequência [Hz]')
plt.subplot(2,1,2)
plt.pcolormesh(t_stft_1, f_stft_1, 2*np.abs(Zxx_1) , vmin=0)
plt.colorbar()
plt.grid(True)
plt.xlabel('tempo [s]')
plt.ylabel('frequência [Hz]')
plt.show()
# Utilizando wavelets 
widths = np.arange(1,30,1)
cwt_tr = sig.cwt(x, sig.morlet2, widths)
plt.figure(figsize=(12, 10))
plt.subplot(2,1,1)
plt.plot(t,x , c = 'red' , linewidth = 1 )
plt.grid(True)
plt.xlabel('tempo [s]')
plt.ylabel('x(t) [mm/s]')
plt.subplot(2,1,2)

plt.pcolormesh(t, widths, 2*np.abs(cwt_tr) , vmin=0)
plt.colorbar()
plt.grid(True)
plt.xlabel('tempo [s]')
plt.ylabel('escala []')
plt.show()


