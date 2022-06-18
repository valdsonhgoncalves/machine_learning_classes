import numpy as np
from parametros_estatisticos import calcula_estatistica as cest 

func = cest()
x = np.random.randn(3,100000)
media = func.Media(x)
print(media)
m2 = func.M2(x)
print(m2)
rms = func.M2(x,rms=True)
print(rms)


