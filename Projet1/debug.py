from model import *
from strategy import *
import numpy as np
i=np.zeros((5,5))+3
j=1+np.zeros((5,5))
j[2][2]=2
print(i)
print(j)
print(i*j)
print(6 in i*j)
print(7 in i*j)
