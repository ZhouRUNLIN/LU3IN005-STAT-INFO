import numpy as np
import pandas as pd
from scipy.special import comb

def getPrior(data):
    N = len(data['target'])
    N1 = 0
    for i in data['target']:
        if i == 1:
            N1 += 1

    estimation = N1/N
    max5pourcent = estimation + 2*math.sqrt((estimation*(1 - estimation))/N)
    min5pourcent = estimation - 2*math.sqrt((estimation*(1 - estimation))/N)

    return{'estimation': estimation,"min5pourcent": min5pourcent, "max5pourcent": max5pourcent}
