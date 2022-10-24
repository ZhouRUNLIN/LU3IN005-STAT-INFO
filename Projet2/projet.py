import numpy as np
import pandas as pd
import utils
import math

def getPrior(data):
    N = 0
    N1 = 0
    for i in data['target']:        #parcours les fichers de .csv pour compter les valeurs
        N += 1
        if i == 1:
            N1 += 1

    estimation = N1/N           #la valeur de p
    variance = math.sqrt(estimation*(1 - estimation)/N) 
    min5pourcent = estimation - 1.96*variance
    max5pourcent = estimation + 1.96*variance

    return{"estimation": estimation,"min5pourcent": min5pourcent, "max5pourcent": max5pourcent}



class APrioriClassifier(utils.AbstractClassifier):
  """
  Un classifier implémente un algorithme pour estimer la classe d'un vecteur d'attributs. 
  Il propose aussi comme service de calculer les statistiques de reconnaissance à partir d'un pandas.dataframe.
  """

  def ___init__(self):
    pass

  def estimClass(self, attrs):
    """
    toujours retounre 1 n'importe
    """
    return 1

  def statsOnDF(self, df):
    """
    à partir d'un pandas.dataframe, calcule les taux d'erreurs de classification et rend un dictionnaire.


    :paramètre 'df': un dataframe à tester
    :return: un dictionnaire incluant les VP,FP,VN,FN,précision et rappel
    """
    
    #Initialisation des variables
    VP=0
    VN=0
    FP=0
    FN=0
    
    for i in range(df.count()["target"]):
        #calculer la valeur des variables
        dictI = utils.getNthDict(df,i)
        tar = dictI["target"]
        pre = self.estimClass(dictI)
        if tar==1 and pre==1:
            VP += 1
        if tar==0 and pre==0:
            VN += 1
        if tar==0 and pre==1:
            FP += 1
        if tar==1 and pre==0:
            FN += 1
            
    return {"VP":VP,"VN":VN,"FP":FP,"FN":FN,"Précision":VP/(VP+FP),"rappel":VP/(VP+FN)}