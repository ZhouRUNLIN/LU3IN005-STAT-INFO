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

def P2D_l(df,attr):
    train0=df.copy()
    dict1=dict() #Un dictionnaire de la fréquence d'occurrence de chaque valeur lorsque target=1
    trainP=train0.groupby("target").get_group(1)
    for line in trainP[attr]:
        if line not in dict1:
            dict1[line]=1
        else:
            dict1[line]+=1
    for key in dict1:
        dict1[key]=(0.0+dict1[key])/trainP[attr].count()
    
    dict0=dict() #Un dictionnaire de la fréquence d'occurrence de chaque valeur lorsque target=0
    trainNP=train0.groupby("target").get_group(0)
    for line in trainNP[attr]:
        if line not in dict0:
            dict0[line]=1
        else:
            dict0[line]+=1
    for key in dict0:
        dict0[key]=(0.0+dict0[key])/trainNP[attr].count()
    return {1:dict1,0:dict0}
    

def P2D_p(df,attr):
    train0=df.copy()
    
    dict0=dict() #Un dictionnaire de la fréquence d'occurrence de chaque valeur
    for line in train0[attr]:
        if line not in dict0:
            dict0[line]=1
        else:
            dict0[line]+=1
    
    dict1=dict() #Un dictionnaire de la fréquence d'occurrence de chaque valeur lorsque target=1
    trainP=train0.groupby("target").get_group(1)
    for key in dict0:
        dict1[key]=0
    for line in trainP[attr]:
        dict1[line]+=1
    
    dictP2Dp=dict()
    for key in dict0:
        dictP2Dp[key]={1:(0.0+dict1[key])/dict0[key],0:1.0-(0.0+dict1[key])/dict0[key]}
    
    return dictP2Dp

def nbParams(df,lAttr=None):
    """
    返回各个特征可取的值数量之积然后乘以8
    """
    if lAttr==None:
        lAttr=list(df.columns)
    nb=8
    for attr in lAttr:
        nb*=df.groupby(attr).size().count()
    return nb

def nbParamsIndep(df,lAttr=None):
    """
    还各取直数之积，以其八乘之。
    """
    if lAttr==None:
        lAttr=list(df.columns)
    nb=0
    for attr in lAttr:
        nb+=df.groupby(attr).size().count()
    return nb*8

class APrioriClassifier(utils.AbstractClassifier):
  """
  Un classifier implémente un algorithme pour estimer la classe d'un vecteur d'attributs. 
  Il propose aussi comme service de calculer les statistiques de reconnaissance à partir d'un pandas.dataframe.
  """

  def __init__(self):
    pass

  def estimClass(self, attrs):
    """
    toujours retourne 1
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

class ML2DClassifier(APrioriClassifier):
  """
  commentaire
  """

  def __init__(self,df,attr):
    self.df=df
    self.attr=attr
    self.dictAttr=P2D_l(df,attr)

  def estimClass(self, attrs):
    """
    retourne 1 si dictAttr[1][attrs[self.attr]]>dictAttr[0][attrs[self.attr]], retourne 0 sinon
    """
    if self.dictAttr[1][attrs[self.attr]]>self.dictAttr[0][attrs[self.attr]]:
        return 1
    return 0

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

class MAP2DClassifier(APrioriClassifier):
  """
  commentaire
  """

  def __init__(self,df,attr):
    self.df=df
    self.attr=attr
    self.dictAttr=P2D_p(df,attr)

  def estimClass(self, attrs):
    """
    retourne 1 si self.dictAttr[attrs[self.attr]][1]>self.dictAttr[attrs[self.attr]][0], retourne 0 sinon
    """
    if self.dictAttr[attrs[self.attr]][1]>self.dictAttr[attrs[self.attr]][0]:
        return 1
    return 0

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




