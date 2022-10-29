import numpy as np
import pandas as pd
import utils
import math

#pour Question 1
def getPrior(data):
    N = 0
    N1 = 0
    for i in data['target']:        #parcours les fichers de .csv pour compter les valeurs
        N += 1
        if i == 1:
            N1 += 1

    estimation = N1/N           #la valeur de p
    variance = math.sqrt(estimation*(1 - estimation)/N) 
    
    #l'intervale de confiance à 95% est [-1.96, 1.96]
    min5pourcent = estimation - 1.96*variance
    max5pourcent = estimation + 1.96*variance

    return{"estimation": estimation,"min5pourcent": min5pourcent, "max5pourcent": max5pourcent}

#pour Question 3a
def P2D_l(df,attr):
    """
    Calcule dans le dataframe(df) la probabilité P(attr|target) sous la forme:
        un dictionnaire asssociant à la valeur 't';
        un dictionnaire associant à la valeur 'a', la probabilité de P(attr=a|target=t) 
    """
    train0=df.copy()
    dict1=dict() #la création d'une dictionnaire vide pour sotcker la fréquence des clés dans le dataframe lorsque target=1
    trainP=train0.groupby("target").get_group(1)
    for line in trainP[attr]:
        if line not in dict1:
            dict1[line]=1
        else:
            dict1[line]+=1
    for key in dict1:
        dict1[key]=(0.0+dict1[key])/trainP[attr].count()
    
    dict0=dict() #la création d'une dictionnaire vide pour sotcker la fréquence des clés dans le dataframe lorsque target=0
    trainNP=train0.groupby("target").get_group(0)
    for line in trainNP[attr]:
        if line not in dict0:
            dict0[line]=1
        else:
            dict0[line]+=1
    
    for key in dict0:
        dict0[key]=(0.0+dict0[key])/trainNP[attr].count() #calculer la probabilité conditionnelle
    return {1:dict1,0:dict0}
    
#pour Question 3a
def P2D_p(df,attr):
     """
    Calcule dans le dataframe(df) la probabilité P(target|attr) sous la forme:
        un dictionnaire asssociant à la valeur 't';
        un dictionnaire associant à la valeur 'a', la probabilité de P(target=t|attr=a) 
    """
    train0=df.copy()
    
    dict0=dict() #la création d'une dictionnaire vide pour sotcker la fréquence des clés dans le dataframe lorsque target=0
    for line in train0[attr]:
        if line not in dict0:
            dict0[line]=1
        else:
            dict0[line]+=1
    
    dict1=dict() #la création d'une dictionnaire vide pour sotcker la fréquence des clés dans le dataframe lorsque target=1
    trainP=train0.groupby("target").get_group(1)
    for key in dict0:
        dict1[key]=0
    for line in trainP[attr]:
        dict1[line]+=1
    
    dictP2Dp=dict()
    for key in dict0:
        dictP2Dp[key]={1:(0.0+dict1[key])/dict0[key],0:1.0-(0.0+dict1[key])/dict0[key]} #calculer la probabilité conditionelle
    
    return dictP2Dp

#pour Question 4.1
def nbParams(df,lAttr=None):
    """
        retourne la taille mémoire d'une table (df)
    (en supposant qu'un float est représenté sur 8octets)
    """
    if lAttr==None:
        lAttr=list(df.columns)
        
    nb=8    
    for attr in lAttr:
        nb*=df.groupby(attr).size().count()
    return nb

#pour Question 4.2
def nbParamsIndep(df,lAttr=None):
    """
        retourne la taille mémoire d'une table (df)
    en supposant qu'un float est représenté sur 8octets 
    et en supposant l'indépendance des variables
    """
    if lAttr==None:
        lAttr=list(df.columns)
    nb=0
    for attr in lAttr:
        nb+=df.groupby(attr).size().count()
    return nb*8

#pour Question 2
class APrioriClassifier(utils.AbstractClassifier):
  def __init__(self):
    pass

  #pour Question 2a
  def estimClass(self, attrs):
    """
    toujours retourne 1
        Raison : selon le résultat de question 1, la probabilité a priori de la classe 1 est plus huat que 70%,
        donc on peut penser que la majorité de gens est positive (égale à 1)
    """
    return 1

  #pour Question 2b
  def statsOnDF(self, df):
    """
    à partir d'un pandas.dataframe, calcule les taux d'erreurs de classification et rend un dictionnaire.

    Entrée 'df': un dataframe à tester
    Sortie : un dictionnaire incluant la valeur de VP, FP, VN, FN, précision et rappel
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

#pour Question 3b
class ML2DClassifier(APrioriClassifier):
  def __init__(self,df,attr):
    self.df=df
    self.attr=attr
    self.dictAttr=P2D_l(df,attr)

  def estimClass(self, attrs):
    if self.dictAttr[1][attrs[self.attr]]>self.dictAttr[0][attrs[self.attr]]:
        return 1
    return 0

  def statsOnDF(self, df):
    """
    à partir d'un pandas.dataframe, calcule les taux d'erreurs de classification et rend un dictionnaire.

    Entrée 'df': un dataframe à tester
    Sortie : un dictionnaire incluant la valeur de VP, FP, VN, FN, précision et rappel
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

#pour Question 3c
class MAP2DClassifier(APrioriClassifier):
  def __init__(self,df,attr):
    self.df=df
    self.attr=attr
    self.dictAttr=P2D_p(df,attr)

  def estimClass(self, attrs):
    if self.dictAttr[attrs[self.attr]][1]>self.dictAttr[attrs[self.attr]][0]:
        return 1
    return 0

  def statsOnDF(self, df):
    """
    à partir d'un pandas.dataframe, calcule les taux d'erreurs de classification et rend un dictionnaire.

    Entrée 'df': un dataframe à tester
    Sortie : un dictionnaire incluant la valeur de VP, FP, VN, FN, précision et rappel
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




