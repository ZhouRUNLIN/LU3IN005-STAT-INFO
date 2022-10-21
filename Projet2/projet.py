import numpy as np
import pandas as pd
import utils
import math

def getPrior(data):
    N = len(data['target'])
    N1 = 0
    for i in data['target']:
        if i == 1:
            N1 += 1

    estimation = N1/N
    max5pourcent = estimation + 2*math.sqrt((estimation*(1 - estimation))/N)
    min5pourcent = estimation - 2*math.sqrt((estimation*(1 - estimation))/N)

    return{"estimation": estimation,"min5pourcent": min5pourcent, "max5pourcent": max5pourcent}

class APrioriClassifier(utils.AbstractClassifier):
  """
  Un classifier implémente un algorithme pour estimer la classe d'un vecteur d'attributs. Il propose aussi comme service
  de calculer les statistiques de reconnaissance à partir d'un pandas.dataframe.
  """

  def ___init__(self):
    pass
    
  def train_model(self):
    """
    Pour faire des prédictions, nous devons construire des modèles à partir de données. Afin de respecter les conditions d'utilisation de l'algorithme apriori, nous encodons d'abord les caractéristiques avec la fonction get_dummies().
    Afin d'obtenir la relation entre l'entité et la cible, nous calculons sa portance. Une valeur de lift supérieure à 1 est une corrélation positive, et inférieure à 1 est une corrélation négative.
    Afin de faciliter la comparaison de la relation, nous utilisons la fonction logarithmique et utilisons log(lift) comme indicateur de relation entre l'entité et la cible.
    La fonction génère un dictionnaire avec chaque caractéristique encodée et son index de relation sous forme de paires clé-valeur.
    """
    dict0=dict()
    train0=pd.read_csv("train.csv")
    for i in train0.keys():
        if i != "target":
            train0[i]=train0[i].astype(str)
    encoded_train=pd.get_dummies(train0)
    pClass1=getPrior(train0)["estimation"]
    dataClass1=encoded_train.groupby("target").get_group(1)
    for j in encoded_train.keys():
        if j != "target":
            pJClass1=max(dataClass1.sum()[j],1)/dataClass1.count()[j]
            pJ=encoded_train.sum()[j]/encoded_train.count()[j]
            dict0[j]=math.log(pJClass1/(pJ*pClass1))
    return dict0
    
  def estimClass(self, attrs):
    """
    à partir d'un dictionanire d'attributs, estime la classe 0 ou 1

    :param attrs: le  dictionnaire nom-valeur des attributs
    :return: la classe 0 ou 1 estimée
    """
    s=0.0
    model=self.train_model()
    for i in attrs.keys():
        tempKey=i+"_"+str(attrs[i])
        if tempKey in model:
            s+=model[tempKey]
    if s>0:
        return 1
    return 0

  def statsOnDF(self, df):
    """
    à partir d'un pandas.dataframe, calcule les taux d'erreurs de classification et rend un dictionnaire.
    VP : nombre d'individus avec target=1 et classe prévue=1
    VN : nombre d'individus avec target=0 et classe prévue=0
    FP : nombre d'individus avec target=0 et classe prévue=1
    FN : nombre d'individus avec target=1 et classe prévue=0

    :param df:  le dataframe à tester
    :return: un dictionnaire incluant les VP,FP,VN,FN,précision et rappel
    """
    VP=0.0
    VN=0.0
    FP=0.0
    FN=0.0
    for i in range(df.count()["target"]):
        dictI=utils.getNthDict(df,i)
        tar=dictI["target"]
        pre=self.estimClass(dictI)
        if tar==1 and pre==1:
            VP+=1
        if tar==0 and pre==0:
            VN+=1
        if tar==0 and pre==1:
            FP+=1
        if tar==1 and pre==0:
            FN+=1
    return {"VP":VP,"VN":VN,"FP":FP,"FN":FN,"Précision":VP/(VP+FP),"rappel":VP/(VP+FN)}