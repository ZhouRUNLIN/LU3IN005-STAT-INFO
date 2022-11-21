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

#pour Question 5.3.1
def drawNaiveBayes(df,attr):
    """
        retourne le graphe du model Naive Bayes a partir 
    d'un dataframe et du nom de la colonne qui est la classe
    """
    str0=""
    for attr0 in list(df.columns):
        if attr0!=attr:
            str0+=attr+"->"+attr0+";"
    str0=str0[0:-1]
    return utils.drawGraph(str0)

#pour Question 5.3.2
def nbParamsNaiveBayes(df,attr,lAttr=None):
    """
        retourne la taille mémoire d'une table (df)
    en supposant qu'un float est représenté sur 8octets 
    et en utilisant l'hypothèse du Naive Bayes.
    """
    nb=0
    if lAttr==None:
        lAttr=list(df.columns)
    for attr0 in lAttr:
        nb+=df.groupby(attr0).size().count()
    nb=nb*df.groupby(attr).size().count()*8
    if nb==0:
        nb=df.groupby(attr).size().count()*8
    else:
        nb-=df.groupby(attr).size().count()*8
    print(nb)
    return nb

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

#pour Question 5.4 classifier naive bayes
from functools import reduce
class MLNaiveBayesClassifier(APrioriClassifier):

    def __init__(self,df,attr='target'):
        self.df=df
        self.attr=attr
        self.dictAttr=P2D_p(df,attr)

        self.x_df = df.iloc[:, 0:-1]
        self.y_df = df.iloc[:, -1]
        self.label = set(self.y_df)
        self.col = len(self.x_df.columns)
        self.row = len(self.x_df)
        self.label_proba = {}
        self.label_side = {}
        self._lambda = 0.01

        self.fit()

    def fit(self):
        for y_label in self.label:
            feature_dict = {}
            for col in range(self.col):
                sub_feature_dict = {}
                for feature in set(self.df.iloc[:, col]):
                    filter_temp = self.df[self.df.iloc[:, -1] == y_label]
                    proba = (sum(filter_temp.iloc[:,col] == feature) + self._lambda) /\
                        (len(filter_temp)+len(set(self.df.iloc[:,col])))
                    sub_feature_dict[feature] = proba
                feature_dict[self.x_df.columns[col]] = sub_feature_dict
            self.label_proba[y_label] = feature_dict
        
        for label_value in self.label:
            self.label_side[label_value] = (sum(self.y_df == label_value) + self._lambda)/\
                                        (self.row + len(self.label) * self._lambda)

    def estimProbas(self, attrs):
        max_proba = 0
        res_probe_dict = {}
        x_list = list(attrs.values())[:-1]
        for label_key, label_value in self.label_proba.items():
            feature_prob = []
            count = 0
            for feature_key, feature_value in label_value.items():
                for prob_key, prob_value in feature_value.items():
                    if prob_key == x_list[count]:
                        feature_prob.append(prob_value)
                count += 1
            
            posterior_prob = reduce(lambda x, y:x * y, feature_prob)
            #posterior_prob *= self.label_side[label_key]
            res_probe_dict[label_key] = posterior_prob
            if posterior_prob > max_proba:
                max_proba = posterior_prob
                res_y = label_key

        return res_probe_dict

    def estimClass(self, attrs):
        max_proba = 0
        res_y = 0
        x_list = list(attrs.values())[:-1]
        for label_key, label_value in self.label_proba.items():
            feature_prob = []
            count = 0
            for feature_key, feature_value in label_value.items():
                for prob_key, prob_value in feature_value.items():
                    if prob_key == x_list[count]:
                        feature_prob.append(prob_value)
                count += 1
            prior_prob = reduce(lambda x, y:x * y, feature_prob)
            #prior_prob *= self.label_side[label_key]
            if prior_prob > max_proba:
                max_proba = prior_prob
                res_y = label_key
        return res_y

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

from functools import reduce
class MAPNaiveBayesClassifier(APrioriClassifier):

    def __init__(self,df,attr='target'):
        self.df=df
        self.attr=attr
        self.dictAttr=P2D_p(df,attr)

        self.x_df = df.iloc[:, 0:-1]
        self.y_df = df.iloc[:, -1]
        self.label = set(self.y_df)
        self.col = len(self.x_df.columns)
        self.row = len(self.x_df)
        self.label_proba = {}
        self.label_side = {}
        self._lambda = 0.01

        self.fit()

    def fit(self):
        for y_label in self.label:
            feature_dict = {}
            for col in range(self.col):
                sub_feature_dict = {}
                for feature in set(self.df.iloc[:, col]):
                    filter_temp = self.df[self.df.iloc[:, -1] == y_label]
                    proba = (sum(filter_temp.iloc[:,col] == feature) + self._lambda) /\
                        (len(filter_temp)+len(set(self.df.iloc[:,col])))
                    sub_feature_dict[feature] = proba
                feature_dict[self.x_df.columns[col]] = sub_feature_dict
            self.label_proba[y_label] = feature_dict
        
        for label_value in self.label:
            self.label_side[label_value] = (sum(self.y_df == label_value) + self._lambda)/\
                                        (self.row + len(self.label) * self._lambda)

    def estimProbas(self, attrs):
        max_proba = 0
        res_probe_dict = {}
        x_list = list(attrs.values())[:-1]
        for label_key, label_value in self.label_proba.items():
            feature_prob = []
            count = 0
            for feature_key, feature_value in label_value.items():
                for prob_key, prob_value in feature_value.items():
                    if prob_key == x_list[count]:
                        feature_prob.append(prob_value)
                count += 1
            
            posterior_prob = reduce(lambda x, y:x * y, feature_prob)
            posterior_prob *= self.label_side[label_key]
            res_probe_dict[label_key] = posterior_prob
            if posterior_prob > max_proba:
                max_proba = posterior_prob
                res_y = label_key

        return res_probe_dict

    def estimClass(self, attrs):
        max_proba = 0
        res_y = 0
        x_list = list(attrs.values())[:-1]
        for label_key, label_value in self.label_proba.items():
            feature_prob = []
            count = 0
            for feature_key, feature_value in label_value.items():
                for prob_key, prob_value in feature_value.items():
                    if prob_key == x_list[count]:
                        feature_prob.append(prob_value)
                count += 1
            
            posterior_prob = reduce(lambda x, y:x * y, feature_prob)
            posterior_prob *= self.label_side[label_key]
            if posterior_prob > max_proba:
                max_proba = posterior_prob
                res_y = label_key
        return res_y

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

#pour Question 6 : feature selection dans le cadre du classifier naive bayes

from scipy import stats 
def isIndepFromTarget(df, attr, x):
    table_sp = pd.crosstab(df[attr], df['target'])
    observed = stats.chi2_contingency(np.array(table_sp))
    chi2, p, dof, ex = stats.chi2_contingency(table_sp, correction=False)
    if p > x:
        return True
    return False

class ReducedMLNaiveBayesClassifier(APrioriClassifier):

    def __init__(self, df, param, attr='target'):
        self.df=df
        self.attr=attr
        self.dictAttr=P2D_p(df,attr)
        self.param = param
        self.reduce_node = []

    def fit(self):
        for y_label in self.label:
            feature_dict = {}
            for col in range(self.col):
                sub_feature_dict = {}
                for feature in set(self.df.iloc[:, col]):
                    filter_temp = self.df[self.df.iloc[:, -1] == y_label]
                    proba = (sum(filter_temp.iloc[:,col] == feature) + self._lambda) /\
                        (len(filter_temp)+len(set(self.df.iloc[:,col])))
                    sub_feature_dict[feature] = proba
                feature_dict[self.x_df.columns[col]] = sub_feature_dict
            self.label_proba[y_label] = feature_dict
        
        for label_value in self.label:
            self.label_side[label_value] = (sum(self.y_df == label_value) + self._lambda)/\
                                        (self.row + len(self.label) * self._lambda)

    def estimProbas(self, attrs):
        for rn in self.reduce_node:
            del attrs[rn]

        max_proba = 0
        res_probe_dict = {}
        x_list = list(attrs.values())[:-1]
        for label_key, label_value in self.label_proba.items():
            feature_prob = []
            count = 0
            for feature_key, feature_value in label_value.items():
                for prob_key, prob_value in feature_value.items():
                    if prob_key == x_list[count]:
                        feature_prob.append(prob_value)
                count += 1
            
            posterior_prob = reduce(lambda x, y:x * y, feature_prob)
            #posterior_prob *= self.label_side[label_key]
            res_probe_dict[label_key] = posterior_prob
            if posterior_prob > max_proba:
                max_proba = posterior_prob
                res_y = label_key

        return res_probe_dict

    def estimClass(self, attrs):
        for rn in self.reduce_node:
            del attrs[rn]

        max_proba = 0
        res_y = 0
        x_list = list(attrs.values())[:-1]
        for label_key, label_value in self.label_proba.items():
            feature_prob = []
            count = 0
            for feature_key, feature_value in label_value.items():
                for prob_key, prob_value in feature_value.items():
                    if prob_key == x_list[count]:
                        feature_prob.append(prob_value)
                count += 1
            
            posterior_prob = reduce(lambda x, y:x * y, feature_prob)
            #posterior_prob *= self.label_side[label_key]
            if posterior_prob > max_proba:
                max_proba = posterior_prob
                res_y = label_key
        return res_y
    
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

    def draw(self):
        """
            retourne le graphe du model Naive Bayes a partir 
        d'un dataframe et du nom de la colonne qui est la classe
        """
        str0=""
        for attr0 in list(self.df.columns):
            if isIndepFromTarget(self.df, attr0, self.param):
                self.reduce_node.append(attr0)
                continue
            if attr0!=self.attr:
                str0+=self.attr+"->"+attr0+";"
        str0=str0[0:-1]
        
        self.df = self.df.drop(columns=self.reduce_node)
        self.x_df = self.df.iloc[:, 0:-1]
        self.y_df = self.df.iloc[:, -1]
        self.label = set(self.y_df)
        self.col = len(self.x_df.columns)
        self.row = len(self.x_df)
        self.label_proba = {}
        self.label_side = {}
        self._lambda = 0.01
        self.fit()
        return utils.drawGraph(str0)

class ReducedMAPNaiveBayesClassifier(APrioriClassifier):

    def __init__(self, df, param, attr='target'):
        self.df=df
        self.attr=attr
        self.dictAttr=P2D_p(df,attr)
        self.param = param
        self.reduce_node = []

    def fit(self):
        for y_label in self.label:
            feature_dict = {}
            for col in range(self.col):
                sub_feature_dict = {}
                for feature in set(self.df.iloc[:, col]):
                    filter_temp = self.df[self.df.iloc[:, -1] == y_label]
                    proba = (sum(filter_temp.iloc[:,col] == feature) + self._lambda) /\
                        (len(filter_temp)+len(set(self.df.iloc[:,col])))
                    sub_feature_dict[feature] = proba
                feature_dict[self.x_df.columns[col]] = sub_feature_dict
            self.label_proba[y_label] = feature_dict
        
        for label_value in self.label:
            self.label_side[label_value] = (sum(self.y_df == label_value) + self._lambda)/\
                                        (self.row + len(self.label) * self._lambda)

    def estimProbas(self, attrs):
        for rn in self.reduce_node:
            del attrs[rn]

        max_proba = 0
        res_probe_dict = {}
        x_list = list(attrs.values())[:-1]
        for label_key, label_value in self.label_proba.items():
            feature_prob = []
            count = 0
            for feature_key, feature_value in label_value.items():
                for prob_key, prob_value in feature_value.items():
                    if prob_key == x_list[count]:
                        feature_prob.append(prob_value)
                count += 1
            
            posterior_prob = reduce(lambda x, y:x * y, feature_prob)
            posterior_prob *= self.label_side[label_key]
            res_probe_dict[label_key] = posterior_prob
            if posterior_prob > max_proba:
                max_proba = posterior_prob
                res_y = label_key

        return res_probe_dict

    def estimClass(self, attrs):
        for rn in self.reduce_node:
            del attrs[rn]

        max_proba = 0
        res_y = 0
        x_list = list(attrs.values())[:-1]
        for label_key, label_value in self.label_proba.items():
            feature_prob = []
            count = 0
            for feature_key, feature_value in label_value.items():
                for prob_key, prob_value in feature_value.items():
                    if prob_key == x_list[count]:
                        feature_prob.append(prob_value)
                count += 1
            
            posterior_prob = reduce(lambda x, y:x * y, feature_prob)
            posterior_prob *= self.label_side[label_key]
            if posterior_prob > max_proba:
                max_proba = posterior_prob
                res_y = label_key
        return res_y
    
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

    def draw(self):
        """
            retourne le graphe du model Naive Bayes a partir 
        d'un dataframe et du nom de la colonne qui est la classe
        """
        str0=""
        for attr0 in list(self.df.columns):
            if isIndepFromTarget(self.df, attr0, self.param):
                self.reduce_node.append(attr0)
                continue
            if attr0!=self.attr:
                str0+=self.attr+"->"+attr0+";"
        str0=str0[0:-1]
        
        self.df = self.df.drop(columns=self.reduce_node)
        self.x_df = self.df.iloc[:, 0:-1]
        self.y_df = self.df.iloc[:, -1]
        self.label = set(self.y_df)
        self.col = len(self.x_df.columns)
        self.row = len(self.x_df)
        self.label_proba = {}
        self.label_side = {}
        self._lambda = 0.01
        self.fit()
        return utils.drawGraph(str0)