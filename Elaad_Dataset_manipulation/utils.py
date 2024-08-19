import pandas as pd
import datetime as dt
import os
import math
import numpy as np
from collections import Counter
from numpy import random
import sklearn_extra.cluster as sk


# Funzione per sostituire i valori minori o uguali a 5 con il rapporto tra la somma e il conteggio
def replace_values(value, sum_values, count_values):
    if ((value <= 5) & (value > 0)):
        return sum_values / count_values
    else:
        return value

def roulette_wheel_startTime(prob_startTime_cumsum):
    global startTime, x
    x = random.rand()
    # if sum(prob_startTime_cumsum) == 0:
    #     startTime = 0
    # else:
    for i in range(len(prob_startTime_cumsum)):
        if x < prob_startTime_cumsum[i]:
            startTime = i
            break
        else:
            continue
    return startTime, x

def roulette_wheel_stopTime(startTime, prob_stopTime_cumsum):
    # global stopTime, x
    x = random.rand()
    # print('startTime: ',startTime)
    # print('x: ',x)
    # if sum(prob_stopTime_cumsum.iloc[startTime, :]) == 0:
    #     stopTime = 0
    # else:
    for k in range(np.shape(prob_stopTime_cumsum)[0]):
        # print('prob_stopTime_cumsum.iloc[startTime, k]: ', prob_stopTime_cumsum.iloc[startTime, k])
        if x < prob_stopTime_cumsum.iloc[int(startTime), k]:
            stopTime = k
            # if stopTime < startTime:
            #     print('aiuto')
            break
        else:
            continue
    return stopTime, x

def roulette_wheel_TotalEnergy_cond(TotalEnergy_cond_cumsum):
    global chargedEnergy, x
    x = random.rand()
    if sum(np.array(TotalEnergy_cond_cumsum)[0]) == 0:
        chargedEnergy = math.nan
    else:
        for i in range(np.shape(TotalEnergy_cond_cumsum)[1]):
            if x < np.array(TotalEnergy_cond_cumsum)[0][i]:
                chargedEnergy = i+1
                break
            else:
                continue
    return chargedEnergy, x

def roulette_wheel_thirdlevel(third_level_cumsum):
    global chargeTime, x
    x = random.rand()
    if sum(np.array(third_level_cumsum)[0]) == 0:
        chargeTime = math.nan
    else:
        for i in range(np.shape(third_level_cumsum)[1]):
            if x < np.array(third_level_cumsum)[0][i]:
                chargeTime = i+1
                break
            else:
                continue
    return chargeTime, x


def centroids_weight(n_cluster, labels, centroids_predict, clustered_list):
    # print('centr =', centroids_predict)
    count_labels = Counter(labels)
    count_centroids = Counter(centroids_predict)
    dict_prob = {}
    for key in count_labels.keys():
        dict_prob[key] = (count_labels[key] / count_centroids[key]) / len(clustered_list)
    prob = []
    for i in range(n_cluster):
        prob.append(dict_prob[centroids_predict[i]])
    return prob