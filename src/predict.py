# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 15:38:30 2019

@author: Wayne
"""

import json
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import keras
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
import time
from keras.models import Sequential
import pandas as pd
from src.final import load_data

TRAINSIZE = 5

def perf_score(stat):
    GOAL = float(stat[24])
    TEAM_RATING = float(stat[3])
    ACC_PASS = float(stat[19])
    TOTAL_PASS = float(stat[20])
    AERIAL_WON = float(stat[15])
    AERIAL_LOSS = float(stat[18])
    POSSESION_PERCENTAGE = float(stat[7])
    TOTAL_SCORING_ATT = float(stat[11])
    
#    score = GOAL*24/5 + ACC_PASS/TOTAL_PASS*13 + \
#        AERIAL_WON/(AERIAL_WON+AERIAL_LOSS)*13+ POSSESION_PERCENTAGE/100*20
#    if float(TOTAL_SCORING_ATT) != 0:
#        score += float(stat[24])/float(stat[11])*13
    score = ACC_PASS/TOTAL_PASS*13 + AERIAL_WON/(AERIAL_WON+AERIAL_LOSS)*13+ POSSESION_PERCENTAGE/100*20
    if float(TOTAL_SCORING_ATT) != 0:
        score += float(stat[24])/float(stat[11])*13
    return score

def calc_score():
    f = open('season_stats_new.csv')
    data = f.readlines()
    header = data[0]
    data = data[1:]

    teams = {}
    for stat in data:
        stat = stat.split(',')
        score = perf_score(stat)
        if stat[1] not in teams:
            teams[stat[1]] = [score]
        else:
            teams[stat[1]].append(score)

    return teams


#sys.exit()

def make_class_data(model,match,teams):

    global class_data
    global calss_label
    global teamsi
    class_data = []
    class_label = []
    teamsi = dict.fromkeys(teams.keys(), 0)
    for m in match.values():
        home = m['home_team_id']
        away = m['away_team_id']
        if int(m['full_time_score'].split(':')[0]) > int(m['full_time_score'].split(':')[1]):
            y = [1,0,0]
        elif int(m['full_time_score'].split(':')[0]) == int(m['full_time_score'].split(':')[1]):
            y = [0,1,0]
        else:
            y = [0,0,1]
        i_home = teamsi[home]
        i_away = teamsi[away]
        if i_home >= TRAINSIZE and i_away >= TRAINSIZE:
            h = np.array(teams[home][i_home-TRAINSIZE:i_home]+[0]).reshape(-1,1)
            a = np.array(teams[away][i_away-TRAINSIZE:i_away]+[0]).reshape(-1,1)
            h = sc.transform(h.reshape(1,-1))
            a = sc.transform(a.reshape(1,-1))
            h_score = model.predict(h.reshape(-1,1)[:TRAINSIZE].reshape(1,TRAINSIZE,1))
            a_score = model.predict(a.reshape(-1,1)[:TRAINSIZE].reshape(1,TRAINSIZE,1))
            class_data.append([h_score[0], a_score[0]])
            class_label.append(y)
        teamsi[home] += 1
        teamsi[away] += 1

    return np.array(class_data), np.array(class_label)



def plot_ROC():
    from sklearn.metrics import roc_curve, auc
    from sklearn import datasets
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import label_binarize
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    
    
    X, y = test_data, test_label
    
    y = label_binarize(y, classes=[0,1,2])
    n_classes = 3
    
    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.33, random_state=0)
    
    # classifier
    clf = OneVsRestClassifier(RandomForestClassifier())
    y_score = clf.fit(X_train, y_train).decision_function(X_test)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot of a ROC curve for a specific class
    for i in range(n_classes):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()


def write_result(match, x_test, model):

    lines = [['Home','Away','Win','Draw','Lose','Prediction']]
    for i,a in enumerate(list(match.values())[-380:]):
        predict = model.predict(x_test[i].reshape(1,-1))
        res = np.argmax(predict)
        line = [a['home_team_name'],a['away_team_name']] + list(predict.ravel()) + ['W' if res==0 else 'D' if res==1 else 'L']
        lines.append(line)
    my_df = pd.DataFrame(lines)
    my_df.to_csv('predict_new.csv', index=False, header = False)
    return lines


def match_prediction():
    match, stat = load_data()
    teams = calc_score()
    teams_train = {}
    
    for team in teams:
        teams_train[team] = {'x': [], 'y': []}
        for i in range(len(teams[team])-TRAINSIZE):
            teams_train[team]['x'].append(teams[team][i:i+TRAINSIZE])
            teams_train[team]['y'].append(teams[team][i+TRAINSIZE])
        teams_train[team]['x'] = np.array(teams_train[team]['x'])
        teams_train[team]['y'] = np.array(teams_train[team]['y'])
    
    x_train = []
    y_train = []
    
    for a in teams_train.values():
        x_train.append(a['x'])
        y_train.append(a['y'])
    
    x_train = np.vstack(x_train)
    y_train = np.hstack(y_train)
    #
    x_all = np.concatenate((x_train,y_train.reshape(-1,1)),axis=1)
    sc = StandardScaler()
    x_all = sc.fit_transform(x_all)
    x_train2 = x_all[:,:TRAINSIZE]
    y_train2 = x_all[:,TRAINSIZE]
        
    #x_train = teams_train['26']['x']
    #y_train = teams_train['26']['y']
    
    model = Sequential()
    model.add(LSTM(32, input_shape=(TRAINSIZE,1), return_sequences=True))
    model.add(Dropout(0.1))
    #model.add(LSTM(32, return_sequences=True))
    #model.add(Dropout(0.1))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='linear'))
    start = time.time()
    adam = keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=False)
    nadam = keras.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
    sgd = keras.optimizers.SGD(learning_rate=0.001, momentum=0.0, nesterov=False)
    model.compile(loss="mse", optimizer=adam)
    print("Compilation Time : ", time.time() - start)
    model.fit(x_train2.reshape(x_train2.shape+(1,)), y_train2, batch_size=16, epochs=10, validation_split=0.05)
    
    plt.figure()
    predict = model.predict(x_train2.reshape(x_train2.shape+(1,)))
    plt.plot(y_train2)
    plt.plot(predict)


    class_data, class_label = make_class_data(model,match,teams)
    
    
    
    s4teams = []
    for a in list(match.values())[-380:]:
        if a['home_team_id'] not in s4teams:
            s4teams.append(a['home_team_id'])
    
    s4_teams = {}
    for a in teams:
        if a in s4teams:
            if len(teams[a]) == 38:
                s4_teams[a] = [46,46,46,46,46] + teams[a]
            else:
                s4_teams[a] = teams[a][-43:]
    
    teamsi = dict.fromkeys(s4_teams.keys(), 0)
    test_data = []
    test_label = []
    for a in list(match.values())[-380:]:
        home = a['home_team_id']
        away = a['away_team_id']
        if int(a['full_time_score'].split(':')[0]) > int(a['full_time_score'].split(':')[1]):
            y = [1,0,0]
        elif int(a['full_time_score'].split(':')[0]) == int(a['full_time_score'].split(':')[1]):
            y = [0,1,0]
        else:
            y = [0,0,1]
        ihome = teamsi[home]
        iaway = teamsi[away]
        h = np.array(s4_teams[home][ihome:ihome+TRAINSIZE]+[0]).reshape(-1,1)
        a = np.array(s4_teams[away][iaway:iaway+TRAINSIZE]+[0]).reshape(-1,1)
        h = sc.transform(h.reshape(1,-1))
        a = sc.transform(a.reshape(1,-1))
        h_score = model.predict(h.reshape(-1,1)[:TRAINSIZE].reshape(1,TRAINSIZE,1))
        a_score = model.predict(a.reshape(-1,1)[:TRAINSIZE].reshape(1,TRAINSIZE,1))
        test_data.append([h_score[0], a_score[0]])
        test_label.append(y)
        teamsi[home] += 1
        teamsi[away] += 1
    test_data = np.array(test_data)
    test_label = np.array(test_label)
    
    
    
    clf = RandomForestClassifier()
    clf = clf.fit(class_data[:1300].reshape(1300,2), class_label[:1300])
    score = clf.score(test_data.reshape(380,2),test_label)
    print(score)