# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 15:12:58 2019

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
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
import time
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


DIR = '.\\data\\datafilev2\\datafile'

def load_data():
    match = {}
    stat = {}
    for a in os.listdir(DIR):
        f = open(os.path.join(DIR,a,'season_match_stats.json'),encoding="utf8")
        match.update(json.load(f))
        
        f = open(os.path.join(DIR,a,'season_stats.json'),encoding="utf8")
        stat.update(json.load(f))
    return match, stat

def pre_processing(match, stat, method = 'player', onehot = False, split = True):
    if method == 'player':
        match_result = [a['full_time_score'] for a in match.values()]
        match_result = np.array([0 if int(a.split(':')[0]) > int(a.split(':')[1]) else \
                        1 if int(a.split(':')[0]) == int(a.split(':')[1]) else 2 for a in match_result])
        feature_player = []
        for match in stat.values():
            players = []
            for team in match.values():
                for player in team['Player_stats'].values():
                    players.append(int(player['player_details']['player_id']))
                if len(players)<=18:
                    while len(players)<18 :
                        players.append(0)
                else:
                    while len(players)<36 :
                        players.append(0)
            feature_player.append(players)
        sc = StandardScaler()
        feature_player = sc.fit_transform(np.array(feature_player))
        if onehot:
            enc = OneHotEncoder()
            match_result = enc.fit_transform(match_result.reshape(-1,1))
        if split:
            return train_test_split(feature_player, match_result, test_size=0.25)
        else:
            return feature_player, match_result
    
    if method == 'team':
        match_result = [a['full_time_score'] for a in match.values()]
        match_result = np.array([0 if int(a.split(':')[0]) > int(a.split(':')[1]) else \
                        1 if int(a.split(':')[0]) == int(a.split(':')[1]) else 2 for a in match_result])
        feature_team = np.array([[int(a) for a in list(a.keys())] for a in stat.values()])
        sc = StandardScaler()
        feature_team = sc.fit_transform(feature_team)
        if onehot:
            enc = OneHotEncoder()
            match_result = enc.fit_transform(match_result.reshape(-1,1))
        if split:
            return train_test_split(feature_team, match_result, test_size=0.25)
        else:
            return feature_team, match_result

    if method == 'LSTM_team':
        teams = {}
        for match in stat.values():
            for team in match.values():
                if team['team_details']['team_id'] not in teams:
                    teams[team['team_details']['team_id']] = [float(team['team_details']['team_rating'])]
                else:
                    teams[team['team_details']['team_id']].append(float(team['team_details']['team_rating']))
        
        for idd in teams:
            feature = []
            label = []
            for i in range(len(teams[idd])-10):
                feature.append(teams[idd][i:i+9])
                label.append(teams[idd][i+10])
            teams[idd] = [np.array(feature), np.array(label)]
        return teams

def train_model(X_train, X_test, y_train, y_test, method = 'SVM'):
    if method == 'SVM':
        gksvm = SVC(kernel='rbf', C=1, random_state=1)
        gksvm.fit(X_train,y_train)
        score = gksvm.score(X_test,y_test)
        return gksvm, score
    
    if method == 'XGB':
        xgbc = XGBClassifier()
        xgbc.fit(X_train,y_train)
        score = xgbc.score(X_test,y_test)
        return xgbc, score
    
    if method == 'Forest':
        clf = RandomForestClassifier()
        clf = clf.fit(X_train,y_train)
        score = clf.score(X_test,y_test)
        return clf, score
    
    if method == 'LSTM':
        model = Sequential()
        model.add(LSTM(32, input_shape=(9,1), return_sequences=False))
        model.add(Dropout(0.1))
#        model.add(LSTM(64, return_sequences=False))
#        model.add(Dropout(0.1))
        model.add(Dense(1, activation='linear'))
        start = time.time()
        model.compile(loss="mse", optimizer="rmsprop")
        print("Compilation Time : ", time.time() - start)
        model.fit(X_train, y_train, batch_size=16, epochs=20, validation_split=0.05)
        return model
    
    if method == 'MLP':
        model = Sequential()
        model.add(Dense(32, activation='relu',input_shape=(36,)))
        model.add(Dropout(0.1))
#        model.add(Dense(64, activation='relu'))
#        model.add(Dropout(0.1))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss="mse", optimizer="adam")
        model.fit(X_train, y_train, batch_size=8, epochs=30, validation_split=0.05)
        return model

def train_score(X, y):
    gksvm = SVC(kernel='rbf', C=1, random_state=1)
    scores = cross_val_score(gksvm, X, y, cv=5)
    return np.mean(scores)

def visualize(X, y, predict):
    data = TSNE(n_components=2).fit_transform(X)
    vis_x = data[:,0]
    vis_y = data[:,1]
    plt.figure()
    plt.title('True result')
    plt.scatter(vis_x, vis_y, c= y, cmap=plt.cm.get_cmap("jet",3))
    plt.show() 
    
    plt.figure()
    plt.title('Predicted result')
    plt.scatter(vis_x, vis_y, c= predict, cmap=plt.cm.get_cmap("jet",3))
    plt.show() 

def write_result(match, x_test, model):

    lines = [['Home','Away','Win','Draw','Lose','Prediction']]
    for i,a in enumerate(list(match.values())[-380:]):
        predict = model.predict(x_test[i].reshape(1,-1))
        res = np.argmax(predict)
        line = [a['home_team_name'],a['away_team_name']] + list(predict.ravel()) + ['W' if res==0 else 'D' if res==1 else 'L']
        lines.append(line)
    my_df = pd.DataFrame(lines)
    my_df.to_csv('predict.csv', index=False, header = False)
    return lines


def compare_classificaion(model,method):

    match, stat = load_data()
    
    if model != 'MLP':
        X_train, X_test, y_train, y_test = pre_processing(match, stat, method = method)
        model,score = train_model(X_train, X_test, y_train, y_test, method = model)
        print(score)

        visualize(X_test, y_test, model.predict(X_test))

    else:
        data = pre_processing(match, stat, method = method, onehot = True, split = True)
        model = train_model(np.vstack([data[0],data[1]]), 0, np.vstack([data[2].toarray(),data[3].toarray()]), 0,'MLP')
        #model = train_model(data[0], 0, data[2].toarray(), 0,'MLP')
        predict = np.argmax(model.predict(data[1]),1)
        y = np.argmax(data[3].toarray(),1)
        visualize(data[1], y, predict)
        #
        acc = 0
        for i in range(380):
            if y[i]==predict[i]:
                acc+=1
        acc = acc/380
        print(acc)

#
#plt.figure()
#predict = model.predict(data[1])
#plt.plot(data[3])
#plt.plot(predict)



