# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 14:38:22 2019

@author: Wayne
"""
import json
import os
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pyspark import SparkConf, SparkContext
import pyspark
from pyspark import sql
from sklearn.cluster import KMeans
import pandas as pd

DIR = '.\\data\\datafilev2\\datafile'
filename = 'player_stats_new.csv'
types = ['touches', 'saves', 'total_pass', 'aerial_won', 'formation_place', 'accurate_pass', 'total_tackle',\
 'aerial_lost', 'fouls', 'yellow_card', 'total_scoring_att', 'man_of_the_match', 'goals', 'won_contest',\
 'blocked_scoring_att', 'goal_assist', 'good_high_claim', 'second_yellow', 'red_card', 'error_lead_to_goal',\
 'last_man_tackle', 'six_yard_block', 'post_scoring_att', 'att_pen_target', 'penalty_save', 'penalty_conceded',\
 'clearance_off_line', 'att_pen_goal', 'att_pen_miss', 'own_goals', 'att_pen_post']


def load_data():
    match = {}
    stat = {}
    for a in os.listdir(DIR):
        f = open(os.path.join(DIR,a,'season_match_stats.json'),encoding="utf8")
        match.update(json.load(f))
        
        f = open(os.path.join(DIR,a,'season_stats.json'),encoding="utf8")
        stat.update(json.load(f))
    return match, stat


def load_data_spark():
    
    sc = pyspark.SparkContext.getOrCreate()
    data = sc.textFile(filename)
    header = data.first()
    data = data.filter(lambda row: row != header)
    data = data.map(lambda line: (int(line.split(',')[0]),np.array([float(a) for a in line.split(',')[7:]])))
    seqOp = (lambda x, y: (x[0] + y, x[1] + 1))
    combOp = (lambda x, y: (x[0] + y[0], x[1] + y[1]))
    data = data.aggregateByKey((0,0),seqOp,combOp)
    data = data.map(lambda line: [line[0]] + [float(a) for a in list(line[1][0]/line[1][1])])
    return data



def get_players(stat):
    players = {}
    for match in stat.values():
        for team in match.values():
            for player in team['Player_stats'].values():
                id = player['player_details']['player_id']
                if id not in players:
                    players[id] = {'details':player['player_details'], 'stats': dict.fromkeys(types, 0), 'ngames': 0}
                for key in player['Match_stats']:
                    players[id]['stats'][key] += int(player['Match_stats'][key])
                players[id]['ngames'] += 1
    return players



def score(stat):
    if 'goals' in stat:
        GOAL = float(stat['goals'])
    else: GOAL = 0
#    TEAM_RATING = float(stat['goals'])
    ACC_PASS = float(stat['accurate_pass'])
    TOTAL_PASS = float(stat['total_pass'])
    AERIAL_WON = float(stat['aerial_won'])
    AERIAL_LOSS = float(stat['aerial_lost'])
    POSSESION_PERCENTAGE = float(stat['possession_percentage'])
    if 'total_scoring_att' in stat:
        TOTAL_SCORING_ATT = float(stat['total_scoring_att'])
    else: TOTAL_SCORING_ATT = 0
    
    score = GOAL*24/5 + ACC_PASS/TOTAL_PASS*13 +\
        AERIAL_WON/(AERIAL_WON+AERIAL_LOSS)*13+ POSSESION_PERCENTAGE/100*20
#        score = GOAL
    if float(TOTAL_SCORING_ATT) != 0:
        score += GOAL/TOTAL_SCORING_ATT*13
    
    return score


def calc_value(stat):
    player_value = {}
    for match in stat.values():
        for team in match.values():
            for player in team['Player_stats'].values():
                pid = player['player_details']['player_id']
                if pid not in player_value:
                    player_value[pid] = [score(team['aggregate_stats'])]
                else:
                    player_value[pid].append(score(team['aggregate_stats']))
    return player_value
    


def cluster_num(data,method):
    if method == 'elbow':
        from scipy.spatial.distance import cdist 
        distortions = [] 
        inertias = [] 
        mapping1 = {} 
        mapping2 = {} 
        K = range(1,16) 
          
        for k in K: 
            #Building and fitting the model 
            kmeanModel = KMeans(n_clusters=k).fit(data) 
            kmeanModel.fit(data)     
              
            distortions.append(sum(np.min(cdist(data, kmeanModel.cluster_centers_, 
                              'euclidean'),axis=1)) / data.shape[0]) 
            inertias.append(kmeanModel.inertia_) 
          
            mapping1[k] = sum(np.min(cdist(data, kmeanModel.cluster_centers_, 
                         'euclidean'),axis=1)) / data.shape[0] 
            mapping2[k] = kmeanModel.inertia_ 
        
        plt.plot(K, distortions, 'bx-') 
        plt.xlabel('Values of K') 
        plt.ylabel('Distortion') 
        plt.title('The Elbow Method using Distortion') 
        plt.show() 

    elif method == 'silhouette':
        from sklearn.metrics import silhouette_score
        score = []
        for n_clusters in range(2,16):
            clusterer = KMeans (n_clusters=n_clusters)
            preds = clusterer.fit_predict(data)
            centers = clusterer.cluster_centers_
            score.append(silhouette_score(data, preds, metric='euclidean'))
        plt.plot(range(2,16), score)
        plt.title("Silhouette score values vs Numbers of Clusters ")
        plt.show()




def cluster():
    data = load_data_spark()
    df = pd.DataFrame(data.collect(), columns = ['id']+types) 
    match,stat = load_data()
    players = get_players(stat)
    
    player_value = calc_value(stat)
    
    data = np.zeros((len(players), len(types)))
    i = 0
    for p in players:
        data[i] = np.array(list(players[p]['stats'].values()),dtype=int)/players[p]['ngames']
        i += 1
    
    match,stat = load_data()
    model = KMeans(n_clusters=10, random_state=0)
    
    model.fit(data)
    
    res = model.fit_predict(data)
    dataa = TSNE(n_components=2).fit_transform(data)
    vis_x = dataa[:,0]
    vis_y = dataa[:,1]
    plt.scatter(
       vis_x,
       vis_y,
       c=res
    )
    
    
def write_csv():
    import math
    
    i = 0
    cluster = []
    for p in players:
        cluster.append([players[p]['details']['player_id'],players[p]['details']['player_name'],res[i],vis_x[i],vis_y[i],round((sum(player_value[p])/math.sqrt(len(player_value[p]))/25),1)])
        i += 1
    
    i = 0
    dd = []
    for p in players:
        dd.append([vis_x[i],vis_y[i],players[p]['details']['player_id'],players[p]['details']['player_name'], res[i]] + [round((sum(player_value[p])/math.sqrt(len(player_value[p]))/25),1)] + list(np.round(data[i],1)))
        i += 1
    
    my_df = pd.DataFrame(dd)
    my_df.to_csv('player.csv', index=False, header = ['x','y','id','name','style','value'] + types)