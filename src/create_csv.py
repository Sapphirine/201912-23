# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 13:45:58 2019

@author: Wayne
"""
import json
import os
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd


DIR = 'C:\\Users\\Wayne\\Documents\\Big Data Analytics\\datafilev2\\datafile'

types = ['touches', 'saves', 'total_pass', 'aerial_won', 'formation_place', 'accurate_pass', 'total_tackle',\
 'aerial_lost', 'fouls', 'yellow_card', 'total_scoring_att', 'man_of_the_match', 'goals', 'won_contest',\
 'blocked_scoring_att', 'goal_assist', 'good_high_claim', 'second_yellow', 'red_card', 'error_lead_to_goal',\
 'last_man_tackle', 'six_yard_block', 'post_scoring_att', 'att_pen_target', 'penalty_save', 'penalty_conceded',\
 'clearance_off_line', 'att_pen_goal', 'att_pen_miss', 'own_goals', 'att_pen_post']

st = ['att_goal_low_left', 'won_contest', 'possession_percentage', 'total_throws', 'att_miss_high_left',\
 'blocked_scoring_att', 'total_scoring_att', 'att_sv_low_left', 'total_tackle', 'att_miss_high_right',\
 'aerial_won', 'att_miss_right', 'att_sv_low_centre', 'aerial_lost', 'accurate_pass', 'total_pass',\
 'won_corners', 'shot_off_target', 'ontarget_scoring_att', 'goals', 'att_miss_left', 'fk_foul_lost',\
 'att_sv_low_right', 'att_goal_low_centre', 'total_offside', 'att_sv_high_left', 'att_goal_high_left',\
 'att_miss_high', 'att_goal_low_right', 'att_goal_high_right', 'att_sv_high_centre', 'att_post_high',\
 'post_scoring_att', 'att_sv_high_right', 'penalty_save', 'att_pen_goal', 'att_post_right', 'att_post_left', 'att_goal_high_centre']

def load_data():
    match = {}
    stat = {}
    for a in os.listdir(DIR):
        f = open(os.path.join(DIR, a, 'season_match_stats.json'), encoding="utf8")
        match.update(json.load(f))

        f = open(os.path.join(DIR, a, 'season_stats.json'), encoding="utf8")
        stat.update(json.load(f))
    return match, stat

match, stat = load_data()

def get_players():
    players = []
    for match in stat.values():
        for team in match.values():
            for player in team['Player_stats'].values():
                player['player_details'].update({'team_id':team['team_details']['team_id'],'team_name':team['team_details']['team_name']})
                player['player_details'].update(dict.fromkeys(types, 0))
                players.append(player['player_details'])
                for key in player['Match_stats']:
                    players[-1][key] += int(player['Match_stats'][key])
    df = pd.DataFrame(players)
    df.to_csv('player_stats_new.csv',index=False)
    return df

players = get_players()

def get_seasons():
    teams = []
    for match_key in stat:
        for team in stat[match_key].values():
            dic = {'match_id': match_key}
            dic.update(team['team_details'])
            dic.update(dict.fromkeys(st, 0))
            teams.append(dic)
            for key in team['aggregate_stats']:
                teams[-1][key] += float(team['aggregate_stats'][key])
    df = pd.DataFrame(teams)
    df.to_csv('season_stats_new.csv',index=False)
    return df

seasons = get_seasons()

def get_teams():
    f = open('season_stats_new.csv')
    data = f.readlines()
    header = data[0]
    data = data[1:]
    
    teams = {}
    for d in data:
        d = d.split(',')
        if d[2] not in teams:
            teams[d[2]] = np.array([1,d[3]]+d[5:],dtype=float)
        else:
            teams[d[2]] += np.array([1,d[3]]+d[5:],dtype=float)
    
    header = header.split(',')
    lines = [[header[2],header[3]] + header[5:]]
    for team in teams:
        lines.append([team]+list(np.round(teams[team][1:]/teams[team][0],1)))
    
    my_df = pd.DataFrame(lines)
    my_df.to_csv('team_stats.csv', index=False)
    return lines


teams = get_teams()