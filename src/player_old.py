import json
import os
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pyspark import SparkConf, SparkContext
import pyspark

DIR = 'C:\\Users\\Wayne\\Documents\\Big Data Analytics\\datafilev2\\datafile'
filename = 'player_stats.csv'
types = ['touches', 'saves', 'total_pass', 'aerial_won', 'formation_place', 'accurate_pass', 'total_tackle',\
 'aerial_lost', 'fouls', 'yellow_card', 'total_scoring_att', 'man_of_the_match', 'goals', 'won_contest',\
 'blocked_scoring_att', 'goal_assist', 'good_high_claim', 'second_yellow', 'red_card', 'error_lead_to_goal',\
 'last_man_tackle', 'six_yard_block', 'post_scoring_att', 'att_pen_target', 'penalty_save', 'penalty_conceded',\
 'clearance_off_line', 'att_pen_goal', 'att_pen_miss', 'own_goals', 'att_pen_post']

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

players = get_players()

data = np.zeros((len(players), len(types)))
i = 0
for p in players:
    data[i] = np.array(list(players[p]['stats'].values()),dtype=int)/players[p]['ngames']
    i += 1



model = DBSCAN(eps = 0.01, min_samples = 5)

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

st = []
for match in stat.values():
    for team in match.values():
        for a in team['aggregate_stats']:
            if a not in st:
                st.append(a)
