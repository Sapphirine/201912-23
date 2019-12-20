# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 21:48:09 2019

@author: Wayne
"""
from src.final import compare_classificaion
from src.predict import match_prediction
from src.player_cluster import cluster
import sys

# count the arguments

if __name__== "__main__":
    if len(sys.argv) == 1:
        cluster()
    else:
        if sys.argv[1]== 'compare':
            compare_classificaion(sys.argv[2],sys.argv[3])
        elif sys.argv[1]== 'predict':
            match_prediction()
        else:
            cluster()