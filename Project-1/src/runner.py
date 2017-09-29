# Project 5
# File  : runner.py
# Author: Rajarshi Biswas
#       : Sayam Ganguly
import sys
from K_Means_Wine import analyze_wine_data
from K_Means_Hard import analyze_hard_data
from K_Easy_Hard import analyze_easy_data

k = int(sys.argv[1])
# Analyze the Wine data
analyze_wine_data(k)
# Analyze the Hard data
analyze_hard_data(k)
# Analyze the Easy data
analyze_easy_data(k)
