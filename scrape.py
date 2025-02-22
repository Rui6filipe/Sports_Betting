# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 22:31:30 2024

@author: ruira
"""

#%%
import requests
from bs4 import BeautifulSoup
import csv
import re
from datetime import datetime
import locale
import time
import traceback
import numpy as np
from operator import itemgetter
import pandas as pd
import os
import math

start_time = time.time()
locale.setlocale(locale.LC_TIME, 'pt_PT.UTF-8')
moving_average_number = 3

#%%
# List of ranges of game IDs

id_ranges = [
    (2, 183),
    (188, 369),
    (1290, 1471),
    (2189, 2321),
    (3780, 3961),
    (5450, 5631),
    (7911, 8092),
    (10304, 10485),
    (12879, 12976)
]

excluded_id_ranges = [319, 1428]

next_games_ids = [(12977, 12983)]

'''
2425 - 12879 to 13060
2324 - 10304 to 10485 e playoffs
2223 - 7911 to 8092 e playoffs
2122 - 5450 to 5631 e playoffs
2021 - 3780 to 3961 e playoffs
1920 - 2189 to 2321
1819 - 1290 to 1471
1718 - 188 to 369
1617 - 2 to 183
'''

data_points = {
    "team1": {"tag": "span", "attrs": {"style": "font-size:1.6em;"}, "index": 0},
    "team2": {"tag": "span", "attrs": {"style": "font-size:1.6em;"}, "index": 1},
    "goals_1": {"tag": "td", "attrs": {"class": "fondo3", "align": "center"}, "index": 10},
    "penalties_1": {"tag": "td", "attrs": {"class": "fondo3", "align": "center"}, "index": 13},
    "direct_free_hits_1": {"tag": "td", "attrs": {"class": "fondo3", "align": "center"}, "index": 14},
    "warnings_1": {"tag": "td", "attrs": {"class": "fondo3", "align": "center"}, "index": 15},
    "blue_cards_1": {"tag": "td", "attrs": {"class": "fondo3", "align": "center"}, "index": 16},
    "fouls_1": {"tag": "div", "attrs": {"class": "box_faltas"}, "index": 0},
    "goals_2": {"tag": "td", "attrs": {"class": "fondo3", "align": "center"}, "index": 28},
    "penalties_2": {"tag": "td", "attrs": {"class": "fondo3", "align": "center"}, "index": 31},
    "direct_free_hits_2": {"tag": "td", "attrs": {"class": "fondo3", "align": "center"}, "index": 32},
    "warnings_2": {"tag": "td", "attrs": {"class": "fondo3", "align": "center"}, "index": 33},
    "blue_cards_2": {"tag": "td", "attrs": {"class": "fondo3", "align": "center"}, "index": 34},
    "fouls_2": {"tag": "div", "attrs": {"class": "box_faltas"}, "index": 1},
    "date": {"tag": "td", "attrs": {"colspan": "2"}, "index": 0},
    "ref1": {"tag": "td", "attrs": {"colspan": "3", "align": "right"}, "index": 0},
    "ref2": {"tag": "td", "attrs": {"colspan": "3"}, "index": 0},
    "ref3": {"tag": "td", "attrs": {"align": "right"}, "index": 2},
    
    "gr1_1_played": {"tag": "td", "attrs": {"class": "fondo2", "align": "center"}, "index": 1},
    "gr1_1_name": {"tag": "a", "attrs": {"href": "#"}, "index": 2},
    "gr1_1_saves": {"tag": "td", "attrs": {"class": "fondo2", "align": "center"}, "index": 5},
    
    "pl1_1_played": {"tag": "td", "attrs": {"class": "fondo4", "align": "center"}, "index": 1},
    "pl1_1_name": {"tag": "a", "attrs": {"href": "#"}, "index": 3},
    "pl1_1_goals": {"tag": "td", "attrs": {"class": "fondo4", "align": "center"}, "index": 3},
    "pl1_1_pens": {"tag": "td", "attrs": {"class": "fondo4", "align": "center"}, "index": 6},
    "pl1_1_dfh": {"tag": "td", "attrs": {"class": "fondo4", "align": "center"}, "index": 7},
    
    "pl2_1_played": {"tag": "td", "attrs": {"class": "fondo2", "align": "center"}, "index": 12},
    "pl2_1_name": {"tag": "a", "attrs": {"href": "#"}, "index": 4},
    "pl2_1_goals": {"tag": "td", "attrs": {"class": "fondo2", "align": "center"}, "index": 14},
    "pl2_1_pens": {"tag": "td", "attrs": {"class": "fondo2", "align": "center"}, "index": 17},
    "pl2_1_dfh": {"tag": "td", "attrs": {"class": "fondo2", "align": "center"}, "index": 18},
    
    "pl3_1_played": {"tag": "td", "attrs": {"class": "fondo4", "align": "center"}, "index": 12},
    "pl3_1_name": {"tag": "a", "attrs": {"href": "#"}, "index": 5},
    "pl3_1_goals": {"tag": "td", "attrs": {"class": "fondo4", "align": "center"}, "index": 14},
    "pl3_1_pens": {"tag": "td", "attrs": {"class": "fondo4", "align": "center"}, "index": 17},
    "pl3_1_dfh": {"tag": "td", "attrs": {"class": "fondo4", "align": "center"}, "index": 18},
    
    "pl4_1_played": {"tag": "td", "attrs": {"class": "fondo2", "align": "center"}, "index": 23},
    "pl4_1_name": {"tag": "a", "attrs": {"href": "#"}, "index": 6},
    "pl4_1_goals": {"tag": "td", "attrs": {"class": "fondo2", "align": "center"}, "index": 25},
    "pl4_1_pens": {"tag": "td", "attrs": {"class": "fondo2", "align": "center"}, "index": 28},
    "pl4_1_dfh": {"tag": "td", "attrs": {"class": "fondo2", "align": "center"}, "index": 29},
    
    "pl5_1_played": {"tag": "td", "attrs": {"class": "fondo4", "align": "center"}, "index": 23},
    "pl5_1_name": {"tag": "a", "attrs": {"href": "#"}, "index": 7},
    "pl5_1_goals": {"tag": "td", "attrs": {"class": "fondo4", "align": "center"}, "index": 25},
    "pl5_1_pens": {"tag": "td", "attrs": {"class": "fondo4", "align": "center"}, "index": 28},
    "pl5_1_dfh": {"tag": "td", "attrs": {"class": "fondo4", "align": "center"}, "index": 29},
    
    "pl6_1_played": {"tag": "td", "attrs": {"class": "fondo2", "align": "center"}, "index": 34},
    "pl6_1_name": {"tag": "a", "attrs": {"href": "#"}, "index": 8},
    "pl6_1_goals": {"tag": "td", "attrs": {"class": "fondo2", "align": "center"}, "index": 36},
    "pl6_1_pens": {"tag": "td", "attrs": {"class": "fondo2", "align": "center"}, "index": 39},
    "pl6_1_dfh": {"tag": "td", "attrs": {"class": "fondo2", "align": "center"}, "index": 40},
    
    "pl7_1_played": {"tag": "td", "attrs": {"class": "fondo4", "align": "center"}, "index": 34},
    "pl7_1_name": {"tag": "a", "attrs": {"href": "#"}, "index": 9},
    "pl7_1_goals": {"tag": "td", "attrs": {"class": "fondo4", "align": "center"}, "index": 36},
    "pl7_1_pens": {"tag": "td", "attrs": {"class": "fondo4", "align": "center"}, "index": 39},
    "pl7_1_dfh": {"tag": "td", "attrs": {"class": "fondo4", "align": "center"}, "index": 40},
    
    "pl8_1_played": {"tag": "td", "attrs": {"class": "fondo2", "align": "center"}, "index": 45},
    "pl8_1_name": {"tag": "a", "attrs": {"href": "#"}, "index": 10},
    "pl8_1_goals": {"tag": "td", "attrs": {"class": "fondo2", "align": "center"}, "index": 47},
    "pl8_1_pens": {"tag": "td", "attrs": {"class": "fondo2", "align": "center"}, "index": 50},
    "pl8_1_dfh": {"tag": "td", "attrs": {"class": "fondo2", "align": "center"}, "index": 51},
    
    "gr2_1_played": {"tag": "td", "attrs": {"class": "fondo4", "align": "center"}, "index": 45},
    "gr2_1_name": {"tag": "a", "attrs": {"href": "#"}, "index": 11},
    "gr2_1_saves": {"tag": "td", "attrs": {"class": "fondo4", "align": "center"}, "index": 49},
    
    
    "gr1_2_played": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                     "tag2": "td", "attrs2": {"class": "fondo2", "align": "center"}, "index2": 1},
    "gr1_2_name": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                   "tag2": "a", "attrs2": {"href": "#"}, "index2": 0},
    "gr1_2_saves": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                    "tag2": "td", "attrs2": {"class": "fondo2", "align": "center"}, "index2": 5},
    
    "pl1_2_played": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                     "tag2": "td", "attrs2": {"class": "fondo4", "align": "center"}, "index2": 1},
    "pl1_2_name": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                   "tag2": "a", "attrs2": {"href": "#"}, "index2": 1},
    "pl1_2_goals": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                    "tag2": "td", "attrs2": {"class": "fondo4", "align": "center"}, "index2": 3},
    "pl1_2_pens": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                   "tag2": "td", "attrs2": {"class": "fondo4", "align": "center"}, "index2": 6},
    "pl1_2_dfh": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                  "tag2": "td", "attrs2": {"class": "fondo4", "align": "center"}, "index2": 7},
    
    "pl2_2_played": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                     "tag2": "td", "attrs2": {"class": "fondo2", "align": "center"}, "index2": 12},
    "pl2_2_name": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                   "tag2": "a", "attrs2": {"href": "#"}, "index2": 2},
    "pl2_2_goals": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                    "tag2": "td", "attrs2": {"class": "fondo2", "align": "center"}, "index2": 14},
    "pl2_2_pens": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                   "tag2": "td", "attrs2": {"class": "fondo2", "align": "center"}, "index2": 17},
    "pl2_2_dfh": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                  "tag2": "td", "attrs2": {"class": "fondo2", "align": "center"}, "index2": 18},
    
    "pl3_2_played": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                     "tag2": "td", "attrs2": {"class": "fondo4", "align": "center"}, "index2": 12},
    "pl3_2_name": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                   "tag2": "a", "attrs2": {"href": "#"}, "index2": 3},
    "pl3_2_goals": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                    "tag2": "td", "attrs2": {"class": "fondo4", "align": "center"}, "index2": 14},
    "pl3_2_pens": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                   "tag2": "td", "attrs2": {"class": "fondo4", "align": "center"}, "index2": 17},
    "pl3_2_dfh": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                  "tag2": "td", "attrs2": {"class": "fondo4", "align": "center"}, "index2": 18},
    
    "pl4_2_played": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                     "tag2": "td", "attrs2": {"class": "fondo2", "align": "center"}, "index2": 23},
    "pl4_2_name": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                   "tag2": "a", "attrs2": {"href": "#"}, "index2": 4},
    "pl4_2_goals": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                    "tag2": "td", "attrs2": {"class": "fondo2", "align": "center"}, "index2": 25},
    "pl4_2_pens": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                   "tag2": "td", "attrs2": {"class": "fondo2", "align": "center"}, "index2": 28},
    "pl4_2_dfh": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                  "tag2": "td", "attrs2": {"class": "fondo2", "align": "center"}, "index2": 29},
    
    "pl5_2_played": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                     "tag2": "td", "attrs2": {"class": "fondo4", "align": "center"}, "index2": 23},
    "pl5_2_name": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                   "tag2": "a", "attrs2": {"href": "#"}, "index2": 5},
    "pl5_2_goals": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                    "tag2": "td", "attrs2": {"class": "fondo4", "align": "center"}, "index2": 25},
    "pl5_2_pens": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                   "tag2": "td", "attrs2": {"class": "fondo4", "align": "center"}, "index2": 28},
    "pl5_2_dfh": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                  "tag2": "td", "attrs2": {"class": "fondo4", "align": "center"}, "index2": 29},
    
    "pl6_2_played": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                     "tag2": "td", "attrs2": {"class": "fondo2", "align": "center"}, "index2": 34},
    "pl6_2_name": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                   "tag2": "a", "attrs2": {"href": "#"}, "index2": 6},
    "pl6_2_goals": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                    "tag2": "td", "attrs2": {"class": "fondo2", "align": "center"}, "index2": 36},
    "pl6_2_pens": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                   "tag2": "td", "attrs2": {"class": "fondo2", "align": "center"}, "index2": 39},
    "pl6_2_dfh": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                  "tag2": "td", "attrs2": {"class": "fondo2", "align": "center"}, "index2": 40},
    
    "pl7_2_played": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                     "tag2": "td", "attrs2": {"class": "fondo4", "align": "center"}, "index2": 34},
    "pl7_2_name": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                   "tag2": "a", "attrs2": {"href": "#"}, "index2": 7},
    "pl7_2_goals": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                    "tag2": "td", "attrs2": {"class": "fondo4", "align": "center"}, "index2": 36},
    "pl7_2_pens": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                   "tag2": "td", "attrs2": {"class": "fondo4", "align": "center"}, "index2": 39},
    "pl7_2_dfh": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                  "tag2": "td", "attrs2": {"class": "fondo4", "align": "center"}, "index2": 40},
    
    "pl8_2_played": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                     "tag2": "td", "attrs2": {"class": "fondo2", "align": "center"}, "index2": 45},
    "pl8_2_name": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                   "tag2": "a", "attrs2": {"href": "#"}, "index2": 8},
    "pl8_2_goals": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                    "tag2": "td", "attrs2": {"class": "fondo2", "align": "center"}, "index2": 47},
    "pl8_2_pens": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                   "tag2": "td", "attrs2": {"class": "fondo2", "align": "center"}, "index2": 50},
    "pl8_2_dfh": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                  "tag2": "td", "attrs2": {"class": "fondo2", "align": "center"}, "index2": 51},

    "gr2_2_played": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                     "tag2": "td", "attrs2": {"class": "fondo4", "align": "center"}, "index2": 45},
    "gr2_2_name": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                   "tag2": "a", "attrs2": {"href": "#"}, "index2": 9},
    "gr2_2_saves": {"tag": "div", "attrs": {"class": "estadisticas", "style": "margin-top:1em;"}, "index": 0,
                    "tag2": "td", "attrs2": {"class": "fondo4", "align": "center"}, "index2": 49}
}

data_points2 = {
    "team1": {"tag": "span", "attrs": {"style": "font-size:1.6em;"}, "index": 0},
    "team2": {"tag": "span", "attrs": {"style": "font-size:1.6em;"}, "index": 1},
    "date": {"tag": "td", "attrs": {"colspan": "2"}, "index": 0},
    "ref1": {"tag": "td", "attrs": {"colspan": "3", "align": "right"}, "index": 0},
    "ref2": {"tag": "td", "attrs": {"colspan": "3"}, "index": 0},
    "ref3": {"tag": "td", "attrs": {"align": "right"}, "index": 2}
}

#%%
def scrape_game_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    game_data = {}
    
    for data_point, config in data_points.items():
        game_data[data_point] = extract_data_point(soup, config)
    
    return game_data


def scrape_next_game_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    game_data = {}
    
    for data_point, config in data_points2.items():
        game_data[data_point] = extract_data_point(soup, config)
    
    return game_data

#%%
def extract_data_point(soup, config):
            
    elements = soup.find_all(config["tag"], attrs=config["attrs"])
    raw_text = elements[config["index"]].text.strip() if elements else None
    
    if data_points["date"] == config:
        date_match = re.search(r"(\d{1,2} de \w+ de \d{4})", raw_text)
        if date_match:
            raw_date = date_match.group(1) 
            raw_date_corrected = correct_month(raw_date)
            formatted_date = datetime.strptime(raw_date_corrected, "%d de %B de %Y")
            return formatted_date
    
    if data_points["ref1"] == config:
        refs_match = re.search(r"Árbitros: (.+)", raw_text)
        if refs_match:
            refs_text = refs_match.group(1)
            refs_names = [name.strip() for name in refs_text.split(',')]
            return refs_names[0]
    
    if data_points["ref2"] == config:
        refs_match = re.search(r"Árbitros: (.+)", raw_text)
        if refs_match:
            refs_text = refs_match.group(1)
            refs_names = [name.strip() for name in refs_text.split(',')]
            return refs_names[1]
    
    if data_points["ref3"] == config:
        refs_match = re.search(r"Árbitros: (.+)", raw_text)
        if refs_match:
            refs_text = refs_match.group(1)
            refs_names = [name.strip() for name in refs_text.split(',')]
            return refs_names[2]
    
    if "tag2" in config and "attrs2" in config:
        nested_elements = elements[config["index"]].find_all(config["tag2"], attrs=config["attrs2"])
        return nested_elements[config["index2"]].text.strip()

    return raw_text


def correct_month(raw_date):
    # Substitute março by marco
    return raw_date.replace("março", "marÃ§o")

def adjust_game_data(game_data):
    # Define the season ranges and corresponding suffixes
    season_ranges = [
        (datetime(2024, 9, 1), datetime(2025, 9, 1), "2425"),
        (datetime(2023, 9, 1), datetime(2024, 9, 1), "2324"),
        (datetime(2022, 9, 1), datetime(2023, 9, 1), "2223"),
        (datetime(2021, 9, 1), datetime(2022, 9, 1), "2122"),
        (datetime(2020, 9, 1), datetime(2021, 9, 1), "2021"),
        (datetime(2019, 9, 1), datetime(2020, 9, 1), "1920"),
        (datetime(2018, 9, 1), datetime(2019, 9, 1), "1819"),
        (datetime(2017, 9, 1), datetime(2018, 9, 1), "1718"),
        (datetime(2016, 9, 1), datetime(2017, 9, 1), "1617")
    ]
    
    # Iterate through the games and apply the season suffix
    for game in game_data:
        date = game["date"]
        
        # Check which season the game belongs to and update team names
        for start_date, end_date, season in season_ranges:
            if start_date <= date < end_date:
                game["team1"] = f"{game['team1']} {season}"
                game["team2"] = f"{game['team2']} {season}"
                break  # Once the correct season is found, no need to check further
        
        if date < datetime.today():
            for key, value in game.items():
                if key.startswith("pl") and "_name" in key and value.endswith("\xa0©"):
                    game[key] = value.replace("\xa0©", "")

    # Sort the game data by the parsed date
    game_data = sorted(game_data, key=lambda x: x['date'])
    
    return game_data


#%%
def generate_features(game_data):
    team_stats = {}
    player_stats = {}
    ref_stats = {}
    features = []

    def calc_trend_avg(stat_list):
        return np.mean(stat_list) if stat_list else 0
    
    def calc_avg(stat, team, denominator):
        if team_stats[team].get(denominator, 1) > 0:
            return team_stats[team].get(stat, 0) / team_stats[team].get(denominator, 1)
        else:
            return 0

    def calc_avg2(stat1, stat2, team, denominator):
        if team_stats[team].get(denominator, 1) > 0:
            return (team_stats[team].get(stat1, 0) + team_stats[team].get(stat2, 0)) / team_stats[team].get(denominator, 1)
        else:
            return 0
        
    def calc_var(stat, team):
        goals = team_stats[team].get(stat, 0)
        if len(goals)>0:
            mean = sum(goals) / len(goals)
            variance = sum((x - mean) ** 2 for x in goals) / len(goals)  
            return math.sqrt(variance)
        else:
            return 0

    def initialize_team_and_player_stats(game):
        for team in [game["team1"], game["team2"]]:
            if team not in team_stats:
                team_stats[team] = initialize_team_stats()
                player_stats[team] = {}
        for key, value in game.items():
            if key.startswith("pl") and "_name" in key:
                team_key = game[f"team{key.split('_')[1]}"]
                player_name = value.strip()
                if player_name and player_name not in player_stats[team_key]:
                    player_stats[team_key][player_name] = initialize_player_stats()

        # Initialize referee stats
        for ref in [game["ref1"], game["ref2"]]:
            if ref not in ref_stats:
                ref_stats[ref] = initialize_ref_stats()

    for game in game_data:
        try:
            team1, team2, game_date = game["team1"], game["team2"], game["date"] 
            initialize_team_and_player_stats(game)
        
            if game_date.date() < datetime.today().date():
                goals_scored = int(game["goals_1"]) + int(game["goals_2"])
           
            avg_goals_scored_1 = calc_avg2("goals_scored_home", "goals_scored_away", team1, "games_played")
            avg_goals_conceded_1 = calc_avg2("goals_conceded_home", "goals_conceded_away", team1, "games_played")
            avg_goals_scored_2 = calc_avg2("goals_scored_home", "goals_scored_away", team2, "games_played")
            avg_goals_conceded_2 = calc_avg2("goals_conceded_home", "goals_conceded_away", team2, "games_played")
            
            team1_goal_var = calc_var("total_goals", team1)
            team2_goal_var = calc_var("total_goals", team2)
            
            avg_goals_scored_home_1 = calc_avg("goals_scored_home", team1, "games_home")
            avg_goals_scored_away_1 = calc_avg("goals_scored_away", team1, "games_away")
            avg_goals_conceded_home_1 = calc_avg("goals_conceded_home", team1, "games_home")
            avg_goals_conceded_away_1 = calc_avg("goals_conceded_away", team1, "games_away")
            avg_goals_scored_home_2 = calc_avg("goals_scored_home", team2, "games_home")
            avg_goals_scored_away_2 = calc_avg("goals_scored_away", team2, "games_away")
            avg_goals_conceded_home_2 = calc_avg("goals_conceded_home", team2, "games_home")
            avg_goals_conceded_away_2 = calc_avg("goals_conceded_away", team2, "games_away")
            
            goals_scored_trend_1 = calc_trend_avg(team_stats[team1]["goals_scored_list"])
            goals_conceded_trend_1 = calc_trend_avg(team_stats[team1]["goals_conceded_list"])
            goals_scored_trend_2 = calc_trend_avg(team_stats[team2]["goals_scored_list"])
            goals_conceded_trend_2 = calc_trend_avg(team_stats[team2]["goals_conceded_list"])

            win_pct_1, draw_pct_1, loss_pct_1 = calc_avg("wins", team1, "games_played"), calc_avg("draws", team1, "games_played"), calc_avg("losses", team1, "games_played")
            win_pct_2, draw_pct_2, loss_pct_2 = calc_avg("wins", team2, "games_played"), calc_avg("draws", team2, "games_played"), calc_avg("losses", team2, "games_played")

            team1_sos, team2_sos = calculate_strength_of_schedule(game_data, team1, game_date), calculate_strength_of_schedule(game_data, team2, game_date)
            penalties_1 = [calc_avg(f"penalties_{stat}", team1, "games_played") for stat in ["tried", "scored", "tried_against", "conceded"]]
            penalties_2 = [calc_avg(f"penalties_{stat}", team2, "games_played") for stat in ["tried", "scored", "tried_against", "conceded"]]
            dfh_1 = [calc_avg(f"dfh_{stat}", team1, "games_played") for stat in ["tried", "scored", "tried_against", "conceded"]]
            dfh_2 = [calc_avg(f"dfh_{stat}", team2, "games_played") for stat in ["tried", "scored", "tried_against", "conceded"]]
            avg_fouls_1 = [calc_avg(f"fouls_{stat}", team1, "games_played") for stat in ["favor", "against"]]
            avg_fouls_2 = [calc_avg(f"fouls_{stat}", team2, "games_played") for stat in ["favor", "against"]]
            avg_cards_1 = [calc_avg(f"cards_{stat}", team1, "games_played") for stat in ["favor", "against"]]
            avg_cards_2 = [calc_avg(f"cards_{stat}", team2, "games_played") for stat in ["favor", "against"]]

            scoring_conc_1, scoring_conc_2 = calculate_scoring_concentration(team1, player_stats), calculate_scoring_concentration(team2, player_stats)
            head_to_head_stats = calculate_head_to_head(game_data, team1, team2, game_date)

            rest_1, rest_2 = calculate_rest(game_data, team1, game_date), calculate_rest(game_data, team2, game_date)
            
            ref1_avg_fouls = ref_stats[game["ref1"]].get("total_fouls", 0) / max(ref_stats[game["ref1"]].get("games_officiated", 1), 1)
            ref1_avg_cards = ref_stats[game["ref1"]].get("total_cards", 0) / max(ref_stats[game["ref1"]].get("games_officiated", 1), 1)
            ref2_avg_fouls = ref_stats[game["ref2"]].get("total_fouls", 0) / max(ref_stats[game["ref2"]].get("games_officiated", 1), 1)
            ref2_avg_cards = ref_stats[game["ref2"]].get("total_cards", 0) / max(ref_stats[game["ref2"]].get("games_officiated", 1), 1)
            
            games_played = team_stats[team1]["games_played"]
            
            if game_date.date() < datetime.today().date():
                features.append({
                    "team1": team1, "team2": team2, "home_team": team1, "games_played": games_played,
                    "team1_goal_var": team1_goal_var, "team2_goal_var": team2_goal_var,
                    "avg_goals_scored_1": avg_goals_scored_1, "avg_goals_conceded_1": avg_goals_conceded_1, 
                    "avg_goals_scored_2": avg_goals_scored_2, "avg_goals_conceded_2": avg_goals_conceded_2, 
                    "avg_goals_scored_home_1": avg_goals_scored_home_1, "avg_goals_scored_away_1": avg_goals_scored_away_1,
                    "avg_goals_conceded_home_1": avg_goals_conceded_home_1, "avg_goals_conceded_away_1": avg_goals_conceded_away_1,
                    "avg_goals_scored_home_2": avg_goals_scored_home_2, "avg_goals_scored_away_2": avg_goals_scored_away_2,
                    "avg_goals_conceded_home_2": avg_goals_conceded_home_2, "avg_goals_conceded_away_2": avg_goals_conceded_away_2,
                    "win_percentage_1": win_pct_1, "draw_percentage_1": draw_pct_1, "loss_percentage_1": loss_pct_1,
                    "win_percentage_2": win_pct_2, "draw_percentage_2": draw_pct_2, "loss_percentage_2": loss_pct_2,
                    "team_sos_avg_points_1": team1_sos["sos_avg_points"], "team_sos_avg_goal_difference_1": team1_sos["sos_avg_goal_difference"],
                    "team_sos_avg_points_2": team2_sos["sos_avg_points"], "team_sos_avg_goal_difference_2": team2_sos["sos_avg_goal_difference"],
                    **{f"penalties_{stat}_1": val for stat, val in zip(["tried", "scored", "tried_against", "conceded"], penalties_1)},
                    **{f"penalties_{stat}_2": val for stat, val in zip(["tried", "scored", "tried_against", "conceded"], penalties_2)},
                    **{f"dfh_{stat}_1": val for stat, val in zip(["tried", "scored", "tried_against", "conceded"], dfh_1)},
                    **{f"dfh_{stat}_2": val for stat, val in zip(["tried", "scored", "tried_against", "conceded"], dfh_2)},
                    **{f"avg_fouls_{stat}_1": val for stat, val in zip(["favor", "against"], avg_fouls_1)},
                    **{f"avg_fouls_{stat}_2": val for stat, val in zip(["favor", "against"], avg_fouls_2)},
                    **{f"avg_cards_{stat}_1": val for stat, val in zip(["favor", "against"], avg_cards_1)},
                    **{f"avg_cards_{stat}_2": val for stat, val in zip(["favor", "against"], avg_cards_2)},
                    "scoring_concentration_1": scoring_conc_1, "scoring_concentration_2": scoring_conc_2,
                    "head_to_head_avg_team_goals_1": head_to_head_stats["avg_team1_goals"], "head_to_head_avg_team_goals_2": head_to_head_stats["avg_team2_goals"],
                    "rest_1": rest_1, "rest_2": rest_2, "game_date": game["date"],
                    "goals_scored_trend_1": goals_scored_trend_1, "goals_conceded_trend_1": goals_conceded_trend_1,
                    "goals_scored_trend_2": goals_scored_trend_2, "goals_conceded_trend_2": goals_conceded_trend_2,
                    "ref1_avg_fouls": ref1_avg_fouls, "ref1_avg_cards": ref1_avg_cards, "ref2_avg_fouls": ref2_avg_fouls,
                    "ref2_avg_cards": ref2_avg_cards, "goals_scored": goals_scored})
    
                update_team_stats(team_stats, game)
                update_player_stats(player_stats, game)
                update_ref_stats(ref_stats, game)
            
            else:
                features.append({
                    "team1": team1, "team2": team2, "home_team": team1, "games_played": games_played,
                    "team1_goal_var": team1_goal_var, "team2_goal_var": team2_goal_var,
                    "avg_goals_scored_1": avg_goals_scored_1, "avg_goals_conceded_1": avg_goals_conceded_1, 
                    "avg_goals_scored_2": avg_goals_scored_2, "avg_goals_conceded_2": avg_goals_conceded_2,
                    "avg_goals_scored_home_1": avg_goals_scored_home_1, "avg_goals_scored_away_1": avg_goals_scored_away_1,
                    "avg_goals_conceded_home_1": avg_goals_conceded_home_1, "avg_goals_conceded_away_1": avg_goals_conceded_away_1,
                    "avg_goals_scored_home_2": avg_goals_scored_home_2, "avg_goals_scored_away_2": avg_goals_scored_away_2,
                    "avg_goals_conceded_home_2": avg_goals_conceded_home_2, "avg_goals_conceded_away_2": avg_goals_conceded_away_2,
                    "win_percentage_1": win_pct_1, "draw_percentage_1": draw_pct_1, "loss_percentage_1": loss_pct_1,
                    "win_percentage_2": win_pct_2, "draw_percentage_2": draw_pct_2, "loss_percentage_2": loss_pct_2,
                    "team_sos_avg_points_1": team1_sos["sos_avg_points"], "team_sos_avg_goal_difference_1": team1_sos["sos_avg_goal_difference"],
                    "team_sos_avg_points_2": team2_sos["sos_avg_points"], "team_sos_avg_goal_difference_2": team2_sos["sos_avg_goal_difference"],
                    **{f"penalties_{stat}_1": val for stat, val in zip(["tried", "scored", "tried_against", "conceded"], penalties_1)},
                    **{f"penalties_{stat}_2": val for stat, val in zip(["tried", "scored", "tried_against", "conceded"], penalties_2)},
                    **{f"dfh_{stat}_1": val for stat, val in zip(["tried", "scored", "tried_against", "conceded"], dfh_1)},
                    **{f"dfh_{stat}_2": val for stat, val in zip(["tried", "scored", "tried_against", "conceded"], dfh_2)},
                    **{f"avg_fouls_{stat}_1": val for stat, val in zip(["favor", "against"], avg_fouls_1)},
                    **{f"avg_fouls_{stat}_2": val for stat, val in zip(["favor", "against"], avg_fouls_2)},
                    **{f"avg_cards_{stat}_1": val for stat, val in zip(["favor", "against"], avg_cards_1)},
                    **{f"avg_cards_{stat}_2": val for stat, val in zip(["favor", "against"], avg_cards_2)},
                    "scoring_concentration_1": scoring_conc_1, "scoring_concentration_2": scoring_conc_2,
                    "head_to_head_avg_team_goals_1": head_to_head_stats["avg_team1_goals"], "head_to_head_avg_team_goals_2": head_to_head_stats["avg_team2_goals"],
                    "rest_1": rest_1, "rest_2": rest_2, "game_date": game["date"],
                    "goals_scored_trend_1": goals_scored_trend_1, "goals_conceded_trend_1": goals_conceded_trend_1,
                    "goals_scored_trend_2": goals_scored_trend_2, "goals_conceded_trend_2": goals_conceded_trend_2,
                    "ref1_avg_fouls": ref1_avg_fouls, "ref1_avg_cards": ref1_avg_cards, "ref2_avg_fouls": ref2_avg_fouls,
                    "ref2_avg_cards": ref2_avg_cards})
                
        except Exception as e:
            print(f"Error processing game data: {e}")
            traceback.print_exc()

    return features



#%%
def initialize_team_stats():
    return {
        "games_played": 0,
        "games_home": 0,
        "games_away": 0,
        "goals_scored_home": 0,
        "goals_scored_away": 0,
        "goals_conceded_home": 0,
        "goals_conceded_away": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "penalties_scored": 0,
        "penalties_tried": 0,
        "penalties_conceded": 0,
        "penalties_tried_against": 0,
        "dfh_scored": 0,
        "dfh_tried": 0,
        "dfh_conceded": 0,
        "dfh_tried_against": 0,
        "fouls_favor": 0,
        "fouls_against": 0,
        "cards_favor": 0,
        "cards_against": 0,
        "goals_scored_list": [],
        "goals_conceded_list": [],
        "total_goals": []
    }


def initialize_player_stats(): 
    return {
        "goals_scored": 0
    }


def initialize_ref_stats(): 
    return {
        "total_fouls": 0,
        "total_cards": 0,
        "games_officiated": 0
    }


#%%
def update_team_stats(team_stats, game):
    team1 = game["team1"]
    team2 = game["team2"]
    goals_1 = int(game["goals_1"])
    goals_2 = int(game["goals_2"])

    # Update games played
    team_stats[team1]["games_played"] += 1
    team_stats[team2]["games_played"] += 1
    
    # Update games at home and away
    team_stats[team1]["games_home"] += 1
    team_stats[team2]["games_away"] += 1

    # Update goals scored and allowed
    team_stats[team1]["goals_scored_home"] += goals_1
    team_stats[team2]["goals_scored_away"] += goals_2
    team_stats[team1]["goals_conceded_home"] += goals_2
    team_stats[team2]["goals_conceded_away"] += goals_1
    
    team_stats[team1]["total_goals"].append(goals_1 + goals_2)
    team_stats[team2]["total_goals"].append(goals_1 + goals_2)
    
    for team, opponent, team_goals, opponent_goals in [(team1, team2, goals_1, goals_2), (team2, team1, goals_2, goals_1)]:
        for stat, goals in [("goals_scored_list", team_goals), ("goals_conceded_list", opponent_goals)]:
            if len(team_stats[team][stat]) >= moving_average_number:
                team_stats[team][stat].pop(0)
            team_stats[team][stat].append(goals)
        
    # Update penalties and direct free hits    
    penalties_scored_1, penalties_tried_1 = extract_penalties_and_dfh(game.get("penalties_1", "0/0"))
    penalties_scored_2, penalties_tried_2 = extract_penalties_and_dfh(game.get("penalties_2", "0/0"))
    dfh_scored_1, dfh_tried_1 = extract_penalties_and_dfh(game.get("direct_free_hits_1", "0/0"))
    dfh_scored_2, dfh_tried_2 = extract_penalties_and_dfh(game.get("direct_free_hits_2", "0/0"))
            
    # Update penalties and direct free hits data
    team_stats[team1]["penalties_scored"] += penalties_scored_1
    team_stats[team1]["penalties_tried"] += penalties_tried_1
    team_stats[team2]["penalties_scored"] += penalties_scored_2
    team_stats[team2]["penalties_tried"] += penalties_tried_2
    
    team_stats[team1]["penalties_conceded"] += penalties_scored_2
    team_stats[team1]["penalties_tried_against"] += penalties_tried_2
    team_stats[team2]["penalties_conceded"] += penalties_scored_1
    team_stats[team2]["penalties_tried_against"] += penalties_tried_1

    team_stats[team1]["dfh_scored"] += dfh_scored_1
    team_stats[team1]["dfh_tried"] += dfh_tried_1
    team_stats[team2]["dfh_scored"] += dfh_scored_2
    team_stats[team2]["dfh_tried"] += dfh_tried_2
    
    team_stats[team1]["dfh_conceded"] += dfh_scored_2
    team_stats[team1]["dfh_tried_against"] += dfh_tried_2
    team_stats[team2]["dfh_conceded"] += dfh_scored_1
    team_stats[team2]["dfh_tried_against"] += dfh_tried_1
    
    # Update fouls and cards
    team_stats[team1]["fouls_favor"] += int(game["fouls_2"])
    team_stats[team1]["fouls_against"] += int(game["fouls_1"])
    team_stats[team1]["cards_favor"] += int(game["blue_cards_2"])
    team_stats[team1]["cards_against"] += int(game["blue_cards_1"])
    
    team_stats[team2]["fouls_favor"] += int(game["fouls_1"])
    team_stats[team2]["fouls_against"] += int(game["fouls_2"])
    team_stats[team2]["cards_favor"] += int(game["blue_cards_1"])
    team_stats[team2]["cards_against"] += int(game["blue_cards_2"])

    # Update wins, losses, and draws
    if goals_1 > goals_2:
        team_stats[team1]["wins"] += 1
        team_stats[team2]["losses"] += 1
    elif goals_1 < goals_2:
        team_stats[team1]["losses"] += 1
        team_stats[team2]["wins"] += 1
    else:
        team_stats[team1]["draws"] += 1
        team_stats[team2]["draws"] += 1



def update_player_stats(player_stats, game):
    
    for key, value in game.items():
        if key.startswith("pl") and "_name" in key:
            # Extract team and player index
            parts = key.split("_")
            player_index = parts[0][2:]  # Extract the "1" from "pl1"
            team_number = parts[1]  # "1" or "2"
            team_key = game[f"team{team_number}"]  # Directly fetch the team name from the game dictionary
           
            player_name = value.strip()
                    
            # Update stats for each player
            if player_name:
                goals_key = f"pl{player_index}_{parts[1]}_goals"
                goals = game.get(goals_key, 0)
                if goals.isdigit():
                    player_stats[team_key][player_name]["goals_scored"] += int(goals)
                else:
                    player_stats[team_key][player_name]["goals_scored"] += 0  


def update_ref_stats(ref_stats, game):
    
    for key, value in game.items():
        if key.startswith("ref1") or key.startswith("ref2"):

            ref_name = value.strip()
                    
            # Update stats for each ref
            if ref_name:
                fouls_1_key = "fouls_1"
                fouls_2_key = "fouls_2"
                cards_1_key = "blue_cards_1"
                cards_2_key = "blue_cards_2"
                
                # Update fouls and cards for the referee
                ref_stats[ref_name]["total_fouls"] += int(game.get(fouls_1_key, 0)) + int(game.get(fouls_2_key, 0))
                ref_stats[ref_name]["total_cards"] += int(game.get(cards_1_key, 0)) + int(game.get(cards_2_key, 0))
                
                # Increment the number of games the referee has officiated
                ref_stats[ref_name]["games_officiated"] += 1
                
#%%

''' AUXILIAR FUNCTIONS FOR FEATURE GENERATION '''

#%%
def calculate_head_to_head(game_data, team1, team2, current_date):
    # Filter matches where team1 played against team2 before the current date
    recent_matches = [
        g for g in game_data
        if ((g["team1"] == team1 and g["team2"] == team2) or (g["team1"] == team2 and g["team2"] == team1))
        and g["date"] < current_date
    ]

    team1_goals = 0
    team2_goals = 0
    match_count = len(recent_matches)

    # Accumulate goals for head-to-head matches
    for match in recent_matches:
        if match["team1"] == team1:
            team1_goals += int(match["goals_1"])
            team2_goals += int(match["goals_2"])
        elif match["team1"] == team2:
            team1_goals += int(match["goals_2"])
            team2_goals += int(match["goals_1"])

    # Compute averages (handle case where there are no matches)
    avg_team1_goals = team1_goals / match_count if match_count > 0 else 0
    avg_team2_goals = team2_goals / match_count if match_count > 0 else 0

    return {
        "avg_team1_goals": avg_team1_goals,
        "avg_team2_goals": avg_team2_goals,
        "recent_match_count": match_count
    }

#%%
def calculate_strength_of_schedule(game_data, team, current_date):
    past_games = [
        g for g in game_data
        if (g["team1"] == team or g["team2"] == team)
        and g["date"] < current_date
    ]

    if not past_games:
        return {"sos_avg_points": 0, "sos_avg_goal_difference": 0}

    sos_avg_points = []
    sos_avg_goal_difference = []

    for game in past_games:
        opponent = game["team2"] if game["team1"] == team else game["team1"]
        opponent_strength = calculate_opponent_strength(game_data, opponent, current_date)
        sos_avg_points.append(opponent_strength["avg_points"])
        sos_avg_goal_difference.append(opponent_strength["avg_goal_difference"])

    avg_sos_points = sum(sos_avg_points) / len(sos_avg_points)
    avg_sos_goal_difference = sum(sos_avg_goal_difference) / len(sos_avg_goal_difference)

    return {
        "sos_avg_points": avg_sos_points,
        "sos_avg_goal_difference": avg_sos_goal_difference
    }

#%%
def calculate_opponent_strength(game_data, team, current_date):
    past_games = [
        g for g in game_data
        if (g["team1"] == team or g["team2"] == team)
        and g["date"] < current_date and g not in game_data[-7:]
    ]

    if not past_games:
        return {"avg_points": 0, "avg_goal_difference": 0}

    total_points = 0
    total_goal_difference = 0
    total_games = len(past_games)

    for game in past_games:
        if game["team1"] == team:
            goals_scored = int(game["goals_1"])
            goals_allowed = int(game["goals_2"])
        else:
            goals_scored = int(game["goals_2"])
            goals_allowed = int(game["goals_1"])

        if goals_scored > goals_allowed:
            total_points += 3
        elif goals_scored == goals_allowed:
            total_points += 1

        total_goal_difference += (goals_scored - goals_allowed)

    avg_points = total_points / total_games
    avg_goal_difference = total_goal_difference / total_games

    return {"avg_points": avg_points, "avg_goal_difference": avg_goal_difference}

#%%
def calculate_rest(game_data, team, current_date):
    
    # Filter games where the team participated and occurred before the current date
    recent_games = [
        g for g in game_data
        if (g["team1"] == team or g["team2"] == team) and 
           g["date"] < current_date
    ]
    
    # Sort the games by date in ascending order
    recent_games.sort(key=lambda g: g["date"])
    
    # Ensure there is at least one prior game to calculate rest
    if recent_games:
        # Get the date of the most recent prior game
        last_game_date = recent_games[-1]["date"]
        
        # Calculate and return the difference in days
        return (current_date - last_game_date).days
    
    # Return None if there are no prior games
    return 0

#%%
def extract_penalties_and_dfh(value):
    # This function extracts the scored and attempted values from a string like "0/2"
    try:
        scored, attempted = value.split("/")  # Split the value by "/"
        scored = int(scored)  # Convert scored to integer
        attempted = int(attempted)  # Convert attempted to integer
        return scored, attempted
    except ValueError:
        # If there's an error in parsing, return 0 scored and 0 attempted
        return 0, 0

#%%
def calculate_scoring_concentration(team, player_stats):
    # Calculates the scoring concentration for a team based on goals scored by individual players
    if team not in player_stats or len(player_stats[team]) == 0:
        return 0  # Return 0 if no player data exists for the team
    
    # Calculate the standard deviation of goals scored by players
    goals_scored = [stats["goals_scored"] for stats in player_stats[team].values()]
    
    if team == "CANDELARIA SC 2425":
        goals_scored = goals_scored[:8]  # Limit the list to the first 8 elements
    else:
        goals_scored = goals_scored[:9]  # Limit the list to the first 9 elements
    
    if sum(goals_scored) == 1:
        return 0  # No variation in scoring if only one player scores
        
    mean_goals = sum(goals_scored) / len(goals_scored)
    variance = sum((x - mean_goals) ** 2 for x in goals_scored) / len(goals_scored)
    
    return math.sqrt(variance) 


#%%

''' TO RUN '''

#%%       
csv_file_path = "final_game_data.csv"

try:
    all_game_data = []
    if os.path.exists(csv_file_path):
        # If the file exists, load the data from it
        print(f"Loading game data from {csv_file_path}")
        with open(csv_file_path, "r") as csv_file:
            reader = csv.DictReader(csv_file)
            
            # Loop through each row in the CSV and convert the 'date' column to datetime
            for row in reader:
                if 'date' in row:  
                    row['date'] = datetime.strptime(row['date'], "%Y-%m-%d %H:%M:%S")
                all_game_data.append(row)
            
            # Adjust the scraped data
            all_game_data_adjusted = adjust_game_data(all_game_data)
    
    else:
        # If the file doesn't exist, scrape the data
        print("Scraping game data...")
        for start_id, end_id in id_ranges:
            for game_id in range(start_id, end_id + 1):
                if game_id in excluded_id_ranges:
                    continue
                game_url = f"https://fpp.assyssoftware.es/intranet/web/partido.asp?id={game_id}"
                try:
                    all_game_data.append(scrape_game_data(game_url))
                except Exception as e:
                    print(f"Error scraping data for game with id {game_id}: {e}")
                    continue  # Continue to the next game_id even if the current one fails

        for start_id2, end_id2 in next_games_ids:
            for game_id2 in range(start_id2, end_id2 + 1):
                if game_id2 in excluded_id_ranges:
                    continue
                game_url = f"https://fpp.assyssoftware.es/intranet/web/partido.asp?id={game_id2}"
                try:
                    all_game_data.append(scrape_next_game_data(game_url))
                except Exception as e:
                    print(f"Error scraping data for next game with id {game_id2}: {e}")
                    continue

        # Adjust the scraped data
        all_game_data_adjusted = adjust_game_data(all_game_data)
        
        # Save the adjusted game data to the CSV file
        with open(csv_file_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=all_game_data_adjusted[0].keys())
            writer.writeheader()
            writer.writerows(all_game_data_adjusted)

except Exception as e:
    print(f"Error scraping data: {e}")

#%% 
try:
    # Generate features from the loaded or scraped data
    all_game_features = generate_features(all_game_data_adjusted)
except Exception as e:
    print(f"Error extracting features: {e}")
    all_game_features = [] 

#%%
if all_game_features:  
    with open("final_game_features.csv", "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames = all_game_features[0].keys())
        writer.writeheader() 
        writer.writerows(all_game_features)
 
#%%
end_time = time.time()
runtime = end_time - start_time
print(f"Runtime: {runtime} seconds")