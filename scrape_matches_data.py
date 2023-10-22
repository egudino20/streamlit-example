# data manipulation and analysis
import pandas as pd
import numpy as np

# data visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch, FontManager
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgba
from matplotlib.patches import ConnectionPatch
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.image as mpimg
import matplotlib.ticker as ticker
import matplotlib.patheffects as path_effects

# web scraping
from selenium import webdriver

# text and annotation
from itertools import combinations
from highlight_text import fig_text, ax_text
from highlight_text import fig_text, ax_text, HighlightText

# machine learning / statiscal analysis
from scipy import stats
from scipy.stats import poisson
from scipy.interpolate import interp1d, make_interp_spline
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp2d
import pickle

# embedded packages
import main
from main import insert_ball_carries
import visuals
from visuals import progressive_pass, progressive_carry, pass_into_box, carry_into_box

# file handling and parsing
import os
import json

# other
import datetime
from datetime import datetime

league_urls = main.getLeagueUrls()

# pick league you want to pull
competition = 'LaLiga'
season = '2023/2024'

# get match urls for that competition and season
match_urls = main.getMatchUrls(comp_urls=league_urls, competition=competition, season=season)

matches_data = main.getMatchesData(match_urls=match_urls[0:10])

# file path and title positioning conditions
if matches_data[0]['region'] == 'Argentina':
    comp = 'liga-profesional'
elif matches_data[0]['region'] == 'Spain':
    comp = 'la-liga'
elif matches_data[0]['region'] == 'England':
    comp = 'premier-league'
elif matches_data[0]['region'] == 'France':
    comp = 'ligue-1'
elif matches_data[0]['region'] == 'Germany':
    comp = 'bundesliga'
else:
    comp = 'serie-a'

# Convert the JSON data to a string
json_string = json.dumps(matches_data)

json_date = matches_data[0]['startDate']
# Convert to a datetime object
json_date = datetime.strptime(json_date, '%Y-%m-%dT%H:%M:%S')
# Format the datetime object as '%m-%d-%y'
json_date = json_date.strftime('%m-%d-%y')

# save the matches_data as a json file
filename = f"matches_data-{json_date}.json"
filepath = os.path.join("Data", f"{comp}", f"{season[5:]}", "match-data", filename)

with open(filepath, "w") as f:
    f.write(json_string)