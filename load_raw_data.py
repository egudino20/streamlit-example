# data manipulation and analysis
import pandas as pd
import numpy as np

# data visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch, FontManager
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgba, Normalize
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.image as mpimg
import matplotlib.ticker as ticker
import matplotlib.patheffects as path_effects
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors

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
from visuals import progressive_pass, progressive_carry, pass_into_box, carry_into_box, team_performance
from visuals import calc_xg, calc_xa, data_preparation, data_preparation_xA, xThreat
from visuals import process_and_export_match_data, load_individual_match_team_dfs, generate_match_week_zone_control_viz, generate_team_zone_control_viz
from visuals import generate_all_teams_zone_control, process_and_export_season_match_data, load_season_match_team_dfs

# file handling and parsing
import os
import json
#import ijson
import glob

# other
import datetime

# Images
from PIL import Image
import requests
from io import BytesIO

# Load Data
import os
import json

# pick season we are loading data for
league_folder = 'premier-league'
season = 2024

def load_matches_data(league_folder, season):

    # Define the folder where your JSON files are located
    folder_path = os.path.join("Data", f"{league_folder}", f"{season}", "match-data")

    # Initialize an empty list to store the concatenated data
    matches_data = []

    # Iterate through files in the folder
    for file_name in os.listdir(folder_path):
        if "matches_data-" in file_name and file_name.endswith(".json"):
            # Construct the full path to the JSON file
            file_path = os.path.join(folder_path, file_name)

            # Open and load the JSON data
            with open(file_path, "r") as f:
                data = json.load(f)

            # Extend the concatenated_data list with the data from the current file
            matches_data.extend(data)

    # Define the output file path for the concatenated data
    # output_file_path = os.path.join("Data", "premier-league", "2024", "match-data", "matches_data.json")

    return matches_data

matches_data = load_matches_data(league_folder, season)

def load_events_df(matches_data):

    # league identifiers
    league = f"{matches_data[0]['region']} {matches_data[0]['league']} {matches_data[0]['season']}"

    season = matches_data[0]['season']

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

    # Define the directory and pattern to search for files
    directory = f'Data/{comp}/{season[5:]}/raw-season-data/'
    pattern = f'{comp}-*.csv'

    # Use glob to find all matching files
    file_list = glob.glob(directory + pattern)

    # Initialize an empty list to store DataFrames
    dfs = []

    # Loop through the list of matching files and read them into DataFrames
    for file in file_list:
        df = pd.read_csv(file, low_memory=False)
        # Drop the 'Unnamed: 0' column if it exists
        if 'Unnamed: 0' in df.columns:
            df.drop('Unnamed: 0', axis=1, inplace=True)
        dfs.append(df)

    # Concatenate all DataFrames into a single DataFrame
    events_df = pd.concat(dfs, ignore_index=True)

    return events_df

    # Now you have events_df containing data from all matching files


# export and process match_data through current date; save process and exported dfs to dfs
dfs = process_and_export_match_data(events_df, matches_data, comp, season)