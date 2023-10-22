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

import os
import json
import pandas as pd

# pick season we are loading data for
league_folder = 'la-liga'
season = 2024

# Specify the path to the folder containing the matches data files
folder_path = os.path.join("Data", f"{league_folder}", f"{season}", "match-data")


# Get a list of all the JSON files in the folder
json_files = [file for file in os.listdir(folder_path) if file.endswith(".json")]

# Extract the dates from the file names
dates = [file.split("-")[1].split(".")[0] for file in json_files]

# Display the available dates to the user
print("Available dates:")
for i, date in enumerate(dates):
    print(f"{i+1}. {date}")

# Ask the user to choose a date
choice = input("Enter the number corresponding to the desired date: ")

# Validate the user's choice
while not choice.isdigit() or int(choice) < 1 or int(choice) > len(dates):
    print("Invalid choice. Please enter a valid number.")
    choice = input("Enter the number corresponding to the desired date: ")

# Get the chosen date and corresponding JSON file
chosen_date = dates[int(choice) - 1]
chosen_file = json_files[int(choice) - 1]

# Construct the full path to the chosen JSON file
json_file_path = os.path.join(folder_path, chosen_file)

# Load the JSON data from the chosen file
with open(json_file_path, "r") as f:
    matches_data = json.load(f)

events_ls = [main.createEventsDF(match) for match in matches_data]
# Add EPV column
events_list = [main.addEpvToDataFrame(match) for match in events_ls]
events_dfs = pd.concat(events_list)

# Print the length of matches_data and unique match IDs
print("Length of matches_data:", len(matches_data))
print("Number of unique match IDs:", len(events_dfs.matchId.unique()))