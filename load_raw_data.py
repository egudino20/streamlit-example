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