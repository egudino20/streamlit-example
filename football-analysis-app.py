# Import your modules here
import streamlit as st
import os
import json
import pandas as pd
import numpy as np

# data visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
#from mplsoccer import Pitch, VerticalPitch, FontManager
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgba
from matplotlib.patches import ConnectionPatch
import matplotlib.gridspec as gridspec
#import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.image as mpimg
import matplotlib.ticker as ticker
import matplotlib.patheffects as path_effects
#from mplsoccer import PyPizza, add_image, FontManager
from matplotlib.colors import Normalize
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors

# web scraping
from selenium import webdriver
import requests

# text and annotation
from itertools import combinations
from highlight_text import fig_text, ax_text
from highlight_text import fig_text, ax_text, HighlightText
from adjustText import adjust_text

# machine learning / statiscal analysis
from scipy import stats
from scipy.stats import poisson
from scipy.interpolate import interp1d, make_interp_spline
from scipy.interpolate import interp2d
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp2d
import pickle
from math import pi

# file handling and parsing
import os
import json
import glob

# Images
from PIL import Image
import requests
from io import BytesIO

# other
import datetime

def team_performance(team, comp, season, league, metric_1, metric_2, web_app=False):

    # Assuming the "Roboto" font is installed on your system, you can specify it as the default font family.
    plt.rcParams['font.family'] = 'Roboto'

    title_font = 'Roboto'
    body_font = 'Roboto'

    if web_app == True:

        df = pd.read_csv(f'Data/{comp}/{season[5:]}/match-logs/{team}-match-logs.csv', index_col=0)

    else:

        df = pd.read_csv(f'Data/{comp}/{season[5:]}/match-logs/{team}-match-logs.csv', index_col=0)
        
    df['yrAvg'] = df[metric_1].rolling(window=10).mean()
    df['zrAvg'] = df[metric_2].rolling(window=10).mean()

    background = '#1d2849'
    text_color='w'
    text_color_2='gray'
    mpl.rcParams['xtick.color'] = text_color
    mpl.rcParams['ytick.color'] = text_color

    filler = 'grey'
    primary = 'red'

    # create figure and axes
    fig, ax = plt.subplots(figsize=(12,6))
    fig.set_facecolor(background)
    ax.patch.set_facecolor(background)

    # add grid
    # ax.grid(ls='dotted', lw="0.5", color='lightgrey', zorder=1)

    x = df.startDate
    x = range(1,39)
    y = df.yrAvg
    z = df.zrAvg

    ax.plot(x, y, color='#4B9CD3', alpha=0.9, lw=1, zorder=2,
            label=f'Rolling 10 game {metric_1}', marker='o')
    ax.plot(x, z, color='#39ff14', alpha=0.9, lw=1, zorder=3,
           label=f'Rolling 10 game {metric_2}', marker='o')

    # Add a dotted horizontal line at y=0
    ax.axhline(y=0, color='gray', lw=1, linestyle='--', zorder=0)

    # Add a legend to the plot with custom font properties and white text color
    legend_font = fm.FontProperties(family='Roboto', weight='bold')
    legend = plt.legend(prop=legend_font, loc='upper left', frameon=False)
    plt.setp(legend.texts, color='white')  # Set legend text color to white

    # add title and subtitle

    df['startDate'] = df['startDate'].astype(str)
    start_date = df['startDate'].unique()[0]
    end_date = df['startDate'].unique()[37]

    fig.text(0.12,1.115, "{}".format(team), 
             fontsize=20, color=text_color, fontfamily=title_font, fontweight='bold')
    fig.text(0.12,1.065, f"{league}", fontsize=14, 
             color='white', fontfamily=title_font, fontweight='bold')
    fig.text(0.12,1.015, f"All matches, {start_date} to {end_date}", fontsize=14, 
             color='white', fontfamily=title_font, fontweight='bold')

    ax.tick_params(axis='both', length=0)

    spines = ['top', 'right', 'bottom', 'left']
    for s in spines:
        if s in ['top', 'right']:
            ax.spines[s].set_visible(False)
        else:
            ax.spines[s].set_color(text_color)
            
    # Set the x-axis ticks to increment by 2
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=2)) 
    for label in ax.get_xticklabels():
        label.set_fontfamily(body_font)
        label.set_fontproperties(fm.FontProperties(weight='bold'))
    for label in ax.get_yticklabels():
        label.set_fontfamily(body_font)
        label.set_fontproperties(fm.FontProperties(weight='bold'))
            
    ax2 = fig.add_axes([0.05,0.99,0.08,0.08])
    ax2.axis('off')

    path = f'Logos/{comp}/{team}.png'
    ax_team = fig.add_axes([-0.02,0.99,0.175,0.175])
    ax_team.axis('off')
    im = plt.imread(path)
    ax_team.imshow(im);

    fig.text(0.05, -0.025, "Viz by @egudi_analysis | Data via Opta", fontsize=9,
             fontfamily=body_font, fontweight='bold', color=text_color)

    fig.text(0.05, -0.05, "Expected goals model trained on ~10k shots from the 2021/2022 EPL season.", fontsize=9,
             fontfamily=body_font, fontweight='bold', color=text_color)

    fig.text(0.05, -0.075, "Expected threat model by Karun Singh", fontsize=9,
             fontfamily=body_font, fontweight='bold', color=text_color)

    plt.tight_layout()

    return fig

    # save figure
    #fig.savefig(f'{main_folder}/Output/{comp}/{season[5:]}/{team}-{start_date}-{end_date}', dpi=None, bbox_inches="tight")

import pandas as pd

def main():
    print("Current working directory:", os.getcwd())
    st.title('Football Analysis')

    # Add sidebar for league
    country = st.sidebar.selectbox('Choose a league', ['Argentina', 'Spain', 'England', 'France', 'Germany', 'Italy'])

    # Add sidebar for season
    year = st.sidebar.selectbox('Choose a season', ['2023', '2024'])

    # Add sidebar for date
   # date = st.sidebar.date_input('Choose a date')

    # Convert date to string in the format "mm.dd.yyyy"
    #date_str = date.strftime("%m.%d.%Y")

    # Map league to competition
    league_to_comp = {
        'Argentina': 'liga-profesional',
        'Spain': 'la-liga',
        'England': 'premier-league',
        'France': 'ligue-1',
        'Germany': 'bundesliga',
        'Italy': 'serie-a'
    }
    league_folder = league_to_comp[country]

    url = "https://storage.googleapis.com/matches-data/matches_data.json"

    try:
        response = requests.get(url)
        if response.status_code == 404:
            st.error("Resource not found. Please check the URL.")
        else:
            matches_data = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")

    # league identifiers
    league = f"{matches_data[0]['region']} {matches_data[0]['league']} {matches_data[0]['season']}"

    season = matches_data[0]['season'] 

    #events_df = pd.read_csv(f'Data/{league_folder}/{season[5:]}/raw-season-data/{league_folder}-{date_str}.csv', low_memory=False)
    #events_df.drop('Unnamed: 0', axis=1, inplace=True)

    options = ['Team Performance', 'Match Data', 'Individual Match Team Data', 
           'Match Week Zone Control Visualization', 'Team Zone Control Visualization', 
           'All Teams Zone Control', 'Season Match Data', 'Season Match Team Data']

    option = st.sidebar.selectbox('Choose an option', options)

    if option == 'Team Performance':
        metrics = ['xG', 'xGA', 'xGD', 'xT', 'xTA', 'xTD']

        if country == 'England':
            premier_league_teams = ['Arsenal', 'Aston Villa', 'Brentford', 'Brighton & Hove Albion', 
                                    'Burnley', 'Chelsea', 'Crystal Palace', 'Everton', 'Leeds United', 
                                    'Leicester City', 'Liverpool', 'Manchester City', 'Manchester United',
                                    'Newcastle United', 'Norwich City', 'Southampton', 'Tottenham Hotspur', 
                                    'Watford', 'West Ham United', 'Wolverhampton Wanderers']        
            team = st.sidebar.selectbox('Select a team', premier_league_teams)
        else:
            team = st.sidebar.text_input('Enter a team name')
        selected_metric_1 = st.sidebar.selectbox('Select metric 1', metrics)
        selected_metric_2 = st.sidebar.selectbox('Select metric 2', metrics)
        try:
            result = team_performance(team, league_folder, season, league, selected_metric_1, selected_metric_2, web_app=True)
            st.pyplot(result)  # Display the plot
        except:
            st.write("No Data")

        # Load the underlying data
        data_path = f'Data/{league_folder}/{season[5:]}/match-logs/{team}-match-logs.csv'
        data_df = pd.read_csv(data_path)
        data_df = data_df[['teamName', 'opponent', 'matchId', 'startDate'] + metrics]

        # Check if data_df is empty
        if len(data_df) == 0:
            st.write("No Data")
        else:
            # Display the data as a table
            st.write(f"{team} Match Logs")
            st.table(data_df)

            # Add a button to export the data as a CSV file
            csv_export_button = st.download_button(
                label="Export Data as CSV",
                data=data_df.to_csv(index=False),
                file_name="team_performance_data.csv",
                mime="text/csv"
            )
    
    #elif option == 'Match Data':
        # Call your process_and_export_match_data method here
        # result = my_module.process_and_export_match_data()
        # st.write(result)

    #elif option == 'Individual Match Team Data':
        # Call your load_individual_match_team_dfs method here
        # result = my_module.load_individual_match_team_dfs()
        # st.write(result)

    #elif option == 'Match Week Zone Control Visualization':
        # Call your generate_match_week_zone_control_viz method here
        # result = my_module.generate_match_week_zone_control_viz()
        # st.write(result)

    #elif option == 'Team Zone Control Visualization':
        # Call your generate_team_zone_control_viz method here
        # result = my_module.generate_team_zone_control_viz()
        # st.write(result)

    #elif option == 'All Teams Zone Control':
        # Call your generate_all_teams_zone_control method here
        # result = my_module.generate_all_teams_zone_control()
        # st.write(result)

    #elif option == 'Season Match Data':
        # Call your process_and_export_season_match_data method here
        # result = my_module.process_and_export_season_match_data()
        # st.write(result)

    #elif option == 'Season Match Team Data':
        # Call your load_season_match_team_dfs method here
        # result = my_module.load_season_match_team_dfs()
        # st.write(result)

if __name__ == "__main__":
    main()
