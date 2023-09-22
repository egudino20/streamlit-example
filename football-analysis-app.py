# Import your modules here
import streamlit as st
import os
import json
import pandas as pd
import numpy as np

# data visualization
import matplotlib
import requests
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgba
from matplotlib.patches import ConnectionPatch
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.image as mpimg
import matplotlib.ticker as ticker
import matplotlib.patheffects as path_effects
#from mplsoccer import PyPizza, add_image, FontManager
from matplotlib.colors import Normalize
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
from mplsoccer import VerticalPitch

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
from tools import xThreat

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

    # Now you have events_df containing data from all matching files

def team_performance(team, comp, season, league, metric_1, metric_2, web_app=False):

    # Assuming the "Roboto" font is installed on your system, you can specify it as the default font family.
    plt.rcParams['font.family'] = 'Roboto'

    title_font = 'Roboto'
    body_font = 'Roboto'

    if web_app == True:

        df = pd.read_csv(f'Data/{comp}/{season}/match-logs/{team}-match-logs.csv', index_col=0)

    else:

        df = pd.read_csv(f'Data/{comp}/{season}/match-logs/{team}-match-logs.csv', index_col=0)
        
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

def process_and_export_match_data(events_df, matches_data, comp, season):

    teamNames = []

    for match in matches_data:

        teamHomeName = match['home']['name']
        teamAwayName = match['away']['name']

        teamNames.append(teamHomeName)
        teamNames.append(teamAwayName)
        
    teamIds = []
        
    for match in matches_data:

        teamHomeId = match['home']['teamId']
        teamAwayId = match['away']['teamId']

        teamIds.append(teamHomeId)
        teamIds.append(teamAwayId)
        
    teams = pd.DataFrame({'teamId': teamIds,
                          'teamName': teamNames})

    teams = teams.drop_duplicates().reset_index(drop=True)

    passes = events_df[events_df['type'] == 'Pass']
    passes = xThreat(passes)  # Assuming xThreat is a function you have defined
    
    carries = events_df[events_df['type'] == 'Carry']
    carries = xThreat(carries)  # Assuming xThreat is a function you have defined
    
    df = pd.concat([passes, carries], axis=0)
    df = pd.merge(df, teams, on='teamId', how='left')
    
    teamNames = teams['teamName']
    
    matchIds = list(df.matchId.unique())
    
    df['startDate'] = pd.to_datetime(df['startDate'])
    df['startDate'] = df['startDate'].dt.date
    
    dfs = []
    
    for matchId in matchIds:
        team_df = df[df['matchId'] == matchId]
        home = team_df.teamName.unique()[0]
        away = team_df.teamName.unique()[1]
        date = team_df.startDate.unique()[0]
        team_df.to_csv(f'Data/{comp}/{season[5:]}/raw-season-data/{home}-{away}-{date}-passes-carries.csv')
        dfs.append(df)
    
    return dfs

def load_individual_match_team_dfs(comp, season, pass_filter='passes-carries'):

    # Enter a team for pass_filter

    path_to_folder = f'Data/{comp}/{season[5:]}/raw-season-data/'
    team_csv_files = glob.glob(os.path.join(path_to_folder, f'*{pass_filter}*.csv'))
    team_dataframes = []
    
    for csv_file in team_csv_files:
        match_data = pd.read_csv(csv_file)
        match_data = match_data.drop(match_data.columns[match_data.columns.str.contains('Unnamed', case=False)], axis=1)
        team_dataframes.append(match_data)
    
    return team_dataframes

def generate_match_week_zone_control_viz(team_dataframes, match_week, league, comp, season, off_week=True):

    # todays date for saving
    output = f'{comp}-{match_week}'

    final_dfs = []

    for match in team_dataframes:   
        
        match = match[match['outcomeType'] == 'Successful']
        match['isOpenPlay'] = np.where((match['passFreekick'] == False) &
                                      ((match['passCorner'] == False)
                                                    ) 
                                                   , 1, 0
                                                   )
        match = match[match['isOpenPlay'] == 1]
        
        match = match[['playerName', 'teamName', 'teamId', 'matchId', 'startDate', 'type', 'x', 'y', 'endX', 'endY', 'xT', 'h_a']]

        match['xT'] = match['xT'].fillna(0)

        # Create a boolean mask to identify rows where "teamName" is not equal to the home team
        home_team_df = match[match['h_a'] == 'h']
        home_team = home_team_df.teamName.unique()[0]
        mask = match['teamName'] != home_team

        # Multiply the values in "xT" column by -1 where the mask is True
        match.loc[mask, 'xT'] *= -1

        # Flip the "x" and "y" coordinates by subtracting them from 100 where the mask is True
        match.loc[mask, ['x', 'y', 'endX', 'endY']] = 100 - match.loc[mask, ['x', 'y', 'endX', 'endY']]
        
        final_dfs.append(match)
        
    final_df = pd.concat(final_dfs, axis=0)

    final_df = final_df.sort_values(by='startDate')

    if comp == 'la-liga':
        
        # Replace values in 'column_name' based on condition
        team_abbreviations =  {'Deportivo Alaves': 'ALV',
                               'Almeria': 'ALM',
                               'Athletic Bilbao': 'ATB',
                               'Atletico': 'ATM',
                               'Barcelona': 'FCB',
                               'Cadiz': 'CCF',
                               'Celta Vigo': 'CLV',
                               'Getafe': 'GET',
                               'Girona': 'GIR',
                               'Granada': 'GCF',
                               'Las Palmas': 'LAP',
                               'Osasuna': 'OSA',
                               'Rayo Vallecano': 'RAY',
                               'Mallorca': 'RCD',
                               'Real Betis': 'BET',
                               'Real Madrid': 'MAD',
                               'Real Sociedad': 'SOC',
                               'Sevilla': 'SEV',
                               'Valencia': 'VAL',
                               'Villarreal': 'VIL'}

    else:
        
        # Replace values in 'column_name' based on condition
        team_abbreviations =  {'Arsenal': 'ARS',
                               'Aston Villa': 'AVL',
                               'Brentford': 'BRE',
                               'Brighton': 'BHA',
                               'Luton': 'LUT',
                               'Chelsea': 'CHE',
                               'Fulham': 'FUL',
                               'Everton': 'EVE',
                               'Sheff Utd': 'SHU',
                               'Burnley': 'BUR',
                               'Liverpool': 'LIV',
                               'Man City': 'MCI',
                               'Man Utd': 'MUN',
                               'Newcastle': 'NEW',
                               'Tottenham': 'TOT',
                               'West Ham': 'WHU',
                               'Wolves': 'WOL',
                               'Nottingham Forest': 'NFO',
                               'Crystal Palace': 'CRY',
                               'Bournemouth': 'BOU'}

    # Add a new column 'team_abbreviation' using the mapping
    final_df['team_abbreviation'] = final_df['teamName'].map(team_abbreviations)

    # Assuming the "Roboto" font is installed on your system, you can specify it as the default font family.
    plt.rcParams['font.family'] = 'Roboto'

    title_font = 'Roboto'
    body_font = 'Roboto'

    # Define the custom colormap with colors for negative, zero, and positive values
    negative_color = '#ff4500'   # Red for negative values
    zero_color = '#1d2849'        # Dark blue for zero values
    positive_color = '#39ff14'    # Green for positive values

    colors = [(negative_color), (zero_color), (positive_color)]
    n_bins = 100  # Number of color bins
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=n_bins)

    # List of matchIds
    match_ids_list = list(final_df.matchId.unique())

    # Define the number of matches per gameweek (adjust as needed)
    matches_per_gameweek = 10

    # Calculate the index range for the desired gameweek
    start_index = (match_week - 1) * matches_per_gameweek

    if off_week == False:
        start_index = start_index - 1
    else:
        pass  # Do nothing
        

    end_index = (match_week * matches_per_gameweek)

    # Adjust the match_ids_list using the calculated index range
    match_ids_list = match_ids_list[start_index:end_index]

    # Setup the pitch
    pitch = VerticalPitch(pitch_type='opta', pitch_color='#1d2849', line_color='w', line_zorder=5,
                          half=False, pad_top=2, axis=False, 
                          positional=True, positional_color='#eadddd', positional_zorder=5)

    # Create the subplot grid using mplsoccer
    fig, axs = pitch.grid(nrows=3, ncols=4, figheight=30,
                          endnote_height=0.01, endnote_space=0.01,
                          axis=False, space=0.1,
                          title_height=0.04, grid_height=0.84)
    fig.set_facecolor('#1d2849')

    # Set the title
    title = f"Match Week {match_week} - Zone Control"
    fig.text(0.0315,1.02,title, fontsize=40,
             fontfamily=body_font, fontweight='bold', color='w')

    if comp == 'la-liga':
        x_coor = 0.11
    else:
        x_coor = 0.165

    ax_text(x_coor, 96, f"{league}", fontsize=28, ha='center',
            fontfamily=body_font, fontweight='bold', color='w')

    ax_text(0, 94, f"<Green> shaded zones represent zones controlled by the home team", 
            fontsize=24,
            fontfamily=body_font, fontweight='bold', color='w',
            highlight_textprops=[{'color': positive_color}])

    ax_text(0, 92.5, "<Red> shaded zones represent zones controlled by the away team", 
            fontsize=24,
            fontfamily=body_font, fontweight='bold', color='w',
            highlight_textprops=[{'color': negative_color}])

    ax_text(0, 91, "Blue shaded zones represent neutral zones, not controlled by the home team or the away team", 
            fontsize=24,
            fontfamily=body_font, fontweight='bold', color='w')

    # Set the footnote
    footnote = "Zone Control is the difference of expected threat (xT) generated (+) and conceded (-)\nby the home team in each zone based on the start location of open play passes and carries."
    footnote2 = "Expected threat model by Karun Singh."
    footnote3 = 'Data via Opta | Created by @egudi_analysis'
    ax_text(0.73, 5.8, f"{footnote}\n{footnote2} {footnote3}", fontsize=17, ha='center',
             fontfamily=body_font, fontweight='bold', color='w')

    # Calculate the title height for positioning the logos
    title_height = 1  # Adjust as needed

    # Cycle through the grid axes and plot the heatmaps for each match
    for idx, ax in enumerate(axs['pitch'].flat):
        if idx < len(match_ids_list):
            match_id = match_ids_list[idx]
            match_test = final_df[final_df['matchId'] == match_id]
            home_team_df = match_test[match_test['h_a'] == 'h']
            home_team = home_team_df.teamName.unique()[0]
            away_team_df = match_test[match_test['h_a'] == 'a']
            away_team = away_team_df.teamName.unique()[0]
            home_abrev = home_team_df.team_abbreviation.unique()[0]
            away_abrev = away_team_df.team_abbreviation.unique()[0]
            
            # Calculate the sum total of 'xT' in each bin
            bin_statistic = pitch.bin_statistic_positional(match_test.x, match_test.y, match_test.xT, statistic='sum', positional='full',
                                                           normalize=True)
           
            # Use the colormap to create the heatmap using mplsoccer
            pitch.heatmap_positional(bin_statistic, ax=ax, edgecolors='#1a223d', cmap=cmap, vmin=-1, vmax=1)
            
            #labels = pitch.label_heatmap(bin_statistic, color='#f4edf0', fontsize=18,
            #                             ax=ax, ha='center', va='center',
            #                             str_format='{:.0}')
            
            ax_text(75,107, f"<{home_abrev}> vs <{away_abrev}>", color='w', fontsize=26,
                    fontfamily=body_font, fontweight='bold', ax=ax,
                    highlight_textprops=[{'color': positive_color},
                                         {'color': negative_color}])
            
            # Load logo images
            logo_paths = f'Logos/{comp}/{away_team}.png'
            team_logo_path = f'Logos/{comp}/{home_team}.png'
            logo = plt.imread(logo_paths)
            team_logo = plt.imread(team_logo_path)

            # Position the logo next to the title
            logo_ax = fig.add_axes([ax.get_position().x0 + 0.17, ax.get_position().y1, 0.03, 0.03])
            logo_ax.imshow(logo)
            logo_ax.axis('off')
            
            # Position the logo next to the title
            logo_ax = fig.add_axes([ax.get_position().x0+0.0025, ax.get_position().y1, 0.03, 0.03])
            logo_ax.imshow(team_logo)
            logo_ax.axis('off')
            
            ax.axis('off')  # Turn off axis for a clean visualization
            
    # Now, remove the last few plots using a separate loop
    for ax in axs['pitch'].flat[len(match_ids_list):]:
        ax.remove()
        
    # league logo
    path = f'Logos/{comp}/{comp}.png'
    ax_team = fig.add_axes([0.88,0.94,0.105,0.105])
    ax_team.axis('off')
    im = plt.imread(path)
    ax_team.imshow(im);

def main():
    print("Current working directory:", os.getcwd())
    st.title('Football Analysis')

    # Add sidebar for league
    country = st.sidebar.selectbox('Choose a league', ['England', 'Argentina', 'Spain', 'France', 'Germany', 'Italy'])

    # Add sidebar for season
    season = st.sidebar.selectbox('Choose a season', ['2023', '2024'])
    # Check if the input is valid
    if season:
        try:
            # Parse the input season as an integer
            input_season = int(season)
            
            # Calculate the prev_season
            prev_season = input_season - 1

        except ValueError:
            st.sidebar.error("Please enter a valid season (e.g., 2023)")
    else:
        st.sidebar.info("Please enter a season in the sidebar.")

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

    #url = "https://storage.googleapis.com/matches-data/matches_data.json"

    #try:
    #    response = requests.get(url)
    #    if response.status_code == 404:
    #        st.error("Resource not found. Please check the URL.")
    #    else:
    #        matches_data = response.json()
    #except requests.exceptions.RequestException as e:
    #    st.error(f"Connection error: {e}")

    # league identifiers
    #league = f"{matches_data[0]['region']} {matches_data[0]['league']} {matches_data[0]['season']}"

    #season = matches_data[0]['season'] 

    #events_df = pd.read_csv(f'Data/{league_folder}/{season[5:]}/raw-season-data/{league_folder}-{date_str}.csv', low_memory=False)
    #events_df.drop('Unnamed: 0', axis=1, inplace=True)

    options = ['Team Performance', 'Match Week Zone Control']

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
            league = f"{country} Premier League {prev_season}/{season}"
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
        data_path = f'Data/{league_folder}/{season}/match-logs/{team}-match-logs.csv'
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
    
    elif option == 'Match Week Zone Control':
        
        match_weeks = [1, 2, 3, 4, 5]

        if country == 'England':
            premier_league_teams = ['Arsenal', 'Aston Villa', 'Brentford', 'Brighton & Hove Albion', 
                                    'Burnley', 'Chelsea', 'Crystal Palace', 'Everton', 'Leeds United', 
                                    'Leicester City', 'Liverpool', 'Manchester City', 'Manchester United',
                                    'Newcastle United', 'Norwich City', 'Southampton', 'Tottenham Hotspur', 
                                    'Watford', 'West Ham United', 'Wolverhampton Wanderers']        
            team = st.sidebar.selectbox('Select a team', premier_league_teams)
            league = f"{country} Premier League {prev_season}/{season}"
        else:
            team = st.sidebar.text_input('Enter a team name')
        
        match_week = st.sidebar.selectbox('Select match week', match_weeks)

        #load matches data and events data
        matches_data = load_matches_data(league_folder, season)
        events_df = load_events_df(matches_data)
        # process and export events data after loading it
        dfs = process_and_export_match_data(events_df, matches_data, league_folder, season)
        team_dataframes = load_individual_match_team_dfs(league_folder, season)

        # plot the final result
        try:
            result = generate_match_week_zone_control_viz(team_dataframes, match_week, league, league_folder, season, off_week=False)
            st.pyplot(result)  # Display the plot
        except:
            st.write("No Data")

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
