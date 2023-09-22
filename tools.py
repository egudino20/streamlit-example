# data manipulation
import pandas as pd
import numpy as np

# machine learning / statistical analysis
from scipy.interpolate import interp2d

def xThreat(events_df, interpolate=True, pitch_length=100, pitch_width=100):
    """ Add expected threat metric to whoscored-style events dataframe
    Function to apply Karun Singh's expected threat model to all successful pass and carry events within a
    whoscored-style events dataframe. This imposes a 12x8 grid of expected threat values on a standard pitch. An
    interpolate parameter can be passed to impose a continous set of expected threat values on the pitch.
    Args:
        events_df (pandas.DataFrame): whoscored-style dataframe of event data. Events can be from multiple matches.
        interpolate (bool, optional): selection of whether to impose a continous set of xT values. True by default.
        pitch_length (float, optional): extent of pitch x coordinate (based on event data). 100 by default.
        pitch_width (float, optional): extent of pitch y coordinate (based on event data). 100 by default.
    Returns:
        pandas.DataFrame: whoscored-style dataframe of events, including expected threat
    """

    # Define function to get cell in which an x, y value falls
    def get_cell_indexes(x_series, y_series, cell_cnt_l, cell_cnt_w, field_length, field_width):
        xi = x_series.divide(field_length).multiply(cell_cnt_l)
        yj = y_series.divide(field_width).multiply(cell_cnt_w)
        xi = xi.astype('int64').clip(0, cell_cnt_l - 1)
        yj = yj.astype('int64').clip(0, cell_cnt_w - 1)
        return xi, yj

    # Initialise output
    events_out = pd.DataFrame()

    # Get Karun Singh expected threat grid
    path = "https://karun.in/blog/data/open_xt_12x8_v1.json"
    xt_grid = pd.read_json(path)
    init_cell_count_w, init_cell_count_l = xt_grid.shape

    # Isolate actions that involve successfully moving the ball (successful carries and passes)
    move_actions = events_df[(events_df['type'].isin(['Carry', 'Pass'])) &
                             (events_df['outcomeType'] == 'Successful')]

    # Set-up bilinear interpolator if user chooses to
    if interpolate:
        cell_length = pitch_length / init_cell_count_l
        cell_width = pitch_width / init_cell_count_w
        x = np.arange(0.0, pitch_length, cell_length) + 0.5 * cell_length
        y = np.arange(0.0, pitch_width, cell_width) + 0.5 * cell_width
        interpolator = interp2d(x=x, y=y, z=xt_grid.values, kind='linear', bounds_error=False)
        interp_cell_count_l = int(pitch_length * 10)
        interp_cell_count_w = int(pitch_width * 10)
        xs = np.linspace(0, pitch_length, interp_cell_count_l)
        ys = np.linspace(0, pitch_width, interp_cell_count_w)
        grid = interpolator(xs, ys)
    else:
        grid = xt_grid.values

    # Set cell counts based on use of interpolator
    if interpolate:
        cell_count_l = interp_cell_count_l
        cell_count_w = interp_cell_count_w
    else:
        cell_count_l = init_cell_count_l
        cell_count_w = init_cell_count_w

    # For each match, apply expected threat grid (we go by match to avoid issues with identical event indicies)
    for match_id in move_actions['matchId'].unique():
        match_move_actions = move_actions[move_actions['matchId'] == match_id]

        # Get cell indices of start location of event
        startxc, startyc = get_cell_indexes(match_move_actions['x'], match_move_actions['y'], cell_count_l,
                                            cell_count_w, pitch_length, pitch_width)
        endxc, endyc = get_cell_indexes(match_move_actions['endX'], match_move_actions['endY'], cell_count_l,
                                        cell_count_w, pitch_length, pitch_width)

        # Calculate xt at start and end of events
        xt_start = grid[startyc.rsub(cell_count_w - 1), startxc]
        xt_end = grid[endyc.rsub(cell_count_w - 1), endxc]

        # Build dataframe of event index and net xt
        ratings = pd.DataFrame(data=xt_end-xt_start, index=match_move_actions.index, columns=['xT'])

        # Merge ratings dataframe to all match events
        match_events_and_ratings = pd.merge(left=events_df[events_df['matchId'] == match_id], right=ratings,
                                            how="left", left_index=True, right_index=True)
        events_out = pd.concat([events_out, match_events_and_ratings], ignore_index=True, sort=False)
        events_out['xT_gen'] = events_out['xT'].apply(lambda xt: xt if (xt > 0 or xt != xt) else 0)

    return events_out