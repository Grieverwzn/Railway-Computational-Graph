"""
Title:          Computational Graph-based Railway Demand Estimation Toolkit
Description:    This script contains functions for performing demand estimation in a railway network
                using a computational graph-based approach. The input data includes ticket sale data
                and station information. The output data includes the estimated boarding passengers,
                OD flows, and OD line flows, line section flows, as well as the behavioral parameters
                for the nested logit model. The behavioral parameters include the frequency, travel time,
                and price coefficients for each line, as well as the economic coefficients for each OD pair.

                The model integrates the nested logit model, the hierarchical model, and a linear regression model
                to consider both the induced demand, diverted demand and the ex-post demand constrained by the
                train seat capacity.

                The model is trained using the TensorFlow library. The training process includes four interconnected
                steps:
                1) training the probability of choosing each OD line using the nested logit model,
                2) training the flows in the hierarchical structure with flow conservation,
                3) training the flows od-based linear regression model using the coupling constraints,
                4) integrating the results from the three steps to obtain the final results.
                The coupling constraints and capacity constraints are enforced using the augmented Lagrangian method.

Authors:        Xin (Bruce) Wu  xwu03@villanova.edu, xinwu8592@gmail.com; Villanova University
                Xinyu Wang Beijing Jiaotong University,
                Yang Liu Beijing Jiaotong University,

Date:           2024-09-02
Version:        1.0
License:        MIT License
"""

import os
import pandas as pd
import time
import numpy as np
import tensorflow as tf
from random import sample
import random
from matplotlib import pyplot as plt

pd.set_option('display.max_columns', None)

# global lists
g_od_list = []
g_od_line_list = []
g_line_list = []
g_section_list = []
g_board_station_list = []
g_alight_station_list = []

# global dictionaries
# global dictionary to store the frequency of each line on each day
g_date_line_freq_dict = {}
# global dictionary to store the relationship between station name and station id
g_station_id_dict = {}
g_id_station_dict = {}
g_board_sta_seq_dict = {}
g_alight_sta_seq_dict = {}
g_trainId_to_lineId_dict = {}
g_trainId_to_lineName_dict = {}
g_sectName_to_sectId_dict = {}
g_sectId_to_sectName_dict = {}
g_sectName_to_lineId_dict = {}
g_lineId_to_sectId_dict = {}
g_lineId_to_sectName_dict = {}
g_od_id_dict = {}
g_lineId_to_lineName_dict = {}
g_odLineId_to_odSectName_dict = {}

# Parameters
CASE_NAME = "small scale case"

TRAIN_SAMPLE_RATIO = 0.75
TRAIN_CAPACITY = 1200
PARAMS = {}
UB_BOARDING = 1000  # upper bound of boarding passengers
UB_ECO_COEFF = 500  # upper bound of economic coefficient
LB_THETA = 0.05  # lower bound of theta
UB_LINE_FREQ_COEFF = 10  # upper bound of frequency coefficient
LB_LINE_TT_COEFF = -1  # lower bound of travel time coefficient
LB_LINE_PRICE_COEFF = -1  # lower bound of price coefficient
UB_OD_FREQ_COEFF = 5000  # upper bound of frequency coefficient

INIT_SECT_LAG_MULT = 0.0  # initial value of lagrange multiplier for section capacity conservation
INIT_SECT_LAG_MULT_UPDATING_STEP = 1e-5
INIT_C_SECT_MULT = 100
C_SECT_MULT_UPDATING_STEP = 1.0
# only use c_sect 0.01
# ony use lag_update 1e-6
# use both c_sect = 1e-8and lag_update = 1e-8

INIT_OD_LAG_MULT = 0.0  # initial value of lagrange multiplier for OD flow conservation
INIT_OD_LAG_MULT_UPDATING_STEP = 0.3  # 0.3
INIT_C_OD_MULT = 5
C_OD_MULT_UPDATING_STEP = 1.00
# both lag 0.3, c 5 update 1.05

BATCH_SIZE = 1  # should be less than the total sample number
NON_ZERO = 1e-16
K_PERSON_CONVERTER = 1000
PENALTY_THRESHOLD = 0.15
SECT_PENALTY_THRESHOLD = 0.15
SAMPLE_SIGMA = 0
TOTAL_NB_SAMPLE = 0

# learning rate
lr_nl = 0.02
lr_boarding = 0.2
lr_regression = 0.05
lr_c_od = 1.05  # for updating c_od_multiplier_updating_var
lr_integrating = 0.01

# Weights of each loss function when training probability from the nested logit model
W_PROB_NL = 1
W_PROB_BOARD = 0
W_PROB_ALIGHT = 0
W_PROB_CAP = 0
W_PROB_LR = 0

# Weights of each loss function when training flows in the hierarchical model
W_FLOW_NL = 0
W_FLOW_BOARD = 1
W_FLOW_ALIGHT = 1
W_FLOW_CAP = 1
W_FLOW_LR = 0

# Weights of each loss function when training flows in the hierarchical model
W_COUPLE_NL = 0
W_COUPLE_BOARD = 0
W_COUPLE_ALIGHT = 0
W_COUPLE_CAP = 0
W_COUPLE_LR = 1

# Weights of each loss function when performing integration
W_INTEGRATION_NL = 2
W_INTEGRATION_BOARD = 1
W_INTEGRATION_ALIGHT = 1
W_INTEGRATION_CAP = 1
W_INTEGRATION_LR = 1

# epoch
TOTAL_EPOCHS = 1000  # 1100
MAX_NESTED_LOGIT_STEP = 300  # 300
MAX_OD_FLOW_STEP = 300  # 400
MAX_COUPLING_STEP = 300  # 300
MAX_ITERATION_STEP = 100  # 100
THRESHOLD_MAX_COUPLING_GAP = 0.00  # 0.05
MIN_GRADIENT = 0.00  # 0.01


# ======================Data processing functions======================================
def fetch_line_from_ticket(_ticket_sale_data_df):
    global g_od_list
    global g_line_list
    global g_od_line_list
    global g_date_line_freq_dict
    global g_trainId_to_lineId_dict
    global g_trainId_to_lineName_dict
    global g_lineId_to_lineName_dict  # Xinyu W
    global g_od_id_dict

    # step 1: generate OD of each train
    _ticket_sale_data_df['OD'] = _ticket_sale_data_df['DEP_STATION'] + '-' + _ticket_sale_data_df['ARR_STATION']
    g_od_list = _ticket_sale_data_df['OD'].unique().tolist()
    # get the unique identifier of od using the sequence in g_od_list
    _ticket_sale_data_df['OD_ID'] = \
        _ticket_sale_data_df['OD'].map(dict(zip(g_od_list, range(len(g_od_list))))).astype(int)
    print("___generate", len(g_od_list), "OD pairs...")

    # step 2: generate the line of each train
    train_date_groups = _ticket_sale_data_df.groupby(['TRAINID', 'BOARDDATE'])

    nb_line = 0
    line_name_to_id_dict = {}
    _ticket_with_line_list = []
    for line_id_date_index, single_train_date_group in train_date_groups:
        depart_sta_df = single_train_date_group[['DEP_STATION_ID', 'DEP_STATION', 'DEP_TIME']]
        arrive_sta_df = single_train_date_group[['ARR_STATION_ID', 'ARR_STATION', 'ARR_TIME']]
        # rank the stations from earliest to latest using departure time
        depart_sta_df = depart_sta_df.drop_duplicates()
        depart_sta_df = depart_sta_df.copy()
        depart_sta_df.loc[:, 'DEP_TIME'] = pd.to_datetime(depart_sta_df['DEP_TIME']).dt.strftime('%H:%M')
        depart_sta_df = depart_sta_df.sort_values(by='DEP_TIME')
        # rank the stations from earliest to latest using arrival time
        arrive_sta_df = arrive_sta_df.drop_duplicates()
        arrive_sta_df = arrive_sta_df.copy()
        arrive_sta_df.loc[:, 'ARR_TIME'] = pd.to_datetime(arrive_sta_df['ARR_TIME']).dt.strftime('%H:%M')
        arrive_sta_df = arrive_sta_df.sort_values(by='ARR_TIME')
        line_name_str = depart_sta_df['DEP_STATION'].iloc[0]
        for i in range(0, len(arrive_sta_df)):
            line_name_str += '-' + arrive_sta_df['ARR_STATION'].iloc[i]
        single_train_date_group['LINE'] = line_name_str
        if line_name_str not in g_line_list:
            g_line_list.append(line_name_str)
            print(f"Line {nb_line}: {line_name_str}")
            line_name_to_id_dict[line_name_str] = nb_line
            nb_line += 1
        single_train_date_group['LINE_ID'] = line_name_to_id_dict[line_name_str]
        single_train_date_group['OD_LINE'] = \
            single_train_date_group['OD'] + '-' + single_train_date_group['LINE_ID'].astype(str)
        _ticket_with_line_list.append(single_train_date_group)
    output_ticket_sale_data_df = pd.concat(_ticket_with_line_list)
    g_od_line_list = output_ticket_sale_data_df['OD_LINE'].unique().tolist()
    output_ticket_sale_data_df['OD_LINE_ID'] = \
        output_ticket_sale_data_df['OD_LINE'].map(dict(zip(g_od_line_list, range(len(g_od_line_list))))).astype(int)
    # output_ticket_sale_data_df.to_csv('ticket_sale_data_with_line.csv', index=False)
    print("___generate", len(g_line_list), "lines...")
    print("___generate", len(g_od_line_list), "OD lines...")

    # step 3: generate the frequency of each line
    link_date_groups = output_ticket_sale_data_df.groupby(['LINE_ID', 'BOARDDATE'])
    for line_id_date_index, line_group in link_date_groups:
        frequency = len(line_group['TRAINID'].unique())
        g_date_line_freq_dict[line_id_date_index] = frequency

    # step 4: calculate the travel time of each line
    output_ticket_sale_data_df['DEP_TIME'] = \
        pd.to_datetime(output_ticket_sale_data_df['DEP_TIME'], format='%H:%M').dt.time
    output_ticket_sale_data_df['ARR_TIME'] = \
        pd.to_datetime(output_ticket_sale_data_df['ARR_TIME'], format='%H:%M').dt.time
    output_ticket_sale_data_df['TRAVEL_TIME_MIN'] = \
        output_ticket_sale_data_df['ARR_TIME'].apply(lambda x: x.hour * 60 + x.minute) - \
        output_ticket_sale_data_df['DEP_TIME'].apply(lambda x: x.hour * 60 + x.minute)
    output_ticket_sale_data_df['TRAVEL_TIME'] = (output_ticket_sale_data_df['TRAVEL_TIME_MIN'] / 60).round(2)

    # step 5: calculate the ticket price of each line
    output_ticket_sale_data_df['PRICE'] = \
        (output_ticket_sale_data_df['PRICESUM'] / output_ticket_sale_data_df['PSGNUM']).round().fillna(0).astype(int)

    # step 6: generate a dictionary to store the relationship between train id and line id
    g_trainId_to_lineId_dict = output_ticket_sale_data_df.set_index('TRAINID')['LINE_ID'].to_dict()
    g_trainId_to_lineName_dict = output_ticket_sale_data_df.set_index('TRAINID')['LINE'].to_dict()
    g_od_id_dict = dict(zip(output_ticket_sale_data_df['OD'], output_ticket_sale_data_df['OD_ID']))

    return output_ticket_sale_data_df  # , _od_line_obs_df


# Level 1: fetch boarding and alighting passengers at stations
def calc_board_alight(
        _ticket_sale_data_df):  # Use function calc_daily_board_alight and calc_total_board_alight;
    global g_board_station_list
    global g_alight_station_list
    global g_board_sta_seq_dict
    global g_alight_sta_seq_dict
    global g_station_id_dict
    global g_id_station_dict
    train_dep_passenger_df = _ticket_sale_data_df[['BOARDDATE', 'DEP_STATION_ID', 'DEP_STATION', 'PSGNUM']]
    train_arr_passenger_df = _ticket_sale_data_df[['BOARDDATE', 'ARR_STATION_ID', 'ARR_STATION', 'PSGNUM']]
    # calculate the daily boarding and alighting passengers for each station
    _daily_station_board_df = calc_daily_board_alight(train_dep_passenger_df, 'DEP_STATION_ID', 'DEP_STATION')
    _daily_station_alight_df = calc_daily_board_alight(train_arr_passenger_df, 'ARR_STATION_ID', 'ARR_STATION')
    # calculate the total boarding and alighting passengers for each station
    _total_board_df = calc_total_board_alight(_daily_station_board_df, 'DEP_STATION', 'DEP_STATION_ID')
    _total_alight_df = calc_total_board_alight(_daily_station_alight_df, 'ARR_STATION', 'ARR_STATION_ID')
    board_station_id_dict = dict(zip(_total_board_df['DEP_STATION'], _total_board_df['DEP_STATION_ID']))
    alight_station_id_dict = dict(zip(_total_alight_df['ARR_STATION'], _total_alight_df['ARR_STATION_ID']))
    board_id_station_dict = dict(zip(_total_board_df['DEP_STATION_ID'], _total_board_df['DEP_STATION']))
    alight_id_station_dict = dict(zip(_total_alight_df['ARR_STATION_ID'], _total_alight_df['ARR_STATION']))
    g_board_station_list = _total_board_df['DEP_STATION'].tolist()
    g_alight_station_list = _total_alight_df['ARR_STATION'].tolist()
    g_board_sta_seq_dict = dict(zip(g_board_station_list, range(len(g_board_station_list))))
    g_alight_sta_seq_dict = dict(zip(g_alight_station_list, range(len(g_alight_station_list))))
    g_station_id_dict = {**board_station_id_dict, **alight_station_id_dict}
    g_id_station_dict = {**board_id_station_dict, **alight_id_station_dict}
    _daily_station_board_df['DEP_STATION_SEQ'] = _daily_station_board_df['DEP_STATION'].apply(
        lambda x: g_board_sta_seq_dict[x])
    _daily_station_alight_df['ARR_STATION_SEQ'] = _daily_station_alight_df['ARR_STATION'].apply(
        lambda x: g_alight_sta_seq_dict[x])
    _total_board_df['DEP_STATION_SEQ'] = _total_board_df['DEP_STATION'].apply(lambda x: g_board_sta_seq_dict[x])
    _total_alight_df['ARR_STATION_SEQ'] = _total_alight_df['ARR_STATION'].apply(lambda x: g_alight_sta_seq_dict[x])

    print("___calculate the daily boarding and alighting passengers for", len(g_station_id_dict), "stations...")
    return _daily_station_board_df, _daily_station_alight_df, _total_board_df, _total_alight_df


def calc_daily_board_alight(data, station_id, station):
    dateList = data['BOARDDATE'].drop_duplicates().to_list()
    _daily_board_alight_df = pd.DataFrame()
    for dt in dateList:
        data_a_day = data[data['BOARDDATE'] == dt]
        sum_a_day = data_a_day.groupby(station_id)['PSGNUM'].sum().reset_index()
        sum_a_day['BOARDDATE'] = dt
        sum_a_day[station] = data_a_day.groupby(station_id)[station].first().reset_index(drop=True)
        _daily_board_alight_df = pd.concat([_daily_board_alight_df, sum_a_day], ignore_index=True)
    _daily_board_alight_df = _daily_board_alight_df[['BOARDDATE', station_id, station, 'PSGNUM']]
    return _daily_board_alight_df


def calc_total_board_alight(data, station, station_id):
    _total_board_alight = data.groupby([station])[['PSGNUM']].sum().reset_index()
    _total_board_alight[station_id] = _total_board_alight[station]. \
        map(data.drop_duplicates([station])[[station, station_id]].set_index(station)[station_id])
    _total_board_alight = _total_board_alight[[station_id, station, 'PSGNUM']]
    _total_board_alight = _total_board_alight.sort_values(by=station_id, ascending=True)

    return _total_board_alight


# Level 2: fetch OD flow observations and od-based economic factors from ticket sale data
def calc_obs_od_flow(_ticket_sale_data_df):
    date_od_groups = _ticket_sale_data_df.groupby(['BOARDDATE', 'OD_ID', 'OD'])
    daily_od_obs_list = []
    for date_od_index, od_group in date_od_groups:
        od_id = date_od_index[1]
        boardDate = date_od_index[0]
        od_name = date_od_index[2]
        od_flow = od_group['PSGNUM'].sum()
        dep_station_id = od_group['DEP_STATION_ID'].iloc[0]
        dep_station = od_group['DEP_STATION'].iloc[0]
        dep_station_seq = g_board_sta_seq_dict[dep_station]
        arr_station_id = od_group['ARR_STATION_ID'].iloc[0]
        arr_station = od_group['ARR_STATION'].iloc[0]
        arr_station_seq = g_alight_sta_seq_dict[arr_station]
        daily_od_obs_list.append([boardDate, od_id, od_name, dep_station_id, dep_station, dep_station_seq,
                                  arr_station_id, arr_station, arr_station_seq, od_flow])

    # convert list to df
    _daily_od_obs_df = pd.DataFrame(daily_od_obs_list,
                                    columns=['BOARDDATE', 'OD_ID', 'OD', 'DEP_STATION_ID', 'DEP_STATION',
                                             'DEP_STATION_SEQ', 'ARR_STATION_ID', 'ARR_STATION', 'ARR_STATION_SEQ',
                                             'PSGNUM'])

    return _daily_od_obs_df


def calc_od_eco_factor(_daily_obs_od_flow_df):
    nb_bins = 200  # set the number of groups for economic impacts of OD pairs
    # group by OD Department and Arrival Station (first), PSGNUM (sum)
    _total_obs_od_df = _daily_obs_od_flow_df.groupby('OD').agg({
        'OD_ID': 'first',
        'DEP_STATION_ID': 'first',
        'DEP_STATION': 'first',
        'DEP_STATION_SEQ': 'first',
        'ARR_STATION_ID': 'first',
        'ARR_STATION': 'first',
        'ARR_STATION_SEQ': 'first',
        'PSGNUM': 'sum'
    }).reset_index()

    max_od_flow = _total_obs_od_df['PSGNUM'].max()
    bin_interval = max_od_flow / nb_bins
    group_rule = pd.DataFrame()
    min_gsum = []
    bound_a_bin_df = pd.DataFrame()
    current_value = 0
    while current_value <= max_od_flow:
        min_gsum.append(current_value)
        current_value += bin_interval
    bound_a_bin_df['min'] = min_gsum
    bound_a_bin_df['max'] = bound_a_bin_df['min'] + bin_interval
    bound_a_bin_df['ECO_FAC'] = list(range(1, len(bound_a_bin_df) + 1))
    eco_fac = []
    for i in _total_obs_od_df['PSGNUM']:
        for j in range(len(bound_a_bin_df)):
            judge_line = bound_a_bin_df.iloc[j]
            if judge_line['min'] <= i < judge_line['max']:
                eco_comm_value = bound_a_bin_df.at[j, 'ECO_FAC']
                eco_fac.append(eco_comm_value)
    _total_obs_od_df['ECO_FAC'] = eco_fac
    od_eco_dict = dict(zip(_total_obs_od_df['OD'], _total_obs_od_df['ECO_FAC']))
    _daily_obs_od_flow_df['ECO_FAC'] = _daily_obs_od_flow_df['OD'].map(od_eco_dict)
    _daily_obs_od_df = _daily_obs_od_flow_df.copy()
    return _daily_obs_od_df, _total_obs_od_df


# Level 3: fetch OD line information from ticket sale data

def fetch_od_line_from_ticket(_ticket_sale_data_df):
    global g_odLineId_to_odSectName_dict
    _ticket_sale_data_df = _ticket_sale_data_df[['BOARDDATE', 'OD_LINE_ID', 'OD_LINE', 'OD_ID', 'OD', 'LINE_ID', 'LINE',
                                                 'DEP_STATION', 'DEP_STATION_ID', 'ARR_STATION', 'ARR_STATION_ID',
                                                 'PSGNUM', 'PRICESUM', 'TRAVEL_TIME', 'PRICE']]
    date_od_line_groups = _ticket_sale_data_df.groupby(['BOARDDATE', 'OD_LINE_ID'])
    date_od_line_list = []
    for date_od_line_index, od_line_group_df in date_od_line_groups:
        od_line_date = date_od_line_index[0]
        od_line_id = date_od_line_index[1]
        od_line_name = od_line_group_df['OD_LINE'].iloc[0]
        od_name = od_line_group_df['OD'].iloc[0]
        od_id = od_line_group_df['OD_ID'].iloc[0]
        train_line_id = od_line_group_df['LINE_ID'].iloc[0]
        line_name = od_line_group_df['LINE'].iloc[0]
        dep_station = od_line_group_df['DEP_STATION'].iloc[0]
        dep_station_id = od_line_group_df['DEP_STATION_ID'].iloc[0]
        dep_station_seq = g_board_sta_seq_dict[dep_station]
        arr_station = od_line_group_df['ARR_STATION'].iloc[0]
        arr_station_id = od_line_group_df['ARR_STATION_ID'].iloc[0]
        arr_station_seq = g_alight_sta_seq_dict[arr_station]
        psg_num = od_line_group_df['PSGNUM'].sum()
        price_sum = od_line_group_df['PRICESUM'].sum()
        travel_time = od_line_group_df['TRAVEL_TIME'].mean()
        price = od_line_group_df['PRICE'].mean()
        date_od_line_list.append([od_line_date, od_line_id, od_line_name, od_id, od_name, train_line_id, line_name,
                                  dep_station, dep_station_id, dep_station_seq, arr_station, arr_station_id,
                                  arr_station_seq, psg_num, price_sum, travel_time, price])
    _daily_od_line_df = pd.DataFrame(date_od_line_list,
                                     columns=['BOARDDATE', 'OD_LINE_ID', 'OD_LINE', 'OD_ID', 'OD', 'LINE_ID', 'LINE',
                                              'DEP_STATION', 'DEP_STATION_ID', 'DEP_STATION_SEQ', 'ARR_STATION',
                                              'ARR_STATION_ID', 'ARR_STATION_SEQ',
                                              'PSGNUM', 'PRICESUM', 'TRAVEL_TIME', 'PRICE'])

    # calculate the probability of each OD line per boarding date
    _daily_od_line_df['OBS_PROB'] = \
        _daily_od_line_df['PSGNUM'] / (
            _daily_od_line_df.groupby(['BOARDDATE', 'DEP_STATION'])['PSGNUM'].transform('sum'))
    _daily_od_line_df['FREQUENCY'] = \
        _daily_od_line_df.apply(lambda x: g_date_line_freq_dict[(x['LINE_ID'], x['BOARDDATE'])], axis=1)

    _daily_od_line_df['AVG_PSGNUM'] = _daily_od_line_df.groupby('OD_LINE_ID')['PSGNUM'].transform('mean')
    _daily_od_line_df['AVG_TRAVEL_TIME'] = _daily_od_line_df.groupby('OD_LINE_ID')['TRAVEL_TIME'].transform('mean')
    _daily_od_line_df['AVG_PRICE'] = _daily_od_line_df.groupby('OD_LINE_ID')['PRICE'].transform('mean')
    print("___generate", len(_daily_od_line_df), "daily OD lines...")

    _total_od_line_df = _daily_od_line_df[['OD_LINE_ID', 'OD_LINE', 'OD_ID', 'OD', 'LINE_ID', 'LINE', 'DEP_STATION',
                                           'DEP_STATION_ID', 'DEP_STATION_SEQ', 'ARR_STATION', 'ARR_STATION_ID',
                                           'ARR_STATION_SEQ', 'AVG_PSGNUM', 'AVG_TRAVEL_TIME', 'AVG_PRICE']]
    _total_od_line_df = _total_od_line_df.drop_duplicates(subset=['OD_LINE_ID', 'OD_LINE'], keep='first')
    _total_od_line_df = _total_od_line_df.reset_index(drop=True)
    _total_od_line_df['OD_LINE_SECTION'] = None
    _total_od_line_df['OD_LINE_SECTION'] = _total_od_line_df['OD_LINE_SECTION'].astype(object)
    for row_index, row_df in _total_od_line_df.iterrows():
        station_list = row_df['LINE'].split('-')
        depart_station = row_df['DEP_STATION']
        arrive_station = row_df['ARR_STATION']
        train_line_id = row_df['LINE_ID']
        line_sect_list = []
        flag_within_od = False
        for sta_index in range(len(station_list) - 1):
            if station_list[sta_index] == depart_station:
                flag_within_od = True
            if flag_within_od:
                line_sect_list.append(f"{train_line_id}-{station_list[sta_index]}-{station_list[sta_index + 1]}")
            if station_list[sta_index + 1] == arrive_station:
                flag_within_od = False
        print(line_sect_list)
        _total_od_line_df.at[row_index, 'OD_LINE_SECTION'] = line_sect_list

    _total_od_line_df['AVG_PROB'] = _total_od_line_df['AVG_PSGNUM'] / _total_od_line_df['AVG_PSGNUM'].sum()
    print("___generate", len(_total_od_line_df), "OD lines...")

    g_odLineId_to_odSectName_dict = dict(zip(_total_od_line_df['OD_LINE_ID'], _total_od_line_df['OD_LINE_SECTION']))
    _daily_od_line_df['OD_LINE_SECTION'] = _daily_od_line_df['OD_LINE_ID'].map(g_odLineId_to_odSectName_dict)

    return _daily_od_line_df, _total_od_line_df


# level 4: fetch section information from ticket sale data
def fetch_line_section(_ticket_sale_data_df):
    global g_section_list
    global g_sectName_to_sectId_dict
    global g_sectId_to_sectName_dict
    global g_sectName_to_lineId_dict
    global g_lineId_to_sectId_dict
    global g_lineId_to_sectName_dict

    _all_sect_df = pd.DataFrame()
    date_groups = _ticket_sale_data_df.groupby('BOARDDATE')
    for date, daily_data_df in date_groups:
        daily_line_plan = daily_data_df[['LINE', 'LINE_ID']]
        daily_line_plan = daily_line_plan.drop_duplicates()
        daily_line_plan = daily_line_plan.reset_index(drop=True)
        _daily_sect_df = fetch_daily_line_section(date, daily_line_plan)  # line section not section
        _all_sect_df = pd.concat([_all_sect_df, _daily_sect_df])
    _all_sect_df.drop_duplicates(subset=['SECTION', 'BOARDDATE'], keep='first', inplace=True)
    _all_sect_df = _all_sect_df.reset_index(drop=True)

    g_section_list = _all_sect_df['SECTION'].unique().tolist()
    g_sectName_to_sectId_dict = dict(zip(g_section_list, range(len(g_section_list))))
    g_sectId_to_sectName_dict = dict(zip(range(len(g_section_list)), g_section_list))

    _all_sect_df['SECTION_ID'] = _all_sect_df['SECTION'].map(g_sectName_to_sectId_dict)
    _all_sect_df = _all_sect_df[['SECTION_ID', 'SECTION', 'LINE_ID', 'FREQUENCY', 'CAPACITY', 'BOARDDATE']]

    g_sectName_to_lineId_dict = dict(zip(_all_sect_df['SECTION'], _all_sect_df['LINE_ID']))

    # add output _all_line_df for which section is in line presented by "SECTION_ID" and "SECTION", from main step 5
    no_time_all_sect_df = _all_sect_df.copy()
    no_time_all_sect_df = no_time_all_sect_df[['SECTION_ID', 'SECTION', 'LINE_ID']]
    no_time_all_sect_df.drop_duplicates(subset=['SECTION_ID', 'LINE_ID'], keep='first', inplace=True)
    line_groups = no_time_all_sect_df.groupby('LINE_ID')

    for train_line_id, line_group in line_groups:
        g_lineId_to_sectId_dict[train_line_id] = line_group['SECTION_ID'].tolist()
        g_lineId_to_sectName_dict[train_line_id] = line_group['SECTION'].tolist()
    print("___generate", len(g_section_list), "line sections...")
    return _all_sect_df


def fetch_daily_line_section(line_date, lines):
    _section_list = []
    line_id_list = []
    for ind in lines['LINE_ID'].index:
        plan_element = lines.at[ind, 'LINE']
        _line_id = lines.at[ind, 'LINE_ID']
        # print(k)
        k_stations_list = plan_element.split('-')
        for a in range(len(k_stations_list) - 1):
            section = f"{k_stations_list[a]}-{k_stations_list[a + 1]}"
            _section_list.append(f"{_line_id}-{section}")
            line_id_list.append(_line_id)
    _daily_sect_df = pd.DataFrame({
        'SECTION': _section_list,
        'LINE_ID': line_id_list,
    })
    _daily_sect_df['FREQUENCY'] = \
        _daily_sect_df['LINE_ID'].apply(lambda x: g_date_line_freq_dict[(x, line_date)])
    _daily_sect_df['CAPACITY'] = _daily_sect_df['FREQUENCY'] * TRAIN_CAPACITY
    _daily_sect_df['BOARDDATE'] = line_date
    _daily_sect_df.sort_values(by='LINE_ID', inplace=True)
    _daily_sect_df = _daily_sect_df.reset_index(drop=True)

    return _daily_sect_df


# ======================Variables definition and incidence matrices===============================

# function to create incidence matrices
def create_inc_mat(_total_obs_od_df, _total_od_line_df):
    print("___create incidence matrices for the hierarchical network...")
    _nb_origin = PARAMS['nb_origin']
    _nb_destination = PARAMS['nb_destination']
    _nb_od = PARAMS['nb_od']
    _nb_od_line = PARAMS['nb_od_line']
    nb_section = PARAMS['nb_section']

    # create incidence matrices between aboard stations and od pairs
    print("___create incidence matrices between aboard stations and od pairs...")
    _o_to_od_inc_mat = np.zeros((_nb_origin, _nb_od))
    for i in range(len(_total_obs_od_df)):
        o_id = int(_total_obs_od_df.loc[i, 'DEP_STATION_SEQ'])
        od_id = int(_total_obs_od_df.loc[i, 'OD_ID'])
        _o_to_od_inc_mat[o_id][od_id] = 1

    # create incidence matrices between od pairs and od lines
    print("___create incidence matrices between od pairs and od lines...")
    _od_to_od_line_inc_mat = np.zeros((_nb_od, _nb_od_line))
    for i in range(len(_total_od_line_df)):
        od_id = int(_total_od_line_df.loc[i, 'OD_ID'])
        od_line_id = int(_total_od_line_df.loc[i, 'OD_LINE_ID'])
        _od_to_od_line_inc_mat[od_id][od_line_id] = 1

    # create incidence matrices between od lines and destination stations
    print("___create incidence matrices between od lines and destination stations...")
    _line_to_destination_inc_mat = np.zeros((_nb_od_line, _nb_destination))
    for i in range(len(_total_od_line_df)):
        od_line_id = int(_total_od_line_df.loc[i, 'OD_LINE_ID'])
        d_id = int(_total_od_line_df.loc[i, 'ARR_STATION_SEQ'])
        _line_to_destination_inc_mat[od_line_id][d_id] = 1

    # create incidence matrices between od lines and line sections
    print("___create incidence matrices between od lines and line sections...")
    _line_to_section_inc_mat = np.zeros((_nb_od_line, nb_section))
    for i in range(len(_total_od_line_df)):
        od_line_id = int(_total_od_line_df.loc[i, 'OD_LINE_ID'])
        section_id_list = _total_od_line_df.loc[i, 'OD_LINE_SECTION']
        for j in range(len(section_id_list)):
            sect_id = g_sectName_to_sectId_dict[section_id_list[j]]
            _line_to_section_inc_mat[od_line_id][sect_id] = 1

    return _o_to_od_inc_mat, _od_to_od_line_inc_mat, _line_to_destination_inc_mat, _line_to_section_inc_mat


# function to define variables
def define_variables(_total_obs_od_df, _total_od_line_df):
    # boarding variables and variables coefficient with od parameters witch used to calculate probability
    print("__O layer: define variables for the boarding layer...")
    # define two variables:
    # 1. est_boarding_var: estimated boarding passengers at each station
    # 2. od_eco_coeff_var: economic coefficient for each od pair
    boarding_var_list = []
    station_eco_coeff_var_list = []

    for station_name in g_board_station_list:
        est_boarding_var = tf.Variable([0, ], name='boarding.' + station_name, dtype=tf.float64,
                                       constraint=lambda z: tf.clip_by_value(z, 1e-16, UB_BOARDING))
        boarding_var_list.append(est_boarding_var)
        sta_eco_coeff = tf.Variable([0, ], name='eco_coeff.' + station_name, dtype=tf.float64,
                                    constraint=lambda z: tf.clip_by_value(z, 1e-16, UB_ECO_COEFF))
        station_eco_coeff_var_list.append(sta_eco_coeff)

    print("__OD layer: define variables for the OD layer...")
    # define variables for the OD layer
    # 1 od_line_freq_coeff: frequency coefficient for each od line
    # 2 od_line_tt_coeff: travel time coefficient for each od line
    # 3 od_line_price_coeff: price coefficient for each od line
    # 4 regression_freq_coeff: frequency coefficient for each od pair in od-based regression model
    # 5 od_lag_mult: multiplier for each od pair for lagrange dualization
    # 6 od_lag_mult_updating_step: parameter for each od pair for updating od multiplier
    # 7 c_od_mult: adding penalty for augmented lagrangian function
    od_freq_coeff_var_list = []
    od_tt_coeff_var_list = []
    od_price_coeff_var_list = []
    regression_od_freq_coeff_var_list = []
    od_lag_mult_var_list = []
    od_lag_mult_updating_step_var_list = []
    c_od_mult_var_list = []
    for od_name in g_od_list:
        # three coefficients for each od line in nested logit model for od line choice
        od_freq_coeff = tf.Variable([0, ], name='line_freq_coeff.' + od_name, dtype=tf.float64,
                                    constraint=lambda z: tf.clip_by_value(z, 1e-16, UB_LINE_FREQ_COEFF))
        od_tt_coeff = tf.Variable([0, ], name='line_tt_coeff.' + od_name, dtype=tf.float64,
                                  constraint=lambda z: tf.clip_by_value(z, LB_LINE_TT_COEFF, -1e-16))
        od_price_coeff = tf.Variable([0, ], name='line_price_coeff.' + od_name, dtype=tf.float64,
                                     constraint=lambda z: tf.clip_by_value(z, LB_LINE_PRICE_COEFF, -1e-16))

        od_freq_coeff_var_list.append(od_freq_coeff)
        od_tt_coeff_var_list.append(od_tt_coeff)
        od_price_coeff_var_list.append(od_price_coeff)

        # one coefficient for each od pair in od-based regression model
        reg_od_freq_coeff = tf.Variable([0, ], name='od_freq_coeff.' + od_name, dtype=tf.float64,
                                        constraint=lambda z: tf.clip_by_value(z, 1e-16, UB_OD_FREQ_COEFF))
        regression_od_freq_coeff_var_list.append(reg_od_freq_coeff)

        # two parameters for each od pair for lagrange dualization
        od_lag_mult = tf.Variable([INIT_OD_LAG_MULT, ], name='od_lag_mult.' + od_name, dtype=tf.float64)

        od_lag_mult_var_list.append(od_lag_mult)

        od_lag_mult_updating_step = tf.Variable([INIT_OD_LAG_MULT_UPDATING_STEP, ],
                                                name='od_lag_mult_updating_step.' + od_name, dtype=tf.float64)
        od_lag_mult_updating_step_var_list.append(od_lag_mult_updating_step)

        # one parameter for each od pair for adding penalty for augmented lagrangian dualization
        c_od_mult = tf.Variable([INIT_C_OD_MULT, ],
                                name='c_od_mult.' + od_name, dtype=tf.float64)
        c_od_mult_var_list.append(c_od_mult)

    # define variables for theta in od-based nested logit model
    od_theta_var_list = []
    for od_id in _total_obs_od_df['OD_ID']:
        od_info = _total_obs_od_df[_total_obs_od_df['OD_ID'] == od_id]
        od_name = od_info['OD'].values[0]
        o_id = od_info['DEP_STATION_ID'].values[0]
        # if the od pair only have one od line, set theta = 1
        if _total_od_line_df['OD_ID'].value_counts()[od_id] == 1:
            od_theta = tf.Variable([1, ], name='theta.' + od_name, dtype=tf.float64,
                                   constraint=lambda z: tf.clip_by_value(z, 1, 1))
            od_theta_var_list.append(od_theta)
        # if the departure station only have one od pair, set theta = 1
        elif _total_obs_od_df['DEP_STATION_ID'].value_counts()[o_id] == 1:
            od_theta = tf.Variable([1, ], name='theta.' + od_name, dtype=tf.float64,
                                   constraint=lambda z: tf.clip_by_value(z, 1, 1))
            od_theta_var_list.append(od_theta)
        # otherwise, set theta as a variable between 0 and 1
        else:
            od_theta = tf.Variable([0.05, ], name='theta.' + od_name, dtype=tf.float64,
                                   constraint=lambda z: tf.clip_by_value(z, LB_THETA, 1))
            od_theta_var_list.append(od_theta)

    # define variables for the section layer
    # 1 sect_lag_mult: multiplier for each od pair for lagrange dualization
    # 2 sect_lag_mult_updating_step: parameter for each od pair for updating od multiplier
    # 3 c_sect_mult: adding penalty for augmented lagrangian function
    sect_lag_mult_var_list = []
    sect_lag_mult_updating_step_var_list = []
    c_sect_mult_var_list = []

    for sect_name in g_section_list:
        # two parameters for each section for lagrange dualization
        sect_lag_mult = tf.Variable([0, ], name='sect_lag_mult.' + sect_name, dtype=tf.float64)
        sect_lag_mult_var_list.append(sect_lag_mult)

        sect_lag_mult_updating_step = tf.Variable([INIT_SECT_LAG_MULT_UPDATING_STEP, ],
                                                  name='sect_lag_mult_updating_step.' + sect_name, dtype=tf.float64)
        sect_lag_mult_updating_step_var_list.append(sect_lag_mult_updating_step)

        # one parameter for each section for adding penalty for augmented lagrangian dualization
        c_sect_mult = tf.Variable([INIT_C_SECT_MULT, ], name='c_sect_mult.' + sect_name,
                                  dtype=tf.float64)
        c_sect_mult_var_list.append(c_sect_mult)

    return (boarding_var_list, od_theta_var_list, station_eco_coeff_var_list, od_freq_coeff_var_list,
            od_tt_coeff_var_list, od_price_coeff_var_list, regression_od_freq_coeff_var_list,
            od_lag_mult_var_list, od_lag_mult_updating_step_var_list, c_od_mult_var_list,
            sect_lag_mult_var_list, sect_lag_mult_updating_step_var_list, c_sect_mult_var_list)


# ======================Utilities for training process===============================

def get_batch_sample(_training_daily_stat_board_df, _training_daily_stat_alight_df, _training_daily_obs_od_df,
                     _training_daily_od_line_df, _training_sample_list, _train_batch_size):
    _batch_training_sample_list = sample(_training_sample_list, _train_batch_size)
    _batch_training_boarding_data = _training_daily_stat_board_df[_training_daily_stat_board_df['BOARDDATE'].isin(
        _batch_training_sample_list)]
    _batch_training_od_data = \
        _training_daily_obs_od_df[_training_daily_obs_od_df['BOARDDATE'].isin(_batch_training_sample_list)]
    _batch_training_od_line_data = \
        _training_daily_od_line_df[_training_daily_od_line_df['BOARDDATE'].isin(_batch_training_sample_list)]
    _batch_training_alight_data = \
        _training_daily_stat_alight_df[_training_daily_stat_alight_df['BOARDDATE'].isin(_batch_training_sample_list)]

    return _batch_training_boarding_data, _batch_training_alight_data, _batch_training_od_data, \
        _batch_training_od_line_data, _batch_training_sample_list


def select_single_day_sample(sample_date, _daily_board_data, _daily_alight_data, _daily_od_obs_data,
                             _daily_od_line_data, _daily_section_data):
    _board_dataset = _daily_board_data[_daily_board_data['BOARDDATE'] == sample_date]
    _od_obs_dataset = _daily_od_obs_data[_daily_od_obs_data['BOARDDATE'] == sample_date]
    _od_line_dataset = _daily_od_line_data[_daily_od_line_data['BOARDDATE'] == sample_date]
    _section_dataset = _daily_section_data[_daily_section_data['BOARDDATE'] == sample_date]
    _alight_dataset = _daily_alight_data[_daily_alight_data['BOARDDATE'] == sample_date]

    _board_dataset = _board_dataset.sort_values(by='DEP_STATION_SEQ', ascending=True)
    _od_obs_dataset = _od_obs_dataset.sort_values(by='OD_ID', ascending=True)
    _od_line_dataset = _od_line_dataset.sort_values(by='OD_LINE_ID', ascending=True)
    _section_dataset = _section_dataset.sort_values(by='SECTION_ID', ascending=True)

    # select the boarding passenger at stations
    # sort the dataset by station sequence
    _board_dataset = _board_dataset.sort_values(by='DEP_STATION_SEQ', ascending=True)
    _boarding_psg_array = _board_dataset['PSGNUM'] / K_PERSON_CONVERTER
    _boarding_psg_array = _boarding_psg_array.values.astype('float64')

    # select the od flow observation
    _od_obs_dataset = _daily_od_obs_data[_daily_od_obs_data['BOARDDATE'] == sample_date]
    _od_psg_array = _od_obs_dataset['PSGNUM'] / K_PERSON_CONVERTER
    _od_psg_array = _od_psg_array.values.astype('float64')
    _od_freq_reg_array = _od_obs_dataset['FREQUENCY'].values.astype('float64')
    _od_eco_array = _od_obs_dataset['ECO_FAC'].values.astype('float64')

    # select the od line observation
    _od_line_dataset = _daily_od_line_data[_daily_od_line_data['BOARDDATE'] == sample_date]
    _od_line_dataset = _od_line_dataset.reset_index(drop=True)
    od_line_param_df = _od_line_dataset[['FREQUENCY', 'TRAVEL_TIME', 'PRICE']]
    _od_line_param_array = od_line_param_df.values.astype('float64')
    _od_line_psg_array = _od_line_dataset['PSGNUM'] / K_PERSON_CONVERTER
    _od_line_psg_array = _od_line_psg_array.values.astype('float64')

    # select the destination alighting passenger
    _alight_dataset = _daily_alight_data[_daily_alight_data['BOARDDATE'] == sample_date]
    _alighting_psg_array = _alight_dataset['PSGNUM'] / K_PERSON_CONVERTER
    _alighting_psg_array = _alighting_psg_array.values.astype('float64')

    # select the section capacity
    _section_capacity_array = _section_dataset['CAPACITY'].values.astype('float64')

    return _boarding_psg_array, _od_psg_array, _od_line_psg_array, _alighting_psg_array, _od_freq_reg_array, \
        _od_eco_array, _od_line_param_array, _section_capacity_array


# ======================Functions for core model==================================

def init_loglikelihood(_training_sample_list, _training_daily_stat_board_df,
                       _training_daily_stat_alight_df, _training_daily_obs_od_df,
                       _training_daily_od_line_df, _training_daily_sect_df,
                       _od_theta_list, _o_eco_coeff_list, _od_freq_coeff_list,
                       _od_tt_coeff_list, _od_price_coeff_list):
    """
    :param _training_sample_list: the total training sample generated through ticket sale data
    :param _training_daily_stat_board_df: daily (original station) boarding information
    :param _training_daily_stat_alight_df: daily (destination station) alighting information
    :param _training_daily_obs_od_df: daily od information
    :param _training_daily_od_line_df: daily od line information
    :param _training_daily_sect_df:  daily section information (line section)
    :param _od_theta_list: for each od pair, the theta value in nested logit model
    :param _o_eco_coeff_list: all od pairs with same original station share one economic coefficient
    :param _od_freq_coeff_list: all od lines with same od pair share one frequency coefficient
    :param _od_tt_coeff_list:  all od lines with same od pair share one travel time coefficient
    :param _od_price_coeff_list: all od lines with same od pair share one price coefficient
    """
    _nb_od_line = len(_training_daily_od_line_df)

    _daily_loglikelihood_list = []
    for date in _training_sample_list:
        _, _, _od_line_psg_array, _, _, _od_eco_array, _od_line_param_array, _ = \
            select_single_day_sample(date, _training_daily_stat_board_df,
                                     _training_daily_stat_alight_df,
                                     _training_daily_obs_od_df,
                                     _training_daily_od_line_df,
                                     _training_daily_sect_df)

        _, _, _final_od_line_prob_tensor = \
            calc_prob_nested_logit_model(_od_theta_list, _o_eco_coeff_list, _od_freq_coeff_list,
                                         _od_tt_coeff_list, _od_price_coeff_list, _od_eco_array, _od_line_param_array)

        _loglikelihood_tensor = calc_loss_loglikelihood(_final_od_line_prob_tensor, _od_line_psg_array)
        _daily_loglikelihood_list.append(_loglikelihood_tensor)
    _sum_loglikelihood = sum(_daily_loglikelihood_list).numpy()[0]
    return _sum_loglikelihood


def calc_prob_nested_logit_model(_od_theta_list, _o_eco_coeff_list, _od_line_freq_coeff_list, _od_line_tt_coeff_list,
                                 _od_line_price_coeff_list, _od_eco_array, _od_line_param_array):
    _nb_od = len(_od_theta_list)
    _nb_od_line = len(_od_line_param_array)
    _nb_origin = len(_o_eco_coeff_list)

    # part 1:calculate conditional probability:

    # step 1: calculate od line utility:
    _od_theta_mat = tf.concat(_od_theta_list, axis=0)
    od_theta_mat = tf.reshape(_od_theta_mat, shape=[_nb_od, 1])
    _od_line_param_coeff_mat = \
        tf.concat([_od_line_freq_coeff_list, _od_line_tt_coeff_list, _od_line_price_coeff_list], axis=1)
    _od_line_param_coeff_mat = tf.reshape(_od_line_param_coeff_mat, [_nb_od, 3])
    _util_od_line = \
        tf.reduce_sum(tf.multiply(tf.transpose(tf.matmul(tf.transpose(_od_line_param_coeff_mat)
                                                         , od_to_od_line_inc_mat)), _od_line_param_array), axis=1)

    # step 2: calculate the sum value of exponent of od line utility in each od
    _util_od_line = tf.reshape(_util_od_line, [_nb_od_line, 1])
    _util_over_theta = tf.math.divide(_util_od_line, tf.transpose(tf.matmul(tf.transpose(od_theta_mat)
                                                                            , od_to_od_line_inc_mat)))
    # makesure all values in _util_over_theta should be within 300 or -300
    # set the upper bound as 300 and lower bound as -300 to avoid 0 and infinity
    _exp_util_od_line_over_theta = tf.exp(tf.clip_by_value(_util_over_theta, -300.0, 300.0))

    _int_od_to_od_line_inc_mat = tf.cast(od_to_od_line_inc_mat, tf.float64)
    _sum_exp_util_od_line_over_theta_for_od = tf.matmul(_int_od_to_od_line_inc_mat, _exp_util_od_line_over_theta)

    # step 3: calculate the probability of selecting an od line under the certain od condition
    _conditional_od_line_prob_tensor = tf.divide(_exp_util_od_line_over_theta,
                                                 tf.matmul(tf.transpose(_int_od_to_od_line_inc_mat),
                                                           _sum_exp_util_od_line_over_theta_for_od))

    # part 2: calculate marginal probability (od probability):

    # step 1: calculate od utility:
    _od_param_coeff_mat = tf.concat(_o_eco_coeff_list, axis=0)
    _od_param_coeff_mat = tf.reshape(_od_param_coeff_mat, shape=[_nb_origin, 1])
    _util_od = tf.reduce_sum(tf.multiply(tf.transpose(tf.matmul(tf.transpose(_od_param_coeff_mat),
                                                                o_to_od_inc_mat)), _od_eco_array), axis=1)
    _util_od = tf.reshape(_util_od, [_nb_od, 1])

    # step 2: calculate the sum value of exponent of od utility from each original station
    _log_sum_exp_util_od_line_over_theta_for_od = tf.math.log(_sum_exp_util_od_line_over_theta_for_od)
    # the numerator of od probability
    _exp_util_od = \
        tf.exp(tf.add(_util_od + NON_ZERO, tf.multiply(od_theta_mat, _log_sum_exp_util_od_line_over_theta_for_od)))
    # the denominator of od probability
    _int_o_to_od_inc_mat = tf.cast(o_to_od_inc_mat, tf.float64)
    _sum_exp_util_od = tf.matmul(_int_o_to_od_inc_mat, _exp_util_od)

    # step 3: calculate the probability of selecting an od from certain original station
    _od_prob_tensor = tf.divide(_exp_util_od, tf.matmul(tf.transpose(_int_o_to_od_inc_mat), _sum_exp_util_od))

    # part 3: calculate the probability of selecting an od line
    _final_od_line_prob_tensor = tf.multiply(_conditional_od_line_prob_tensor,
                                             tf.matmul(tf.transpose(_int_od_to_od_line_inc_mat), _od_prob_tensor))

    return _od_prob_tensor, _conditional_od_line_prob_tensor, _final_od_line_prob_tensor


@tf.function
def est_flow_by_nested_logit_model(_est_boarding_list, _od_prob_tensor, _conditional_od_line_prob_tensor):
    _nb_origin = len(_est_boarding_list)

    # level 1: estimating boarding flow
    _est_boarding_tensor = tf.concat(_est_boarding_list, axis=0)
    _est_boarding_tensor = tf.reshape(_est_boarding_tensor, [_nb_origin, 1])

    # level 2: estimating od flow
    _est_od_flow_nl_tensor = \
        tf.multiply(tf.transpose(tf.matmul(tf.transpose(_est_boarding_tensor), o_to_od_inc_mat)), _od_prob_tensor)

    # level 3: estimating od line flow
    _est_od_line_flow_tensor = \
        tf.multiply(tf.transpose(tf.matmul(tf.transpose(_est_od_flow_nl_tensor), od_to_od_line_inc_mat)),
                    _conditional_od_line_prob_tensor)

    # level 4: estimating alighting flow
    _est_alighting_tensor = tf.transpose(tf.matmul(tf.transpose(_est_od_line_flow_tensor), line_to_destination_inc_mat))

    # level 5: estimating section flow
    _est_section_flow_tensor = tf.transpose(tf.matmul(tf.transpose(_est_od_line_flow_tensor), line_to_section_inc_mat))
    return (_est_boarding_tensor, _est_od_flow_nl_tensor, _est_od_line_flow_tensor,
            _est_alighting_tensor, _est_section_flow_tensor)


def est_od_flow_by_regression(_reg_od_freq_coeff_list, _od_freq_reg_array):
    _nb_od = len(_reg_od_freq_coeff_list)

    # estimating od flow
    od_regression_param_coeff_tensor = tf.reshape(_reg_od_freq_coeff_list, [1, _nb_od])
    _est_od_flow_reg_tensor = tf.multiply(od_regression_param_coeff_tensor, _od_freq_reg_array)
    _est_od_flow_reg_tensor = tf.reshape(_est_od_flow_reg_tensor, [_nb_od, 1])
    return _est_od_flow_reg_tensor


# loss functions
def calc_loss_loglikelihood(_final_od_line_prob_tensor, _od_line_psg_array):
    _nb_od_line = len(_final_od_line_prob_tensor)
    _od_line_psg_tensor = tf.reshape(_od_line_psg_array, [_nb_od_line, 1])
    _loglikelihood_tensor = -tf.reduce_sum(tf.multiply(tf.math.log(_final_od_line_prob_tensor), _od_line_psg_tensor),
                                           axis=0)
    return _loglikelihood_tensor


def loss_func_boarding_mse(_est_board_tensor, _boarding_psg_array):
    _nb_origin = len(_est_board_tensor)

    _boarding_psg_tensor = tf.reshape(_boarding_psg_array, [_nb_origin, 1])
    _boarding_mse = tf.reduce_sum(tf.square(_boarding_psg_tensor - _est_board_tensor), axis=0)
    return _boarding_mse


def loss_func_alighting_mse(_est_alight_tensor, _alighting_psg_array):
    _nb_destination = len(_est_alight_tensor)

    _alighting_psg_tensor = tf.reshape(_alighting_psg_array, [_nb_destination, 1])
    _alighting_mse = tf.reduce_sum(tf.square(_alighting_psg_tensor - _est_alight_tensor), axis=0)
    return _alighting_mse


def loss_func_sect_cap_constraint(_est_section_flow, _sect_lag_mult_list,
                                  _c_sect_mult_list,
                                  _training_sample_sect_df):
    # training_daily_sect_df just for capacity

    _nb_sect = len(_est_section_flow)

    _sect_multiplier_tensor = tf.concat(_sect_lag_mult_list, axis=0)
    _sect_multiplier_tensor = tf.reshape(_sect_multiplier_tensor, shape=[_nb_sect, 1])

    _c_sect_multiplier_tensor = tf.concat(_c_sect_mult_list, axis=0)
    _c_sect_multiplier_tensor = tf.reshape(_c_sect_multiplier_tensor, shape=[_nb_sect, 1])

    _sect_cap_array = _training_sample_sect_df[
        'CAPACITY']  # Xinyu W, _training_daily_sect_df --> _training_sample_sect_df
    _sect_cap_array = _sect_cap_array.values.astype('float64')
    _sect_cap_tensor = tf.reshape(_sect_cap_array, [_nb_sect, 1])
    _cap_error_tensor = tf.nn.relu(_est_section_flow - _sect_cap_tensor / K_PERSON_CONVERTER)
    _cap_error_tensor_plot = _est_section_flow - _sect_cap_tensor / K_PERSON_CONVERTER
    _sect_constraint = tf.reduce_sum(tf.multiply(_sect_multiplier_tensor, _cap_error_tensor), axis=0) + \
                       tf.reduce_sum(tf.multiply(_c_sect_multiplier_tensor / 2,
                                                 tf.square(tf.nn.relu(_cap_error_tensor))), axis=0)
    _mae_sect_gap = tf.reduce_sum(tf.nn.relu(_cap_error_tensor_plot), axis=0)  # mean absolute error

    return _sect_constraint, _mae_sect_gap


def loss_func_od_flow_coupling_constraint(_est_od_flow_nl_tensor, _est_od_flow_reg_tensor,
                                          _od_lag_mult_list, _c_od_mult_list):
    _nb_od = len(_est_od_flow_nl_tensor)

    _od_multiplier_tensor = tf.concat(_od_lag_mult_list, axis=0)
    _od_multiplier_tensor = tf.reshape(_od_multiplier_tensor, shape=[_nb_od, 1])

    _c_od_multiplier_tensor = tf.concat(_c_od_mult_list, axis=0)
    _c_od_multiplier_tensor = tf.reshape(_c_od_multiplier_tensor, shape=[_nb_od, 1])

    _od_flow_error_tensor = _est_od_flow_reg_tensor - _est_od_flow_nl_tensor
    _od_flow_coupling_constraint = tf.reduce_sum(tf.multiply(_od_multiplier_tensor, _od_flow_error_tensor), axis=0) + \
                                   tf.reduce_sum(tf.multiply(_c_od_multiplier_tensor / 2,
                                                             tf.square(_od_flow_error_tensor)), axis=0)
    _mse_od_gap = tf.reduce_sum(tf.square(_od_flow_error_tensor), axis=0)
    # _mse_od_gap is used to plot the convergence of the od flow coupling constraint
    return _od_flow_coupling_constraint, _mse_od_gap


def update_section_multiplier(_est_sect_flow_tensor, _sect_lag_mult_list, _c_sect_mult_list,
                              _sect_lag_mult_updating_step_list,
                              _training_sample_sect_df):
    _nb_sect = len(_est_sect_flow_tensor)

    # Update sect_mult using fixed step
    _sect_lag_mult_updating_step_tensor = tf.concat(_sect_lag_mult_updating_step_list, axis=0)
    _sect_lag_mult_updating_step_tensor = tf.reshape(_sect_lag_mult_updating_step_tensor, shape=[_nb_sect, 1])

    _sect_cap = _training_sample_sect_df['CAPACITY']
    _sect_cap = _sect_cap.values.astype('float64')
    _sect_cap_tensor = tf.reshape(_sect_cap, [_nb_sect, 1])
    # _sect_cap_error = tf.nn.relu(_est_sect_flow_tensor - _sect_cap_tensor)
    _sect_cap_error = _est_sect_flow_tensor * K_PERSON_CONVERTER - _sect_cap_tensor
    _update_value_tensor = tf.multiply(_sect_cap_error, _sect_lag_mult_updating_step_tensor)

    for _lamda_id in range(len(_sect_lag_mult_list)):
        _section_multiplier = _sect_lag_mult_list[_lamda_id]
        _sect_lag_mult_list[_lamda_id].assign(_section_multiplier + _update_value_tensor[_lamda_id])
        _sect_lag_mult_list[_lamda_id].assign(tf.nn.relu(_sect_lag_mult_list[_lamda_id]))

    # Update c_sect_mult using variable step
    for _error_id in range(len(_sect_cap_error)):
        error_abs = _sect_cap_error[_error_id]
        if error_abs > SECT_PENALTY_THRESHOLD:
            _c_sect_mult_list[_error_id] = _c_sect_mult_list[_error_id] * C_OD_MULT_UPDATING_STEP


def update_od_multiplier(_est_od_flow_nl_tensor, _est_od_flow_reg_tensor, _od_lag_mult_list,
                         _od_lag_mult_updating_step_list, _c_od_mult_list):
    _nb_od = len(_est_od_flow_nl_tensor)

    # Update od_lag_mult using fixed step
    _od_lag_mult_updating_step_tensor = tf.concat(_od_lag_mult_updating_step_list, axis=0)
    _od_lag_mult_updating_step_tensor = tf.reshape(_od_lag_mult_updating_step_tensor, shape=[_nb_od, 1])
    _od_flow_error = _est_od_flow_reg_tensor - _est_od_flow_nl_tensor
    _update_value_tensor = tf.multiply(_od_flow_error, _od_lag_mult_updating_step_tensor)
    for _lamda_id in range(len(_od_lag_mult_list)):
        _od_multiplier = _od_lag_mult_list[_lamda_id]
        _od_lag_mult_list[_lamda_id].assign(_od_multiplier + _update_value_tensor[_lamda_id])

    # Update c_od_mult_list using variable step
    _abs_od_flow_error = [abs(error) for error in _od_flow_error]
    for _error_id in range(len(_abs_od_flow_error)):
        error_abs = _abs_od_flow_error[_error_id]
        if error_abs > PENALTY_THRESHOLD:
            _c_od_mult_list[_error_id] = _c_od_mult_list[_error_id] * C_OD_MULT_UPDATING_STEP

    return _c_od_mult_list, _od_lag_mult_list


def get_od_flow_gap(_est_od_flow_nl_tensor, _est_od_flow_reg_tensor):
    _od_flow_error = _est_od_flow_reg_tensor - _est_od_flow_nl_tensor
    _abs_od_flow_error = tf.abs(_od_flow_error)
    _percentage_gap = tf.math.divide(_abs_od_flow_error, _est_od_flow_nl_tensor + NON_ZERO)
    return _percentage_gap


# calculating r-square
def calc_r_square(est_flow, psg_mean, num_shape):
    psg_mean1 = tf.reshape(psg_mean, [num_shape, 1])
    error1 = tf.reduce_sum(tf.square(est_flow - psg_mean1))
    error2 = tf.reduce_sum(tf.square(psg_mean1 - tf.reduce_mean(psg_mean1)))
    return 1 - error1 / error2


def calc_mean_psg_df(df, groupby_list):
    mean_psg_df = df.groupby(groupby_list).agg({'PSGNUM': 'mean'})
    mean_psg = mean_psg_df['PSGNUM'] / K_PERSON_CONVERTER
    mean_psg = mean_psg.values.astype('float64')
    return mean_psg


if __name__ == '__main__':
    # ===Step 1: Reading data ===
    ticket_sale_data_df = pd.read_csv('ticket_sale_data.csv')
    print("Step 1: Reading data...")
    print("Initialization: Fetch line plan from ticket sale data...")
    ticket_sale_data_df = fetch_line_from_ticket(ticket_sale_data_df)

    print("Level 1 : fetch boarding and alight PSGNUM at stations ... ")
    daily_stat_board_df, daily_stat_alight_df, total_stat_board_df, total_stat_alight_df = \
        calc_board_alight(ticket_sale_data_df)

    print("Level 2 : fetch OD PSGNUM and od-based ECO_FAC from ticket sale data... ")
    daily_obs_od_flow_df = calc_obs_od_flow(ticket_sale_data_df)
    daily_obs_od_df, total_obs_od_df = calc_od_eco_factor(daily_obs_od_flow_df)

    print("Level 3 : fetch OD line PSGNUM, PRICENUM, FREQUNCY, TRAVEL TIME, LINE_SECTION, AVG_PROB from ticket data")
    daily_od_line_df, total_od_line_df = fetch_od_line_from_ticket(ticket_sale_data_df)

    print("Level 4 : fetch LINE_SECTION from ticket sale data ... ")
    daily_sect_df = fetch_line_section(ticket_sale_data_df)

    print("Supplement of daily OD FREQUENCY ... ")
    date_line_freq_dict = daily_od_line_df.groupby(['OD', 'BOARDDATE'])['FREQUENCY'].sum().to_dict()
    daily_obs_od_df['FREQUENCY'] = \
        daily_obs_od_df.apply(lambda x: date_line_freq_dict[(x['OD'], x['BOARDDATE'])], axis=1)

    PARAMS['nb_origin'] = len(g_board_station_list)
    PARAMS['nb_destination'] = len(g_alight_station_list)
    PARAMS['nb_od'] = len(g_od_list)
    PARAMS['nb_od_line'] = len(g_od_line_list)
    PARAMS['nb_section'] = len(g_section_list)

    print("Step 2: Create incidence matrices for the hierarchical network...")
    o_to_od_inc_mat, od_to_od_line_inc_mat, line_to_destination_inc_mat, line_to_section_inc_mat = \
        create_inc_mat(total_obs_od_df, total_od_line_df)
    # od_to_line_inc_mat = tf.cast(od_to_od_line_inc_mat, tf.float64)

    print("Step 3: Define variables for the hierarchical network...")
    (est_boarding_list, od_theta_list, o_eco_coeff_list, od_freq_coeff_list, od_tt_coeff_list, od_price_coeff_list,
     reg_od_freq_coeff_list, od_lag_mult_list, od_lag_mult_updating_step_list, c_od_mult_list,
     sect_lag_mult_list, sect_lag_mult_updating_step_list, c_sect_mult_list) = \
        define_variables(total_obs_od_df, total_od_line_df)
    print(type(est_boarding_list))
    prob_var_list = od_theta_list + o_eco_coeff_list + od_freq_coeff_list + od_tt_coeff_list + od_price_coeff_list
    all_variable_list = prob_var_list + est_boarding_list

    print("Step 4: generate training and testing samples...")
    all_sample_list = daily_stat_board_df['BOARDDATE'].unique().tolist()
    total_samples = len(all_sample_list)
    train_sample_number = int(total_samples * TRAIN_SAMPLE_RATIO)
    training_sample_list = all_sample_list[:train_sample_number]
    validation_sample_list = all_sample_list[train_sample_number:]

    # generate training samples
    training_daily_stat_board_df = daily_stat_board_df[daily_stat_board_df['BOARDDATE'].isin(training_sample_list)]
    training_daily_stat_alight_df = daily_stat_alight_df[daily_stat_alight_df['BOARDDATE'].isin(training_sample_list)]
    training_daily_obs_od_df = daily_obs_od_df[daily_obs_od_df['BOARDDATE'].isin(training_sample_list)]
    training_daily_od_line_df = daily_od_line_df[daily_od_line_df['BOARDDATE'].isin(training_sample_list)]
    training_daily_sect_df = daily_sect_df[daily_sect_df['BOARDDATE'].isin(training_sample_list)]
    training_daily_stat_board_df = training_daily_stat_board_df.reset_index(drop=True)
    training_daily_stat_alight_df = training_daily_stat_alight_df.reset_index(drop=True)
    training_daily_obs_od_df = training_daily_obs_od_df.reset_index(drop=True)
    training_daily_od_line_df = training_daily_od_line_df.reset_index(drop=True)
    training_daily_sect_df = training_daily_sect_df.reset_index(drop=True)

    # generate validation samples
    valid_daily_stat_board_df = daily_stat_board_df[daily_stat_board_df['BOARDDATE'].isin(validation_sample_list)]
    valid_daily_stat_alight_df = daily_stat_alight_df[daily_stat_alight_df['BOARDDATE'].isin(validation_sample_list)]
    valid_daily_obs_od_df = daily_obs_od_df[daily_obs_od_df['BOARDDATE'].isin(validation_sample_list)]
    valid_daily_od_line_df = daily_od_line_df[daily_od_line_df['BOARDDATE'].isin(validation_sample_list)]
    valid_daily_sect_df = daily_sect_df[daily_sect_df['BOARDDATE'].isin(validation_sample_list)]
    valid_daily_stat_board_df = valid_daily_stat_board_df.reset_index(drop=True)
    valid_daily_stat_alight_df = valid_daily_stat_alight_df.reset_index(drop=True)
    valid_daily_obs_od_df = valid_daily_obs_od_df.reset_index(drop=True)
    valid_daily_od_line_df = valid_daily_od_line_df.reset_index(drop=True)
    valid_daily_sect_df = valid_daily_sect_df.reset_index(drop=True)

    print("Step 5: Training process...")

    # ============================== Settings of training process ================================
    # step 1: define optimizers
    # opt_nl = tf.keras.optimizers.Adagrad(learning_rate=lr_nl)
    # opt_nl = tf.keras.optimizers.SGD(learning_rate=lr_nl, momentum=0.9)
    opt_nl = tf.keras.optimizers.Adam(learning_rate=lr_nl, weight_decay=1e-5)
    opt_boarding = tf.keras.optimizers.Adam(learning_rate=lr_boarding, weight_decay=1e-5)
    opt_regression = tf.keras.optimizers.Adam(learning_rate=lr_regression)
    opt_integrating = tf.keras.optimizers.Adam(learning_rate=lr_integrating)

    # ============================== Variables and functions in model ================================
    # step 2: connect variables with optimizers
    opt_nl.build(est_boarding_list + prob_var_list + reg_od_freq_coeff_list)
    opt_boarding.build(est_boarding_list + prob_var_list + reg_od_freq_coeff_list)
    opt_regression.build(est_boarding_list + prob_var_list + reg_od_freq_coeff_list)
    opt_integrating.build(est_boarding_list + prob_var_list + reg_od_freq_coeff_list)

    # step 3: define list of loss function for generating loss figure
    plot_loss_total_list = []  # total_loss_plot for generating total loss figure
    plot_loss_boarding_list = []  # boarding_loss_plot for generating boarding loss figure
    plot_loss_nest_logit_list = []  # nest_logit_loss_plot for generating nest logit loss figure
    plot_loss_alighting_list = []  # alighting_loss_plot for generating alighting loss figure
    plot_loss_section_constraint_list = []  # section_constraint_loss_plot for generating section constraint loss figure
    plot_loss_od_flow_coupling_constraint_list = []  # od_flow_coupling_constraint_loss_plot for generating od flow coupling constraint loss figure
    # ============================== Statistics related goodness of fit ================================

    # step 4: fetch data for calculating r-square
    mean_training_board = calc_mean_psg_df(training_daily_stat_board_df, ['DEP_STATION_ID', "DEP_STATION"])
    mean_training_alight = calc_mean_psg_df(training_daily_stat_alight_df, ['ARR_STATION_ID', "ARR_STATION"])
    mean_training_od_psg = calc_mean_psg_df(training_daily_obs_od_df, ['OD_ID', 'OD'])
    mean_training_od_line_psg = calc_mean_psg_df(training_daily_od_line_df, ['OD_LINE_ID', 'OD_LINE'])

    boarding_r_square_list = []
    od_flow_r_square_list = []
    od_line_flow_r_square_list = []
    alighting_r_square_list = []

    # step 5: calculating initial value of loglikelihood
    init_log_likelihood = init_loglikelihood(training_sample_list, training_daily_stat_board_df,
                                             training_daily_stat_alight_df, training_daily_obs_od_df,
                                             training_daily_od_line_df, training_daily_sect_df,
                                             od_theta_list, o_eco_coeff_list, od_freq_coeff_list, od_tt_coeff_list,
                                             od_price_coeff_list)

    # ================================= Training process ===================================
    flag_nested_logit_learning = 1  # Whether to learn variables in prob_var_list and est_boarding_list,
    flag_flow_learning = 0  # Whether to learn variables in hierarch_net_var_list,
    flag_coupling_learning = 0  # Whether to learn variables in reg_od_freq_coeff_list,
    flag_integrating_learning = 0  # Whether to learn all variables in the model
    epochs = TOTAL_EPOCHS
    nested_logit_step = 0
    flow_learning_step = 0
    coupling_learning_step = 0
    integrating_learning_step = 0
    for epoch in range(epochs):
        batch_training_board_data, batch_training_alight_data, batch_training_od_data, batch_training_od_line_data, \
            batch_training_sample_list = get_batch_sample(training_daily_stat_board_df, training_daily_stat_alight_df,
                                                          training_daily_obs_od_df, training_daily_od_line_df,
                                                          training_sample_list, BATCH_SIZE)
        # ======================= stage 1: Learn variables in prob_var_list ============================
        if flag_nested_logit_learning == 1:
            with tf.GradientTape() as tape1:
                # initial value of each kind of loss function
                weighted_loss = 0
                loss_plot = 0
                total_loss_nest_logit = 0
                total_loss_boarding = 0
                total_loss_alighting = 0
                total_loss_section_constraint = 0
                total_loss_od_flow_coupling_constraint = 0

                for sample_id in batch_training_sample_list:
                    # origin_psg for calculate boarding loss
                    # _od_line_param, _od_param for calculate probability by function nested_logit_model

                    boarding_psg_array, od_psg_array, od_line_psg_array, alighting_psg_array, od_freq_reg_array, \
                        od_eco_array, od_line_param_array, section_capacity_array = \
                        select_single_day_sample(sample_id, training_daily_stat_board_df, training_daily_stat_alight_df,
                                                 training_daily_obs_od_df, training_daily_od_line_df,
                                                 training_daily_sect_df)

                    # calculate probability by function nested_logit_model in this sample
                    od_prob_tensor, conditional_od_line_prob_tensor, final_od_line_prob_tensor = \
                        calc_prob_nested_logit_model(od_theta_list, o_eco_coeff_list, od_freq_coeff_list,
                                                     od_tt_coeff_list, od_price_coeff_list, od_eco_array,
                                                     od_line_param_array)

                    # estimate flow by function est_flow_by_nested_logit_model in this sample
                    # including boarding, alighting, od, od line and section flow
                    est_board_tensor, est_od_flow_nl_tensor, \
                        est_line_flow_tensor, est_alight_tensor, est_sect_flow_tensor = \
                        est_flow_by_nested_logit_model(est_boarding_list, od_prob_tensor,
                                                       conditional_od_line_prob_tensor)

                    # estimate flow by function est_flow_by_regression in this sample
                    # reg_od_freq_coeff_list can be extended to general parameters metrics
                    est_od_flow_reg_tensor = \
                        est_od_flow_by_regression(reg_od_freq_coeff_list, od_freq_reg_array)

                    training_sample_sect_df = training_daily_sect_df[
                        training_daily_sect_df['BOARDDATE'] == sample_id]

                    # calculate loss function
                    # MLE of nest logit, MSE of boarding and alighting,
                    # section capacity constraint, od flow coupling constraint
                    loss_nest_logit = calc_loss_loglikelihood(final_od_line_prob_tensor, od_line_psg_array)
                    loss_boarding = loss_func_boarding_mse(est_board_tensor, boarding_psg_array)
                    loss_alighting = loss_func_alighting_mse(est_alight_tensor, alighting_psg_array)
                    loss_section_constraint, mae_sect_gap = \
                        loss_func_sect_cap_constraint(est_sect_flow_tensor, sect_lag_mult_list,
                                                      c_sect_mult_list,
                                                      training_sample_sect_df)  # Xinyu W, training_daily_sect_df --> training_sample_sect_df

                    loss_od_flow_coupling_constraint, mse_od_gap = \
                        loss_func_od_flow_coupling_constraint(est_od_flow_nl_tensor, est_od_flow_reg_tensor,
                                                              od_lag_mult_list, c_od_mult_list)

                    weighted_loss += W_PROB_NL * loss_nest_logit + W_PROB_BOARD * loss_boarding + W_PROB_ALIGHT * loss_alighting + \
                                     W_PROB_CAP * loss_section_constraint + W_PROB_LR * loss_od_flow_coupling_constraint
                    # loss_plot for generating total loss figure
                    loss_plot += loss_nest_logit + loss_boarding + loss_alighting + \
                                 mae_sect_gap + mse_od_gap

                    total_loss_nest_logit += loss_nest_logit
                    total_loss_boarding += loss_boarding
                    total_loss_alighting += loss_alighting
                    total_loss_section_constraint += mae_sect_gap  # use mse to depict convergence
                    total_loss_od_flow_coupling_constraint += mse_od_gap

                # calculate r_square
                nb_origin = len(boarding_psg_array)
                nb_od = len(od_psg_array)
                nb_od_line = len(od_line_psg_array)
                nb_destination = len(alighting_psg_array)

                boarding_r_square = calc_r_square(est_board_tensor, mean_training_board, nb_origin)
                od_flow_r_square = calc_r_square(est_od_flow_nl_tensor, mean_training_od_psg, nb_od)
                od_line_flow_r_square = calc_r_square(est_line_flow_tensor, mean_training_od_line_psg, nb_od_line)
                alighting_r_square = calc_r_square(est_alight_tensor, mean_training_alight, nb_destination)

                boarding_r_square_list.append(boarding_r_square)
                od_flow_r_square_list.append(od_flow_r_square)
                od_line_flow_r_square_list.append(od_line_flow_r_square)
                alighting_r_square_list.append(alighting_r_square)

                # loss for generating each loss figure
                plot_loss_nest_logit_list.append(total_loss_nest_logit.numpy()[0] / BATCH_SIZE)
                plot_loss_boarding_list.append(total_loss_boarding.numpy()[0] / BATCH_SIZE)
                plot_loss_alighting_list.append(total_loss_alighting.numpy()[0] / BATCH_SIZE)
                plot_loss_section_constraint_list.append(total_loss_section_constraint.numpy()[0] / BATCH_SIZE)
                plot_loss_od_flow_coupling_constraint_list.append(
                    total_loss_od_flow_coupling_constraint.numpy()[0] / BATCH_SIZE)
                plot_loss_total_list.append(loss_plot.numpy()[0] / BATCH_SIZE)

                # loss for gradient
                stage1_loss = weighted_loss / BATCH_SIZE

            prob_grads = tape1.gradient(stage1_loss, prob_var_list)
            if np.isnan(tf.reduce_mean(prob_grads).numpy()):
                print(1)
            opt_nl.apply_gradients(zip(prob_grads, prob_var_list))
            nested_logit_step = nested_logit_step + 1
            # if tf.reduce_mean(prob_grads).numpy() is nan, stop the trainin

            print('total_epoch:', epoch, ', nested_logit_step:', nested_logit_step, ', mean_grads:',
                  tf.reduce_mean(prob_grads).numpy())
            print('total_epoch:', epoch, ', nested_logit_step:',
                  nested_logit_step, ', total_weighted_loss:', weighted_loss[0].numpy() / BATCH_SIZE)

            if nested_logit_step == MAX_NESTED_LOGIT_STEP:
                flag_nested_logit_learning = 0
                flag_flow_learning = 1
                flag_coupling_learning = 0
                flag_integrating_learning = 0
                nested_logit_step = 0
                continue
            if abs(tf.reduce_mean(prob_grads).numpy()) < MIN_GRADIENT:
                flag_nested_logit_learning = 0
                flag_flow_learning = 1
                flag_coupling_learning = 0
                flag_integrating_learning = 0
                nested_logit_step = 0
                continue

        # ====================== stage 2: Learn variables in est_boarding_list =========================
        if flag_flow_learning == 1:
            with tf.GradientTape() as tape2:
                # initial value of each kind of loss function
                weighted_loss = 0
                loss_plot = 0
                total_loss_nest_logit = 0
                total_loss_boarding = 0
                total_loss_alighting = 0
                total_loss_section_constraint = 0
                total_loss_od_flow_coupling_constraint = 0
                for sample_id in batch_training_sample_list:
                    boarding_psg_array, od_psg_array, od_line_psg_array, alighting_psg_array, od_freq_reg_array, \
                        od_eco_array, od_line_param_array, section_capacity_array = \
                        select_single_day_sample(sample_id, training_daily_stat_board_df, training_daily_stat_alight_df,
                                                 training_daily_obs_od_df, training_daily_od_line_df,
                                                 training_daily_sect_df)

                    # calculate probability by function nested_logit_model in this sample
                    od_prob_tensor, conditional_od_line_prob_tensor, final_od_line_prob_tensor = \
                        calc_prob_nested_logit_model(od_theta_list, o_eco_coeff_list, od_freq_coeff_list,
                                                     od_tt_coeff_list,
                                                     od_price_coeff_list, od_eco_array, od_line_param_array)

                    # estimate flow by function est_flow_by_nested_logit_model in this sample
                    est_board_tensor, est_od_flow_nl_tensor, \
                        est_line_flow_tensor, est_alight_tensor, est_sect_flow_tensor = \
                        est_flow_by_nested_logit_model(est_boarding_list, od_prob_tensor,
                                                       conditional_od_line_prob_tensor)

                    # estimate flow by function est_flow_by_regression in this sample
                    # reg_od_freq_coeff_list can be extended to general parameters metrics
                    est_od_flow_reg_tensor = \
                        est_od_flow_by_regression(reg_od_freq_coeff_list, od_freq_reg_array)

                    training_sample_sect_df = training_daily_sect_df[
                        training_daily_sect_df['BOARDDATE'] == sample_id]

                    # calculate loss function
                    loss_nest_logit = calc_loss_loglikelihood(final_od_line_prob_tensor, od_line_psg_array)
                    loss_boarding = loss_func_boarding_mse(est_board_tensor, boarding_psg_array)
                    loss_alighting = loss_func_alighting_mse(est_alight_tensor, alighting_psg_array)
                    loss_section_constraint, mae_sect_gap = \
                        loss_func_sect_cap_constraint(est_sect_flow_tensor, sect_lag_mult_list,
                                                      c_sect_mult_list,
                                                      training_sample_sect_df)  # Xinyu W, training_daily_sect_df --> training_sample_sect_df

                    loss_od_flow_coupling_constraint, mse_od_gap = \
                        loss_func_od_flow_coupling_constraint(est_od_flow_nl_tensor, est_od_flow_reg_tensor,
                                                              od_lag_mult_list, c_od_mult_list)

                    weighted_loss += W_FLOW_NL * loss_nest_logit + W_FLOW_BOARD * loss_boarding + \
                                     W_FLOW_ALIGHT * loss_alighting + W_FLOW_CAP * loss_section_constraint + \
                                     W_FLOW_LR * loss_od_flow_coupling_constraint

                    # loss_plot for generating total loss figure
                    loss_plot += loss_nest_logit + loss_boarding + loss_alighting + \
                                 mae_sect_gap + mse_od_gap

                    total_loss_nest_logit += loss_nest_logit
                    total_loss_boarding += loss_boarding
                    total_loss_alighting += loss_alighting
                    total_loss_section_constraint += mae_sect_gap
                    total_loss_od_flow_coupling_constraint += mse_od_gap

                # calculate r_square
                nb_origin = len(boarding_psg_array)
                nb_od = len(od_psg_array)
                nb_od_line = len(od_line_psg_array)
                nb_destination = len(alighting_psg_array)

                boarding_r_square = calc_r_square(est_board_tensor, mean_training_board, nb_origin)
                od_flow_r_square = calc_r_square(est_od_flow_nl_tensor, mean_training_od_psg, nb_od)
                od_line_flow_r_square = calc_r_square(est_line_flow_tensor, mean_training_od_line_psg, nb_od_line)
                alighting_r_square = calc_r_square(est_alight_tensor, mean_training_alight, nb_destination)

                boarding_r_square_list.append(boarding_r_square)
                od_flow_r_square_list.append(od_flow_r_square)
                od_line_flow_r_square_list.append(od_line_flow_r_square)
                alighting_r_square_list.append(alighting_r_square)

                # loss for generating each loss figure
                plot_loss_nest_logit_list.append(total_loss_nest_logit.numpy()[0] / BATCH_SIZE)
                plot_loss_boarding_list.append(total_loss_boarding.numpy()[0] / BATCH_SIZE)
                plot_loss_alighting_list.append(total_loss_alighting.numpy()[0] / BATCH_SIZE)
                plot_loss_section_constraint_list.append(total_loss_section_constraint.numpy()[0] / BATCH_SIZE)
                plot_loss_od_flow_coupling_constraint_list.append(
                    total_loss_od_flow_coupling_constraint.numpy()[0] / BATCH_SIZE)
                plot_loss_total_list.append(loss_plot.numpy()[0] / BATCH_SIZE)

                # loss for gradient
                stage2_loss = weighted_loss / BATCH_SIZE

            boarding_grads = tape2.gradient(stage2_loss, est_boarding_list)
            if np.isnan(tf.reduce_mean(boarding_grads).numpy()):
                print(1)
            opt_boarding.apply_gradients(zip(boarding_grads, est_boarding_list))
            update_section_multiplier(est_sect_flow_tensor, sect_lag_mult_list, c_sect_mult_list,
                                      sect_lag_mult_updating_step_list,
                                      training_sample_sect_df)
            flow_learning_step = flow_learning_step + 1

            print('total_epoch:', epoch, ', hierarch_netflow_step:', flow_learning_step, ', mean_grads:',
                  tf.reduce_mean(boarding_grads).numpy())
            print('total_epoch:', epoch, ', hierarch_netflow_step:', flow_learning_step, ', total_weighted_loss:',
                  weighted_loss[0].numpy() / BATCH_SIZE)

            if flow_learning_step == MAX_OD_FLOW_STEP:
                flag_hierarch_net_learning = 0
                flag_flow_learning = 0
                flag_coupling_learning = 1
                flag_integrating_learning = 0
                flow_learning_step = 0
                print('epoch:', epoch, ', change_to_regression')
                continue
            if abs(tf.reduce_mean(boarding_grads).numpy()) < MIN_GRADIENT:
                flag_hierarch_net_learning = 0
                flag_flow_learning = 0
                flag_coupling_learning = 1
                flag_integrating_learning = 0
                flow_learning_step = 0
                continue
        # ====================== stage 3: Learn variables in reg_od_freq_coeff_list =========================
        if flag_coupling_learning == 1:
            with tf.GradientTape() as tape3:
                weighted_loss = 0
                loss_plot = 0
                total_loss_nest_logit = 0
                total_loss_boarding = 0
                total_loss_alighting = 0
                total_loss_section_constraint = 0
                total_loss_od_flow_coupling_constraint = 0

                for sample_id in batch_training_sample_list:
                    boarding_psg_array, od_psg_array, od_line_psg_array, alighting_psg_array, od_freq_reg_array, \
                        od_eco_array, od_line_param_array, section_capacity_array = \
                        select_single_day_sample(sample_id, training_daily_stat_board_df, training_daily_stat_alight_df,
                                                 training_daily_obs_od_df, training_daily_od_line_df,
                                                 training_daily_sect_df)

                    # calculate probability by function nested_logit_model in this sample
                    od_prob_tensor, conditional_od_line_prob_tensor, final_od_line_prob_tensor = \
                        calc_prob_nested_logit_model(od_theta_list, o_eco_coeff_list, od_freq_coeff_list,
                                                     od_tt_coeff_list, od_price_coeff_list, od_eco_array,
                                                     od_line_param_array)

                    # estimate flow by function est_flow_by_nested_logit_model in this sample
                    est_board_tensor, est_od_flow_nl_tensor, \
                        est_line_flow_tensor, est_alight_tensor, est_sect_flow_tensor = \
                        est_flow_by_nested_logit_model(est_boarding_list, od_prob_tensor,
                                                       conditional_od_line_prob_tensor)

                    # estimate flow by function est_flow_by_regression in this sample
                    # reg_od_freq_coeff_list can be extended to general parameters metrics
                    est_od_flow_reg_tensor = \
                        est_od_flow_by_regression(reg_od_freq_coeff_list, od_freq_reg_array)

                    training_sample_sect_df = training_daily_sect_df[
                        training_daily_sect_df['BOARDDATE'] == sample_id]  # Xinyu W

                    # calculate loss function
                    loss_nest_logit = calc_loss_loglikelihood(final_od_line_prob_tensor, od_line_psg_array)
                    loss_boarding = loss_func_boarding_mse(est_board_tensor, boarding_psg_array)
                    loss_alighting = loss_func_alighting_mse(est_alight_tensor, alighting_psg_array)
                    loss_section_constraint, mae_sect_gap = \
                        loss_func_sect_cap_constraint(est_sect_flow_tensor, sect_lag_mult_list,
                                                      c_sect_mult_list,
                                                      training_sample_sect_df)

                    loss_od_flow_coupling_constraint, mse_od_gap = \
                        loss_func_od_flow_coupling_constraint(est_od_flow_nl_tensor, est_od_flow_reg_tensor,
                                                              od_lag_mult_list, c_od_mult_list)

                    weighted_loss += W_COUPLE_NL * loss_nest_logit + W_COUPLE_BOARD * loss_boarding \
                                     + W_COUPLE_ALIGHT * loss_alighting + W_COUPLE_CAP * loss_section_constraint \
                                     + W_COUPLE_LR * loss_od_flow_coupling_constraint

                    # loss_plot for generating total loss figure
                    loss_plot += loss_nest_logit + loss_boarding + loss_alighting \
                                 + mae_sect_gap + mse_od_gap

                    total_loss_nest_logit += loss_nest_logit
                    total_loss_boarding += loss_boarding
                    total_loss_alighting += loss_alighting
                    total_loss_section_constraint += mae_sect_gap
                    total_loss_od_flow_coupling_constraint += mse_od_gap

                # loss for gradient
                stage3_loss = weighted_loss / BATCH_SIZE

                # calculate r_square
                nb_origin = len(boarding_psg_array)
                nb_od = len(od_psg_array)
                nb_od_line = len(od_line_psg_array)
                nb_destination = len(alighting_psg_array)

                boarding_r_square = calc_r_square(est_board_tensor, mean_training_board, nb_origin)
                od_flow_r_square = calc_r_square(est_od_flow_nl_tensor, mean_training_od_psg, nb_od)
                od_line_flow_r_square = calc_r_square(est_line_flow_tensor, mean_training_od_line_psg, nb_od_line)
                alighting_r_square = calc_r_square(est_alight_tensor, mean_training_alight, nb_destination)

                boarding_r_square_list.append(boarding_r_square)
                od_flow_r_square_list.append(od_flow_r_square)
                od_line_flow_r_square_list.append(od_line_flow_r_square)
                alighting_r_square_list.append(alighting_r_square)

                # loss for generating each loss figure
                plot_loss_nest_logit_list.append(total_loss_nest_logit.numpy()[0] / BATCH_SIZE)
                plot_loss_boarding_list.append(total_loss_boarding.numpy()[0] / BATCH_SIZE)
                plot_loss_alighting_list.append(total_loss_alighting.numpy()[0] / BATCH_SIZE)
                plot_loss_section_constraint_list.append(total_loss_section_constraint.numpy()[0] / BATCH_SIZE)
                plot_loss_od_flow_coupling_constraint_list.append(
                    total_loss_od_flow_coupling_constraint.numpy()[0] / BATCH_SIZE)
                plot_loss_total_list.append(loss_plot.numpy()[0] / BATCH_SIZE)

            coupling_grads = tape3.gradient(stage3_loss, reg_od_freq_coeff_list)
            opt_regression.apply_gradients(zip(coupling_grads, reg_od_freq_coeff_list))

            c_od_mult_list, od_lag_mult_list = \
                update_od_multiplier(est_od_flow_nl_tensor, est_od_flow_reg_tensor,
                                     od_lag_mult_list, od_lag_mult_updating_step_list, c_od_mult_list)
            # print(od_lag_mult_list)
            # print(od_lag_mult_updating_step_list)
            od_flow_gap = get_od_flow_gap(est_od_flow_nl_tensor, est_od_flow_reg_tensor)
            coupling_learning_step = coupling_learning_step + 1

            print('total_epoch:', epoch, ', regression_step:', coupling_learning_step, ', mean_grads:',
                  tf.reduce_mean(coupling_grads).numpy())
            print('Linear_Regression_epoch:', epoch, 'od_flow_gap:', tf.reduce_mean(od_flow_gap).numpy())
            print('total_epoch:', epoch, ', regression_step:', coupling_learning_step, ', total_weighted_loss:',
                  weighted_loss[0].numpy() / BATCH_SIZE)

            if coupling_learning_step == MAX_COUPLING_STEP:
                flag_nested_logit_learning = 0
                flag_flow_learning = 0
                flag_coupling_learning = 0
                flag_integrating_learning = 1
                coupling_learning_step = 0
                continue

            if max(od_flow_gap) <= THRESHOLD_MAX_COUPLING_GAP:
                flag_nested_logit_learning = 0
                flag_flow_learning = 0
                flag_coupling_learning = 0
                flag_integrating_learning = 1
                coupling_learning_step = 0
                continue

        # ====================== stage 4: learning all variables together =========================
        if flag_integrating_learning == 1:
            with tf.GradientTape() as tape4:
                weighted_loss = 0
                loss_plot = 0
                total_loss_nest_logit = 0
                total_loss_boarding = 0
                total_loss_alighting = 0
                total_loss_section_constraint = 0
                total_loss_od_flow_coupling_constraint = 0
                # renew od_lag_mult_list and c_od_mult_list to initial value
                # for i in range(len(od_lag_mult_list)):
                #     od_lag_mult_list[i] = tf.Variable(INIT_SECT_LAG_MULT, trainable=True)

                for sample_id in batch_training_sample_list:
                    boarding_psg_array, od_psg_array, od_line_psg_array, alighting_psg_array, od_freq_reg_array, \
                        od_eco_array, od_line_param_array, section_capacity_array = \
                        select_single_day_sample(sample_id, training_daily_stat_board_df, training_daily_stat_alight_df,
                                                 training_daily_obs_od_df, training_daily_od_line_df,
                                                 training_daily_sect_df)

                    # calculate probability by function nested_logit_model in this sample
                    od_prob_tensor, conditional_od_line_prob_tensor, final_od_line_prob_tensor = \
                        calc_prob_nested_logit_model(od_theta_list, o_eco_coeff_list, od_freq_coeff_list,
                                                     od_tt_coeff_list, od_price_coeff_list, od_eco_array,
                                                     od_line_param_array)

                    # estimate flow by function est_flow_by_nested_logit_model in this sample
                    est_board_tensor, est_od_flow_nl_tensor, \
                        est_line_flow_tensor, est_alight_tensor, est_sect_flow_tensor = \
                        est_flow_by_nested_logit_model(est_boarding_list, od_prob_tensor,
                                                       conditional_od_line_prob_tensor)

                    # estimate flow by function est_flow_by_regression in this sample
                    # reg_od_freq_coeff_list can be extended to general parameters metrics
                    est_od_flow_reg_tensor = \
                        est_od_flow_by_regression(reg_od_freq_coeff_list, od_freq_reg_array)

                    training_sample_sect_df = training_daily_sect_df[
                        training_daily_sect_df['BOARDDATE'] == sample_id]  # Xinyu W

                    # calculate loss function
                    loss_nest_logit = calc_loss_loglikelihood(final_od_line_prob_tensor, od_line_psg_array)
                    loss_boarding = loss_func_boarding_mse(est_board_tensor, boarding_psg_array)
                    loss_alighting = loss_func_alighting_mse(est_alight_tensor, alighting_psg_array)
                    loss_section_constraint, mae_sect_gap = \
                        loss_func_sect_cap_constraint(est_sect_flow_tensor, sect_lag_mult_list,
                                                      c_sect_mult_list,
                                                      training_sample_sect_df)

                    loss_od_flow_coupling_constraint, mse_od_gap = \
                        loss_func_od_flow_coupling_constraint(est_od_flow_nl_tensor, est_od_flow_reg_tensor,
                                                              od_lag_mult_list, c_od_mult_list)

                    weighted_loss += W_INTEGRATION_NL * loss_nest_logit + W_INTEGRATION_BOARD * loss_boarding \
                                     + W_INTEGRATION_ALIGHT * loss_alighting + W_INTEGRATION_CAP * loss_section_constraint \
                                     + W_INTEGRATION_LR * loss_od_flow_coupling_constraint

                    # loss_plot for generating total loss figure
                    loss_plot += loss_nest_logit + loss_boarding + loss_alighting \
                                 + mae_sect_gap + mse_od_gap

                    total_loss_nest_logit += loss_nest_logit
                    total_loss_boarding += loss_boarding
                    total_loss_alighting += loss_alighting
                    total_loss_section_constraint += mae_sect_gap
                    total_loss_od_flow_coupling_constraint += mse_od_gap

                # loss for gradient
                stage4_loss = weighted_loss / BATCH_SIZE
                # if weighted_loss[0].numpy() / BATCH_SIZE > 200:
                #     print(1)

                # calculate r_square
                nb_origin = len(boarding_psg_array)
                nb_od = len(od_psg_array)
                nb_od_line = len(od_line_psg_array)
                nb_destination = len(alighting_psg_array)

                boarding_r_square = calc_r_square(est_board_tensor, mean_training_board, nb_origin)
                od_flow_r_square = calc_r_square(est_od_flow_nl_tensor, mean_training_od_psg, nb_od)
                od_line_flow_r_square = calc_r_square(est_line_flow_tensor, mean_training_od_line_psg, nb_od_line)
                alighting_r_square = calc_r_square(est_alight_tensor, mean_training_alight, nb_destination)

                boarding_r_square_list.append(boarding_r_square)
                od_flow_r_square_list.append(od_flow_r_square)
                od_line_flow_r_square_list.append(od_line_flow_r_square)
                alighting_r_square_list.append(alighting_r_square)

                # loss for generating each loss figure
                plot_loss_nest_logit_list.append(total_loss_nest_logit.numpy()[0] / BATCH_SIZE)
                plot_loss_boarding_list.append(total_loss_boarding.numpy()[0] / BATCH_SIZE)
                plot_loss_alighting_list.append(total_loss_alighting.numpy()[0] / BATCH_SIZE)
                plot_loss_section_constraint_list.append(total_loss_section_constraint.numpy()[0] / BATCH_SIZE)
                plot_loss_od_flow_coupling_constraint_list.append(
                    total_loss_od_flow_coupling_constraint.numpy()[0] / BATCH_SIZE)
                plot_loss_total_list.append(loss_plot.numpy()[0] / BATCH_SIZE)

            # all_variable_list = prob_var_list + est_boarding_list
            integrating_grads = tape4.gradient(stage4_loss, all_variable_list)
            opt_integrating.apply_gradients(zip(integrating_grads, all_variable_list))

            update_section_multiplier(est_sect_flow_tensor, sect_lag_mult_list, c_sect_mult_list,
                                      sect_lag_mult_updating_step_list,
                                      training_sample_sect_df)

            c_od_mult_list, od_lag_mult_list = \
                update_od_multiplier(est_od_flow_nl_tensor, est_od_flow_reg_tensor,
                                     od_lag_mult_list, od_lag_mult_updating_step_list, c_od_mult_list)
            # print(od_lag_mult_list)
            # print(od_lag_mult_updating_step_list)
            od_flow_gap = get_od_flow_gap(est_od_flow_nl_tensor, est_od_flow_reg_tensor)
            integrating_learning_step = integrating_learning_step + 1

            print('total_epoch:', epoch, ', integrating_step:', integrating_learning_step, ', mean_grads:',
                  tf.reduce_mean(integrating_grads).numpy())
            print('total_epoch:', epoch, ', integrating_step:', integrating_learning_step, ', total_weighted_loss:',
                  weighted_loss[0].numpy() / BATCH_SIZE)

            if integrating_learning_step == MAX_ITERATION_STEP:
                flag_nested_logit_learning = 0
                flag_flow_learning = 0
                flag_coupling_learning = 0
                flag_integrating_learning = 0
                coupling_learning_step = 0
                epoch = TOTAL_EPOCHS
                continue

            if abs(tf.reduce_mean(integrating_grads).numpy()) < MIN_GRADIENT:
                flag_nested_logit_learning = 0
                flag_flow_learning = 0
                flag_coupling_learning = 0
                flag_integrating_learning = 0
                coupling_learning_step = 0
                epoch = TOTAL_EPOCHS
                continue

        final_log_likelihood = init_loglikelihood(training_sample_list, training_daily_stat_board_df,
                                                  training_daily_stat_alight_df, training_daily_obs_od_df,
                                                  training_daily_od_line_df, training_daily_sect_df,
                                                  od_theta_list, o_eco_coeff_list, od_freq_coeff_list,
                                                  od_tt_coeff_list, od_price_coeff_list)
    print('Finish training process!')

    print("Step 7: Generate loss figure...")
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 10))
    plt.subplots_adjust(wspace=0.5, hspace=0.4)

    ax1.plot(range(len(plot_loss_total_list)), plot_loss_total_list)
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('total loss function')
    ax1.grid()
    ax1.set_title('The convergence curve of total loss function', loc='center')

    ax2.plot(range(len(plot_loss_boarding_list)), np.array(plot_loss_boarding_list))
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('boarding loss function')
    ax2.grid()
    ax2.set_title('The convergence curve of boarding loss function', loc='center')

    ax3.plot(range(len(plot_loss_nest_logit_list)), np.array(plot_loss_nest_logit_list))
    ax3.set_xlabel('epochs')
    ax3.set_ylabel('NL loss function')
    ax3.grid()
    ax3.set_title('The convergence curve of NL loss function', loc='center')

    ax4.plot(range(len(plot_loss_alighting_list)), np.array(plot_loss_alighting_list))
    ax4.set_xlabel('epochs')
    ax4.set_ylabel('alighting loss function')
    ax4.grid()
    ax4.set_title('The convergence curve of alighting loss function', loc='center')

    ax5.plot(range(len(plot_loss_section_constraint_list)), np.array(plot_loss_section_constraint_list))
    ax5.set_xlabel('epochs')
    ax5.set_ylabel('section constraines function')
    ax5.grid()
    ax5.set_title('The convergence curve of section constraines function', loc='center')

    ax6.plot(range(len(plot_loss_od_flow_coupling_constraint_list)),
             np.array(plot_loss_od_flow_coupling_constraint_list))
    ax6.set_xlabel('epochs')
    ax6.set_ylabel('od flow equal constrains function')
    ax6.grid()
    ax6.set_title('The convergence curve of od flow equal constrains function', loc='center')

    print("Step 8: Generate report...")
    print('init_log_likelihood:', init_log_likelihood)
    print('final_log_likelihood:', final_log_likelihood)
    print('rou_square:', 1 - final_log_likelihood / init_log_likelihood)

    # output_folder = (f"train result/train result {CASE_NAME} sample number "
    #                  f"= {TOTAL_NB_SAMPLE} sigma = {SAMPLE_SIGMA} batch = {BATCH_SIZE}")
    output_folder = (f"train result/result {CASE_NAME} sample number "
                     f"= {TOTAL_NB_SAMPLE} sigma = {SAMPLE_SIGMA} batch = {BATCH_SIZE}")
    # output_folder = f"train result/train result sample number = 202 batch = {BATCH_SIZE}"
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, 'loss_figure.png'))
    # Akaike information criterion (AIC)
    Estimated_parameters = len(prob_var_list)
    AIC = -2 * final_log_likelihood + 2 * Estimated_parameters
    print("AIC:         ", AIC)
    SGD_report_df = pd.DataFrame([['Init Loglikelihood', init_log_likelihood],
                                  ['Final Loglikelihood', final_log_likelihood],
                                  ['Rou Square', 1 - final_log_likelihood / init_log_likelihood],
                                  ['AIC', AIC]], columns=['Statistical Indicators', 'Value'])
    SGD_report_df.to_csv(os.path.join(output_folder, 'nested_logit_statistic_metrics.csv'), encoding='gbk', index=False)

    # out put
    # estimate of flow
    # Xinyu W : for each date
    est_od_flow_df = pd.DataFrame()
    est_od_line_flow_df = pd.DataFrame()
    est_alighting_tensor_df = pd.DataFrame()
    est_section_flow_df = pd.DataFrame()

    date_station_board_dict = {}
    date_od_flow_dict = {}
    date_od_flow_reg_dict = {}
    date_od_line_flow_dict = {}
    date_od_line_prob_dict = {}
    date_alighting_dict = {}
    date_section_flow_dict = {}
    for date in all_sample_list:
        boarding_psg_array, od_psg_array, od_line_psg_array, alighting_psg_array, od_freq_reg_array, \
            od_eco_array, od_line_param_array, section_capacity_array = \
            select_single_day_sample(date, daily_stat_board_df, daily_stat_alight_df,
                                     daily_obs_od_df, daily_od_line_df, daily_sect_df)

        # calculate probability by function nested_logit_model in this sample
        od_prob_tensor, conditional_od_line_prob_tensor, daily_final_od_line_prob_tensor = \
            calc_prob_nested_logit_model(od_theta_list, o_eco_coeff_list, od_freq_coeff_list,
                                         od_tt_coeff_list, od_price_coeff_list, od_eco_array,
                                         od_line_param_array)

        # estimate flow by function est_flow_by_nested_logit_model in this sample
        daily_est_board_tensor, daily_est_od_flow_nl_tensor, \
            daily_est_line_flow_tensor, daily_est_alight_tensor, daily_est_sect_flow_tensor = \
            est_flow_by_nested_logit_model(est_boarding_list, od_prob_tensor, conditional_od_line_prob_tensor)

        # estimate flow by function est_flow_by_regression in this sample
        # reg_od_freq_coeff_list can be extended to general parameters metrics
        daily_est_od_flow_reg_tensor = est_od_flow_by_regression(reg_od_freq_coeff_list, od_freq_reg_array)
        daily_est_od_flow_reg_tensor = tf.where(tf.equal(daily_est_od_flow_reg_tensor, 0.0),
                                          NON_ZERO * tf.ones_like(daily_est_od_flow_reg_tensor),
                                          daily_est_od_flow_reg_tensor)

        daily_est_board_flow_df = pd.DataFrame({'BOARDDATE': date, 'DEP_STATION': g_board_station_list,
                                           'EST_PSGNUM': daily_est_board_tensor.numpy().tolist()})

        daily_est_od_flow_df = pd.DataFrame({'BOARDDATE': date, 'OD': g_od_list,
                                           'EST_PSGNUM': daily_est_od_flow_nl_tensor.numpy().tolist(),
                                           'EST_PSGNUM_REG': daily_est_od_flow_reg_tensor.numpy().tolist()})
        daily_est_od_line_flow_df = pd.DataFrame({'BOARDDATE': date, 'OD_LINE': g_od_line_list,
                                           'EST_PSGNUM': daily_est_line_flow_tensor.numpy().tolist(),
                                                  'EST_PROB': daily_final_od_line_prob_tensor.numpy().tolist(),})
        daily_est_alighting_tensor_df = pd.DataFrame({'BOARDDATE': date, 'ARR_STATION': g_alight_station_list,
                                           'EST_PSGNUM': daily_est_alight_tensor.numpy().tolist()})
        daily_est_section_flow_df = pd.DataFrame({'BOARDDATE': date, 'SECTION': g_section_list,
                                           'EST_PSGNUM': daily_est_sect_flow_tensor.numpy().tolist()})

        # use date and g_od_list as pair keys (date, od) to generate dict for EST_PSGNUM and EST_PSGNUM_REG
        for row in daily_est_board_flow_df.itertuples():
            date_station_board_dict[(row.BOARDDATE, row.DEP_STATION)] = row.EST_PSGNUM[0]

        for row in daily_est_od_flow_df.itertuples():
            date_od_flow_dict[(row.BOARDDATE, row.OD)] = row.EST_PSGNUM[0]
            date_od_flow_reg_dict[(row.BOARDDATE, row.OD)] = row.EST_PSGNUM_REG[0]

        for row in daily_est_od_line_flow_df.itertuples():
            date_od_line_flow_dict[(row.BOARDDATE, row.OD_LINE)] = row.EST_PSGNUM[0]

        for row in daily_est_od_line_flow_df.itertuples():
            date_od_line_prob_dict[(row.BOARDDATE, row.OD_LINE)] = row.EST_PROB[0]

        for row in daily_est_alighting_tensor_df.itertuples():
            date_alighting_dict[(row.BOARDDATE, row.ARR_STATION)] = row.EST_PSGNUM[0]

        for row in daily_est_section_flow_df.itertuples():
            date_section_flow_dict[(row.BOARDDATE, row.SECTION)] = row.EST_PSGNUM[0]

    # estimate of variables
    var_name_list = []
    var_value_list = []
    for var in est_boarding_list + prob_var_list + reg_od_freq_coeff_list:
        var_name = var.name
        var_value = var.numpy()[0]

        var_name_list.append(var_name)
        var_value_list.append(var_value)

    all_var_df = pd.DataFrame([var_name_list, var_value_list])
    all_var_df = all_var_df.T
    all_var_df.columns = ['VAR_NAME', 'VAR_VALUE']
    all_var_df.to_csv(os.path.join(output_folder, 'est_behavior_params.csv'), encoding='gbk', index=False)

    # level 1: compare estimated boarding flow with observed boarding flow in training and validation samples
    training_daily_stat_board_df['EST_PSGNUM'] = \
        training_daily_stat_board_df.apply(lambda x: date_station_board_dict[(x.BOARDDATE, x.DEP_STATION)], axis=1)\
        * K_PERSON_CONVERTER
    training_daily_stat_board_df['DATA_TYPE'] = 'TRAINING'
    valid_daily_stat_board_df['EST_PSGNUM'] = \
        valid_daily_stat_board_df.apply(lambda x: date_station_board_dict[(x.BOARDDATE, x.DEP_STATION)], axis=1)\
        * K_PERSON_CONVERTER
    valid_daily_stat_board_df['DATA_TYPE'] = 'VALIDATION'
    complete_daily_stat_board_df = pd.concat([training_daily_stat_board_df, valid_daily_stat_board_df], axis=0)
    complete_daily_stat_board_df.to_csv(os.path.join(output_folder, 'est_board.csv'), encoding='gbk', index=False)

    # level 2: compare estimated od flow with observed od flow in training and validation samples
    training_daily_obs_od_df['EST_PSGNUM'] = \
        training_daily_obs_od_df.apply(lambda x: date_od_flow_dict[(x.BOARDDATE, x.OD)], axis=1) * K_PERSON_CONVERTER
    training_daily_obs_od_df['EST_PSGNUM_REG'] = \
        training_daily_obs_od_df.apply(lambda x: date_od_flow_reg_dict[(x.BOARDDATE, x.OD)], axis=1)\
        * K_PERSON_CONVERTER
    training_daily_obs_od_df['DATA_TYPE'] = 'TRAINING'
    valid_daily_obs_od_df['EST_PSGNUM'] = \
        valid_daily_obs_od_df.apply(lambda x: date_od_flow_dict[(x.BOARDDATE, x.OD)], axis=1) * K_PERSON_CONVERTER
    valid_daily_obs_od_df['EST_PSGNUM_REG'] = \
        valid_daily_obs_od_df.apply(lambda x: date_od_flow_reg_dict[(x.BOARDDATE, x.OD)], axis=1) * K_PERSON_CONVERTER
    valid_daily_obs_od_df['DATA_TYPE'] = 'VALIDATION'
    complete_daily_obs_od_df = pd.concat([training_daily_obs_od_df, valid_daily_obs_od_df], axis=0)
    complete_daily_obs_od_df.to_csv(os.path.join(output_folder, 'est_od.csv'), encoding='gbk', index=False)

    # level 3: compare estimated od line flow with observed od line flow in training and validation samples
    training_daily_od_line_df['EST_PSGNUM'] = \
        training_daily_od_line_df.apply(lambda x: date_od_line_flow_dict[(x.BOARDDATE, x.OD_LINE)], axis=1)\
        * K_PERSON_CONVERTER
    training_daily_od_line_df['EST_PROB'] =\
        training_daily_od_line_df.apply(lambda x: date_od_line_prob_dict[(x.BOARDDATE, x.OD_LINE)], axis=1)
    training_daily_od_line_df['DATA_TYPE'] = 'TRAINING'
    valid_daily_od_line_df['EST_PSGNUM'] = \
        valid_daily_od_line_df.apply(lambda x: date_od_line_flow_dict[(x.BOARDDATE, x.OD_LINE)], axis=1)\
        * K_PERSON_CONVERTER
    valid_daily_od_line_df['EST_PROB'] =\
        valid_daily_od_line_df.apply(lambda x: date_od_line_prob_dict[(x.BOARDDATE, x.OD_LINE)], axis=1)
    valid_daily_od_line_df['DATA_TYPE'] = 'VALIDATION'
    complete_daily_od_line_df = pd.concat([training_daily_od_line_df, valid_daily_od_line_df], axis=0)
    complete_daily_od_line_df.to_csv(os.path.join(output_folder, 'est_od_line.csv'), encoding='gbk', index=False)

    # Level 4: compare estimated alighting flow with observed alighting flow in training and validation samples
    training_daily_stat_alight_df['EST_PSGNUM'] = \
        training_daily_stat_alight_df.apply(lambda x: date_alighting_dict[(x.BOARDDATE, x.ARR_STATION)], axis=1)\
        * K_PERSON_CONVERTER
    training_daily_stat_alight_df['DATA_TYPE'] = 'TRAINING'
    valid_daily_stat_alight_df['EST_PSGNUM'] = \
        valid_daily_stat_alight_df.apply(lambda x: date_alighting_dict[(x.BOARDDATE, x.ARR_STATION)], axis=1)\
        * K_PERSON_CONVERTER
    valid_daily_stat_alight_df['DATA_TYPE'] = 'VALIDATION'
    complete_daily_stat_alight_df = pd.concat([training_daily_stat_alight_df, valid_daily_stat_alight_df], axis=0)
    complete_daily_stat_alight_df.to_csv(os.path.join(output_folder, 'est_alight.csv'), encoding='gbk', index=False)

    # level 5: compare estimated section flow with observed section flow in training and validation samples
    training_daily_sect_df['EST_PSGNUM'] = \
        training_daily_sect_df.apply(lambda x: date_section_flow_dict[(x.BOARDDATE, x.SECTION)], axis=1)\
        * K_PERSON_CONVERTER
    training_daily_sect_df['DATA_TYPE'] = 'TRAINING'
    valid_daily_sect_df['EST_PSGNUM'] = \
        valid_daily_sect_df.apply(lambda x: date_section_flow_dict[(x.BOARDDATE, x.SECTION)], axis=1)\
        * K_PERSON_CONVERTER
    valid_daily_sect_df['DATA_TYPE'] = 'VALIDATION'
    complete_daily_sect_df = pd.concat([training_daily_sect_df, valid_daily_sect_df], axis=0)
    complete_daily_sect_df.to_csv(os.path.join(output_folder, 'est_sect_cap.csv'), encoding='gbk', index=False)

    # level 5: estimate of section flow
    _nb_line = len(total_od_line_df)
    all_sec_df = pd.DataFrame()
    for date in daily_od_line_df['BOARDDATE'].drop_duplicates().to_list():
        sub_sect_df = pd.DataFrame(columns=['BOARDDATE', 'SECTION_ID', 'SECTION', 'PSGNUM'])
        current_day_sect_df = daily_sect_df[daily_sect_df['BOARDDATE'] == date]
        sub_sect_df['SECTION_ID'] = current_day_sect_df['SECTION_ID']
        sub_sect_df['SECTION'] = current_day_sect_df['SECTION']
        sub_sect_df['BOARDDATE'] = date

        # calculate daily section flow
        single_date_df = daily_od_line_df[daily_od_line_df['BOARDDATE'] == date]
        ticket_od_line_flow = single_date_df['PSGNUM']
        ticket_od_line_flow_np = ticket_od_line_flow.values.astype('float64')
        ticket_od_line_flow_tensor = tf.reshape(ticket_od_line_flow_np, [1, _nb_line])
        ticket_section_flow = tf.transpose(tf.matmul(ticket_od_line_flow_tensor, line_to_section_inc_mat))
        sub_sect_df['PSGNUM'] = ticket_section_flow.numpy()
        all_sec_df = pd.concat([all_sec_df, sub_sect_df], axis=0)
    training_all_sec_df = all_sec_df[all_sec_df['BOARDDATE'].isin(training_sample_list)]
    valid_all_sec_df = all_sec_df[all_sec_df['BOARDDATE'].isin(validation_sample_list)]
    training_all_sec_df = training_all_sec_df.copy()
    training_all_sec_df['EST_PSGNUM'] = \
        training_all_sec_df.apply(lambda x: date_section_flow_dict[(x.BOARDDATE, x.SECTION)], axis=1) \
        * K_PERSON_CONVERTER
    training_all_sec_df['DATA_TYPE'] = 'TRAINING'
    valid_all_sec_df = valid_all_sec_df.copy()
    valid_all_sec_df['EST_PSGNUM'] = \
        valid_all_sec_df.apply(lambda x: date_section_flow_dict[(x.BOARDDATE, x.SECTION)], axis=1) \
        * K_PERSON_CONVERTER
    valid_all_sec_df['DATA_TYPE'] = 'VALIDATION'
    all_sec_df = pd.concat([training_all_sec_df, valid_all_sec_df], axis=0)
    all_sec_df.to_csv(os.path.join(output_folder, 'est_sect_flow.csv'), encoding='gbk', index=False)

    # R-square
    output_boarding_r_square = []
    output_od_flow_r_square = []
    output_od_line_flow_r_square = []
    output_alighting_r_square = []
    for i in range(len(boarding_r_square_list)):
        epoch_aboarding_r_square = boarding_r_square_list[i].numpy()
        epoch_od_flow_r_square = od_flow_r_square_list[i].numpy()
        epoch_od_line_flow_r_square = od_line_flow_r_square_list[i].numpy()
        epoch_alighting_r_square = alighting_r_square_list[i].numpy()
        output_boarding_r_square.append(epoch_aboarding_r_square)
        output_od_flow_r_square.append(epoch_od_flow_r_square)
        output_od_line_flow_r_square.append(epoch_od_line_flow_r_square)
        output_alighting_r_square.append(epoch_alighting_r_square)

    R_square_report_dict = {'boarding': output_boarding_r_square,
                            'od_flow': output_od_flow_r_square,
                            'od_line_flow': output_od_line_flow_r_square,
                            'alighting': output_alighting_r_square}

    R_square_report_df = pd.DataFrame(R_square_report_dict)
    R_square_report_df.insert(0, 'epoch', list(range(len(R_square_report_df))))
    R_square_report_df.to_csv(os.path.join(output_folder, 'r_square.csv'), encoding='gbk', index=False)

    # output losses
    output_losses = {'epochs': [i for i in range(len(plot_loss_total_list))],
                     'total loss function': plot_loss_total_list,
                     'boarding losses train': plot_loss_boarding_list, 'nest_losses_train': plot_loss_nest_logit_list,
                     'alighting losses train': plot_loss_alighting_list,
                     'section constraint train': plot_loss_section_constraint_list,
                     'od flow equal constraint train f': plot_loss_od_flow_coupling_constraint_list}
    output_losses_df = pd.DataFrame(output_losses)
    output_losses_df.to_csv(os.path.join(output_folder, 'loss.csv'))

    # Output the current hyperparameters
    Hyperparameters = {
        "TOTAL_NB_SAMPLE": TOTAL_NB_SAMPLE,
        'TRAIN_SAMPLE_RATIO': TRAIN_SAMPLE_RATIO,
        "TRAIN_CAPACITY": TRAIN_CAPACITY,
        "UB_BOARDING": UB_BOARDING,
        "UB_ECO_COEFF": UB_ECO_COEFF,
        "LB_THETA": LB_THETA,
        "UB_LINE_FREQ_COEFF": UB_LINE_FREQ_COEFF,
        "LB_LINE_TT_COEFF": LB_LINE_TT_COEFF,
        "LB_LINE_PRICE_COEFF": LB_LINE_PRICE_COEFF,
        "UB_OD_FREQ_COEFF": UB_OD_FREQ_COEFF,
        "INIT_SECT_LAG_MULT": INIT_SECT_LAG_MULT,
        "INIT_SECT_LAG_MULT_UPDATING_STEP": INIT_SECT_LAG_MULT_UPDATING_STEP,
        "INIT_C_SECT_MULT": INIT_C_SECT_MULT,
        "C_SECT_MULT_UPDATING_STEP": C_SECT_MULT_UPDATING_STEP,
        "INIT_OD_LAG_MULT": INIT_OD_LAG_MULT,
        "INIT_OD_LAG_MULT_UPDATING_STEP": INIT_OD_LAG_MULT_UPDATING_STEP,
        "INIT_C_OD_MULT": INIT_C_OD_MULT,
        "C_OD_MULT_UPDATING_STEP": C_OD_MULT_UPDATING_STEP,
        "BATCH_SIZE": BATCH_SIZE,
        "NON_ZERO": NON_ZERO,
        "K_PERSON_CONVERTER": K_PERSON_CONVERTER,
        "PENALTY_THRESHOLD": PENALTY_THRESHOLD,
        "SECT_PENALTY_THRESHOLD": SECT_PENALTY_THRESHOLD,
        "SAMPLE_SIGMA": SAMPLE_SIGMA,
        "lr_nl": lr_nl,
        "lr_boarding": lr_boarding,
        "lr_regression": lr_regression,
        "lr_c_od": lr_c_od,
        "lr_integrating": lr_integrating,
        "W_PROB_NL": W_PROB_NL,
        "W_PROB_BOARD": W_PROB_BOARD,
        "W_PROB_ALIGHT": W_PROB_ALIGHT,
        "W_PROB_CAP": W_PROB_CAP,
        "W_PROB_LR": W_PROB_LR,
        "W_FLOW_NL": W_FLOW_NL,
        "W_FLOW_BOARD": W_FLOW_BOARD,
        "W_FLOW_ALIGHT": W_FLOW_ALIGHT,
        "W_FLOW_CAP": W_FLOW_CAP,
        "W_FLOW_LR": W_FLOW_LR,
        "W_COUPLE_NL": W_COUPLE_NL,
        "W_COUPLE_BOARD": W_COUPLE_BOARD,
        "W_COUPLE_ALIGHT": W_COUPLE_ALIGHT,
        "W_COUPLE_CAP": W_COUPLE_CAP,
        "W_COUPLE_LR": W_COUPLE_LR,
        "W_INTEGRATION_NL": W_INTEGRATION_NL,
        "W_INTEGRATION_BOARD": W_INTEGRATION_BOARD,
        "W_INTEGRATION_ALIGHT": W_INTEGRATION_ALIGHT,
        "W_INTEGRATION_CAP": W_INTEGRATION_CAP,
        "W_INTEGRATION_LR": W_INTEGRATION_LR,
        "MAX_NESTED_LOGIT_STEP": MAX_NESTED_LOGIT_STEP,
        "MAX_OD_FLOW_STEP": MAX_OD_FLOW_STEP,
        "MAX_COUPLING_STEP": MAX_COUPLING_STEP,
        "MAX_ITERATION_STEP": MAX_ITERATION_STEP,
        "THRESHOLD_MAX_COUPLING_GAP": THRESHOLD_MAX_COUPLING_GAP,
        "MIN_GRADIENT": MIN_GRADIENT
    }

    Hyperparameters_df = pd.DataFrame([Hyperparameters], index=['Value']).T
    Hyperparameters_df.to_csv(os.path.join(output_folder, 'params_setting.csv'), index=True)
