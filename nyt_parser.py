import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import re

def calc_newCases(df, state):
    df_filt = df.loc[df.state == state]

    if df_filt.shape[0] < 2:
        pass
    else:
        confirmed = df_filt.cases.to_list()

        new_cases=[]
        for i in list(range(len(confirmed)-1)):
            new_cases.append(confirmed[i+1] - confirmed[i])
#             print(confirmed[i+1] - confirmed[i])

        df_out = df_filt.iloc[1:,:].copy()
        df_out['new_cases'] = new_cases
        df_out['tooltip_newcases'] = df_out.apply(lambda x : x.state + ": " + str(x.new_cases), axis=1)

        return(df_out)
#         return(df_out[['Country','Date', 'Region', 'Lat', 'Lon', 'tooltip_newcases', 'new_cases']])

def calculate_new_cases(df):
    unique_countries = df.state.unique().tolist()

    output = [calc_newCases(df, c) for c in unique_countries]


    df_output = pd.concat(output)
    df_output.reset_index().to_feather("./parsed_data/new_confirmed.feather")

    return (df_output)

############# Parsing County-level Data #############
us_states = pd.read_csv("./us-states.csv")
us_states["date"] = pd.to_datetime(us_states.date)
us_states.drop(columns=["fips"], inplace=True)
us_states.to_feather("./parsed_data/us_states.feather")

# Calculating the new cases from the confirmed cases
states_newcases = calculate_new_cases(us_states)
states_newcases.reset_index().to_feather("./parsed_data/us_states_newcases.feather")


############# Parsing County-level Data #############
us_counties = pd.read_csv("./us-counties.csv")
us_counties["date"] = pd.to_datetime(us_counties.date)
us_counties.drop(columns=["fips"], inplace=True)
us_counties.to_feather("./parsed_data/us_counties.feather")
