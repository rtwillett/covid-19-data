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
        deaths = df_filt.deaths.to_list()

        new_cases=[]
        new_deaths=[]
        for i in list(range(len(confirmed)-1)):
            new_cases.append(confirmed[i+1] - confirmed[i])
            new_deaths.append(deaths[i+1] - deaths[i])
#             print(confirmed[i+1] - confirmed[i])

        df_out = df_filt.iloc[1:,:].copy()
        df_out['new_cases'] = new_cases
        df_out['new_deaths'] = new_deaths
        df_out['tooltip_newcases'] = df_out.apply(lambda x : x.state + ": " + str(x.new_cases), axis=1)
        df_out['tooltip_deaths'] = df_out.apply(lambda x : x.state + ": " + str(x.deaths), axis=1)

        return(df_out)

def calculate_new_cases(df):
    '''
    This function calculates iterates through each state in the dataframe and uses calc_newCases to compute a dataframe of new cases for that state base on the confirmed cases for that state. It then concatenates all of the state dataframe to produce a continuous dataset, which is then returned.
    '''
    unique_countries = df.state.unique().tolist()

    output = [calc_newCases(df, c) for c in unique_countries]

    df_output = pd.concat(output)
    df_output.reset_index().to_feather("./parsed_data/us_states_newcases.feather")

    return (df_output)

def calc_newCases_county(df, state, county):
    df_filt = df.loc[df.state == state].loc[df.county == county]
    df_out = df_filt.iloc[1:, :].copy()

    cases = df_filt.cases.to_numpy()
    cases_leading = np.roll(cases, -1)[:-1]
    cases_trailing = cases[:-1]

    deaths = df_filt.deaths.to_numpy()
    deaths_leading = np.roll(deaths, -1)[:-1]
    deaths_trailing = deaths[:-1]

    new_cases = cases_leading - cases_trailing
    new_deaths = deaths_leading - deaths_trailing

    df_out["new_cases"] = new_cases
    df_out["new_deaths"] = new_deaths

#     new_cases=[]
#     new_deaths=[]
#     for i in list(range(len(confirmed)-1)):
#         new_cases.append(confirmed[i+1] - confirmed[i])
#         new_deaths.append(deaths[i+1] - deaths[i])

#     print(new_cases.tolist())
    return(df_out)

def calculate_new_cases_county(df):
    state_counties = list(zip(df.state.tolist(), df.county.tolist()))
    state_counties = list(set(state_counties))
    s, c = list(zip(*state_counties)) # Unpacking the tuple set. Comes out as a list of tuples (states, counties)
    s = list(s) # Convert state tuple to list
    c = list(c) # Convert county tuple to list

    df_list = [calc_newCases_county(df, s[i], c[i]) for i in list(range(len(s)))]

    return pd.concat(df_list)

def top_rates_counties(df, var, n=10):
    top10 = df.loc[df.date == max(df.date)].sort_values(by = var, ascending = False).iloc[:n, :].reset_index().drop(columns=['index'])
    return top10

############# Importing Coordinates #############

state_coords = pd.read_csv("./additional_data/state_geocodes.csv")
state_coords.columns = ["abbrev", 'lat', 'lon', 'state']

############# Parsing County-level Data #############
us_states = pd.read_csv("./us-states.csv")
us_states["date"] = pd.to_datetime(us_states.date)
us_states.drop(columns=["fips"], inplace=True)
us_states = us_states.merge(state_coords, on=['state'])
us_states['tooltip_cases'] = us_states.apply(lambda x : x.state + ": " + str(x.cases), axis=1)
us_states.to_feather("./parsed_data/us_states.feather")

# Calculating the new cases from the confirmed cases
states_newcases = calculate_new_cases(us_states)
# states_newcases.reset_index().to_feather("./parsed_data/us_states_newcases.feather")


############# Parsing County-level Data #############

def calc_newCases_county(df, state, county):
    '''

    '''

    df_filt = df.loc[df.state == state].loc[df.county == county]
    df_out = df_filt.iloc[1:, :].copy()

    cases = df_filt.cases.to_numpy()
    cases_leading = np.roll(cases, -1)[:-1]
    cases_trailing = cases[:-1]

    deaths = df_filt.deaths.to_numpy()
    deaths_leading = np.roll(deaths, -1)[:-1]
    deaths_trailing = deaths[:-1]

    new_cases = cases_leading - cases_trailing
    new_deaths = deaths_leading - deaths_trailing

    df_out["new_cases"] = new_cases
    df_out["new_deaths"] = new_deaths

#     new_cases=[]
#     new_deaths=[]
#     for i in list(range(len(confirmed)-1)):
#         new_cases.append(confirmed[i+1] - confirmed[i])
#         new_deaths.append(deaths[i+1] - deaths[i])

#     print(new_cases.tolist())
    return(df_out)

def calculate_new_cases_county(df):
    '''

    '''

    state_counties = list(zip(df.state.tolist(), df.county.tolist()))
    state_counties = list(set(state_counties))
    s, c = list(zip(*state_counties)) # Unpacking the tuple set. Comes out as a list of tuples (states, counties)
    s = list(s) # Convert state tuple to list
    c = list(c) # Convert county tuple to list

    df_list = [calc_newCases_county(df, s[i], c[i]) for i in list(range(len(s)))]

    return pd.concat(df_list)

def top_rates_counties(df, var, n=10):
    '''

    '''
    top_filt = df.loc[df.date == max(df.date)].sort_values(by = var, ascending = False).iloc[:n, :].reset_index().drop(columns=['index'])
    top_counties = top_filt.county
    top_state = top_filt.state

    return top_state.to_list(), top_counties.to_list()


county_geocodes = pd.read_csv("./additional_data/Geocodes_USA_with_Counties.csv")
county_geocodes = county_geocodes[["state", "latitude", "longitude", "county", "country"]].loc[county_geocodes.country == "US"].drop_duplicates(keep=False)

# The dataset does not report data for each NYC county, but aggregates it for all of NYC.
drop_list = ["New York", "Queens", "Kings"]
# county_geocodes = county_geocodes.loc[county_geocodes.state == "NY"]
county_geocodes = county_geocodes.loc[~county_geocodes.county.isin(drop_list)]
ny_df = pd.DataFrame([("NY", 40.7128, -74.0060, "New York City", "US")], columns = county_geocodes.columns)
county_geocodes = pd.concat([county_geocodes, ny_df])

county_geocodes = county_geocodes.groupby(['state', 'county', 'country']).mean().reset_index()
county_geocodes.columns = ["abbrev", "county", "country", "lat", "lon"]

us_counties = pd.read_csv("./us-counties.csv")
us_counties["date"] = pd.to_datetime(us_counties.date)
us_counties.drop(columns=["fips"], inplace=True)
us_counties = us_counties.merge(state_coords[["abbrev", "state"]], on=["state"]).merge(county_geocodes, on=["abbrev", "county"])

# Creating a column with the labels for the mapdeck visualization.
us_counties["tooltip_cases"] = us_counties.apply(lambda x : x.county + ": " + str(x.cases), axis=1)
us_counties["tooltip_deaths"] = us_counties.apply(lambda x : x.county + ": " + str(x.deaths), axis=1)

df_rates = calculate_new_cases_county(us_counties)
# Save the data as a feather file.
df_rates.reset_index(drop=True).to_feather("./parsed_data/county_rates.feather")

# cases_rates_county = top_rates_counties(df_rates, "new_cases")
# death_rates_county = top_rates_counties(df_rates, "new_deaths")

t_states, t_counties = top_rates_counties(df_rates, "new_cases")

top_list = []
for i in list(range(len(t_states))):
    top_list.append(df_rates.loc[(df_rates.county == t_counties[i]) & (df_rates.state == t_states[i])])

top_list = pd.concat(top_list)
top_list["label"] = top_list.apply(lambda x : x.county + ", " + x.state, axis=1)

top_list.reset_index(drop=True).to_feather("./parsed_data/county_rates.feather")

top_rates_counties(df_rates, "new_deaths", 20)

us_counties.to_feather("./parsed_data/us_counties.feather")

# cases_rates_county.to_feather("./parsed_data/case_rates_county.feather")
# death_rates_county.to_feather("./parsed_data/death_rates_county.feather")



us_counties.to_feather("./parsed_data/us_counties.feather")
