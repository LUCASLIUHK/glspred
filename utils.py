# import SQL_connect
import pandas as pd
import numpy as np
from varname import nameof
from typing import List, Tuple, Any
import requests
import random
import time
from datetime import datetime


def upload(dbconn, df: List[pd.DataFrame], tbl_name: List[str], auto_check_conditions: List[bool] = None):
    # upload list of dfs to database
    # set conditions for auto-check before uploading, by assigning auto_check_conditions, this will turn on auto-upload
    # if auto_check_conditions is None, it will ask manual confirmation before uploading (press Y/N)
    if isinstance(df, pd.DataFrame):
        df = [df]
    if isinstance(tbl_name, str):
        tbl_name = [tbl_name]
    if isinstance(auto_check_conditions, bool):
        auto_check_conditions = [auto_check_conditions]
    if all(isinstance(item, list) for item in [df, tbl_name]):
        df_list = df
        tbl_list = tbl_name
        acc_list = auto_check_conditions
        print("-" * 67)

        if len(df_list) != len(tbl_list):
            print(f"Unequal number of dataframes and table names: {len(df_list)} and {len(tbl_list)}")
            return None

        for i in range(len(df_list)):
            print(f"Shape of dataframe{i + 1}: {df_list[i].shape}", f"Name of table{i + 1}: {tbl_list[i]}\n", sep='\n')
        check_upload = ''

        if acc_list is not None:
            check_upload = 'Y'
            for i in range(max(len(df_list), len(acc_list))):
                try:
                    statement = acc_list[i]
                    if statement is True:
                        dbconn.copy_from_df(df_list[i], tbl_list[i])
                        print(f"{i+1}/{len(df_list)}: Dataframe{i + 1} {df_list[i].shape} uploaded as {tbl_list[i]}")
                except IndexError:
                    print(f"Unequal number of dataframes and conditions: {len(df_list)} and {len(acc_list)}")
                    if len(df_list) > len(acc_list):
                        if len(df_list) - len(acc_list) > 1:
                            addition = f" to dataframe{len(df_list)}"
                        else:
                            addition = ''
                        print(f"Unable to check for dataframe{len(acc_list) + 1}{addition}")
                    break
            print("Auto-check and uploading finished")

        while check_upload not in ['Y', 'y'] and check_upload not in ['N', 'n']:
            check_upload = input("Confirm uploading (Y/N)\n")
            if check_upload in ['Y', 'y']:
                for i in range(len(df_list)):
                    dbconn.copy_from_df(df_list[i], tbl_list[i])
                    print(f"{i+1}/{len(df_list)}: Dataframe{i + 1} {df_list[i].shape} uploaded as {tbl_list[i]}\n")
                print("Uploading finished")
                break
            elif check_upload in ['N', 'n']:
                print('Uploading aborted')
                break
            else:
                print(f'"{check_upload}" is not a valid command')
    else:
        print("Type error")


# using google geocoding to get coordinates for address
def random_time_delay(range=None):
    if range is None:
        range = [1, 5]
    time.sleep(random.uniform(range[0], range[1]))


def geocode_get_location(address, region=None, api_key=None,
                         google_maps_api_url="https://maps.googleapis.com/maps/api/geocode/json",
                         boundary=None):
    if boundary is None:
        boundary = [90, -90, 180, -180]  # boundary = [N, S, E, W]
    random_time_delay([1, 3])
    coord_na = (np.nan, np.nan)
    params = {
        'key': api_key,
        'address': address,
        'region': region,
        'sensor': 'false'
    }

    try:
        response = requests.get(url=google_maps_api_url, params=params, timeout=120).json()
    except:
        print("\nError sending request")
        return coord_na

    if response['status'] != 'OK':
        print(f"\nError getting response: {response['status']}[{params['address']}]")
        return coord_na

    try:
        lat = response['results'][0]['geometry']['location']['lat']
        lng = response['results'][0]['geometry']['location']['lng']
        if boundary[1] <= lat <= boundary[0] and boundary[3] <= lng <= boundary[2]:
            return lat, lng
        else:
            print(f"\nCoordinates out of bound, try strict search[{params['address']}]")
            params['address'] = address + ',' + region
            try:
                random_time_delay([1, 3])
                response_strict = requests.get(url=google_maps_api_url, params=params, timeout=120).json()
            except:
                print("\nError sending request")
                return coord_na

            if response_strict['status'] != 'OK':
                print(f"\nError getting response: {response['status']}[{params['address']}]")
                return coord_na
            else:
                try:
                    lat = response_strict['results'][0]['geometry']['location']['lat']
                    lng = response_strict['results'][0]['geometry']['location']['lng']
                    return lat, lng
                except:
                    print("\nError parsing response")
                    return coord_na

    except:
        print("\nError parsing response")
        return coord_na


def date_format(datetime_text: str, input_format: str, output_format: str = '%d/%m/%Y'):
    datetime_object = datetime.strptime(datetime_text, input_format)
    try:
        return datetime.strftime(datetime_object, output_format)
    except:
        return np.nan
