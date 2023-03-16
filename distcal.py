# to calculate pairwise distance between poi (for land parcels, can also calculate the time difference in launch)
# input: poi table a, poi table b (both with cols [id, latitude, longitude])
# output: df[id_a, id_b, distance]

import pandas as pd
import numpy as np
from geopy.distance import geodesic
from tqdm import tqdm
from datetime import date, datetime


class PairwiseDistCalculator:

    def __init__(self, poi_master=None, poi_ref=None, master_id=None, ref_id=None):
        self.poi_master = poi_master
        self.poi_reference = poi_ref
        self.master_id = master_id
        self.reference_id = ref_id

    def calculate_distance(self, poi_master=None, poi_ref=None, master_id=None, ref_id=None, distance_limit=5000, cutoff=False):
        """
        To calculate pairwise distances base on latitude and longitude (geodesic distance)
        Usage: feed in two tables a) master (sites with lat & lng) and b) reference (sites with lat & lng), and calculate pairwise distances btw two tables
        :param poi_master: master poi table
        :param poi_ref: reference poi table
        :param master_id: name of the column to loop through in the master/objective table
        :param ref_id: name of the column to loop through in the reference table
        :param distance_limit: (in meters) the nearest place found should fall within this distance limit
        :param cutoff: if True, break the loop when one location whose distance is within limit has been found
        :return: a dataframe with 3 columns: master_id, ref_id, distance
        """

        if poi_master is not None:
            self.poi_master = poi_master
        if poi_ref is not None:
            self.poi_reference = poi_ref
        if master_id is not None:
            self.master_id = master_id
        if ref_id is not None:
            self.reference_id = ref_id

        key_a = self.master_id
        key_b = self.reference_id

        # columns to loop through in master and ref tbl respectively
        loop_a, loop_b = self.poi_master[key_a], self.poi_reference[key_b]
        self.poi_master.index = loop_a
        self.poi_reference.index = loop_b
        output = pd.DataFrame(columns=['poi_a',
                                       'poi_b',
                                       'distance_m'])
        # start looping, for each master site, loop all reference places, calculate distances
        for id_a in tqdm(loop_a, desc='Calculating pairwise distance'):
            coord_a = (self.poi_master.loc[id_a].latitude, self.poi_master.loc[id_a].longitude)
            for id_b in loop_b:
                coord_b = (self.poi_reference.loc[id_b].latitude, self.poi_reference.loc[id_b].longitude)
                try:
                    distance = round(geodesic(coord_a, coord_b).m, 2)
                except ValueError:
                    distance = -1
                to_append = [id_a, id_b, distance]
                # append output table if the distance falls within the limit
                if 0 < distance <= distance_limit:
                    output.loc[len(output)] = to_append
                    # if cutoff is True, stop current loop when one site satisfies the limit; or append all sites satisfying the limit
                    if cutoff:
                        break

        return output

    @staticmethod
    def find_nearby(pairwise_tbl: pd.DataFrame, rename_col: dict = None):
        """
        To find the nearest place
        :param pairwise_tbl: pairwise distances table
        :param rename_col: dictionary, columns to rename
        :return: dataframe
        """
        header = pairwise_tbl.columns
        nearby_df = pairwise_tbl.groupby(header[0])[header[-1]].min()  # for pandas==1.0.5
        # nearby_df = pairwise_tbl.groupby(header[0])min(header[-1])
        if len(nearby_df.shape) < 2:
            nearby_df = nearby_df.to_frame(list(rename_col.keys())[0])
        if rename_col is not None:
            nearby_df = nearby_df.rename(columns=rename_col)
        return nearby_df


class TimeMaster:

    def __init__(self):
        pass

    @staticmethod
    def time_diff_by_index(df, time_index_col_a, time_index_col_b, time_format='%Y%m%d', time_unit='D'):
        from numpy import timedelta64
        from datetime import datetime
        return (
                    df[time_index_col_a].apply(lambda x: datetime.strptime(str(x), time_format)) -
                    df[time_index_col_b].apply(lambda x: datetime.strptime(str(x), time_format))
               ) / timedelta64(1, time_unit)
