import pandas as pd
import numpy as np
from glspred.distcal import PairwiseDistCalculator
from dateutil.parser import ParserError
from datetime import datetime


class Preprocess:

    def __init__(self, pred: pd.DataFrame = None, master: pd.DataFrame = None):
        if pred is not None:
            self.pred_origin = pred.copy()  # store original pred data
            self.pred = pred
        if master is not None:
            self.master = master
        self.res_keywords = ['residential',
                             'condo',
                             'hdb',
                             'apartment',
                             'apt',
                             'executive condo',
                             'ec',
                             'flat',
                             'landed',
                             'terrace',
                             'bungalow',
                             'condo-w-house',
                             'detach-house',
                             'apt-w-house',
                             'landed-housing-group',
                             'semid-house',
                             'cluster-house',
                             'terrace-house'
                             ]
        self.com_keywords = ['commercial',
                             'recreation']
        self.res_kw_hdb_condo = ['CO', 'AP', 'LP', 'FT', 'BH', 'TH', 'EC']
        hdb_condo_name = ['condo', 'apt', 'landed', 'flat', 'bungalow', 'terrace', 'exec condo']
        self.res_kw_hdb_condo_dict = dict(zip(self.res_kw_hdb_condo, hdb_condo_name))

    @staticmethod
    def get_uuid(text_id: str, mode='hashlib'):
        # create 64-digit uuid
        import hashlib
        if mode == 'hashlib':
            return hashlib.sha256(text_id.encode('utf-8')).hexdigest()
        else:
            txt_cleaned = text_id.lower().replace(' ', '')
            return ''.join([s for s in txt_cleaned if s.isalnum()]).ljust(64, '0')

    @staticmethod
    def get_time_index(date: str, format=None, dayfirst: bool = True, yearfirst: bool = False, unit: str = 'M',
                       asint=True):
        # if format is None, machine will infer format
        if format is not None:
            try:
                date_parsed = datetime.strptime(date, format)
                time_index = str(date_parsed.year).zfill(4) + str(date_parsed.month).zfill(2)
                if unit.lower() == 'd':
                    time_index = time_index + str(date_parsed.day).zfill(2)
                elif unit.lower() == 'y':
                    time_index = str(date_parsed.year).zfill(4)
            except ValueError:
                print(f'Date parsing failed for {date}')
                time_index = np.nan
        else:
            try:
                date_inferred = pd.to_datetime(date, dayfirst=dayfirst, yearfirst=yearfirst, infer_datetime_format=True)
                time_index = str(date_inferred.year).zfill(4) + str(date_inferred.month).zfill(2)
                if unit.lower() == 'd' or 'day':
                    time_index = time_index + str(date_inferred.day).zfill(2)
                elif unit.lower() == 'y' or 'year':
                    time_index = str(date_inferred.year).zfill(4)

            except ParserError:
                print(f'Date parsing failed for {date}')
                time_index = np.nan

        if asint and pd.notna(time_index):
            return int(time_index)
        return time_index

    # to create uuid for land parcel and gls sale record
    def encode(self, pred: pd.DataFrame = None) -> pd.DataFrame:
        # if param passed in, overwrite self
        if pred is not None:
            self.pred = pred

        pred = self.pred
        input_cols = pred.columns  # store original input columns
        # define dimensions used to create uuid
        land_parcel_dimensions = ['land_parcel_name',
                                  'latitude',
                                  'longitude']
        gls_dimensions = ['land_parcel_id',
                          'award_month_index',
                          'site_area_sqm']

        # if key dimensions not exist, create with values of NA
        for col in land_parcel_dimensions + gls_dimensions:
            if col not in pred.columns:
                pred[col] = np.nan

        # create land parcel uuid
        # in case of lacking any dimension, create uuid as <name><00000...> (total 64-digit), same for gls uuid
        # pred['land_parcel_id'] = pred.land_parcel_name.apply(lambda x: x.lower().replace(' ', ''))\
        #     .apply(lambda x: ''.join([s for s in x if s.isalnum()]).ljust(64, '0'))
        pred['land_parcel_id'] = pred.land_parcel_name.apply(self.get_uuid, mode='default')
        land_parcel_non_na_index = pred[(pred.land_parcel_name.notna())
                                        & (pred.latitude.notna())
                                        & (pred.longitude.notna())].index
        land_parcel_id_text = pred.land_parcel_name + pred.latitude.astype(str) + pred.longitude.astype(str)
        pred.loc[land_parcel_non_na_index, 'land_parcel_id'] = land_parcel_id_text.loc[land_parcel_non_na_index]\
                                                                                .apply(self.get_uuid, mode='hashlib')

        # create gls uuid
        if "date_award" in pred.columns:
            pred['award_month_index'] = pred.date_award.apply(self.get_time_index, format='%d/%m/%Y', dayfirst=True)

        pred['sg_gls_id'] = pred.land_parcel_name.apply(self.get_uuid, mode='default')
        gls_non_na_index = pred[(~pred.land_parcel_id.str.contains('0'*5))
                                & (pred.award_month_index.notna())
                                & (pred.site_area_sqm.notna())].index
        gls_id_text = pred.land_parcel_id + pred.award_month_index.astype(str)
        pred.loc[gls_non_na_index, 'sg_gls_id'] = gls_id_text.loc[gls_non_na_index].apply(self.get_uuid, mode='hashlib')

        output_cols = list(input_cols)
        # add id cols to output cols
        if 'sg_gls_id' not in input_cols:
            output_cols = ['sg_gls_id'] + output_cols
        if 'land_parcel_id' not in input_cols:
            output_cols = ['land_parcel_id'] + output_cols

        # update self.pred
        self.pred = pred[output_cols]

        return self.pred

    # to balance metrics of gfa, gpr and site area (just in case)
    # only fill in values for NA; will not overwrite non-NA values
    def balance_metrics(self, pred: pd.DataFrame = None) -> pd.DataFrame:
        # if param passed in, overwrite self
        if pred is not None:
            self.pred = pred

        pred = self.pred
        # get index of NA values
        gfa_na_index = pred[(pred.gfa_sqm.isna()) | (pred.gfa_sqm == 0)].index
        gpr_na_index = pred[(pred.gpr.isna()) | (pred.gpr == 0)].index
        area_na_index = pred[(pred.site_area_sqm.isna()) | (pred.site_area_sqm == 0)].index

        # calculate values and fill into table
        for i in gfa_na_index:
            try:
                pred.loc[i, 'gfa_sqm'] = pred.loc[i, 'site_area_sqm'] * pred.loc[i, 'gpr']
            except (TypeError, ZeroDivisionError, ValueError):
                pass

        for i in gpr_na_index:
            try:
                pred.loc[i, 'gpr'] = pred.loc[i, 'gfa_sqm'] / pred.loc[i, 'site_area_sqm']
            except (TypeError, ZeroDivisionError, ValueError):
                pass

        for i in area_na_index:
            try:
                pred.loc[i, 'site_area_sqm'] = pred.loc[i, 'gfa_sqm'] * pred.loc[i, 'gpr']
            except (TypeError, ZeroDivisionError, ValueError):
                pass

        # update
        self.pred = pred

        return self.pred

    # to align the columns between master and pred tables, by adjusting pred columns
    def align_columns(self, pred: pd.DataFrame = None, master: pd.DataFrame = None, fill_value=np.nan) -> pd.DataFrame:
        # if param passed in, overwrite self
        if pred is not None:
            self.pred = pred
        if master is not None:
            self.master = master

        # create NA cols for columns in master but not in pred
        pred_cols = list(self.pred.columns)
        master_cols = list(self.master.columns)
        additional_cols = [col for col in master_cols if col not in pred_cols]
        for col in additional_cols:
            self.pred[col] = fill_value
        # update
        self.pred = self.pred[self.master.columns]

        return self.pred

    def land_use_regroup(self, land_use_type_raw: str):
        if land_use_type_raw and pd.notna(land_use_type_raw):
            type_code_lower = land_use_type_raw.lower()
            if (any(kw in type_code_lower for kw in self.res_keywords) and any(kw in type_code_lower for kw in self.com_keywords)) \
                    or ('mix' in type_code_lower) or ('white' in type_code_lower):
                return 'mixed'
            elif any(kw in type_code_lower for kw in self.res_keywords):
                return 'residential'
            elif any(kw in type_code_lower for kw in self.com_keywords):
                return 'commercial'
            else:
                return 'others'

        else:
            return np.nan

    def project_type_regroup(self, project_type_raw: str):
        if project_type_raw:
            prj_type = ''.join([s for s in project_type_raw if s.isalpha()])
            if prj_type[:2] in self.res_kw_hdb_condo_dict.keys():
                return self.res_kw_hdb_condo_dict[prj_type[:2]]
            else:
                prj_type_lower = prj_type.lower()
                if (any(kw in prj_type_lower for kw in self.res_keywords)
                    and any(kw in prj_type_lower for kw in self.com_keywords)) \
                        or ('mix' in prj_type_lower) \
                        or ('white' in prj_type_lower):
                    return 'mixed'
                elif any(kw in prj_type_lower for kw in self.res_keywords):
                    return 'condo'
                elif any(kw in prj_type_lower for kw in self.com_keywords):
                    return 'commercial'
                elif 'hotel' in prj_type_lower:
                    return 'hotel'
                else:
                    return 'others'
        else:
            return np.nan

    # to merge func into one
    def process(self, pred: pd.DataFrame = None, master: pd.DataFrame = None, fill_value=np.nan) -> pd.DataFrame:
        if pred is not None:
            self.pred = pred
            self.pred_origin = pred.copy()
        if master is not None:
            self.master = master

        self.pred = self.pred_origin.copy()
        pred_encoded = self.encode()
        pred_balanced = self.balance_metrics()
        pred_aligned = self.align_columns()
        pred_aligned.loc[:, 'land_use_type'] = pred_aligned.land_use_type.apply(self.land_use_regroup)
        self.pred = pred_aligned

        return self.pred

    def find_region_zone(self, poi_a: pd.DataFrame, ref_a, dist_lim=3000, cutoff=False):
        """
        To find the region & zone of site based on historical data
        Usage: for each site, find one place within the distance limit and assign region & zone of the place to the site
        :param poi_a: dataframe poi data of sites whose region & zone info to be found
        :param ref_a: name of the column containing site id/names in poi_a (to loop through)
        :param dist_lim: (in meters) the nearest place found should fall within this distance limit
        :param cutoff: if True, break the loop when one location whose distance is within limit has been found
        :return: a dataframe with 3 columns: site id/names, region, zone
                * if cannot find any places to match, the returned table won't have record for the very site
        """

        distcal = PairwiseDistCalculator()

        # read in location_df as a reference providing info on region & zone of different addresses
        # location_df = self.dbconn.read_data('''select address_dwid, city_area as "zone", region_admin_district as region, latitude, longitude from masterdata_sg.address''')
        location_df = pd.read_csv("https://raw.githubusercontent.com/LUCASLIUHK/datasets/main/masterdata_sg_address.csv")
        location_df.rename(columns={'address_dwid': 'poi_b'}, inplace=True)
        location_df['coordinates'] = list(zip(location_df.latitude.tolist(), location_df.longitude.tolist()))

        # calculate distances
        dist_df = distcal.calculate_distance(poi_a, location_df, ref_a, 'poi_b', dist_lim, cutoff)
        location_df.reset_index(drop=True, inplace=True)
        poi_a.reset_index(drop=True, inplace=True)
        region_zone = dist_df.merge(location_df[['poi_b', 'region', 'zone']], how='left', on='poi_b').rename(columns={'poi_a': ref_a})
        return region_zone[[ref_a, 'region', 'zone']]
