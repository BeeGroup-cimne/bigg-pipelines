import argparse
import hashlib
import re
from datetime import datetime

import pandas as pd
from neo4j import GraphDatabase
from rdflib import Namespace

from utils.hbase import save_to_hbase, get_hbase_data_batch
from utils.utils import read_config

tz_info = 'Europe/Madrid'


def script_input(config):
    hbase_table = "harmonized_ts_invoices_invoices_icaen"
    hbase_conn = config['hbase_store_harmonized_data']

    dic_list = []
    for data in get_hbase_data_batch(hbase_conn, hbase_table):
        print("parsing hbase")
        for id_, x in data:
            item = dict()
            for k, v in x.items():
                k1 = re.sub("^info:", "", k.decode("utf-8"))
                item[k1] = v.decode("utf-8")
            cups, ts = id_.decode("utf-8").split("~")
            item.update({"CUPS": cups, "start": ts, "ts": ts})
            dic_list.append(item)

    df = pd.DataFrame(dic_list)
    df.rename(columns={"v:value": "value"}, inplace=True)

    return df


def split_dataframe(dataframe):
    list_of_dataframes = []
    if True in dataframe['is_gap'].unique():

        last_status = None
        index = None

        for row in dataframe.itertuples():

            if last_status is None:
                index = 0
                list_of_dataframes.append([])

            if last_status is not None and row.is_gap is True:
                index += 1
                list_of_dataframes.append([])

            if last_status is not None and row.is_gap is False and last_status is True:
                index += 1
                list_of_dataframes.append([])

            list_of_dataframes[index].append(row)
            last_status = row.is_gap

        for i in range(len(list_of_dataframes)):
            new_df = pd.DataFrame(list_of_dataframes[i])
            list_of_dataframes[i] = new_df

    else:
        list_of_dataframes.append(dataframe)

    return list_of_dataframes


def harmonize_dataframe(dataframe):
    if True in dataframe['isReal'].unique():
        # Drop n rows until REAL
        first_real = dataframe[dataframe['isReal'] == True].index[0]
        df_i = dataframe.loc[first_real:]
    else:
        return

    df_i_real = df_i[df_i["isReal"] == True].copy()

    df_i_real['cumsum'] = df_i_real['value'].cumsum()

    df_i_real.set_index('measurementStart_dt', inplace=True)

    interpolate_df = df_i_real[['cumsum']].resample('1D').interpolate()

    interpolate_df['diff'] = interpolate_df['cumsum'].diff()

    interpolate_df.at[interpolate_df.index[0], 'diff'] = interpolate_df.iloc[0]['cumsum'] # todo: monthly_value/ nº days of invoice

    interpolate_df['measurementEnd_dt'] = interpolate_df.index + pd.Timedelta(hours=23)

    interpolate_df.reset_index(inplace=True)

    interpolate_df.rename(columns={'measurementStart_dt': 'start', 'diff': 'value',
                                   'measurementEnd_dt': 'end'}, inplace=True)
    interpolate_df['CUPS'] = cups

    interpolate_df['start'] = interpolate_df['start'].astype('int') / 10 ** 9
    interpolate_df['start'] = interpolate_df['start'].astype('int')

    interpolate_df['end'] = interpolate_df['end'].astype('int') / 10 ** 9
    interpolate_df['end'] = interpolate_df['end'].astype('int')

    return interpolate_df[['CUPS', 'start', 'end', 'value']]


def create_measurement_list(df, device_id, args, config):
    neo4j_connection = config['neo4j']
    neo = GraphDatabase.driver(**neo4j_connection)

    n = Namespace(args.namespace)

    with neo.session() as session:
        query = f"""
        MATCH (n:ns0__MeasurementList{{ns0__measurementKey:"{device_id}"}})-[:ns0__hasMeasurementLists]-(b) return b
        """
        for i in list(session.run(query)):
            cups = i['b']['ns0__deviceName']
            uri = n[f"{cups}-DEVICE-{args.source}"]

            list_uri = n[f"{cups}-DEVICE-{args.source}-LIST-PROJECTED-P1D"]
            new_d_id = hashlib.sha256(list_uri.encode("utf-8"))
            new_d_id = new_d_id.hexdigest()

            dt_ini = df['start'].iloc[0]
            dt_end = df['end'].iloc[-1]

            query_measures = f"""
                MATCH (device: ns0__Device {{uri:"{uri}"}})
                MERGE (list: ns0__MeasurementList{{uri: "{list_uri}", ns0__measurementKey: "{new_d_id}",
                ns0__measurementFrequency: "P1D"}} )<-[:ns0__hasMeasurementLists]-(device)
                SET
                    list.ns0__measurementUnit= "kWh",
                    list.ns0__measuredProperty= "gasConsumption",
                    list.ns0__measurementListStart = CASE
                        WHEN list.ns0__measurementListStart <
                         datetime("{datetime.fromtimestamp(dt_ini).isoformat()}")
                            THEN list.ns0__measurementListStart
                            ELSE datetime("{datetime.fromtimestamp(dt_ini).isoformat()}")
                        END,
                    list.ns0__measurementListEnd = CASE
                        WHEN list.ns0__measurementListEnd >
                         datetime("{datetime.fromtimestamp(dt_end).isoformat()}")
                            THEN list.ns0__measurementListEnd
                            ELSE datetime("{datetime.fromtimestamp(dt_end).isoformat()}")
                        END
                return list
            """

        session.run(query_measures)


def save_ts_to_hbase(df, device_id, args, config):
    hbase_conn2 = config['hbase_store_harmonized_data']

    n = Namespace(args.namespace)

    list_uri = n[f"{device_id}-DEVICE-{args.source}-LIST-PROJECTED-P1D"]
    new_d_id = hashlib.sha256(list_uri.encode("utf-8"))
    new_d_id = new_d_id.hexdigest()

    df['listKey'] = new_d_id

    save_to_hbase(df.to_dict(orient="records"), f"harmonized_ts_EnergyConsumptionGas_P1D_{args.user}", hbase_conn2,
                  [("info", ['end']), ("v", ['value'])],
                  row_fields=['listKey', 'start'])

    save_to_hbase(df.to_dict(orient="records"), f"harmonized_analyticsTs_EnergyConsumptionGas_P1D_{args.user}",
                  hbase_conn2,
                  [("info", ['end']), ("v", ['value'])],
                  row_fields=['start', 'listKey'])


if __name__ == '__main__':
    # Arguments
    ap = argparse.ArgumentParser(description='Mapping of Gas data to neo4j.')
    ap.add_argument("--user", "-u", help="The user importing the data", required=True)
    ap.add_argument("--source", "-so", help="The source importing the data", required=True)
    ap.add_argument("--namespace", "-n", help="The subjects namespace uri", required=True)
    ap.add_argument("--timezone", "-tz", help="The local timezone", required=True, default='Europe/Madrid')
    args = ap.parse_args()

    # Read Config
    config = read_config('../config.json')

    # Load Input
    df = script_input(config)

    # Group by CUPS
    for cups, sub_df in df.groupby('CUPS'):
        # Convert Timestamps to Datetime
        sub_df.sort_values(by=['start'], inplace=True)
        sub_df['measurementStart_dt'] = pd.to_datetime(sub_df['start'], unit='s').dt.tz_localize(
            'UTC').dt.tz_convert(tz_info)

        sub_df['measurementEnd_dt'] = pd.to_datetime(sub_df['end'], unit='s').dt.tz_localize(
            'UTC').dt.tz_convert(tz_info)

        sub_df['measurementEnd_dt'] -= pd.Timedelta(hours=23)

        sub_df['isReal'] = sub_df['isReal'].astype(bool)
        sub_df['value'] = sub_df['value'].astype(int)

        # https://towardsdatascience.com/efficiently-iterating-over-rows-in-a-pandas-dataframe-7dd5f9992c01

        # Shift
        sub_df['measurementStart_dt_shifted'] = sub_df['measurementStart_dt'].shift(-1)

        # Compare invoices start and end dates
        sub_df['is_gap'] = (sub_df['measurementStart_dt_shifted'] - sub_df['measurementEnd_dt']).to_list()

        # Find gaps between invoices
        sub_df.loc[sub_df["is_gap"] != pd.Timedelta(days=1), "is_gap"] = True
        sub_df.loc[sub_df["is_gap"] == pd.Timedelta(days=1), "is_gap"] = False
        sub_df.loc[sub_df["measurementStart_dt_shifted"].isnull(), "is_gap"] = False

        # Split Dataframes
        list_of_dataframes = split_dataframe(dataframe=sub_df)

        # Loop for each dataframe
        for df_i in list_of_dataframes:
            df_res = harmonize_dataframe(df_i)
            if df_res is not None:
                create_measurement_list(df=df_res, device_id=cups, args=args, config=config)
                save_ts_to_hbase(df=df_res, device_id=cups, args=args, config=config)
