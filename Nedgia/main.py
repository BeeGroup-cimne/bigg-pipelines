import pandas as pd

tz_info = 'Europe/Madrid'


def script_input():
    df = pd.read_excel('../data/Generalitat Extracción_2018.xlsx', skiprows=2)

    # Transform Raw Data

    df['Fecha fin Docu. cálculo'] += pd.Timedelta(hours=23)

    df['Fecha inicio Docu. cálculo'] = df['Fecha inicio Docu. cálculo'].dt.tz_localize(tz_info)
    df['Fecha fin Docu. cálculo'] = df['Fecha fin Docu. cálculo'].dt.tz_localize(tz_info)

    # datatime64 [ns] to unix time
    df['measurementStart'] = df['Fecha inicio Docu. cálculo'].astype('int') / 10 ** 9
    df['measurementStart'] = df['measurementStart'].astype('int')

    df['measurementEnd'] = df['Fecha fin Docu. cálculo'].astype('int') / 10 ** 9
    df['measurementEnd'] = df['measurementEnd'].astype('int')
    df['ts'] = df['measurementStart']

    # Calculate kWh
    df['measurementValue'] = df['Consumo kWh ATR'].fillna(0) + df['Consumo kWh GLP'].fillna(0)

    df = df[['CUPS', 'ts', 'measurementStart', 'measurementEnd', 'measurementValue', 'Tipo Lectura']]

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
            new_df.rename(columns={'_6': 'Tipo Lectura'}, inplace=True)
            list_of_dataframes[i] = new_df

    else:
        list_of_dataframes.append(dataframe)

    return list_of_dataframes


def harmonize_dataframe(dataframe):
    if 'REAL' in dataframe['Tipo Lectura'].unique():
        # Drop n rows until REAL
        first_real = dataframe[dataframe['Tipo Lectura'] == 'REAL'].index[0]

        df_i = dataframe.loc[first_real:]
    else:
        return

    # Filter by Tipo Lectura == REAL
    df_i_real = df_i[df_i["Tipo Lectura"] == 'REAL'].copy()

    df_i_real['cumsum'] = df_i_real['measurementValue'].cumsum()

    df_i_real.set_index('measurementStart_dt', inplace=True)

    interpolate_df = df_i_real[['cumsum']].resample('1D').interpolate()

    interpolate_df['diff'] = interpolate_df['cumsum'].diff()

    interpolate_df.at[interpolate_df.index[0], 'diff'] = interpolate_df.iloc[0]['cumsum']

    interpolate_df['measurementEnd_dt'] = interpolate_df.index + pd.Timedelta(hours=23)

    interpolate_df.reset_index(inplace=True)

    interpolate_df.rename(columns={'measurementStart_dt': 'measurementStart', 'diff': 'measurementValue',
                                   'measurementEnd_dt': 'measurementEnd'}, inplace=True)
    interpolate_df['CUPS'] = cups

    interpolate_df['measurementStart'] = interpolate_df['measurementStart'].astype('int') / 10 ** 9
    interpolate_df['measurementStart'] = interpolate_df['measurementStart'].astype('int')

    interpolate_df['measurementEnd'] = interpolate_df['measurementEnd'].astype('int') / 10 ** 9
    interpolate_df['measurementEnd'] = interpolate_df['measurementEnd'].astype('int')

    return interpolate_df[['CUPS', 'measurementStart', 'measurementEnd', 'measurementValue']]


if __name__ == '__main__':
    df = script_input()

    # Group by CUPS
    for cups, sub_df in df.groupby('CUPS'):
        # Convert Timestamps to Datetime
        sub_df.sort_values(by=['measurementStart'], inplace=True)
        sub_df['measurementStart_dt'] = pd.to_datetime(sub_df['measurementStart'], unit='s').dt.tz_localize(
            'UTC').dt.tz_convert(tz_info)

        sub_df['measurementEnd_dt'] = pd.to_datetime(sub_df['measurementEnd'], unit='s').dt.tz_localize(
            'UTC').dt.tz_convert(tz_info)

        sub_df['measurementEnd_dt'] -= pd.Timedelta(hours=23)

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
            print(df_res)

    # todo: MeasureList Day
    # todo: time series -> HBASE
    # todo: inputs, workflows
