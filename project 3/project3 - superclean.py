import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
from pathlib import Path

pd.options.mode.chained_assignment = None

def clean():
    print('cleaning')
    df = pd.read_csv("NCSA_Reserves_DA_412_2.csv")
    df['Region'] = df['Region'].str.strip()
    df['Installment'] = df['Installment'].str.strip()
    df['Platform'] = df['Platform'].str.strip()
    df['Channel'] = df['Channel'].str.strip()
    df['Edition'] = df['Edition'].str.strip()
    df['RegionOrd'] = pd.Categorical(df.Region).codes
    df['PlatformOrd'] = pd.Categorical(df.Platform).codes
    df['ChannelOrd'] = pd.Categorical(df.Channel).codes
    df['EditionOrd'] = pd.Categorical(df.Edition).codes
    df['TitleOrd'] = pd.Categorical(df.Installment).codes

    d = pd.DataFrame(columns=['ReservesLevel','Installment','Region','Platform','Channel','Edition','RelativeWeek'])

    x = 0

    for u in df['Installment'].value_counts().index.tolist():
        for m in df['RelativeWeek'].value_counts().index.tolist():
            # print(m)
            d.loc[x] = [df['ReservesLevel'][np.logical_and(df['RelativeWeek'] == m, df['Installment'] == u)].sum(),
                        u, 'All', 'All', 'All', 'All', m]
            x = x + 1
            for i in df['Region'].value_counts().index.tolist():
                # print(i)
                d.loc[x] = [df['ReservesLevel'][np.logical_and(np.logical_and(df['Region'] == i, df['RelativeWeek'] == m), df['Installment'] == u)].sum(),
                            u, i, 'All', 'All', 'All', m]
                x = x + 1
                for j in df['Platform'].value_counts().index.tolist():
                    # print(j)
                    d.loc[x] = [df['ReservesLevel'][np.logical_and(
                        np.logical_and(np.logical_and(df['Region'] == i, df['Platform'] == j),
                        df['RelativeWeek'] == m), df['Installment'] == u)].sum(),
                                u, i, j, 'All', 'All', m]
                    x = x + 1
                    for k in df['Channel'].value_counts().index.tolist():
                        # print(k)
                        d.loc[x] = [df['ReservesLevel'][np.logical_and(np.logical_and(
                            np.logical_and(np.logical_and(df['Region'] == i, df['Platform'] == j), df['Channel'] == k), df['RelativeWeek'] == m), df['Installment'] == u)].sum(),
                                    u, i, j, k, 'All', m]
                        x = x + 1
                        for l in df['Edition'].value_counts().index.tolist():
                            # print(l)
                            # print(df['ReservesLevel'][np.logical_and(np.logical_and(
                            #     np.logical_and(np.logical_and(df['Region'] == i, df['Platform'] == j), df['Channel'] == k),
                            #     df['Edition'] == l), df['RelativeWeek'] == m)].sum())
                            d.loc[x] = [df['ReservesLevel'][np.logical_and(np.logical_and(np.logical_and(
                                np.logical_and(np.logical_and(df['Region'] == i, df['Platform'] == j), df['Channel'] == k),
                                df['Edition'] == l), df['RelativeWeek'] == m),df['Installment'] == u)].sum(),
                                         u,i,j,k,l,m]
                            x = x + 1

    d = d.sort_values(by=['RelativeWeek','Edition'])
    d.to_csv('cleaned2.csv')


clean()
