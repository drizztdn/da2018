import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

d = pd.read_csv("DA_Clean NCSA Reserves_4.14.18-FINAL.csv")

d = d.drop(d[d['Region'] != 'USA'].index)


def consolidate_values(data, column, old_value, new_value):
    for i in data['Installment'].unique():
        platforms = data.loc[data['Installment'] == i,column].unique()
        if(old_value in platforms and new_value in platforms):
            weeks = data[np.logical_and(data[column] == old_value, data['Installment'] == i)]['RelativeWeek'].unique()
            for week in weeks:
                old = data[(data[column] == old_value) & (data['RelativeWeek'] == week) & (data['Installment'] == i)]['ReservesLevel']
                new = data[(data[column] == new_value) & (data['RelativeWeek'] == week) & (data['Installment'] == i)]['ReservesLevel']
                data[(data[column] == new_value) & (data['RelativeWeek'] == week) & (data['Installment'] == i)][
                    'ReservesLevel'] = old + new
            data.drop(data[(data[column] == old_value) & (data['Installment'] == i)].index, inplace=True)
        else:
            if (old_value in platforms):
                data.loc[(data[column] == old_value) & (data['Installment'] == i),column] = new_value

consolidate_values(d, 'Platform', 'PS VITA', 'PS')
consolidate_values(d, 'Platform', 'PS4', 'PS')
consolidate_values(d, 'Platform', 'PS3', 'PS')
consolidate_values(d, 'Platform', 'XBOX360', 'XBOX')
consolidate_values(d, 'Platform', 'XBOX ONE', 'XBOX')
consolidate_values(d, 'Platform', 'Wii U', 'Nintendo')
consolidate_values(d, 'Platform', '3DS', 'Nintendo')
consolidate_values(d, 'Platform', 'Switch', 'Nintendo')


consolidate_values(d, 'Edition', 'Deluxe', 'Standard')
consolidate_values(d, 'Edition', 'Collector', 'Standard')
consolidate_values(d, 'Edition', 'Gold', 'Standard')

d['FinalPrice'] = '59.99'

d['FinalPrice'][d['Installment'] == 'Annie'] = '59.99'
d['FinalPrice'][d['Installment'] == 'RPG L'] = '19.99'
d['FinalPrice'][d['Installment'] == 'RPG E'] = '39.99'
d['FinalPrice'][d['Installment'] == 'RPG 3'] = '39.99'


d.to_csv('cleaned.csv')
