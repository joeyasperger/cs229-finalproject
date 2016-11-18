import csv
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt
import pickle


players = []
headers = []
rookieYears = {}

with open('Pitching.csv', 'rb') as csvfile:
    statsreader = csv.reader(csvfile)
    for row in statsreader:
        if len(headers) == 0:
            headers = row
        else:
            player = {}
            year = int(row[1])
            if year > 1990 and row[headers.index('IPouts')] != '':
                ipOuts = int(row[headers.index('IPouts')])
                playerId = row[0]
                if ipOuts > 150 and year < rookieYears.get(playerId, 2020):
                    rookieYears[playerId] = year
            if year > 2008:
                for i in range(len(headers)):
                    player[headers[i]] = row[i]
                players.append(player)



# x.append(feature_vec)
# y.append(result)
# scaler.fit(vec.fit_transform(x).toarray())
# reg.fit(scaler.transform(vec.transform(x).toarray()),y)