import csv
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle


def getWar():
    headers = []
    years = []
    for i in range(9):
        years.append({})
    with open('war_daily_pitch.txt', 'rb') as war_file:
        war_reader = csv.reader(war_file)
        for row in war_reader:
            if len(headers) == 0:
                headers = row
            else:
                year = int(row[4])
                if year < 2008:
                    continue
                player_id = row[3]
                war_string = row[28];
                if war_string != 'NULL':
                    war = float(war_string)
                    years[year - 2008][player_id] = war
    return years

def getStats():
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
    return players, rookieYears

def getFeatureDict(player):
    features = {}
    ip = int(player['IPouts']) / 3.0
    features['ERA'] = int(player['ER']) / ip * 9
    features['W'] = float(player['W'])
    features['L'] = float(player['L'])
    features['IP'] = ip
    features['SO/IP'] = float(player['SO']) / ip
    return features

def getPlayersFutureWar(players, rookieYears, warYears):
    data = []
    for player in players:
        if int(player['yearID']) == rookieYears.get(player['playerID']) and int(player['yearID']) < 2013:
            player['futureWAR'] = warYears[int(player['yearID'])-2008].get(player['playerID'])
            if player['futureWAR'] != None:
                data.append(player)
    return data

def squared_loss(data, reg, scaler, vec):
    loss = 0.0
    count = 0
    for player in data:
        features = getFeatureDict(player)
        predicted = reg.predict(scaler.transform(vec.transform(features).toarray()))[0]
        loss += (predicted - player['futureWAR']) ** 2
        count += 1
    return loss / count


players, rookieYears = getStats()
warYears = getWar()
data = getPlayersFutureWar(players, rookieYears, warYears)
trainingData = data[0:200]
testData = data[200:]

x = []
y = []
for player in trainingData:
    features = getFeatureDict(player)
    x.append(features)
    y.append(player['futureWAR'])


vec = DictVectorizer()
scaler = StandardScaler()
reg = SGDRegressor(loss='squared_loss', n_iter=1000, verbose=2, penalty='l2', \
    alpha= 0.001, learning_rate="invscaling", eta0=0.002, power_t=0.4)
scaler.fit(vec.fit_transform(x).toarray())
reg.fit(scaler.transform(vec.transform(x).toarray()),y)

print 'Training loss = ', squared_loss(trainingData, reg, scaler, vec)
print 'Test loss = ', squared_loss(testData, reg, scaler, vec)


