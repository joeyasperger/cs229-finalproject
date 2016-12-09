import csv
import numpy as np
from sklearn.svm import SVR
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
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
    features['ERA'] = float(player['ERA'])
    features['W'] = int(player['W'])
    features['L'] = int(player['L'])
    features['IP'] = ip
    features['SO'] = float(player['SO'])
    features['SO/IP'] = features['SO']/ip
    features['GS'] = int(player['GS'])
    features['H'] = int(player['H'])
    features['H/IP'] = features['H']/ip
    features['HR'] = int(player['HR'])
    features['BB'] = int(player['BB'])
    features['BB/IP'] = features['BB']/ip
    features['BAOpp'] = float(player['BAOpp'])
    features['R'] = int(player['R'])
    features['ER'] = int(player['ER'])
    features['rookieWAR'] = player['rookieWAR']
    return features

def getPlayersFutureWar(players, rookieYears, warYears):
    data = []
    for player in players:
        if int(player['yearID']) == rookieYears.get(player['playerID']) and int(player['yearID']) < 2013:
            war0 = warYears[int(player['yearID'])-2008].get(player['playerID'])
            war1 = warYears[int(player['yearID'])-2008 + 1].get(player['playerID'])
            war2 = warYears[int(player['yearID'])-2008 + 2].get(player['playerID'])
            war3 = warYears[int(player['yearID'])-2008 + 3].get(player['playerID'])
            if war0 != None and war1 != None and war2 != None and war3 != None:
            # if war0 != None and war1 != None:
                player['futureWAR'] = war1 + war2 + war3
                # player['futureWAR'] = war1
                player['rookieWAR'] = war0
                data.append(player)
    print len(data)
    return data

# def squared_loss(feature_list, war, reg, scaler, vec):
#     loss = 0.0
#     count = 0
#     for features in feature_list:
#         predicted = reg.predict(scaler.transform(vec.transform(features).toarray()))[0]
#         loss += (predicted - war[count]) ** 2
#         count += 1
#     return loss / count

def squared_loss(x, y, reg, scaler, vec):
    predicted = []
    for player in x:
        predicted.append(reg.predict(scaler.transform(vec.transform(player).toarray())));
    return mean_squared_error(y, predicted)


players, rookieYears = getStats()
warYears = getWar()
data = getPlayersFutureWar(players, rookieYears, warYears)

x = []
y = []
for player in data:
    features = getFeatureDict(player)
    x.append(features)
    y.append(player['futureWAR'])
    #y.append(player['rookieWAR'])

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=.20, random_state=0)

vec = DictVectorizer()
scaler = StandardScaler()
reg = SGDRegressor(loss='squared_loss', n_iter=1000, verbose=2, penalty='l2', \
    alpha= 0.001, learning_rate="invscaling", eta0=0.002, power_t=0.4)
svr = SVR(kernel='rbf', C=100, gamma=.001)
scaler.fit(vec.fit_transform(x_train).toarray())
scores = cross_val_score(reg, scaler.transform(vec.transform(x_train).toarray()),y_train, cv=5)
print scores.mean()

scores = cross_val_score(svr, scaler.transform(vec.transform(x_train).toarray()),y_train, cv=5)
print scores.mean()
print(len(data))

reg.fit(scaler.transform(vec.transform(x_train).toarray()),y_train)
svr.fit(scaler.transform(vec.transform(x_train).toarray()),y_train)

print 'Training loss SGDRegressor = ', squared_loss(x_train, y_train, reg, scaler, vec)
print 'Training loss SVR = ', squared_loss(x_train, y_train, svr, scaler, vec)
# print 'Test loss = ', squared_loss(x_test, y_test, reg, scaler, vec)


