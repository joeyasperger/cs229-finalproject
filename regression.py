import csv
import numpy as np
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle
import optunity
import optunity.metrics
import pandas

YEAR = 1975

def getWar():
    headers = []
    years = []
    for i in range(2015 - YEAR + 2):
        years.append({})
    with open('war_daily_pitch.txt', 'rb') as war_file:
        war_reader = csv.reader(war_file)
        for row in war_reader:
            if len(headers) == 0:
                headers = row
            else:
                year = int(row[4])
                if year < YEAR:
                    continue
                player_id = row[3]
                war_string = row[28];
                if war_string != 'NULL':
                    war = float(war_string)
                    years[year - YEAR][player_id] = war
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
                if row[headers.index('IPouts')] != '':
                    ipOuts = int(row[headers.index('IPouts')])
                    playerId = row[0]
                    if ipOuts > 150 and year < rookieYears.get(playerId, 2020):
                        rookieYears[playerId] = year
                if year > YEAR:
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
    if features['W'] + features['L'] > 0:
        features['W%'] = features['W'] / (features['W'] + features['L'])
    else:
        features['W%'] = 0
    features['IP'] = ip
    features['SO'] = float(player['SO'])
    features['SO/IP'] = features['SO']/ip
    features['GS'] = int(player['GS'])
    features['H'] = int(player['H'])
    features['H/IP'] = features['H']/ip
    features['HR'] = int(player['HR'])
    features['HR/IP'] = features['HR']/ip
    features['BB'] = int(player['BB'])
    features['BB/IP'] = features['BB']/ip
    if '.' not in player['BAOpp']:
        features['BAOpp'] = 0.0
    else:
        features['BAOpp'] = float(player['BAOpp'])
    features['R'] = int(player['R'])
    features['ER'] = int(player['ER'])
    features['rookieWAR'] = player['rookieWAR']
    return features

def getPlayersFutureWar(players, rookieYears, warYears):
    data = []
    for player in players:
        if int(player['yearID']) == rookieYears.get(player['playerID']) and int(player['yearID']) < 2013:
            war0 = warYears[int(player['yearID'])-YEAR].get(player['playerID'])
            war1 = warYears[int(player['yearID'])-YEAR + 1].get(player['playerID'])
            war2 = warYears[int(player['yearID'])-YEAR + 2].get(player['playerID'])
            war3 = warYears[int(player['yearID'])-YEAR + 3].get(player['playerID'])
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
    differences = [];
    big_misses = [];
    for index in range(len(predicted)):
        print('predicted ', predicted[index], ' actual ', y[index], ' difference ', predicted[index] - y[index])
        differences.append(abs(predicted[index] - y[index]));
        if(differences[len(differences) - 1] > 5):
            big_misses.append(y[index])
    # print(differences)
    print(big_misses)
    print(max(predicted))
    return mean_squared_error(y, predicted)

def assignClassLabels(y):
    # stud = 1, scrub = 0
    classes = []
    for war in y:
        if war > 5:
            classes.append(1)
        else:
            classes.append(0)
    return classes  


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


classes = assignClassLabels(y_train)

space = {'kernel': {'linear': {'C': [0, 2]},
                    'rbf': {'logGamma': [-5, 0], 'C': [0, 10]},
                    'poly': {'degree': [2, 5], 'C': [0, 5], 'coef0': [0, 2]}
                    }
         }

def train_model(x_train, y_train, kernel, C, logGamma, degree, coef0):
    """A generic SVM training function, with arguments based on the chosen kernel."""
    if kernel == 'linear':
        model = SVC(kernel=kernel, C=C, class_weight='balanced')
    elif kernel == 'poly':
        model = SVC(kernel=kernel, C=C, degree=degree, coef0=coef0, class_weight='balanced')
    elif kernel == 'rbf':
        model = SVC(kernel=kernel, C=C, gamma=10 ** logGamma, class_weight='balanced')
    else:
        raise ArgumentError("Unknown kernel function: %s" % kernel)
    model.fit(x_train, y_train)
    return model

# @optunity.cross_validated(x=scaler.fit_transform(vec.fit_transform(x_train).toarray()), y=classes, num_folds=3)
cv_decorator = optunity.cross_validated(x=scaler.fit_transform(vec.fit_transform(x_train).toarray()), y=classes, num_folds=3)

def svm_rbf_tuned_auroc(x_train, y_train, x_test, y_test, C, logGamma):
    model = SVC(C=C, gamma=10 ** logGamma, class_weight='balanced').fit(x_train, y_train)
    decision_values = model.decision_function(x_test)
    auc = optunity.metrics.roc_auc(y_test, decision_values)
    return auc

def svm_tuned_auroc(x_train, y_train, x_test, y_test, kernel='linear', C=0, logGamma=0, degree=0, coef0=0):
    model = train_model(x_train, y_train, kernel, C, logGamma, degree, coef0)
    decision_values = model.decision_function(x_test)
    return optunity.metrics.roc_auc(y_test, decision_values)

svm_tuned_auroc = cv_decorator(svm_tuned_auroc)

#print svm_default_auroc(C=1.0, logGamma=0.0)

# optimal_rbf_pars, info, _ = optunity.maximize(svm_rbf_tuned_auroc, num_evals=150, C=[0, 10], logGamma=[-5, 0])
optimal_svm_pars, info, _ = optunity.maximize_structured(svm_tuned_auroc, space, num_evals=150)

# when running this outside of IPython we can parallelize via optunity.pmap
# optimal_rbf_pars, _, _ = optunity.maximize(svm_rbf_tuned_auroc, 150, C=[0, 10], gamma=[0, 0.1], pmap=optunity.pmap)

print("Optimal parameters: " + str(optimal_svm_pars))
print("AUROC of tuned SVM with RBF kernel: %1.3f" % info.optimum)

# df = optunity.call_log2dataframe(info.call_log)
# print df.sort('value', ascending=False)

svc = SVC(C=optimal_svm_pars['C'], kernel='rbf', gamma=10 ** optimal_svm_pars['logGamma'], class_weight='balanced')

# scaler.fit_transform(vec.fit_transform(x_train).toarray())
scores = cross_val_score(svc, scaler.transform(vec.transform(x_train).toarray()), classes)
print 'cross validation scores: ', scores
svc.fit(scaler.transform(vec.transform(x_train).toarray()), classes)

index = 0
count = 0
vals = []
for feature in scaler.transform(vec.transform(x_train).toarray()):
    val = svc.predict([feature])
    if(val[0] != classes[index]):
        vals.append(val[0])
        count += 1
    index += 1
print 'incorrectly predicted:', vals
print 'num incorrect:', count, 'out of:', index


index = 0
count = 0
test_classes = assignClassLabels(y_test)
vals = []
for feature in vec.transform(x_test).toarray():
    val = svc.predict([feature])
    if(val[0] != test_classes[index]):
        vals.append(val[0])
        count += 1
    index += 1

print 'test set classes = ', test_classes
print 'incorrectly predicted in test set:', vals
print 'num incorrect:', count, 'out of:', index




# reg = SGDRegressor(loss='squared_loss', n_iter=1000, verbose=2, penalty='l2', \
#     alpha= 0.001, learning_rate="invscaling", eta0=0.002, power_t=0.4)
# svr = SVR(kernel='rbf', C=100, gamma=.001)
# scaler.fit(vec.fit_transform(x_train).toarray())
# scores = cross_val_score(reg, scaler.transform(vec.transform(x_train).toarray()),y_train, cv=5)
# print scores.mean()

# scores = cross_val_score(svr, scaler.transform(vec.transform(x_train).toarray()),y_train, cv=5)
# print scores.mean()
# print(len(data))

# reg.fit(scaler.transform(vec.transform(x_train).toarray()),y_train)
# svr.fit(scaler.transform(vec.transform(x_train).toarray()),y_train)

# print 'Training loss SGDRegressor = ', squared_loss(x_train, y_train, reg, scaler, vec)
# print 'Training loss SVR = ', squared_loss(x_train, y_train, svr, scaler, vec)

# print 'Test loss SGD= ', squared_loss(x_test, y_test, reg, scaler, vec)
# print 'Test loss SVR= ', squared_loss(x_test, y_test, svr, scaler, vec)
# print min(y_test)
# print min(y_train)

# plt.boxplot(y)
# plt.show()




