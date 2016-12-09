import csv
from pymongo import MongoClient
from bson.objectid import ObjectId

client = MongoClient('localhost', 27017)
db = client["pitchfx"]

# players = {}
# headers = []
# rookieYears = {}
# with open('Pitching.csv', 'rb') as csvfile:
#     statsreader = csv.reader(csvfile)
#     for row in statsreader:
#         if len(headers) == 0:
#             headers = row
#         else:
#             player = {}
#             year = int(row[1])
#             if year > 2004 and row[headers.index('IPouts')] != '':
#                 ipOuts = int(row[headers.index('IPouts')])
#                 playerId = row[0]
#                 if ipOuts > 150 and year < rookieYears.get(playerId, 2020):
#                     rookieYears[playerId] = year
#             if year > 2008:
#                 for i in range(len(headers)):
#                     player[headers[i]] = row[i]
#                 players[playerId] = player

# print len(players)
# for playerId in players.keys():
#     if rookieYears.get(playerId) == None:
#         del players[playerId]
# print len(players)

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

def getPlayersFutureWar(players, rookieYears, warYears):
    data = {}
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
                data[player['playerID']] = player
    print len(data)
    return data

playersRaw, rookieYears = getStats()
warYears = getWar()
players = getPlayersFutureWar(playersRaw, rookieYears, warYears)

headers = []
with open('Master.csv', 'rb') as csvfile:
    master = csv.reader(csvfile)
    for row in master:
        if len(headers) == 0:
            headers = row
        else:
            playerId = row[headers.index('playerID')]
            if playerId in players.keys():
                players[playerId]['nameFirst'] = row[headers.index('nameFirst')]
                players[playerId]['nameLast'] = row[headers.index('nameLast')]

found = 0
missing = 0
fxtoID = {}
for playerId in players.keys():
    lastname = players[playerId]['nameLast']
    firstname = players[playerId]['nameFirst']
    if lastname == 'Happ':
        firstname = 'J.A.'
    elif lastname == 'Niese':
        firstname = 'Jon'
    elif lastname == 'Deduno':
        firstname = 'Samuel'

    if lastname == 'Hernandez' and firstname == 'David':
        pitchfxID = '456696'
    elif lastname == 'Gonzalez' and firstname == 'Miguel':
        pitchfxID = '456068'
    elif lastname == 'Rodriguez' and firstname == 'Henry':
        pitchfxID = '469159'
    else:
        for player in db.players.find({'last_name':lastname, 'first_name': firstname}):
            pitchfxID = player['player_id']
    players[playerId]['pitchfxID'] = pitchfxID
    fxtoID[pitchfxID] = playerId

for playerId in players.keys():
    player = players[playerId]
    player['spd_total'] = 0
    player['spd_count'] = 0

for pitch in db.pitches.find({"pitch_type":{"$in": ["FF", "FT"]}}):
    pitchfxID = pitch['pitcher']
    if pitchfxID in fxtoID.keys():
        playerId = fxtoID[pitch['pitcher']]
        players[playerId]['spd_total'] += pitch['start_speed']
        players[playerId]['spd_count'] += 1

for playerId in players.keys():
    player = players[playerId]
    if player['spd_count'] > 0:
        avg = player['spd_total']/player['spd_count']
        player['avg_fastball_speed'] = avg
    print player['nameFirst'], player['nameLast'], avg





