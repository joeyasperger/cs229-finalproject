import csv

def main():

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

if __name__ == "__main__":
    main()