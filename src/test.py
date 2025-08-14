import requests
import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
from io import StringIO
from bs4 import BeautifulSoup, Comment
ADVANCED_STATS = 'nba_24-25_advanced_team_stats.csv'

teamDict = {'GSW': 'Warriors', 'CHI': 'Bulls', 'CLE': 'Cavaliers', 
            'ATL': 'Hawks', 'BOS': 'Celtics', 'BKN': 'Nets', 'CHA': 'Hornets', 
            'DAL': 'Mavericks', 'DEN': 'Nuggets', 'DET': 'Pistons', 
            'HOU': 'Rockets', 'IND': 'Pacers', 'LAC': 'Clippers', 
            'LAL': 'Lakers', 'MEM': 'Grizzlies', 'MIA': 'Heat', 'MIL': 'Bucks', 
            'MIN': 'Timberwolves', 'NOP': 'Pelicans', 'NYK': 'Knicks',
            'OKC': 'Thunder', 'ORL': 'Magic', 'PHI': '76ers', 'PHX': 'Suns',
            'POR': 'Blazers', 'SAC': 'Kings', 'SAS': 'Spurs', 
            'TOR': 'Raptors','UTA': 'Jazz', 'WAS': 'Wizards'}

df = pd.read_csv(ADVANCED_STATS)
teamStats = df[['Team', 'SRS','NRtg', 'OeFG%','OTOV%','OORB%', 'DRB%']].copy()
teamStats['Team'] = teamStats['Team'].apply(lambda x: x.replace('*', ''))
# print(teamStats.head())

def getTeamID(team): #helper for getGameLog
    allTeams = teams.get_teams()
    teamX = [t for t in allTeams if t['abbreviation'] == team.upper()][0]
    teamID = teamX['id']
    return teamID

# print(getTeamID('MIL')) #

def getOpponent(str): #helper for getGameLog
    if 'vs.' in str:
        oppIndex = str.find('.')
        opponent = str[oppIndex+2:]
        return opponent, 1
    elif '@' in str:
        oppIndex = str.find('@')
        opponent = str[oppIndex+2:]
        return opponent, 0
    

teamID = '1610612749'
gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=str(teamID),
                                    season_type_nullable='Regular Season')
games = gamefinder.get_data_frames()[0]
print(games.head())
opponents = []
winLoss = []
HomeAway = []
for index, row in games.head(200).iterrows():
    matchup = games.loc[index, 'MATCHUP']
    opponent, homeAway = getOpponent(matchup)
    mascot = teamDict[opponent]
    opponents.append(mascot)
    HomeAway.append(homeAway)
    outcome = games.loc[index, 'WL']
    if outcome == 'W':
        winLoss.append(1)
    else:
        winLoss.append(0)
gameLog = pd.DataFrame(
    {
        'Opponent': opponents[::-1],
        'H1/A0': HomeAway[::-1],
        'W/L': winLoss[::-1]
    }
)
print(gameLog.head())