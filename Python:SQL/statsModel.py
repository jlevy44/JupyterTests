# Adopted from Jordan Levy's code
import requests
import urllib
import urllib.request
import numpy as np, pandas as pd

from bs4 import BeautifulSoup

totalArr = np.empty
def make_soup(url):
    url = urllib.request.urlopen(url)
    soupdata = BeautifulSoup(url, "html.parser")
    return soupdata
#data = []
def index_containing_substring(the_list, substring):
    for i, s in enumerate(the_list):
        if substring in s:
              return i
    return -1
for year in map(str,range(2010,2017)):
    for week in range(1,18):
        savedplayerdata=""
        #playerdata = ""
        soup_proj = make_soup("http://www.fftoday.com/rankings/playerwkproj.php?Season=%s&GameWeek=%d&PosID=10&LeagueID=189999"%(year,week))
        # print(soup.find('table',{"width":"100%", "cellpadding":"2"}))
        

        table = soup_proj.find('table', {"width": "100%", "cellpadding": "2"})
        
        #for rows in table.findAll('tr'):
        #playerdata=""
        playerdata = ','.join([data.text.strip(' ') for data in table.findAll('td', {"class":"bodycontent"})]).replace('\xa0,\xa0','').replace('\xa0','').split(',')
        playerdata = list(filter(lambda a: a != '', playerdata))
        playerdata = [playerdata[i:i+12] for i in range(len(playerdata))[::12]]
        
        soup_actual = make_soup('http://www.fftoday.com/stats/playerstats.php?Season=%s&GameWeek=%d&PosID=10&LeagueID=189999'%(year,week))
        table_actual = soup_actual.find('table', {"width": "100%", "cellpadding": "2"})


        lines = table_actual.text.splitlines()
        playerdata_act = lines[index_containing_substring(lines,'.'):]
        #print(soup_actual.text)
        #playerdata_act = ','.join([data.text.strip(' ') for data in table_actual.findAll('TD', {"CLASS":"sort1"})]).replace('\xa0,\xa0','').replace('\xa0','').split(',')
        playerdata_act = list(filter(lambda a: a != '', playerdata_act))
        #print(playerdata_act)
        #print(playerdata_act[481])
        #print(list(range(len(playerdata_act))[::13]))
        playerdata_act = {playerdata_act[i].split('. ')[1]:playerdata_act[i+12] for i in range(len(playerdata_act))[::13]}#playerdata_act[i+11]
        #print(playerdata_act)


        
        #playerdata = [player.replace('\xa0,\xa0','').replace('\xa0','')]

        #playerdata+=data.text+","
        #print(playerdata)
        
        playerArr = np.array(playerdata)
        playerArr[:,1] = 'QB'
        playerArr[:,2] = week
        playerArr = np.insert(playerArr,2,int(year),axis=1)
        playerArr = np.column_stack([playerArr,np.ones((len(playerArr),1,))])#np.concatenate((playerArr,np.nan((len(playerArr),1,))),axis=1)
        #print(playerArr)
        playerArr[:,-1] = np.nan
        #print(playerArr)
        #rmRows = []
        for i,player in enumerate(playerArr[:,1]):
            try:
                playerArr[i,-1] = playerdata_act[playerArr[i,0]]
            except:
                pass#rmRows.append(i)
        #print(rmRows)
        #print(playerArr[:,-1])
        #print([playerdata_act[player] for player in playerArr[:,0]])
        playerArr = playerArr[playerArr[:,-1] != 'nan']#np.delete(playerArr,rmRows,1)
        #print(playerArr)
        
        
        #print(playerArr)
        #print(playerArr)
        if week == 1 and int(year) == 2010:
            totalArr = playerArr
        else:
            totalArr=np.concatenate((totalArr,playerArr),axis=0)
            #if totalArr.shape[1] != playerArr.shape[1]:
            #    print(playerArr)
            #    print(playerdata)
            #    print(playerdata_act)
#print(totalArr.shape)
#print(len(totalArr[1,:]))
#print(totalArr)
    #print(totalArr)
            #print(totalArr)
        #playerArr[:,3:] = playerArr[:,3:])
        #print(playerArr[10,6]-playerArr[10,7])
    #if int(year) == 2010:
    #    totalArr2 = totalArr
    #else:
    #    totalArr2=np.concatenate((totalArr2,totalArr),axis=0)
players_train = pd.DataFrame(totalArr)
print(players)
#data.append(players)

#print(data)
#players = players.set_index(0)

#print(players)
#savedplayerdata= savedplayerdata + "\n" + playerdata[1:]

#print(savedplayerdata)

############################

# grab 2017 projection data
year = '2016'
def make_soup(url):
    url = urllib.request.urlopen(url)
    soupdata = BeautifulSoup(url, "html.parser")
    return soupdata
#data = []
def index_containing_substring(the_list, substring):
    for i, s in enumerate(the_list):
        if substring in s:
              return i
    return -1
for week in range(1,18):
    savedplayerdata=""
    #playerdata = ""
    soup_proj = make_soup("http://www.fftoday.com/rankings/playerwkproj.php?Season=%s&GameWeek=%d&PosID=10&LeagueID=189999"%(year,week))
    # print(soup.find('table',{"width":"100%", "cellpadding":"2"}))


    table = soup_proj.find('table', {"width": "100%", "cellpadding": "2"})

    #for rows in table.findAll('tr'):
    #playerdata=""
    playerdata = ','.join([data.text.strip(' ') for data in table.findAll('td', {"class":"bodycontent"})]).replace('\xa0,\xa0','').replace('\xa0','').split(',')
    #print(playerdata)
    playerdata = list(filter(lambda a: a != '', playerdata))
    playerdata = [playerdata[i:i+12] for i in range(len(playerdata))[::12]]
    playerArr = np.array(playerdata)
    playerArr[:,1] = 'QB'
    playerArr[:,2] = week
    playerArr = np.insert(playerArr,2,int(year),axis=1)
    soup_actual = make_soup('http://www.fftoday.com/stats/playerstats.php?Season=%s&GameWeek=%d&PosID=10&LeagueID=189999'%(year,week))
    table_actual = soup_actual.find('table', {"width": "100%", "cellpadding": "2"})


    lines = table_actual.text.splitlines()
    playerdata_act = lines[index_containing_substring(lines,'.'):]
    playerdata_act = list(filter(lambda a: a != '', playerdata_act))

    playerdata_act = {playerdata_act[i].split('. ')[1]:playerdata_act[i+12] for i in range(len(playerdata_act))[::13]}#playerdata_act[i+11]

    playerArr = np.array(playerdata)
    playerArr[:,1] = 'QB'
    playerArr[:,2] = week
    playerArr = np.insert(playerArr,2,int(year),axis=1)
    playerArr = np.column_stack([playerArr,np.ones((len(playerArr),1,))])#np.concatenate((playerArr,np.nan((len(playerArr),1,))),axis=1)
    playerArr[:,-1] = np.nan

    for i,player in enumerate(playerArr[:,1]):
        try:
            playerArr[i,-1] = playerdata_act[playerArr[i,0]]
        except:
            pass#rmRows.append(i)

    playerArr = playerArr[playerArr[:,-1] != 'nan']
    if week == 1:
        totalArr = playerArr
    else:
        totalArr=np.concatenate((totalArr,playerArr),axis=0)
    

players_test = pd.DataFrame(totalArr)
del players_test[13]
print(players_test)

##################################

#print(np.vectorize(float)(players_test[:,3:]))
#players_test = pd.DataFrame(totalArr)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
#print(players_train.loc[:,3:13].values)
stdS = StandardScaler()

trainData = np.vectorize(float)(players_train.loc[:,3:12].values)
stdS.fit(trainData)
trainData = stdS.transform(trainData)
trainLabels = np.vectorize(float)(players_train.loc[:,13].values)
testData = stdS.transform(np.vectorize(float)(players_test.loc[:,3:].values))
#print(trainData.shape,trainLabels.shape,testData.shape)
#print(pd.DataFrame(trainData),pd.DataFrame(trainLabels),pd.DataFrame(testData))
mlp = MLPRegressor()
#print(np.vectorize(float)(players_train.loc[:,3:13].values),np.vectorize(float)(players_train.loc[:,13].values,players_test.loc[:,3:].values)
mlp.fit(trainData,trainLabels)
a = mlp.predict(testData)
#print(players_train[players_train[2] == '2016'].loc[:,13].values)
print(np.vectorize(float)(players_train[players_train[2] == '2016'].loc[:,13].values))
playernames = players_test.loc[:,0].values
actual = np.vectorize(float)(players_train[players_train[2] == '2016'].loc[:,13].values)
predicted = np.vectorize(float)(players_train[players_train[2] == '2016'].loc[:,12].values)
err1, err2 = actual - predicted , actual - a#, predicted - actual
av1, av2 = tuple(np.mean(x) for x in [err1,err2])#,err3])#tuple([(np.sum(np.vectorize(lambda x: x**(2))(err))/len(a))**(1/2) for err in [err1,err2,err3]])
std1, std2 = np.std(err1), np.std(err2)
print([x.shape for x in [playernames,predicted,actual,a]])
print(pd.DataFrame(np.column_stack([playernames,predicted,actual,a,err1,err2]),columns=['Player','Projection','Actual Score','My Computed Score','actual-projection','actual-model']))
print('Average Residual Error for actual-projection = %f ± %f and actual - model =  %f ± %f\n For model to work, second number must be less than first number'%(av1,std1,av2,std2))
