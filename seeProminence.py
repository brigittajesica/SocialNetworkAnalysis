import pandas as pd

df = pd.DataFrame()
tweetsinfo = pd.read_csv("infosfromtweets.csv")

alreadycheckedlist = []
listofname = []
listofcount = []
alreadycheckedusername = []

#iterate through every list (current list = i)
for x in range(1000):
    currentusername1 = tweetsinfo.iloc[x].loc['username']
    if currentusername1 in alreadycheckedusername:
        continue
    alreadycheckedusername.append(currentusername1)
    listname1 = "followinglist" + str(x)
    filename1 = listname1 + ".csv" 
    print('---------------------')
    print("going through list " + str(x))
    with open(filename1) as file1:
        for line in file1:
            #check first if the username is already in the alreadycheckedlist[]
            #if yes just increase count, do not save a new item of 'name'
            if line in alreadycheckedlist:
                continue
            print('list ' + str(x) + ' checking for the username ' + line)
            listofname.append(line)
            currentlinefile1 = line
            alreadycheckedlist.append(line)
            count = 1
            for y in range(10):
                listname2 = "followinglist" + str(y)
                filename2 = listname2 + ".csv"
                currentusername2 = tweetsinfo.iloc[y].loc['username']
                if currentusername1==currentusername2:
                    continue
                with open(filename2) as file2:
                    for line in file2:
                        currentlinefile2 = line
                        if currentlinefile1==currentlinefile2:
                            count += 1
            listofcount.append(count)

df['name'] = listofname
df['count'] = listofcount
df.to_csv('seeProminence.csv')

#get 10o of the usernames that people follow the most
analysis = df.nlargest(100, 'count')
analysis.to_csv('analysis.csv')

