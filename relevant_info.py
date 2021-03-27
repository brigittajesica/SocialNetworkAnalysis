import json
import pandas as pd
import tweepy
import csv
from datetime import datetime
import time

#THIS FIELD IS FROM MY TWITTER DEVELOPER ACCOUNT
API_KEY="xxxxxxxx"
API_SECRET="xxxxxxxx"
ACCESS_TOKEN="xxxxxxxx"
ACCESS_TOKEN_SECRET="xxxxxxxx"

#create oauthhandler object and set access token and secret
auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
#wait on rate limit= true will tell tweepy to pause until rate limit expires
api = tweepy.API(auth, wait_on_rate_limit=True)

#insert path to file containing tweets JSON object
filepath = "tweets.txt"

#read file line by line
file = open(filepath, 'r')
Lines = file.readlines()

#CREATE LISTS FOR USER ID AND FULL TEXT
userid = []
username = []
fulltext = []

#iterate line by line through tweets to get user id, 
#fulltext and save it to list
count = 0
for line in Lines:
    json_object_line = json.loads(line)
    userid.append(json_object_line['user']['id'])
    username.append(json_object_line['user']['screen_name'])
    #if the tweet is more than 140 characters, we need to get
    #the full text from 'extended_tweet'
    #otherwise extended tweet do not exist and we can simply take 'text'
    try:
        temptext = json_object_line['extended_tweet']['full_text']
        temptext = temptext.replace('\n', '')
        fulltext.append(temptext)
    except KeyError:
        temptext = json_object_line['text']
        temptext = temptext.replace('\n', '')
        fulltext.append(json_object_line['text'])
    count += 1

#convert list to dataframe columns
df = pd.DataFrame()
df['userid'] = userid
df['fulltext'] = fulltext
df['username'] = username
#converting to csv for easy checking  & visualization
df.to_csv('infosfromtweets.csv')

#save all usernames that have been checked
checkedusernames = []

#get the followings from each userid
for x in range(300):
    listname = "followinglist" + str(x)
    print("-----------------------------")
    print(listname)
    filename = listname + ".csv" 
    i = 1

    #if the same username is encountered twice, skip it
    currentusername = df.iloc[x].loc['username']
    if currentusername in checkedusernames:
        continue
    checkedusernames.append(currentusername)

    #save it to a csv file to have data locally
    with open(filename, 'w') as writecsv:
        #using csv writer method from CSV package
        write = csv.writer(writecsv)

        for friend in tweepy.Cursor(api.friends, screen_name=username[x]).items():
            #getting the time to see the last request's time
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)
            print('getting name' + str(i))
            print(friend.screen_name)
            i += 1
            
            #put the item in [] brackets otherwise the username
            #will be written letter by letter 
            write.writerow([friend.screen_name])
        
