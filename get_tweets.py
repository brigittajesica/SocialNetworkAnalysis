import tweepy
import json

#THIS FIELD IS FROM MY TWITTER DEVELOPER ACCOUNT
API_KEY="xxxxxxxx"
API_SECRET="xxxxxxxx"
ACCESS_TOKEN="xxxxxxxx"
ACCESS_TOKEN_SECRET="xxxxxxxx"

#create oauthhandler object and set access token and secret
auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

import tweepy
tweet_list=[]
#streaming for tweets
class MyStreamListener(tweepy.StreamListener):
    def __init__(self,api=None):
        super(MyStreamListener,self).__init__()
        self.num_tweets=0
        self.file=open("tweets.txt","w")
    #on_status describe what is done during reading a status (Tweet)
    def on_status(self,status):
        tweet=status._json
        self.file.write(json.dumps(tweet)+ '\n')
        tweet_list.append(status)
        self.num_tweets+=1
        if self.num_tweets<1000:
            return True
        else:
            return False
        self.file.close()
    #error handling is important, since if you reach a max limit and
    #keep streaming, the amount of time you won't be able to get
    #tweets will increase exponentially
    #hence stop stream request once there is error code
    #(analog to entering a wrong password again and again on phone)
    def on_error(self, status_code):
        if status_code == 420:
            #returning False in on_error disconnects the stream
            return False

            #returning non-False reconnects the stream, with backoff.

#create streaming object and authenticate
l = MyStreamListener()
stream =tweepy.Stream(auth,l)
#this line filters twiiter streams to capture data by keywords
stream.filter(track=['#cryptocurrency'])



