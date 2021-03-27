# SocialNetworkAnalysis
Social Network Analysis using Twitter API, Python and Gephi

## Getting Tweets
Getting the Tweets is done by streaming tweets with the Twitter API. The filter used to obtain the tweets is a certain hashtag. 
After enough tweets are obtained, only relevant informations are extracted. This is done by accessing the json value of the wanted attributes.

## Obtaining Following List
For each of the user that tweeted the tweet obtained in streaming, their following list is obtained (a list of all people they follow).

## Clustering
Clustering done with elbow method and k-means clustering. 
1. Based on connectivity of the following graph
2. Based on words in tweets

## Social Network Graph
Done with NetworkX, and the graph is later processed for a better visualization via gephi.
