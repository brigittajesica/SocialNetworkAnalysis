import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from textblob import TextBlob
from wordcloud import WordCloud
import string
import en_core_web_sm
from nltk.tokenize import RegexpTokenizer, WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans

bag_of_words = []

# https://github.com/pjwebdev/Basic-Data-Science-Projects/blob/master/8-Twitter-Sentiment-Analysis/Tweeter%20Sentiment%20Analysis.ipynb
def cleanUpTweet(text):
    # Remove capital letters
    text = text.lower()
    # Remove line breaks
    text = text.replace('\n', ' ')
    # Remove double spaces
    text = re.sub(' +', ' ', text)
    # Remove mentions
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    # Remove hashtags
    text = re.sub(r'#', '', text)
    # Remove retweets
    text = re.sub(r'rt : ', '', text)
    # Remove urls
    text = re.sub(r'https?:\/\/[A-Za-z0-9\.\/]+', '', text)
    # Remove emojis
    # https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
    pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                           "]+", flags = re.UNICODE)
    text = pattern.sub(r'', text)
    return text

# https://medium.com/swlh/tweets-classification-and-clustering-in-python-b107be1ba7c7
def furnished(text):
    final_text = []
    for i in w_tokenizer.tokenize(text):
        if i.lower() not in stop:
            word = lemmatizer.lemmatize(i)
            final_text.append(word.lower())
    return " ".join(final_text)

def getTextSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getTextPolarity(text):
    return TextBlob(text).sentiment.polarity

def getTextAnalysis(a):
    if a < 0:
        return "Negative"
    elif a == 0:
        return "Neutral"
    else:
        return "Positive"

# https://medium.com/swlh/tweets-classification-and-clustering-in-python-b107be1ba7c7
def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)

# https://medium.com/swlh/tweets-classification-and-clustering-in-python-b107be1ba7c7
def get_scores(group,tweets):
    scores = []
    for tweet in tweets:
        s = jaccard_similarity(group, tweet)
        scores.append(s)
    return scores

# Read the CSV into a DataFrame
info_df = pd.read_csv('infosfromtweets.csv')

# Initialize everything for text preprocessing
nlp = en_core_web_sm.load()
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)
w_tokenizer = WhitespaceTokenizer()

# Preprocess the tweets
info_df['fulltext'] = info_df['fulltext'].apply(cleanUpTweet)
info_df['fulltext'] = info_df['fulltext'].apply(furnished)

# Compute and save subjectivity, polarity, and scores in the Dataframe 
info_df['Subjectivity'] = info_df['fulltext'].apply(getTextSubjectivity)
info_df['Polarity'] = info_df['fulltext'].apply(getTextPolarity)
info_df['Score'] = info_df['Polarity'].apply(getTextAnalysis)

# Parse through all tweets
for tweet in info_df['fulltext']:
    # Store every word seperated by a space into a list
    temp = tweet.split(' ')
    bag_of_words = bag_of_words + temp
# Get all the unique words used in the tweets
# unique_words = list(set(bag_of_words))

# A list of unwanted strings for the WordCloud
unwanteds = ['', 'crypto', 'cryptocurrency', 't', 'o', 'i', 'y', 'k', 'u', 'b', 'g', 'u', 'v', 's', 'x', 'h']
# Filter out unwanted strings
bag_of_words = [el for el in bag_of_words if el not in unwanteds]

# Concatenate all string into one string since WordCloud only accepts a string as input
all_words = ' '.join(bag_of_words)

# Create, show and save the WordCloud 
wordcloud = WordCloud(width=3200, height=1600, collocations=False).generate(all_words)
plt.axis('off')
plt.tight_layout(pad=0)
# plt.savefig('wordcloud.pdf')
plt.imshow(wordcloud)
plt.show()

# https://www.finder.com/cryptocurrency-glossary
# A list of keywords used for cryptocurrency
keywords = '''adresses airdrop algorithm all-time-high all-time-low altcoins aml anti-money-laundring
        application-specific-integrated-circuit arbitrage asic ath atomic-swap bag bear bearish bear-trap bitcoin
        block block-explorer block-height block-reward blockchain btfd bull bullish burned buy-wall cap central-ledger
        chain-linking cipher circulating-supply cold-storage confirmed consensus consensus-process consortium-blockchain
        cryptocurrency cyrtographic-hash-function cryptography dao dapp decentralized-application decentralized-autonomous-organization
        decryption deflation depth-chart deterministic-wallet difficulty digital-commodity digital-currency digital-signature
        distributed-ledger double-spend dump dumping dust-transaction dyor encryption erc erc20 escrow ethereum ethereum-virtual-machine
        evm exchange fa faucet fiat fomo fork frictionless fud full-node fundamental-analysis futures-contract gas gas-limit gas-price
        genesis-block group-mining gwei halving hard-cap hard-fork hardware-wallet hash hash-rate hashing-power hodl ico
        initial-coin-offering jomo kyc lambo ledger leverage lightning-network limit-order limit-buy limit-sell liquidity
        locktime long macd margin-bear-position margin-bull-position market-capitalization margin-trading market-order mcap
        mine mining mining-contract mining-pool money-services-business moon moving-average-convergence-divergence msb
        multipool-mining multi-signature-wallet multi-sig-wallet network node nonce oco one-cancels-the-other-cancels oracles 
        overbought oversold paper-wallet p2p peer-to-peer pnd pre-sale private-key poa proof-of-authority pos proof-of-stake 
        pow proof-of-work protocols public-blockchain public-key pumppump-and-dump rekt relative-strength-index ring-signature 
        rsi satoshi-nakamoto sats scrypt seed segregated-witness segwit selfish-mining sell-wall sha256 sha-256 sharding
        shit-coin short smart-contracts soft-fork software-market solidity ta technical-analysis test-net timestamp token 
        tokenless-ledger tor transaction transaction-fee turing-completeness unconfirmed unspent-transaction-output utxo 
        volatility wallet whale whitelist white-paper zero-confirmation-transaction'''

# Calculate the Jaccard scores by crossreferencing the keywords
jaccard_scores = get_scores(keywords, info_df.fulltext.to_list())
# Store the results in the DataFrame
temp = pd.DataFrame(jaccard_scores, columns=['Jaccard Similarity'])
info_df = pd.concat([info_df, temp], axis=1)

# Store three features to a numpy array for using as input for K-Means
X = info_df.loc[:, ["Polarity", "Subjectivity", "Jaccard Similarity"]].to_numpy()

# Use Elbow Method to find a good number of clusters
# https://medium.com/swlh/tweets-classification-and-clustering-in-python-b107be1ba7c7
# WCSS: Within-Cluster Sums of Squares
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(X)
    # inertia: sum of squared distances of samples to their closest cluster center
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
# plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.savefig('elbow_method.pdf')
plt.show()

# Fitting K-Means to dataset
kmeans = KMeans(n_clusters=3, init='k-means++')
y = kmeans.fit_predict(X)
# Visualize the clusters
plt.scatter(X[y==0, 0], X[y==0, 1], c='cornflowerblue', label= 'Cluster 1')
plt.scatter(X[y==1, 0], X[y==1, 1], c='forestgreen', label= 'Cluster 2')
plt.scatter(X[y==2, 0], X[y==2, 1], c='rebeccapurple', label= 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=70, c='black', label='Centroids' )
plt.legend()
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.savefig('kmeans_clustering.pdf')
plt.show()

# Store the clusters in the DataFrame
temp = pd.DataFrame(y, columns=['Cluster'])
info_df = pd.concat([info_df, temp], axis=1)

# Rearrange the columns
new_cols = ['Unnamed: 0', 'Cluster', 'Score', 'Polarity', 'Subjectivity', 'Jaccard Similarity', 'userid', 'username', 'fulltext']
info_df = info_df[new_cols]

# Rename some of the columns
info_df = info_df.rename({'Unnamed: 0': 'Tweet Number', 'userid': 'User ID', 'username': 'Username', 'fulltext': 'Full Text'}, axis='columns')

# Export the DataFrame to a CSV file
info_df.to_csv('clustered_df.csv', index=False)
