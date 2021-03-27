import networkx as nx 
import pandas as pd
import matplotlib.pyplot as plt
from networkx.algorithms import community

#initialize the graph
H = nx.DiGraph() #H is a graph that we keeps combining with every G iterations

#read the data
df=pd.read_csv("infosfromtweets.csv")
useralreadyexist = [] #save the list of usernames that are already checked

#iterate through every list (current list = i)
for x in range(300):
    print('---------------------------')
    print('creating list for user ' + str(x))
    listname = "followinglist" + str(x)
    filename = listname + ".csv" 
    G = nx.DiGraph() #initialize a graph for each time a new user is loaded
    #Create a node from the username of the user in currentlist
    currentusername = df.iloc[x].loc['username']
    if currentusername in useralreadyexist:
        print('graph not created because this user already existed before')
        continue
    useralreadyexist.append(currentusername)
    G.add_node(currentusername)
    with open(filename) as file:
        i=1
        for line in file:
            G.add_edge(currentusername, line)
            i += 1
    H = nx.compose(H, G)
    print('joined ' + str(x+1) + ' graph')

#pick nodes 'label' parameter and draw labels
#only for nodes that have 3 or more neighbors
labels = {}
for node in H.nodes():
    if len(list(nx.all_neighbors(H, node))) > 3:
        labels[node] = node
    else:
        labels[node] = ''

print('drawing graph')
#nx.draw(H, with_labels = True, labels=labels) 
#print('drawn graph. saving file...')
#plt.savefig("tesgraph.pdf")

#create file for gephy
nx.write_gexf(H, "file2.gexf", version="1.2draft")
print('drawn graph. saving file...')