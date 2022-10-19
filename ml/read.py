import networkx as nx
import csv
import os, glob

csv_file = open('output.csv') 
csv_reader = csv.reader(csv_file, delimiter=',')
graph = []
for row in csv_reader:
    graph.append(int(row[0]))


path = "./graph/"
all_files = sorted(glob.glob(path + "/*.graphml.xml"))
f_edge = open('edge.csv', 'w') 
writer_edge = csv.writer(f_edge)
head = ['graph_id','src','dst','etype']
writer_edge.writerow(head)

f_graph = open('graph.csv', 'w')
head = ['graph_id','label','num_nodes']
writer_graph = csv.writer(f_graph)
writer_graph.writerow(head)

f_node = open('node.csv', 'w')
writer_node = csv.writer(f_node)
head = ['graph_id','node_id','type_id','prop']
writer_node.writerow(head)
id = 0
for filename in all_files:
    G = nx.read_graphml(filename)
    

    index ={}
    i = 0
    for node in G.nodes(data=True):
        index[node[0]] = i
        i = i + 1
    connect = {'DOOR':0, 'OPEN':1, 'ENTRANCE':2, 'VERTICAL':3}
    space = {'OUTSIDE': 0, 'BASEMENT':1, 'STAIRS':2, 'CORRIDOR': 3, 'TECHNICAL ROOM':4, 'BEDROOM':5, 
    'BATHROOM':6, 'DINING/LIVING ROOM':7, 'KITCHEN':8, 'HALL':9, 'WC':10, 'WALK-IN CLOSET':11, 
    'GALLERY':12, 'SHOWER/WC':13, 'PANTRY':14, 'STORAGE':15, 'STUDY':16, 'COAT RACK':17, 'DIELE  ':18}
        
    for edge in G.edges(data=True):
        writer_edge.writerow([id, index[edge[0]],index[edge[1]],connect[edge[2]['edgetype']]])
    for node in G.nodes(data=True):
        writer_node.writerow([id, index[node[0]],space[node[1]['label']],node[1]['area']])
    writer_graph.writerow([id, graph[id],G.number_of_nodes()])
    
    id = id + 1

    










