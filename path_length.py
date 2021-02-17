import sys
import pandas as pd
import numpy as np

if len(sys.argv) < 5:
    print('Usage:')
    print('  python3 {} [edge file] [flow file] [path file] [weights file]'.format(sys.argv[0]))
    sys.exit(0)

#Read in the files
edges = pd.read_csv(sys.argv[1])
edges_arr = np.array(edges)
edges_arr = np.c_[edges_arr, np.array(range(len(edges)))]

flow = pd.read_csv(sys.argv[2], skiprows = 1)
flow_arr = np.array(flow)

paths_ = pd.read_csv(sys.argv[3], sep='delimiter', header=None)
paths_ = paths_.iloc[2:]
paths_ = paths_[0].str.split(',', 2, expand = True)
paths_arr = np.array(paths_)

weights = pd.read_csv(sys.argv[4], skiprows = 1)
weights_arr = np.array(weights)

#Check if weight is at least 1/10th of the maximum weight
small_arr = []
for i in range(len(weights_arr)):
    if weights_arr[i, 1] > max(weights_arr[:, 1])/2:
        small_arr.append(i)
#Find unique paths
paths_arr[:, 0] = [int(i) for i in paths_arr[:, 0]]
paths_arr = paths_arr[paths_arr[:, 0]>=min(small_arr)]
paths_arr_new = paths_arr[:, 1:]
paths_arr_new = np.vstack({tuple(row) for row in paths_arr_new})

def find_edge2(tail_, head_):
    '''Finds edge index for given edge'''
    
    edges_filtered = edges_arr[edges_arr[:, 0] == tail_]
    edges_filtered = edges_filtered[edges_filtered[:, 1] == head_]
    
    return edges_filtered[0, -1]

#Update Flow data
edge_num_arr = []
for j in flow.index:
    tail_value = int(flow_arr[j, 1])
    head_value = int(flow_arr[j, 2])
    edge_num_arr.append(find_edge2(tail_value, head_value))

flow_arr = np.c_[flow_arr, edge_num_arr]

#Sort flow data by edge index
flow_arr = flow_arr[flow_arr[:,-1].argsort()]

#Store the flow value
x_sol = flow_arr[:, 6]
x_sol = np.array([float(i) for i in x_sol])

#Store the flow cost
cost_sol = flow_arr[:, 6]
cost_sol = np.array([float(i) for i in x_sol])


#print(x_sol)

#Define the link latency functions
def link_latency(a_val, power_val, link_ff_cost, link_capacity, link_flow):
    '''Calculates latency of link'''
    return link_ff_cost*(1+a_val*(link_flow/link_capacity)**power_val)

def get_travel_time_path(sol_implement):
    '''Find the total travel time for a given path in the network'''
    
    return sum([link_latency(0.15, 4, int(flow_so_arr[j_val, 3]),  edges_arr[j_val, 3], sol_implement[j_val]) for j_val in range(len(sol_implement))])

def path_edge_finder2(edge_string):
    '''Finds the edge index for a given edge in path'''
    
    simplify_string = edge_string[1:]
    simplify_string2 = simplify_string[:-1]
    simplify_string3 = simplify_string2.split(' ')
    
    return find_edge2(int(simplify_string3[0]), int(simplify_string3[1]))

#Find unique Paths
#paths_arr_new = paths_arr[:, 1:]
#paths_arr_new = np.vstack({tuple(row) for row in paths_arr_new})

#Return the length of each path for each OD pair
OD_dict_ = {}
OD_path_lengths = {}

for OD_pair in range(int(paths_arr[-1, 1])):
    OD_arr = paths_arr_new[paths_arr_new[:, 0] == str(OD_pair)]
    all_paths_OD = []
    OD_tt_paths = []
    for path_val in OD_arr[:, 1]:
        all_edges = path_val.split(',')
        all_edges = [int(i) for i in all_edges]
        list_edges = []
        path_tt = 0
        for edge_values in all_edges:
            list_edges.append(edge_values)
            #path_tt += cost_sol[edge_values]
            path_tt += link_latency(0.15, 4, int(flow_arr[edge_values, 3]),  edges_arr[edge_values, 3], x_sol[edge_values])

        OD_tt_paths.append(path_tt)

        if list_edges not in all_paths_OD:
            all_paths_OD.append(list_edges)
    if OD_pair%100 == 0:
        print(OD_pair)
    OD_dict_[OD_pair] = all_paths_OD
    OD_path_lengths[OD_pair] = OD_tt_paths

with open('OD_paths_tts2.csv', 'w') as f:
    for key in OD_path_lengths.keys():
        #print(key)
        str_val = str(OD_path_lengths[key])[1:-1]
        
        f.write("%s,%s\n"%(key,str_val))
#        f.write("%s,%s\n"%(key,[i for i in OD_path_lengths[key]]))
