import numpy as np
import pandas as pd
import zipfile

class SolutionImplementations():
    def __init__(self):
        '''Init'''

    def Jahn_Et_Al(list_cities, percentile_val):
        '''Jahn et al implementation'''
        alpha_values_2005 = ['1.00', '1.05', '1.10', '1.15', '1.20', '1.25', '1.30', '1.35', '1.40', '1.45', '1.50',
               '1.55', '1.60', '1.65', '1.70', '1.75', '1.80', '1.85', '1.90', '1.95', '2.00']

        final_beta_2005_dict = {}
        percentile_beta_2005_dict = {}
        total_cost_2005_dict = {}

        for city_name in list_cities:

            main_zip = 'Alpha_Objective/' + city_name + '/' + city_name + '-'
            main_zip_loc = 'Alpha_Objective/' + city_name + '/'
            edges_file = 'Locations/' + city_name + '/edges.csv'
            new_folder = 'Alpha_Objective'
            specific_city = city_name + '.zip'
            od_file = 'Locations/' + city_name + '/od.csv'
            loc_ = 'constrained_approach/' + city_name + '/' + city_name +  '-constrained-'

            #Edges Data
            edges = pd.read_csv(edges_file)
            edges_arr = np.array(edges)
            edges_arr = np.c_[edges_arr, np.array(range(len(edges)))]

            OD_mat = pd.read_csv(od_file)
            OD_arr__ = OD_mat['volume'].values

            total_cost_2005 = []
            final_beta_2005 = []
            percentile_beta_2005 = []
            for alpha_val in alpha_values_2005:
                print(alpha_val)

                df_flow = pd.read_csv(loc_+alpha_val+'/flow.csv', skiprows = 1)
                actual_cost = np.array(df_flow['actualCost'])
                flow = np.array(df_flow['flow'])
                total_cost_2005.append(sum([actual_cost[i]*flow[i] for i in range(len(actual_cost))]))

                flow_arr = np.array(df_flow)

                def find_edge2(tail_, head_):
                    '''Finds edge index for given edge'''

                    edges_filtered = edges_arr[edges_arr[:, 0] == tail_]
                    edges_filtered = edges_filtered[edges_filtered[:, 1] == head_]

                    return edges_filtered[0, -1]

                #Update Flow data
                edge_num_arr = []
                for j in df_flow.index:
                    tail_value = int(flow_arr[j, 1])
                    head_value = int(flow_arr[j, 2])
                    edge_num_arr.append(find_edge2(tail_value, head_value))

                flow_arr = np.c_[flow_arr, edge_num_arr]
                flow_arr = flow_arr[flow_arr[:,-1].argsort()]

                cost_sol = flow_arr[:, 4]
                cost_sol = np.array([float(i) for i in cost_sol])

                paths_df = pd.read_csv(loc_+alpha_val+'/paths.csv', sep='delimiter', header=None)
                paths_df = paths_df.iloc[2:]
                paths_df = paths_df[0].str.split(',', 2, expand = True)
                paths_arr = np.array(paths_df)

                weights_df = pd.read_csv(loc_+alpha_val+'/weights.csv', skiprows = 1)
                weights_df_arr = np.array(weights_df)

                #Check if weight is at least 1/10th of the maximum weight
                small_arr = []
                for i in range(len(weights_df_arr)):
                    if weights_df_arr[i, 1] > max(weights_df_arr[:, 1])/10:
                        small_arr.append(i)

                paths_arr[:, 0] = [int(i) for i in paths_arr[:, 0]]
                #paths_arr_ue = np.append(paths_arr_ue, np.array(weig), axis=1)
                paths_arr = paths_arr[paths_arr[:, 0]>=min(small_arr)]
                paths_arr_new = paths_arr[:, 1:]
                paths_arr_new = np.vstack({tuple(row) for row in paths_arr_new})

                beta_val = []
                for OD_pair in range(int(paths_arr[-1, 1])):
                    OD_arr = paths_arr_new[paths_arr_new[:, 0] == str(OD_pair)]
                    all_paths_OD = []
                    OD_tt_paths = []
                    for path_idx, path_val in enumerate(OD_arr[:, 1]):
                        all_edges = path_val.split(',')
                        all_edges = [int(i) for i in all_edges]
                        list_edges = []
                        path_tt = 0
                        for edge_values in all_edges:
                            list_edges.append(edge_values)
                            path_tt += cost_sol[edge_values]
                        OD_tt_paths.append(path_tt)

                    #Store max and min paths per OD pair
                    path_OD_max = max(OD_tt_paths)
                    path_OD_min = min(OD_tt_paths)

                    #Find ratio of max and min
                    if path_OD_max-path_OD_min>0.001:
                        path_OD_beta = path_OD_max/path_OD_min
                    else:
                        path_OD_beta = 1

                    beta_val.append(path_OD_beta)

                #Find percentile value of beta for this value of alpha
                beta_new_val = [item for item, count in zip(beta_val, OD_arr__) for i in range(count)]
                beta_dist = np.percentile(beta_new_val, percentile_val)

                #Find maximum value of beta for this value of alpha
                beta = max(beta_val) 
                final_beta_2005.append(beta)
                percentile_beta_2005.append(beta_dist)
                
            final_beta_2005 = np.array(final_beta_2005)
            final_beta_plot = [final_beta_2005[0]]
            total_cost_plot = [total_cost_2005[0]]
            for i in range(1, len(alpha_values_2005)):
                restricted_beta = final_beta_2005[final_beta_2005<final_beta_2005[i]]
                if len(restricted_beta)>1:
                    idx_val_arr = []
                    total_cost_arr = []
                    for j in range(len(restricted_beta)):
                        idx_val_arr.append(list(final_beta_2005).index(restricted_beta[j]))
                        total_cost_arr.append(total_cost_2005[idx_val_arr[-1]])

                    if min(total_cost_arr)>total_cost_2005[i]:
                        total_cost_plot.append(total_cost_2005[i])
                        final_beta_plot.append(final_beta_2005[i])
                        
                else:
                    total_cost_plot.append(total_cost_2005[i])
                    final_beta_plot.append(final_beta_2005[i])
                
            total_cost_2005_dict[city_name] = total_cost_plot
            final_beta_2005_dict[city_name] = final_beta_plot
            percentile_beta_2005_dict[city_name] = percentile_beta_2005

        return final_beta_2005_dict, total_cost_2005_dict, percentile_beta_2005_dict


    def ITAP_001(list_cities, percentile_val):
        '''I-TAP with step size of 0.01'''
        alpha_values = []
        for i in range(100):
            if i<10:
                alpha_values.append('.0'+str(i))
            else:
                alpha_values.append('.'+str(i))
        alpha_values[0] = '0'
        alpha_values.append('1.00')

        final_beta_dict = {}
        percentile_beta_dict = {}
        total_cost_dict = {}
        total_cost_original_dict = {}
        final_beta_original_dict = {}

        for city_name in list_cities:
            edges_file = 'Locations/' + city_name + '/edges.csv'
            percentile_val = 98
            od_file = 'Locations/' + city_name + '/od.csv'
            main_zip = 'alpha_0.01_increments_I-TAP/' + city_name + '-'
            #Edges Data
            edges = pd.read_csv(edges_file)
            edges_arr = np.array(edges)
            edges_arr = np.c_[edges_arr, np.array(range(len(edges)))]

            OD_mat = pd.read_csv(od_file)
            OD_arr__ = OD_mat['volume'].values

            total_cost = []
            final_beta = []
            percentile_beta = []
            total_tt_solution2 = []
            path_max_total = {}
            path_min_total = {}
            paths_total_save = {}
            for alpha_val in alpha_values:
                print(alpha_val)
                #with zipfile.ZipFile(main_zip+alpha_val+'.zip', 'r') as zip_ref:
                #    zip_ref.extractall(main_zip_loc)

                df_flow = pd.read_csv(main_zip+alpha_val+'/flow.csv', skiprows = 1)
                actual_cost = np.array(df_flow['actualCost'])
                flow = np.array(df_flow['flow'])
                total_cost.append(sum([actual_cost[i]*flow[i] for i in range(len(actual_cost))]))


                flow_arr = np.array(df_flow)

                def find_edge2(tail_, head_):
                    '''Finds edge index for given edge'''

                    edges_filtered = edges_arr[edges_arr[:, 0] == tail_]
                    edges_filtered = edges_filtered[edges_filtered[:, 1] == head_]

                    return edges_filtered[0, -1]

                #Update Flow data
                edge_num_arr = []
                for j in df_flow.index:
                    tail_value = int(flow_arr[j, 1])
                    head_value = int(flow_arr[j, 2])
                    edge_num_arr.append(find_edge2(tail_value, head_value))

                flow_arr = np.c_[flow_arr, edge_num_arr]
                flow_arr = flow_arr[flow_arr[:,-1].argsort()]

                cost_sol = flow_arr[:, 4]
                cost_sol = np.array([float(i) for i in cost_sol])

                paths_df = pd.read_csv(main_zip+alpha_val+'/paths.csv', sep='delimiter', header=None)
                paths_df = paths_df.iloc[2:]
                paths_df = paths_df[0].str.split(',', 2, expand = True)
                paths_arr = np.array(paths_df)

                weights_df = pd.read_csv(main_zip+alpha_val+'/weights.csv', skiprows = 1)
                weights_df_arr = np.array(weights_df)

                #Check if weight is at least 1/10th of the maximum weight
                small_arr = []
                for i in range(len(weights_df_arr)):
                    if weights_df_arr[i, 1] > max(weights_df_arr[:, 1])/10:
                        small_arr.append(i)

                paths_arr[:, 0] = [int(i) for i in paths_arr[:, 0]]
                #paths_arr_ue = np.append(paths_arr_ue, np.array(weig), axis=1)
                paths_arr = paths_arr[paths_arr[:, 0]>=min(small_arr)]
                paths_arr_new = paths_arr[:, 1:]
                paths_arr_new = np.vstack({tuple(row) for row in paths_arr_new})
                print(len(paths_arr_new))

                #Define the link latency functions
                def link_latency(a_val, power_val, link_ff_cost, link_capacity, link_flow):
                    '''Calculates latency of link'''
                    return link_ff_cost*(1+a_val*(link_flow/link_capacity)**power_val)

                def get_travel_time(sol_implement):
                    '''Find the total travel time in a given network'''

                    return sum([sol_implement[j_val]*link_latency(0.15, 4, int(flow_arr[j_val, 3]),  edges_arr[j_val, 3], sol_implement[j_val]) for j_val in range(len(sol_implement))])

                x_sol = flow_arr[:, 6]
                x_sol = np.array([float(i) for i in x_sol])
                total_tt_alpha = get_travel_time(x_sol)
                total_tt_solution2.append(total_tt_alpha)

                beta_val = []
                path_max_OD_pair = []
                path_min_OD_pair = []
                for OD_pair in range(int(paths_arr[-1, 1])):
                    OD_arr = paths_arr_new[paths_arr_new[:, 0] == str(OD_pair)]
                    all_paths_OD = []
                    OD_tt_paths = []
                    for path_idx, path_val in enumerate(OD_arr[:, 1]):
                        all_edges = path_val.split(',')
                        all_edges = [int(i) for i in all_edges]
                        list_edges = []
                        path_tt = 0
                        for edge_values in all_edges:
                            list_edges.append(edge_values)
                            path_tt += cost_sol[edge_values]
                        OD_tt_paths.append(path_tt)

                    #Store max and min paths per OD pair
                    path_OD_max = max(OD_tt_paths)
                    path_OD_min = min(OD_tt_paths)

                    if alpha_val == '.05' and OD_pair == 148:
                        print(OD_arr[OD_tt_paths.index(path_OD_max), 1])
                        print(OD_arr[:, 1])
                    path_max_OD_pair.append(OD_arr[OD_tt_paths.index(path_OD_max), 1])
                    path_min_OD_pair.append(OD_arr[OD_tt_paths.index(path_OD_min), 1])

                    #Find ratio of max and min
                    if path_OD_max-path_OD_min>0.001:
                        path_OD_beta = path_OD_max/path_OD_min
                    else:
                        path_OD_beta = 1

                    beta_val.append(path_OD_beta)

                #Find percentile value of beta for this value of alpha
                beta_new_val = [item for item, count in zip(beta_val, OD_arr__) for i in range(count)]
                beta_dist = np.percentile(beta_new_val, percentile_val)

                #Find maximum value of beta for this value of alpha
                beta = max(beta_val) 
                final_beta.append(beta)
                percentile_beta.append(beta_dist)
                path_max_total[alpha_val] = path_max_OD_pair
                path_min_total[alpha_val] = path_min_OD_pair
                paths_total_save[alpha_val] = paths_arr_new
                
            final_beta = np.array(final_beta)
            final_beta_plot = [final_beta[0]]
            total_cost_plot = [total_cost[0]]
            for i in range(1, len(alpha_values)):
                restricted_beta = final_beta[final_beta<final_beta[i]]
                if len(restricted_beta)>1:
                    idx_val_arr = []
                    total_cost_arr = []
                    for j in range(len(restricted_beta)):
                        idx_val_arr.append(list(final_beta).index(restricted_beta[j]))
                        total_cost_arr.append(total_cost[idx_val_arr[-1]])

                    if min(total_cost_arr)>total_cost[i]:
                        total_cost_plot.append(total_cost[i])
                        final_beta_plot.append(final_beta[i])
                        
                else:
                    total_cost_plot.append(total_cost[i])
                    final_beta_plot.append(final_beta[i])
            
            if city_name == 'Massachusetts':
                print(total_cost_plot/total_cost_plot[-1])
                print(final_beta_plot)
            total_cost_dict[city_name] = total_cost_plot
            final_beta_dict[city_name] = final_beta_plot
            percentile_beta_dict[city_name] = percentile_beta 
            final_beta_original_dict[city_name] = final_beta
            total_cost_original_dict[city_name] = total_cost

        return final_beta_dict, total_cost_dict, percentile_beta_dict, final_beta_original_dict, total_cost_original_dict

    def ITAP_005(list_cities, percentile_val):
        '''I-TAP with step size of 0.05'''
        alpha_values = ['0', '.05', '.10', '.15', '.20', '.25', '.30', '.35', '.40', '.45', '.50',
               '.55', '.60', '.65', '.70', '.75', '.80', '.85', '.90', '.95', '1.00']

        final_beta_dict_05 = {}
        percentile_beta_dict_05 = {}
        total_cost_dict_05 = {}

        for city_name in list_cities:

            main_zip = 'Alpha_Objective/' + city_name + '/' + city_name + '-'
            main_zip_loc = 'Alpha_Objective/' + city_name + '/'
            edges_file = 'Locations/' + city_name + '/edges.csv'
            new_folder = 'Alpha_Objective'
            specific_city = city_name + '.zip'
            od_file = 'Locations/' + city_name + '/od.csv'
            loc_ = 'constrained_approach/' + city_name + '/' + city_name +  '-constrained-'
            
            with zipfile.ZipFile(specific_city, 'r') as zip_ref:
                zip_ref.extractall(new_folder)

            #Edges Data
            edges = pd.read_csv(edges_file)
            edges_arr = np.array(edges)
            edges_arr = np.c_[edges_arr, np.array(range(len(edges)))]

            OD_mat = pd.read_csv(od_file)
            OD_arr__ = OD_mat['volume'].values

            total_cost = []
            final_beta = []
            percentile_beta = []
            for alpha_val in alpha_values:
                print(alpha_val)
                with zipfile.ZipFile(main_zip+alpha_val+'.zip', 'r') as zip_ref:
                    zip_ref.extractall(main_zip_loc)

                df_flow = pd.read_csv(main_zip+alpha_val+'/flow.csv', skiprows = 1)
                actual_cost = np.array(df_flow['actualCost'])
                flow = np.array(df_flow['flow'])
                total_cost.append(sum([actual_cost[i]*flow[i] for i in range(len(actual_cost))]))


                flow_arr = np.array(df_flow)

                def find_edge2(tail_, head_):
                    '''Finds edge index for given edge'''

                    edges_filtered = edges_arr[edges_arr[:, 0] == tail_]
                    edges_filtered = edges_filtered[edges_filtered[:, 1] == head_]

                    return edges_filtered[0, -1]

                #Update Flow data
                edge_num_arr = []
                for j in df_flow.index:
                    tail_value = int(flow_arr[j, 1])
                    head_value = int(flow_arr[j, 2])
                    edge_num_arr.append(find_edge2(tail_value, head_value))

                flow_arr = np.c_[flow_arr, edge_num_arr]
                flow_arr = flow_arr[flow_arr[:,-1].argsort()]

                cost_sol = flow_arr[:, 4]
                cost_sol = np.array([float(i) for i in cost_sol])

                paths_df = pd.read_csv(main_zip+alpha_val+'/paths.csv', sep='delimiter', header=None)
                paths_df = paths_df.iloc[2:]
                paths_df = paths_df[0].str.split(',', 2, expand = True)
                paths_arr = np.array(paths_df)

                weights_df = pd.read_csv(main_zip+alpha_val+'/weights.csv', skiprows = 1)
                weights_df_arr = np.array(weights_df)

                #Check if weight is at least 1/10th of the maximum weight
                small_arr = []
                for i in range(len(weights_df_arr)):
                    if weights_df_arr[i, 1] > max(weights_df_arr[:, 1])/10:
                        small_arr.append(i)

                paths_arr[:, 0] = [int(i) for i in paths_arr[:, 0]]
                #paths_arr_ue = np.append(paths_arr_ue, np.array(weig), axis=1)
                paths_arr = paths_arr[paths_arr[:, 0]>=min(small_arr)]
                paths_arr_new = paths_arr[:, 1:]
                paths_arr_new = np.vstack({tuple(row) for row in paths_arr_new})

                beta_val = []
                for OD_pair in range(int(paths_arr[-1, 1])):
                    OD_arr = paths_arr_new[paths_arr_new[:, 0] == str(OD_pair)]
                    all_paths_OD = []
                    OD_tt_paths = []
                    for path_idx, path_val in enumerate(OD_arr[:, 1]):
                        all_edges = path_val.split(',')
                        all_edges = [int(i) for i in all_edges]
                        list_edges = []
                        path_tt = 0
                        for edge_values in all_edges:
                            list_edges.append(edge_values)
                            path_tt += cost_sol[edge_values]
                        OD_tt_paths.append(path_tt)

                    #Store max and min paths per OD pair
                    path_OD_max = max(OD_tt_paths)
                    path_OD_min = min(OD_tt_paths)

                    #Find ratio of max and min
                    if path_OD_max-path_OD_min>0.001:
                        path_OD_beta = path_OD_max/path_OD_min
                    else:
                        path_OD_beta = 1

                    beta_val.append(path_OD_beta)

                #Find percentile value of beta for this value of alpha
                beta_new_val = [item for item, count in zip(beta_val, OD_arr__) for i in range(count)]
                beta_dist = np.percentile(beta_new_val, percentile_val)

                #Find maximum value of beta for this value of alpha
                beta = max(beta_val) 
                final_beta.append(beta)
                percentile_beta.append(beta_dist)
                
            final_beta = np.array(final_beta)
            final_beta_plot = [final_beta[0]]
            total_cost_plot = [total_cost[0]]
            for i in range(1, len(alpha_values)):
                restricted_beta = final_beta[final_beta<final_beta[i]]
                if len(restricted_beta)>1:
                    idx_val_arr = []
                    total_cost_arr = []
                    for j in range(len(restricted_beta)):
                        idx_val_arr.append(list(final_beta).index(restricted_beta[j]))
                        total_cost_arr.append(total_cost[idx_val_arr[-1]])

                    if min(total_cost_arr)>total_cost[i]:
                        total_cost_plot.append(total_cost[i])
                        final_beta_plot.append(final_beta[i])
                        
                else:
                    total_cost_plot.append(total_cost[i])
                    final_beta_plot.append(final_beta[i])
                
            total_cost_dict_05[city_name] = total_cost_plot
            final_beta_dict_05[city_name] = final_beta_plot
            percentile_beta_dict_05[city_name] = percentile_beta 

        return final_beta_dict_05, total_cost_dict_05, percentile_beta_dict_05

    def ISolution_001(list_cities, percentile_val):
        '''I-Solution with step size of 0.01'''
        beta_alpha_solution_dict = {}
        beta_alpha_dist_solution_dict = {}
        total_tt_solution_dict = {}
        global_city_so = {}
        for city_name in list_cities:


            percentile_val = 98
            main_zip = 'Alpha_Objective/' + city_name + '/' + city_name + '-'
            main_zip_loc = 'Alpha_Objective/' + city_name + '/'
            edges_file = 'Locations/' + city_name + '/edges.csv'
            new_folder = 'Alpha_Objective'
            specific_city = city_name + '.zip'
            od_file = 'Locations/' + city_name + '/od.csv'
            loc_ = 'constrained_approach/' + city_name + '/' + city_name +  '-constrained-'

            edges = pd.read_csv(edges_file)
            edges_arr = np.array(edges)
            edges_arr = np.c_[edges_arr, np.array(range(len(edges)))]

            flow_so = pd.read_csv(main_zip+'1.00/flow.csv', skiprows = 1)
            flow_so_arr = np.array(flow_so)

            flow_ue = pd.read_csv(main_zip+'0/flow.csv', skiprows = 1)
            flow_ue_arr = np.array(flow_ue)

            def find_edge2(tail_, head_):
                '''Finds edge index for given edge'''

                edges_filtered = edges_arr[edges_arr[:, 0] == tail_]
                edges_filtered = edges_filtered[edges_filtered[:, 1] == head_]

                return edges_filtered[0, -1]

            #Update Flow UE data
            edge_num_ue_arr = []
            for j in flow_ue.index:
                tail_value = int(flow_ue_arr[j, 1])
                head_value = int(flow_ue_arr[j, 2])
                edge_num_ue_arr.append(find_edge2(tail_value, head_value))

            flow_ue_arr = np.c_[flow_ue_arr, edge_num_ue_arr]

            #Update Flow SO data
            edge_num_so_arr = []
            for j in flow_so.index:
                tail_value = int(flow_so_arr[j, 1])
                head_value = int(flow_so_arr[j, 2])
                edge_num_so_arr.append(find_edge2(tail_value, head_value))

            flow_so_arr = np.c_[flow_so_arr, edge_num_so_arr]

            #Sort flows based on edge indices
            flow_ue_arr = flow_ue_arr[flow_ue_arr[:,-1].argsort()]
            flow_so_arr = flow_so_arr[flow_so_arr[:,-1].argsort()]

            #Store the UE and SO solutions
            x_UE_sol = flow_ue_arr[:, 6]
            y_SO_sol = flow_so_arr[:, 6]

            x_UE_sol = np.array([float(i) for i in x_UE_sol])
            y_SO_sol = np.array([float(i) for i in y_SO_sol])

            #Define the link latency functions
            def link_latency(a_val, power_val, link_ff_cost, link_capacity, link_flow):
                '''Calculates latency of link'''
                return link_ff_cost*(1+a_val*(link_flow/link_capacity)**power_val)

            def get_travel_time(sol_implement):
                '''Find the total travel time in a given network'''

                return sum([sol_implement[j_val]*link_latency(0.15, 4, int(flow_so_arr[j_val, 3]),  edges_arr[j_val, 3], sol_implement[j_val]) for j_val in range(len(sol_implement))])


            paths_so = pd.read_csv(main_zip+'1.00/paths.csv', sep='delimiter', header=None)
            paths_ue = pd.read_csv(main_zip+'0/paths.csv', sep='delimiter', header=None)

            paths_so = paths_so.iloc[2:]
            paths_so = paths_so[0].str.split(',', 2, expand = True)
            paths_arr_so = np.array(paths_so)

            paths_ue = paths_ue.iloc[2:]
            paths_ue = paths_ue[0].str.split(',', 2, expand = True)
            paths_arr_ue = np.array(paths_ue)

            weights_ue = pd.read_csv(main_zip+'0/weights.csv', skiprows = 1)
            weights_ue_arr = np.array(weights_ue)

            #Check if weight is at least 1/10th of the maximum weight
            small_arr = []
            for i in range(len(weights_ue_arr)):
                if weights_ue_arr[i, 1] > max(weights_ue_arr[:, 1])/10:
                    small_arr.append(i)

            #Find unique paths
            paths_arr_ue[:, 0] = [int(i) for i in paths_arr_ue[:, 0]]
            paths_arr_ue = paths_arr_ue[paths_arr_ue[:, 0]>=min(small_arr)]
            paths_arr_ue_new = paths_arr_ue[:, 1:]
            paths_arr_ue_new = np.vstack({tuple(row) for row in paths_arr_ue_new})

            weights_so = pd.read_csv(main_zip+'1.00/weights.csv', skiprows = 1)
            weights_so_arr = np.array(weights_so)

            #Check if weight is at least 1/10th of the maximum weight
            small_arr = []
            for i in range(len(weights_so_arr)):
                if weights_so_arr[i, 1] > max(weights_so_arr[:, 1])/10:
                    small_arr.append(i)

            #Find unique paths
            paths_arr_so[:, 0] = [int(i) for i in paths_arr_so[:, 0]]
            paths_arr_so = paths_arr_so[paths_arr_so[:, 0]>=min(small_arr)]
            paths_arr_so_new = paths_arr_so[:, 1:]
            paths_arr_so_new = np.vstack({tuple(row) for row in paths_arr_so_new})

            OD_dict_UE = {}
            OD_path_lengths_UE = {}
            for OD_pair in range(int(paths_arr_ue[-1, 1])):
                OD_arr = paths_arr_ue_new[paths_arr_ue_new[:, 0] == str(OD_pair)]
                all_paths_OD = []
                OD_tt_paths = []
                for path_idx, path_val in enumerate(OD_arr[:, 1]):
                    all_edges = path_val.split(',')
                    all_edges = [int(i) for i in all_edges]
                    list_edges = []
                    path_tt = 0

                    for edge_values in all_edges:
                        list_edges.append(edge_values)
                        #path_tt += cost_sol[edge_values]
                        path_tt += link_latency(0.15, 4, int(flow_ue_arr[edge_values, 3]),  edges_arr[edge_values, 3], x_UE_sol[edge_values])

                    OD_tt_paths.append(path_tt)

                    #if list_edges not in all_paths_OD:
                    all_paths_OD.append(list_edges)
                if OD_pair%100 == 0:
                    print(OD_pair)
                OD_dict_UE[OD_pair] = all_paths_OD
                OD_path_lengths_UE[OD_pair] = OD_tt_paths

            OD_dict_SO = {}
            OD_path_lengths_SO = {}
            for OD_pair in range(int(paths_arr_so[-1, 1])):
                OD_arr = paths_arr_so_new[paths_arr_so_new[:, 0] == str(OD_pair)]
                all_paths_OD = []
                OD_tt_paths = []
                for path_val in OD_arr[:, 1]:
                    all_edges = path_val.split(',')
                    all_edges = [int(i) for i in all_edges]
                    list_edges = []
                    path_tt = 0

                    for edge_values in all_edges:
                        list_edges.append(edge_values)
                        path_tt += link_latency(0.15, 4, int(flow_so_arr[edge_values, 3]),  edges_arr[edge_values, 3], y_SO_sol[edge_values])

                    OD_tt_paths.append(path_tt)

                    #if list_edges not in all_paths_OD:
                    all_paths_OD.append(list_edges)
                if OD_pair%100 == 0:
                    print(OD_pair)
                OD_dict_SO[OD_pair] = all_paths_OD
                OD_path_lengths_SO[OD_pair] = OD_tt_paths

            def dict_union(dict_UE, dict_SO):
                '''Find the union of two dictionaries'''
                dict_total = {}
                for OD_pair in dict_SO:
                    a = []#dict_total[OD_pair]
                    for j in dict_SO[OD_pair]:
                        #if j not in dict_UE[OD_pair]:
                        a.append(j)
                    for j in dict_UE[OD_pair]:
                        if j not in dict_SO[OD_pair]:
                            a.append(j)

                    dict_total[OD_pair] = a

                return dict_total

            #Take the union of the two dictionaries to find total set of OD pairs
            OD_dict = dict_union(OD_dict_UE, OD_dict_SO)

            OD_mat = pd.read_csv(od_file)
            OD_arr = OD_mat['volume'].values

            def beta_calc(OD_dict, convex_comb):
                '''Calculate beta for a given alpha value'''

                beta_val = []
                for OD_pair in OD_dict:
                    path_OD = []
                    for path in OD_dict[OD_pair]:
                        path_tt = 0
                        for edge in path:
                            path_tt += link_latency(0.15, 4, int(flow_so_arr[edge, 3]),  edges_arr[edge, 3], convex_comb[edge])

                        path_OD.append(path_tt)

                    #Store max and min paths per OD pair
                    path_OD_max = max(path_OD)
                    path_OD_min = min(path_OD)

                    if path_OD_max-path_OD_min>0.001:
                        path_OD_beta = path_OD_max/path_OD_min
                    else:
                        path_OD_beta = 1

                    #Find ratio of max and min
                    beta_val.append(path_OD_beta)

                #Find maximum value of beta for this value of alpha
                beta = max(beta_val)

                return beta

            def beta_calc_dist(OD_dict, convex_comb, percentile_val):
                '''Calculate beta for a given alpha value'''

                beta_val = []
                for OD_pair in OD_dict:
                    path_OD = []
                    for path_idx, path in enumerate(OD_dict[OD_pair]):
                        path_tt = 0
                        for edge in path:
                            path_tt += link_latency(0.15, 4, int(flow_so_arr[edge, 3]),  edges_arr[edge, 3], convex_comb[edge])

                        path_OD.append(path_tt)



                    #Store max and min paths per OD pair
                    path_OD_max = max(path_OD)
                    path_OD_min = min(path_OD)

                    if path_OD_max-path_OD_min>0.001:
                        path_OD_beta = path_OD_max/path_OD_min
                    else:
                        path_OD_beta = 1

                    #Find ratio of max and min
                    beta_val.append(path_OD_beta)

                #Find percentile value of beta for this value of alpha
                beta_new_val = [item for item, count in zip(beta_val, OD_arr) for i in range(count)]
                beta_dist = np.percentile(beta_new_val, percentile_val)

                return beta_dist, beta_new_val

            #Calculate for each value of alpha the corresponding travel time
            total_tt_solution = []
            beta_alpha_solution = []
            beta_alpha_dist_solution = []
            alpha_vals_solution = np.linspace(0, 1, 101)
            global_beta = {}
            for i in alpha_vals_solution:
                convex_comb = i*x_UE_sol + (1-i)*y_SO_sol

                total_tt_alpha = get_travel_time(convex_comb)
                total_tt_solution.append(total_tt_alpha)
                print(i)
                #If at extremes then calculate beta according to UE and SO solutions respectively
                if i == 0:
                    beta = beta_calc(OD_dict_SO, convex_comb)
                    beta2, beta_array = beta_calc_dist(OD_dict_SO, convex_comb, 98)
                    global_beta[i] = beta_array
                    global_city_so[city_name] = beta_array
                elif i == 1:
                    beta2, beta_array = beta_calc_dist(OD_dict_UE, convex_comb, 98)
                    global_beta[i] = beta_array
                    beta = beta_calc(OD_dict_UE, convex_comb)
                else:
                    beta2, beta_array = beta_calc_dist(OD_dict, convex_comb, 98)
                    global_beta[i] = beta_array
                    beta = beta_calc(OD_dict, convex_comb)
                print(i)
                beta_alpha_dist_solution.append(beta2)
                beta_alpha_solution.append(beta)
            
            beta_alpha_solution = np.array(beta_alpha_solution)
            final_beta_plot = [beta_alpha_solution[0]]
            total_cost_plot = [total_tt_solution[0]]
            for i in range(1, len(alpha_vals_solution)):
                restricted_beta = beta_alpha_solution[beta_alpha_solution<beta_alpha_solution[i]]
                if len(restricted_beta)>1:
                    idx_val_arr = []
                    total_cost_arr = []
                    for j in range(len(restricted_beta)):
                        idx_val_arr.append(list(beta_alpha_solution).index(restricted_beta[j]))
                        total_cost_arr.append(total_tt_solution[idx_val_arr[-1]])

                    if min(total_cost_arr)>total_tt_solution[i]:
                        total_cost_plot.append(total_tt_solution[i])
                        final_beta_plot.append(beta_alpha_solution[i])
                        
                else:
                    total_cost_plot.append(total_tt_solution[i])
                    final_beta_plot.append(beta_alpha_solution[i])
            
            print(total_cost_plot)
            total_tt_solution_dict[city_name] = total_cost_plot
            beta_alpha_solution_dict[city_name] = final_beta_plot
            beta_alpha_dist_solution_dict[city_name] = beta_alpha_dist_solution

        return beta_alpha_solution_dict, total_tt_solution_dict, beta_alpha_dist_solution_dict