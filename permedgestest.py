

import numpy as np
import random
node_num = 4
permute_num = 2
edge_index = np.array([[0,1,0,2,3,1],[1,0,2,0,1,3]])
max_index = edge_index[0].size-1

indices=random.sample(range(0, max_index), permute_num)
tuples = []
for p_i in indices:
    tuples.append((edge_index[0][p_i], edge_index[1][p_i]))
print(tuples)
print(indices)
corr_indices=[]
for tup in tuples:
    for e_i in range(max_index+1):
        if edge_index[0][e_i]==tup[1] and edge_index[1][e_i]==tup[0]:
            corr_indices.append(e_i)

print(corr_indices)


