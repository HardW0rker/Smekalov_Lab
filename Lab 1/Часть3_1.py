import os
import numpy as np
import networkx as nx

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'vk.gml')
vk = nx.read_gml(file_path)

i = 0
len_i = [0]*7

print("Кол-во уникальных пользователей ", len(vk.nodes))

friends = {}
fv = []
for j in vk.nodes: 
    friends[j]=0

for j in vk.edges:
    friends[j[0]]+=1
    friends[j[1]]+=1

sorted_friends = sorted(friends.items(), key=lambda x: x[1], reverse=True)

print("Пользователи с наибольшим количеством друзей ")

for pair in sorted_friends[:15]:
    i+=1
    print("id ", pair[0], " друзья ", pair[1])

for pair in sorted_friends:
    fv.append(pair[1])

print('Медианное число друзей:  ', np.median(fv))
print('Среднее число друзей: ', round(np.mean(fv)))

smallp = nx.all_pairs_shortest_path_length(vk)
print(smallp)
for pair in smallp:
    for ln in pair[1].values():
        if ln>=1 and ln<=6:
            len_i[ln]+=1
        else:
            len_i[0]+=1

overall_ln = len(vk.nodes)**2

for i in range(1,7):
    print(f'Доля пар с L={i} {len_i[i]/overall_ln}')
print(f'Доля несвязанных пар или пар с L>6 или {len_i[0]/overall_ln}')