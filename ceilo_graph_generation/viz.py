import pickle
from os import path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

ceilographs_dir = '/home/rigel/MDammann/PycharmProjects/CC4Graphs/ceilo_graph_generation/graphs_processed.pickle'
img_dir = '/media/rigel/Data/DWD/allimgs'

with open(ceilographs_dir, 'rb') as handle:
    ceilographs = pickle.load(handle)

#print(ceilographs)
clustercolors = {0:'blue', 1:'red', 2:'green', 3:'orange', 4:'cyan', 5:'magenta', 6:'yellow', 7:'darkred', 8:'teal',
                 9:'darkviolet', 10:'chocolate', 11:'pink'}

all_keys = list(ceilographs.keys())
#print(ceilographs['20140725_hohenpeissenberg_CHM060028_000.nc_original._'])
keys_idx = 30
#bild laden

im = Image.open(path.join(img_dir, all_keys[keys_idx][:-1] + 'png'))
im_dims = im.size #(w,h)
print(im_dims)

#quadrate in originalbild zeichnen
#print(ceilographs[all_keys[keys_idx]][0].x.cpu().detach().numpy())
all_nodes = ceilographs[all_keys[keys_idx]][0].x.cpu().detach().numpy()

all_edges = ceilographs[all_keys[keys_idx]][0].edge_index.cpu().detach().numpy()
all_edges_transf = [(all_edges[0][idx], all_edges[1][idx]) for idx in range(len(all_edges[0]))]
all_edges_final = []
for edge in all_edges_transf:
    if not (edge[1], edge[0]) in all_edges_final:
        all_edges_final.append(edge)
all_node_centercoords = []
draw = ImageDraw.Draw(im)

for node in all_nodes:
    x_min_rel, x_max_rel, y_min_rel, y_max_rel = node[12:16]
    cluster = np.argmax(node[0:12])
    x_min, x_max, y_min, y_max = x_min_rel*im_dims[0], x_max_rel*im_dims[0], y_min_rel*im_dims[1], y_max_rel*im_dims[1]
    all_node_centercoords.append(((x_min+x_max)/2,(y_min+y_max)/2))
    draw.rounded_rectangle([x_min, y_min, x_max, y_max], width=2, outline=clustercolors[cluster])
    '''
    font = ImageFont.truetype("sans-serif.ttf", 16)
    # draw.text((x, y),"Sample Text",(r,g,b))
    draw.text(((x_max+x_min)/2, (y_max+y_min)/2), "Sample Text", (255, 255, 255), font=font)
    '''

#knoten verbinden

for edge in all_edges_final:
    x0= all_node_centercoords[edge[0]][0]
    y0=all_node_centercoords[edge[0]][1]
    x1= all_node_centercoords[edge[1]][0]
    y1=all_node_centercoords[edge[1]][1]
    draw.line([(x0, y0), (x1, y1)], width=3, fill='grey')


im.save('test_pil.png')