import pandas as pd
import numpy as np
from pyecharts import Graph
import json

path = "C:\\Users\\NLSDE\\Desktop\\GZ_kdd\\"
# adj = pd.read_csv(path+"gy_contest_link_top.txt", delimiter=';', dtype={'in_links': np.str, 'out_links': str})
adj = pd.read_csv(path+"new_top.txt", delimiter=';', dtype={'in_links': np.str, 'out_links': np.str})
adj = adj.fillna('')
linkDict={}
with open(path+"linkDict.json") as f:
    linkDict=json.loads(f.read())
info = pd.read_csv(path+"gy_contest_link_info.txt", delimiter=';')
nodes = [{"name": str(n), "symbolSize": int(info[info.link_ID == int(linkDict[str(n)])]['width'].values[0])} for n in adj.link_ID.values]
links = []
for (link, inl, outl) in adj.values:
    for ol in outl.split("#"):
        links.append({"source": str(link), "target": str(ol), "value": int(info[info.link_ID==int(linkDict[str(link)])]['length'].values[0])
                                                             +(0 if ol == "" else int(info[info.link_ID==int(linkDict[ol])]['length'].values[0]))})
    for il in inl.split("#"):
        links.append({"source": str(il), "target": str(link), "value": int(info[info.link_ID==int(linkDict[str(link)])]['length'].values[0])
                                                             +(0 if il == "" else int(info[info.link_ID==int(linkDict[il])]['length'].values[0]))})
graph = Graph("road_graph", width=1600, height=800)
graph.add("", nodes, links, is_label_show=True, repulsion=1000, label_text_color=None, gravity=0.001)
# # graph.show_config()
graph.render()