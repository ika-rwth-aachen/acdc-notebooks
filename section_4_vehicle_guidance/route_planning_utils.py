#!/usr/bin/env python
import matplotlib.pyplot as plt
import xml.dom.minidom
import os

def plot_line_string(ids):
    """
    Visualizes a line-string 
    Arguments:
    ids -- line-string given as a list of points
    
    Returns:
    None
    """
    x=[]
    y=[]
    for p in ids:
        x.append(p.x)
        y.append(p.y)
    plt.plot(x,y,'k')

def plot_map(ll_map):
    """
    Visualizes a Lanelet2 map 
    Arguments:
    ll_map -- Lanelet2 map object
    
    Returns:
    None
    """
    for ls in ll_map.lineStringLayer:
        if(len(ls)>1):
            ids = []
            for p in ls:
                ids.append(p)
            plot_line_string(ids)

        
def print_osm(osm_file):
    """
    Display an osm-file
    Arguments:
    osm_file -- Filepath to a given osm-file
    
    Returns:
    None
    """
    if osm_file is None:
        return
    file = xml.dom.minidom.parse(osm_file)
    xml_file = file.toprettyxml()
    # remove newline issue:
    xml_file = os.linesep.join([s for s in xml_file.splitlines() if s.strip()])
    print(xml_file)    

        

        



