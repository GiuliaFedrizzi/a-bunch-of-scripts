#!/usr/bin/env python

"""Extract a graph given a raster image. 
Save a csv file with info about nodes and edges of a network obtained from the skeleton of an image.

The part of the image that is analysed is the part with the colour specified in the "analyse_png" function 
(e.g. color = '(255, 255, 255)' to keep the parts in white)

source of original script: https://github.com/danvk/extract-raster-network


USAGE:
python3 /path/to/extract_topology.py              (don't crop the image) 
python3 /path/to/extract_topology.py w            (whole domain of images generated with python script save_bb_figure.py)
or 
python3 /path/to/extract_topology.py t
python3 /path/to/extract_topology.py b            (top part or bottom part only)

INPUT:
reads all the images called "py_bb_*.png" in the directory it is run from

OUTPUT:
"py_branch_info.csv", "py_branch_info_top.csv" or "py_branch_info_bot.csv", depending on which part was analysed (whole figure, top only or bottom only)
which contains 

Giulia's edits: 
- use thin instead of skeletonize
- can open rectangular images, not just square
- save plot using networkx graphs
- get info about branches (e.g. length) and nodes according to their degree (= n of connections)
    1 = isolated, I nodes
    3 = Y node
    4 = X node

    2 = n_2
    5 = n_5

crop images:

# Setting the points for cropped image
left = 316    # limit
top = 147     # limit. 0 is at top, so top < bottom
right = 996   # limit
bottom = 819  # limit. 0 is at top, so bottom > top

# Cropped image of above dimension
# (It will not change original image)
im1 = im.crop((left, top, right, bottom))

On ARC: works with python 3.6
"""

import sys
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

import cv2          #  conda install -c conda-forge opencv
import networkx as nx
import numpy as np
import scipy.ndimage.measurements
import shapely.geometry
from PIL import Image, ImageFilter
from skimage import morphology, segmentation  # most problematic: install this first
import matplotlib.pyplot as plt
import os
import csv
import glob
import re   # regex

# sys.path.append('/home/home01/scgf/myscripts/post_processing')   # where to look for useful_functions.py
# from useful_functions import getTimeStep


def find_color(im: Image, rgb: Tuple[int]) -> np.ndarray:
    """Given an RGB image, return an ndarray with 1s where the pixel is the given color."""
    px = np.asarray(im)
    print(f'px shape: {px.shape}')
    print(f'im size: {im.size}')
    width, height = im.size  # get width and height because I need to invert these
    # print(f'width {width}, height {height}')
    out = np.zeros((height,width), dtype=np.uint8)  # h,w instead of w,h because PIL treats them the other way round
    print(f'out shape: {out.shape}')

    r, g, b = rgb
    out[(px[:, :, 0] == r) & (px[:, :, 1] == g) & (px[:, :, 2] == b)] = 1
    return out

def getTimeStep(inputFile):
    with open(inputFile) as iFile:
        for num, line in enumerate(iFile,1):
            if "Tstep" in line:
                input_tstep = line.split(" ")[1]  # split before and after space, take the second word (value of timestep)
                return input_tstep 

def zhang_suen_node_detection(skel: np.ndarray) -> List[Tuple[int]]:
    """Find nodes based on a skeletonized bitmap.

    (From nefi) Node detection based on criteria put forward in "A fast parallel algorithm
    for thinning digital patterns" by T. Y. Zhang and C. Y. Suen. Pixels p of the skeleton
    are categorized as nodes/non-nodes based on the value of a function A(p) depending on
    the pixel neighborhood of p. Please check the above paper for details.

    A(p1) == 1: The pixel p1 sits at the end of a skeleton line, thus a node
    of degree 1 has been found.
    A(p1) == 2: The pixel p1 sits in the middle of a skeleton line but not at
    a branching point, thus a node of degree 2 has been found. Such nodes are
    ignored and not introduced to the graph.
    A(p1) >= 3: The pixel p1 belongs to a branching point of a skeleton line,
    thus a node of degree >=3 has been found.

    Args:
        *skel* : Skeletonised source image. The skeleton must be exactly 1 pixel wide.

    Returns:
        *nodes* : List of (x, y) coordinates of nodes
    """
    skel = np.pad(skel, 1)
    item = skel.item

    def check_pixel_neighborhood(x, y, skel):
        """
        Check the number of components around a pixel.
        If it is either 1 or more than 3, it is a node.
        """
        p2 = item(x - 1, y)
        p3 = item(x - 1, y + 1)
        p4 = item(x, y + 1)
        p5 = item(x + 1, y + 1)
        p6 = item(x + 1, y)
        p7 = item(x + 1, y - 1)
        p8 = item(x, y - 1)
        p9 = item(x - 1, y - 1)

        # The function A(p1),
        # where p1 is the pixel whose neighborhood is beeing checked
        components = (
            (p2 == 0 and p3 == 1)
            + (p3 == 0 and p4 == 1)
            + (p4 == 0 and p5 == 1)
            + (p5 == 0 and p6 == 1)
            + (p6 == 0 and p7 == 1)
            + (p7 == 0 and p8 == 1)
            + (p8 == 0 and p9 == 1)
            + (p9 == 0 and p2 == 1)
        )
        return (components >= 3) or (components == 1)

    nodes = []
    w, h = skel.shape
    for x in range(1, w - 1):
        for y in range(1, h - 1):
            if item(x, y) != 0 and check_pixel_neighborhood(x, y, skel):
                nodes.append((x - 1, y - 1))
    return nodes


def find_dense_skeleton_nodes(skel: np.ndarray) -> List[Tuple[int, int]]:
    """Find "dense" (2x2 or larger) regions in the skeleton."""
    eroded = morphology.binary_erosion(np.pad(skel, 1), np.ones((2, 2)))[1:-1, 1:-1]

    # Find the centers of mass of connected components
    labeled_array, num_features = scipy.ndimage.measurements.label(eroded)
    centers = scipy.ndimage.measurements.center_of_mass(eroded, labeled_array, [*range(1, num_features+1)])
    return [(int(x), int(y)) for (x, y) in centers]


def add_dense_nodes(nodes: List[Tuple[int, int]], dense_nodes: List[Tuple[int, int]], min_distance = 5) -> List[Tuple[int, int]]:
    """Add in new nodes which are distinct from the old ones."""
    keep = []
    min_d2 = min_distance ** 2
    for node in dense_nodes:
        x, y = node
        is_ok = True
        for nx, ny in nodes:
            d2 = (x - nx) **2 + (y - ny) ** 2
            if d2 < min_d2:
                is_ok = False
                break
        if is_ok:
            keep.append(node)

    print(f'Adding {len(keep)}/{len(dense_nodes)} dense nodes to existing {len(nodes)} nodes.')
    return [*nodes, *keep]


@dataclass
class Path:
    start: Tuple[int, int]
    stop: Tuple[int, int]
    path: List[Tuple[int, int]]


def is_new_path(paths: List[Path], path: Path) -> bool:
    """Is this a new path, or does it overlap signficantly with existing paths?"""
    candidates = [p for p in paths if p.start == path.start and p.stop == path.stop]
    other_points = {coord for p in candidates for coord in p.path[1:-1]}
    interior = set(path.path[1:-1])
    if other_points & interior:
        return False
    return True


def is_valid_self_loop(path: List[Tuple[int, int]], min_self_loop_distance: int) -> bool:
    if len(path) < min_self_loop_distance:
        return False
    # Only the end node can appear twice in a self-loop
    return len([c for c, n in Counter(path).items() if n >= 2]) == 1


def find_paths(skel: np.ndarray, nodes: List[Tuple[int]], min_distance=5) -> List[Path]:
    """Find paths between nodes in the graph using the connectivity in the skeleton.

    This returns a list of edges (pairs of nodes) with the following properties.
        - path: list of coordinates connecting the nodes (including the nodes)
        - d: length of the path

    This will early-out if a path shorter than min_distance is found.

    There may be multiple distinct paths between the same nodes, or a path between a node and itself.
    """
   
    width, height = skel.shape
    # if width > 2000:
    #     min_distance=4
    def neighbors(x, y):
        for dy in (-1, 0, 1):
            cy = y + dy
            if cy < 0 or cy >= height:
                continue
            for dx in (-1, 0, 1):
                cx = x + dx
                if (dx != 0 or dy != 0) and 0 <= cx < width and skel[cx, cy]:
                    yield cx, cy

    # each cell points back to its parent
    parents = {n: None for n in nodes}

    def trace_back(node):
        trace = []
        while node:
            trace.append(node)
            node = parents.get(node)
        return trace

    d = {n: 0 for n in nodes}  # used to avoid backtracking

    edges = []
    frontier = [*nodes]
    while frontier:
        next_frontier = []
        for n in frontier:
            x, y = n
            for c in neighbors(x, y):
                if c not in parents:
                    parents[c] = n
                    next_frontier.append(c)
                    d[c] = 1 + d[n]
                else:
                    if d[c] >= d[n]:
                        # we've got a connection! Follow both cells back to trace it out
                        tn = trace_back(n)
                        tc = trace_back(c)
                        tc.reverse()
                        path = [*tc, *tn]
                        endpoints = (path[0], path[-1])
                        start, stop = min(endpoints), max(endpoints)
                        new_path = Path(start, stop, path)
                        # Ignore redundant paths and short self-loops
                        if is_new_path(edges, new_path) and (
                            start != stop or is_valid_self_loop(path, min_distance)
                        ):
                            edges.append(new_path)
                            if len(path) - 1 < min_distance:
                                # This edge will get pruned out anyway, so no need to keep looking.
                                return edges

        frontier = next_frontier

    return edges


def merge_nodes(
    nodes: List[Tuple[int, int]], edges: List[Path], n1: Tuple[int, int], n2: Tuple[int, int]
) -> List[Tuple[int, int]]:
    ends = {n1, n2}
    paths = [e.path for e in edges if {e.start, e.stop} == ends]
    assert paths
    path = min(paths, key=lambda p: len(p))
    idx = len(path) // 2
    new_node = path[idx]
    return [new_node] + [n for n in nodes if n != n1 and n != n2]  # add new, ignore the two merged

# def check_angle_between_branches()

def make_graph(nodes: List[Tuple[int, int]], edges: List[Path]) -> nx.MultiGraph:
    g = nx.MultiGraph()
    g.add_nodes_from(nodes)
    for edge in edges:
        g.add_edge(edge.start, edge.stop, path=edge.path, d=len(edge.path) - 1)
    return g


def connect_graph(skel: np.ndarray, min_distance: int) -> nx.MultiGraph:
    """Iteratively produce a graph, merging nodes until none are < min_distance apart.
    zhang_suen_node_detection -> nodes
    find_dense_skeleton_nodes -> dense nodes
    add_dense_nodes -> nodes (old+new)
    find_paths -> edges

    while loop until no further changes:
        if d < min_distance:
            merge_nodes

    """
    nodes = zhang_suen_node_detection(skel)  # original

    # the following lines are to save the graph before the edits in the while loop
    # edges = find_paths(skel, nodes, min_distance)
    # g = make_graph(nodes, edges)

    # plot
    # im = Image.open('medianfilter.png') # open the image saved earlier
    # ax = draw_nx_graph(im,g)
    # plt.savefig('g01zhang')
    # plt.clf()

    dense_nodes = find_dense_skeleton_nodes(skel)  # original
    nodes = add_dense_nodes(nodes, dense_nodes)  # original
    edges = find_paths(skel, nodes, min_distance)  # original
    

    any_changed = True
    while any_changed:
        any_changed = False
        for edge in edges:
            d = len(edge.path) - 1
            if d < min_distance:  # if they are too close to each other: merge them
                n1 = edge.start
                n2 = edge.stop
                nodes = merge_nodes(nodes, edges, n1, n2)
                edges = find_paths(skel, nodes, min_distance)
                print(f'Merged {n1} and {n2}, d={d}')
                any_changed = True
                break

    # All good!
    return make_graph(nodes, edges)


def simplify_paths(g: nx.Graph, tolerance=1) -> nx.Graph:
    for n1, n2, k in g.edges(keys=True):
        g[n1][n2][k]['path'] = shapely.geometry.LineString(g[n1][n2][k]['path']).simplify(tolerance)
    return g


def extract_network(px: np.ndarray, im: Image, min_distance=12) -> nx.Graph:
    skel = morphology.thin(px)
    print(f'Skeleton px={skel.sum()}')
    g = connect_graph(skel, min_distance)
    # ax = draw_nx_graph(im, g)
    # plt.savefig("g04connect_graph.png",dpi=200)
    # plt.clf()

    # simplify:
    g = simplify_paths(g)
    return g


def draw_nx_graph(im: Image, g: nx.Graph) -> None:
    """ Takes the graph "g", plots it according to node and edge coordinates. 
    Returns a figure, which can then be visualised with plt.show() or saved with plt.savefig(out_path) """
    # fig, ax = plt.subplots()
    # print(list(g.degree)[0])
    all_degrees = dict(nx.degree(g)).values()
    # degrees = [x for x in list(g.degree)]
    lab = dict(zip(g.nodes(), all_degrees))
    pos = {point: point for point in g.nodes()}  # save nodes in a format that can be used as position when plotting
    ax = nx.draw(g,pos=pos,node_size=5)
    # nx.draw_networkx_labels(g,pos=pos,labels=degrees,ax=ax)
    
    # node labels:
    pos_higher = {}
    x_off = 5  # offset on the x axis

    for k, v in pos.items():
        pos_higher[k] = (v[0]+x_off, v[1])
    nx.draw_networkx_labels(g,pos=pos_higher,ax=ax,labels=lab,font_weight='bold',font_color='r',font_size=3)  # degrees only
    
    #edge labels
    edge_d = [d for (u,v,d) in g.edges.data('d')]  # distance values are stored under the attribute "d"
    edge_lab = dict(zip(g.edges(),edge_d))   # create a dictionary that can be used to plot labels
    # print(f'edge_lab{edge_lab}')
    # print('edge_lab '+str(edge_lab))
    bbox_options = {"boxstyle":'round', "ec":(1.0, 1.0, 1.0), "fc":(1.0, 1.0, 1.0), "alpha": 0.7}
    nx.draw_networkx_edge_labels(g,pos=pos,edge_labels=edge_lab,ax=ax,font_size=2,bbox=bbox_options)

    plt.axis("on")  # show the axes
    plt.axis('equal')  # keep true aspect ratio
    plt.gca().invert_yaxis() # invert y axis because images start y=0 in the top left corner
    plt.imshow(im)
    return ax
    # plt.gca().invert_xaxis()
    #plt.savefig(out_path)  
    # plt.show()

def topo_analysis(g: nx.Graph,tstep_number: float) -> dict:
    """
    Perform topological analysis. 
    Mostly based on Sanderson, D. J., & Nixon, C.
    W. (2015). The use of topology in fracture network characterization. Journal
    of Structural Geology, 72, 55-66. https://doi.org/10.1016/j.jsg.2015.01.005

    """
    input_tstep = 0   # in case there is no input.txt (e.g. field image)
    if os.path.isfile("input.txt"):
        input_tstep = float(getTimeStep("input.txt"))  # needed to calculate time (1st row of csv)
    edge_lengths = [d for (u,v,d) in g.edges.data('d')]  # distance values are stored under the attribute "d"
    print(f'n of branches: {len(edge_lengths)}')
    print(f'branch lengths: {edge_lengths}')
    # print(f'edge_d: {edge_d}, type: {type(edge_d)}') 
    # all_degrees = (nx.degree(g)).values()
    all_degrees = [x for ((u,v),x) in list(g.degree)]
    n_1 = all_degrees.count(1) # number of isolated (I) nodes
    n_2 = all_degrees.count(2) # numbner of nodes with 2 connections
    n_3 = all_degrees.count(3) # number of nodes with 3 connections
    n_4 = all_degrees.count(4) # etc
    n_5 = all_degrees.count(5) 
    n_0 = all_degrees.count(0) 

    n_I = n_1 + n_0   # I nodes
    n_Y = n_2 + n_3   # Y nodes
    n_X = n_4 + n_5   # X nodes

    #connected = [x for x in all_degrees if (x!=1)]
    print(f'all_degrees: {all_degrees}')
    print(f'num of I: {n_I}, Y: {n_Y}, X: {n_X}. tot = {n_I+n_Y+n_X}')

    # assert len(g.nodes())==n_I+n_Y+n_X, "Error: some of the nodes have not been taken into account. There might have too many connections."
    
    n_of_lines = 0.5*(n_I+n_Y)
    print(f'lines: {n_of_lines}')
    n_of_branches = 0.5*(n_I+3*n_3+4*n_4+5*n_5) + n_2 + n_0
    # branches_to_line = n_of_branches / n_of_lines
    # print(f'branches/lines: {branches_to_line}')
    branches_tot_length = sum(edge_lengths)
    print(f'branches tot length: {branches_tot_length}')
    branch_info = {"time":tstep_number*input_tstep,"n_I":n_I,"n_2":n_2,"n_3":n_3,"n_4":n_4,"n_5":n_5,
                   "branches_tot_length":branches_tot_length} # dictionary with info that I just calculated
    return branch_info

def analyse_png(png_file: str, part_to_analyse: str) -> dict:
    color = '(255, 255, 255)'  # select the colour: do the analysis on the white parts
    assert color[0] == '('
    assert color[-1] == ')'
    rgb = tuple(int(v) for v in color[1:-1].split(','))
    assert len(rgb) == 3
    assert png_file.endswith('.png')
    
    timestep_number = float(re.findall(r'\d+',png_file)[0])   # regex to find numbers in string. Then convert to float. Will be used to name the csv file 
    im = Image.open(png_file)

    crop_im = 1 # flag for cropping the image or not. default is yes (it's no for part_to_analyse = f)
    if part_to_analyse == 'w': # whole domain
    # Setting the points for cropped image
    # left = 316; top = 147; right = 996; bottom = 819 # worked when images were generated on my laptop
        # left = 475; top = 223; right = 1490; bottom = 1228 # worked when images were generated on ARC
        left = 428; top = 162; right = 1492; bottom = 1211  # worked for threeAreas/prod/p01
        out_path = "p_"+png_file.replace('.png', '_nx.grid.png')
    elif part_to_analyse == 'b':
        left = 475; top = 1027; right = 1490; bottom = 1228 # BOTTOM - melt-production zone
        out_path = "p_bot_"+png_file.replace('.png', '_nx.grid.png')
    elif part_to_analyse == 't':
        #left = 475; top = 223; right = 1490; bottom = 1027 # TOP = melt-production zone - if prod zone is 0-0.2
        #left = 475; top = 223; right = 1490; bottom = 1178 # TOP = melt-production zone  - if prod zone is 0-0.05:  1228-(1228-223)*0.05
        left = 475; top = 223; right = 1490; bottom = 1128 # TOP = melt-production zone  - if prod zone is 0-0.1
        
        out_path = "p_top_"+png_file.replace('.png', '_nx.grid.png')
        
    elif part_to_analyse == 'f': # full, or field = do not crop
        crop_im = 0  # in this case, do not crop 
        out_path = "p_"+png_file.replace('.png', '_nx.grid.png')

    # im.show()
    if crop_im:  # crop the image if flag is true
        # Cropped image of above dimension
        im = im.crop((left, top, right, bottom))
        # Apply median filter to smooth the edges
    im = im.filter(ImageFilter.ModeFilter(size=7)) # https://stackoverflow.com/questions/62078016/smooth-the-edges-of-binary-images-face-using-python-and-open-cv 
    # out_path = png_file.replace('.png', '_median.png')
    # im.save(out_path)


    px = find_color(im, rgb).T

    print(f'Street RGB: {rgb}')
    print(f'Street pixels: {px.sum()}')
    g = extract_network(px,im)
    print(f'Extracted street network:')
    print(f'  - {len(g.nodes())} nodes')
    print(f'  - {len(g.edges())} edges')
    
    # do some statistics
    branch_info = topo_analysis(g,timestep_number)

    # viz grid with networkx's plot
    ax = draw_nx_graph(im, g)
    plt.savefig(out_path,dpi=300)
    plt.clf()
    # plt.show()

    return branch_info

def file_loop(parent_dir: str,part_to_analyse: str) -> None:
    """ given the parent directory, cd there and go through all png files"""
    # os.chdir("/Users/giuliafedrizzi/Library/CloudStorage/OneDrive-UniversityofLeeds/PhD/arc/myExperiments/wavedec2022/wd05_visc/visc_4_5e4/vis5e4_mR_09")
    os.chdir(parent_dir)
    print(os.getcwd())
    print(f'n of files: {len(glob.glob("py_bb_*.png"))}')

    branch_info = []  # create an empty list. One row = one dictionary for each simulation
    for f,filename in enumerate(sorted(glob.glob("py_bb_*.png"))):
        """ Get the file name, run 'analyse_png', get the info on the branches,
        save it into a csv   """
        branch_info.append(analyse_png(filename,part_to_analyse))  # build the list of dictionaries
    
    keys = branch_info[0].keys()  #  read the command line arguments

    if part_to_analyse == 'w' or part_to_analyse == 'f': # whole domain
        csv_file_name = "py_branch_info.csv" 
    elif part_to_analyse == 'b':
        csv_file_name = "py_branch_info_bot.csv" 
    elif part_to_analyse == 't':
        csv_file_name = "py_branch_info_top.csv" 

    # write to csv file
    with open(csv_file_name, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(branch_info)

# starting here:           
d = os.getcwd()  # save path to current directory (will be given as input to file_loop() function)
if (len(sys.argv) == 1):
    part_to_analyse = 'f'  # if no command line argument is provided, set the option to "f". It means it won't crop the image.
else:
    part_to_analyse = sys.argv[1]
assert (part_to_analyse == 'w' or part_to_analyse == 't' or part_to_analyse == 'b' or part_to_analyse == 'f'), "Error: specify w for whole domain, b for bottom (melt zone), t for top (through zone), f for full (or field) for field images"


file_loop(d,part_to_analyse)
