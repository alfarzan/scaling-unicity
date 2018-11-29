"""
This file provides two funcitons used for generating the correct graph from a
series of geographic locations corresponding to antennas or any other points of
interest.

Author: Ali Farzanehfar
"""

import numpy as np
import scipy.spatial as sp
import pandas as pd
from collections import defaultdict


def find_neighbors(pindex, triang):
    """This function uses the built in scipy.spatial.Delaunay objects features
    to fetch the neighbouring nodes of a particular point.

    Inputs:
        - pindex: int, index of point to find neighbours for
        - triang: scipy.spatial.Delaunay object

    Outputs:
        - ndarray of inidces of points which neighbour pindex
    -------
    AF
    """
    a = triang.vertex_neighbor_vertices[1]
    b = triang.vertex_neighbor_vertices[0][pindex]
    c = triang.vertex_neighbor_vertices[0][pindex + 1]
    return a[b:c]


def get_geo(inputdir, fname, pandas_sep):
    """This function reads lat and long information from input files and returns
    a dictionary linking antennas to neighbouring antennas.

    Inputs:
        - inputdir: str, indicates path of where input files are located
        - fname: str, name of file containing antenna ids, lat and long
        - pandas_sep: str, the seperator for the fields inside elements of
                      fname

    Outputs:
        - dict with keys being integers which enumerate antennas in the order
          they appear in fname. Values are sets of antenna ids which reside in
          the towers which neighbour the tower of the antenna which is the
          dictionary entry key.
    -------
    AF
    """
    pdf = pd.read_csv(inputdir + fname,
                      names=['antid', 'lat', 'long'], sep=pandas_sep)
    pdf['antid'] = np.int16(pdf['antid'])
    tower_dict = dict(pdf.groupby(['long', 'lat']).indices)
    points, ant_in_tower = zip(*tower_dict.items())
    ant_in_tower = [np.int16(i) for i in ant_in_tower]
    points = np.array(points)
    tri = sp.Delaunay(points)

    ant_neighbour_ant = defaultdict(set)
    for i in range(len(points)):
        t_neighbours = find_neighbors(i, tri)
        a = set()
        for t in t_neighbours:
            ants = ant_in_tower[t]
            a = a.union(set(ants))
            for ant in ant_in_tower[i]:
                ant_neighbour_ant[ant] = a
    return dict(ant_neighbour_ant)
