"""
Copyright Mechanisms Underlying Behavior Lab, Singapore
https://mechunderlyingbehavior.wordpress.com/

animal.py is part of the locomotion package comparing animal behaviours, developed
to support the work discussed in the paper "Computational geometric tools for
modeling inherent variability in animal behavior" by MT Stamps, S Go, and AS Mathuru.

This python module defines the Animal class object used throughout the package to store
and manipulate the tracked data of the animal. On initialization, the animal object
extracts various pieces of information from the JSON files, such as the experiment
settings and the tracked data of the animal, and prepares them for use.
"""
import sys
import os
import csv
import re
import math
import json
import warnings
from math import ceil
import numpy as np

SMOOTH_RANGE = 5 #technically (range-1)/2

################################################################################
#### Animal class ####
################################################################################

class Animal():
    """
    Animal class object used to capture and store experimental information, initialized
    with a JSON file containing experiment information as well as the path to the tracked
    animal data.
    """
    # pylint: disable=too-many-instance-attributes
    # Class contains a large number of attributes, but is necessary for its usecase.
    # pylint: disable=too-many-public-methods
    # Class contains a large number of methods, but is done to improve readability.
    def __init__(self, json_item):
        self.__name = json_item["name"]
        self.__data_file = os.path.abspath(json_item["data_file_location"])
        self.__filename = os.path.basename(self.__data_file)
        self.__animal_type = json_item["animal_attributes"]["species"]
        self.__exp_type = json_item["animal_attributes"]["exp_type"]
        self.__animal_id = json_item["animal_attributes"]["ID"]
        self.__is_control = json_item["animal_attributes"]["control_group"].lower() == 'true'
        self.__dim_x = json_item["capture_attributes"]["dim_x"] # Pixels
        self.__dim_y = json_item["capture_attributes"]["dim_y"] # Pixels
        self.__pix = json_item["capture_attributes"]["pixels_per_mm"]         # Pixels per MM
        self.__frame_rate = json_item["capture_attributes"]["frames_per_sec"] # Frames per Second
        self.__start = json_item["capture_attributes"]["start_time"] # In Minutes
        self.__end = json_item["capture_attributes"]["end_time"]         # In Minutes
        self.__baseline_start = json_item["capture_attributes"]["baseline_start_time"] # In Minutes
        self.__baseline_end = json_item["capture_attributes"]["baseline_end_time"]     # In Minutes
        self.__raw_vals = {}
        self.__means = {}
        self.__stds = {}
        self.__grid_size = None
        self.__num_x_grid = None
        self.__num_y_grid = None
        self.__perturbation = None
        self.__tolerance = None
        self.__num_verts = None
        self.__num_triangles = None
        self.__colors = None
        self.__reg_coords = None
        self.__flat_coords = None
        self.__triangulation = None
        self.__boundary_vertices = None
        self.__boundary_edges = None
        self.__central_vertex = None
        self.__vertex_bfs = None
        self.__triangle_triangle_adjacency = None

    def get_name(self):
        """Getter function for self.__name."""
        return self.__name

    def get_data_file_location(self):
        """Getter function for self.__data_file."""
        return self.__data_file

    def get_data_file_name(self):
        """Getter function for self.__filename."""
        return self.__filename

    def get_animal_type(self):
        """Getter function for self.__name."""
        return self.__animal_type

    def get_exp_type(self):
        """Getter function for self.__exp_type."""
        return self.__exp_type

    def get_id(self):
        """Getter function for self.__animal_id."""
        return self.__animal_id

    def get_exp_times(self):
        """
        Getter function for both self.__start and self.__end.
        :Returns:
            tuple : (self.__start, self.__end)
        """
        return (self.__start, self.__end)

    def get_exp_start_time(self):
        """Getter function for self.__start."""
        return self.__start

    def get_exp_end_time(self):
        """Getter function for self.__end."""
        return self.__end

    def get_baseline_times(self):
        """
        Getter function for both self.__baseline_start and self.__baseline_end.
        :Returns:
            tuple : (self.__baseline_start, self.__baseline_end)
        """
        return (self.__baseline_start, self.__baseline_end)

    def get_baseline_start_time(self):
        """Getter function for self.__baseline_start."""
        return self.__baseline_start

    def get_baseline_end_time(self):
        """Getter function for self.__baseline_end."""
        return self.__baseline_end

    def in_control_group(self):
        """Getter function for self.__is_control."""
        return self.__is_control

    def get_dims(self):
        """Getter function for self.__dim_x and self.__dim_y."""
        return self.__dim_x, self.__dim_y

    def get_pixel_density(self):
        """Getter function for self.__pix."""
        return self.__pix

    def get_frame_rate(self):
        """Getter function for self.__frame_rate."""
        return self.__frame_rate

    def add_raw_vals(self, var_name, val_list):
        """
        Adds an entry to the dict self.__raw_vals.
        :Parameters:
         var_name : hashable key point to variables in animal object
         val_list : list of data values corresponding to var_name
        """
        self.__raw_vals.update({var_name:val_list})

    def get_raw_vals(self, var_name, start_frame=None, end_frame=None):
        """
        Return the raw vals stored in animal object.
        :Parameters:
         var_name : hashable key pointing to variables in animal object
         start_frame : starting frame of portion to extract
         end_frame : ending frame of portion to extract
        """
        if start_frame is None:
            start_frame = self.__start*60*self.__frame_rate
        if end_frame is None:
            end_frame = self.__end*60*self.__frame_rate
        # logic check
        try:
            values = self.__raw_vals[var_name]
        except KeyError:
            raise KeyError("get_raw_vals: {} not found in animal object.".format(var_name))
        if start_frame > end_frame:
            raise ValueError("get_raw_vals: Start frame is after End frame.")
        if start_frame > len(values):
            raise ValueError("get_raw_vals: Start frame comes after existing frames.")
        if end_frame > len(values):
            warnings.warn("get_raw_vals: End frame comes after existing frames. "
                          "Defaulting to the final frame stored.")
        return values[start_frame:end_frame]

    def get_mult_raw_vals(self, var_names, start_frame=None, end_frame=None):
        """
        Runs self.get_raw_vals for multiple variables stored in animal object.
        :Parameters:
         var_names : list of hashable keys pointing to variables in animal object
         start_frame : starting frame of portion to extract
         end_frame : ending frame of portion to extract
        """
        return [self.get_raw_vals(v, start_frame, end_frame) for v in var_names]

    def init_stats(self, var_name):
        """
        Utility function for initializing a dictionary entry in self.__means and
        self.__stds for key var_name.
        :Parameters:
         var_name : hashable key pointing to variable in animal object
        """
        self.__means.update({var_name:{}})
        self.__stds.update({var_name:{}})

    def add_stats(self, var_name, scope, start_frame, end_frame):
        """
        Calculates statistics of var_name over a specific scope, as defined by
        start_frame and end_frame.
        :Parameters:
         var_name : hashable key pointing to variables in animal object
         scope : hashable key representing the scope defined by start_frame and end_frame
         start_frame : starting frame of scope
         end_frame : ending frame of scope
        """
        if var_name not in self.__means:
            self.init_stats(var_name)
        means, stds = norm(self.__raw_vals[var_name][start_frame:end_frame])
        self.__means[var_name].update({scope:means})
        self.__stds[var_name].update({scope:stds})

    def get_stats(self, var_name, scope):
        """
        Retrieve statistics of var_name calculated over scope period.
        :Parameters:
         var_name : hashable key pointing to variables in animal object
         scope : hashable key pointing to predefined scope
        :Returns:
         means, stds
        """
        return self.__means[var_name][scope], self.__stds[var_name][scope]

    def set_grid_size(self, grid_size):
        """
        Setter function for self.__grid_size, self.__num_x_grid, and self.__num_y_grid
        by dividing self.__dim_x and self.__dim_y by grid_size.
        :Parameters:
         grid_size : int. Size of each grid. Should divide self.__dim_x and self.__dim_y.
        """
        if self.__dim_x % grid_size != 0 or self.__dim_y % grid_size != 0:
            _throw_error("grid_size does not divide dim x or dim y.")
        self.__grid_size = grid_size
        self.__num_x_grid = int(ceil(self.__dim_x/grid_size))
        self.__num_y_grid = int(ceil(self.__dim_y/grid_size))

    def get_num_grids(self):
        """Getter functions for self.__num_x_grid and self.__num_y_grid"""
        return self.__num_x_grid, self.__num_y_grid

    def get_grid_size(self):
        """Getter functions for self.__num_x_grid and self.__num_y_grid"""
        return self.__grid_size

    def set_perturbation(self, perturbation):
        """
        Setter functions for self.__perturbation
        :Parameters:
         perturbation : float. A small number. Used to create an offset value
         from which we can check if an animal's X, Y values are within bounds.
        """
        self.__perturbation = perturbation

    def get_perturbation(self):
        """Getter functions for self.__perturbation"""
        return self.__perturbation

    def set_tolerance(self, tolerance):
        """
        Setter functions for self.__tolerance
        :Parameters:
         tolerance : float. A small number. Used to check if an animal's centre
         of mass is close enough to the origin.
        """
        self.__tolerance = tolerance

    def get_tolerance(self):
        """Getter functions for self.__tolerance"""
        return self.__tolerance

    def set_num_verts(self, num_verts):
        """
        Setter functions for self.__num_verts
        :Parameters:
         num_verts : int. The number of vertices in an animal's heat map.
        """
        self.__num_verts = num_verts

    def get_num_verts(self):
        """Getter functions for self.__num_verts"""
        return self.__num_verts

    def set_num_triangles(self, num_triangles):
        """
        Setter functions for self.__num_triangles
        :Parameters:
         num_triangles : int. The number of triangles in the triangulation
         associated with an animal's heat map.
        """
        self.__num_triangles = num_triangles

    def get_num_triangles(self):
        """Getter functions for self.__num_triangles"""
        return self.__num_triangles

    def set_colors(self, colors):
        """
        Setter functions for self.__colors
        :Parameters:
         colors : list of triples of floats. The RGB coordinates for each 
         triangle in the triangulation associated to an animal's heat map.
        """
        self.__colors = colors

    def get_colors(self):
        """Getter functions for self.__colors"""
        return self.__colors

    def set_regular_coordinates(self, coordinates):
        """
        Setter functions for self.__reg_coords
        :Parameters:
         coordinates : list of triples of floats. The the x-, y-,
         and z-coordinates of the vertices for a triangulation of the 
         animal's heat map.
        """
        self.__reg_coords = coordinates

    def get_regular_coordinates(self):
        """Getter functions for self.__reg_coords"""
        return self.__reg_coords

    def set_flattened_coordinates(self, coordinates):
        """
        Setter functions for self.__flat_coords
        :Parameters:
         coordinates : list of pairs of floats. The x- and y-coordinates of 
         the vertices of a triangulation that have been conformally flattened
         to the unit disk.
        """
        self.__flat_coords = coordinates

    def get_flattened_coordinates(self):
        """Getter functions for self.__flat_coords"""
        return self.__flat_coords

    def set_triangulation(self, triangles):
        """
        Setter functions for self.__triangulation
        :Parameters:
         triangles : list of triples of ints. The indices of the vertices
         for each triangle in the triangulation of a surface.
        """
        self.__triangulation = triangles

    def get_triangulation(self):
        """Getter functions for self.__triangulation"""
        return self.__triangulation

    def set_boundary_vertices(self, vertices):
        """
        Setter functions for self.__boundary_vertices
        :Parameters:
         vertices : numpy array of ints. The indices of the vertices that 
         are on the boundary of this animal in counter-clockwise order.
        """
        self.__boundary_vertices = vertices

    def get_boundary_vertices(self):
        """Getter functions for self.__boundary_vertices"""
        return self.__boundary_vertices

    def set_boundary_edges(self, edges):
        """
        Setter functions for self.__boundary_edges
        :Parameters:
         edges : list of int tuple pairs. The edges of the boundary loop in
         counter-clockwise order, where each edge is a tuple of the two 
         vertices it connects.
        """
        self.__boundary_edges = edges

    def get_boundary_edges(self):
        """Getter functions for self.__boundary_edges"""
        return self.__boundary_edges

    def set_central_vertex(self, central_vertex):
        """
        Setter functions for self.__central_vertex
        :Parameters:
         central_vertex : int. The index of the vertex at the topological
         centre of the animal's heat map in the x-y plane.
        """
        self.__central_vertex = central_vertex

    def get_central_vertex(self):
        """Getter functions for self.__central_vertex"""
        return self.__central_vertex

    def set_interior_vertex_bfs(self, vertex_bfs):
        """
        Setter functions for self.__vertex_bfs
        :Parameters:
         vertex_bfs : A tuple of int numpy arrays, (bfs_ordering, bfs_ancestors).

            bfs_ordering is an array containing the interior vertices in an animal's
            heat map in order of discovery in the breadth-first-search.

            bfs_ancestors is an array with length corresponding to the interior vertex
            with the largest index. Each element bfs_ancestors[i] is the index of the
            vertex that preceded vertex i in the breadth-first-search, where -1 
            indicates either the root vertex (where the breadth-first-search started) 
            or a vertex that was not discovered in the breadth-first-search (in this 
            case, it must be a boundary vertex).
        """
        self.__vertex_bfs = vertex_bfs

    def get_interior_vertex_bfs(self):
        """Getter functions for self.__vertex_bfs"""
        return self.__vertex_bfs

    def set_triangle_triangle_adjacency(self, triangle_triangle_adjacency):
        """
        Setter functions for self.__triangle_triangle_adjacency
        :Parameters:
         triangle_triangle_adjacency : num_triangles x 3 numpy array of ints.
         Each 3 X 1 element of triangle_triangle_adjacency[i] corresponds to the
         indices of the triangle in the triangulation of the heat map that is 
         adjacent to the three edges of the triangle with index i. -1 indicates
         that no triangles are adjacent to that particular edge of the the triangle.
        """
        self.__triangle_triangle_adjacency = triangle_triangle_adjacency

    def get_triangle_triangle_adjacency(self):
        """Getter functions for self.__triangle_triangle_adjacency"""
        return self.__triangle_triangle_adjacency

################################################################################
### Basic Functions
################################################################################

# Sure, I suppose I could use the actual error handling, but...
def _throw_error(errmsg):
    """Wrapper function for throwing an error message"""
    print("ERROR: %s" % errmsg)
    sys.exit(1)


def calculate_frame_num(animal, time_in_minutes):
    """
    Calculate the frame number given the time in minutes using the
    frame rate stored in the animal object
    :Parameters:
     time_in_minutes : float.
    """
    return int(animal.get_frame_rate() * time_in_minutes * 60)


def find_col_index(header, col_name):
    """
    Finds the column index of the given variable in the data
    :Parameters:
     header : list of headers in the dataset
     col_name : name of column to be indexed
    """
    # TO-DO: make this case insensitive
    pat = re.compile('^(")*%s(")*$' % col_name)
    for i, _ in enumerate(header):
        if re.match(pat, header[i]):
            return i
    # if we didn't find the index, the column name input is incorrect
    _throw_error("invalid column name: %s" % col_name)
    return None


def _remove_outliers(data):
    """
    Given a numpy array, removes outliers using 1.5 Interquartile Range standard
    :Parameters:
     data : numpy array
    :Returns:
     numpy array, without outliers
    """
    first_quart = np.percentile(data, 25)
    third_quart = np.percentile(data, 75)
    IQR = third_quart - first_quart
    idx = (data > first_quart - 1.5 * IQR) & (data < third_quart + 1.5 * IQR)
    return data[idx]


def norm(data, rm_outliers = True):
    """
    Given data, find the mean and standard deviation.
    :Parameters:
     data : list of data values
     rm_outliers : bool. If True, function removes outliers. True by default.
    :Returns:
     mean, stds
    """
    data_array = np.array(data, dtype=np.float)
    if rm_outliers:
        data_array = _remove_outliers(data_array) #Calculate norm without outliers
    mean = np.mean(data_array)
    std = np.std(data_array)
    return mean, std


def normalize(data, mean, std):
    """
    Normalize data given mean and standard deviation.
    :Parameters:
     data : list of data values
     mean : mean of data values
     std : standard deviation of data values
    :Returns:
     list of normalized data. list of 0 if std == 0
    """
    if std != 0:
        if std < 1:
            std = 1
            print("WARNING: Normalization attempted for data with std < 1. " +
                  "Normalization done with std set to 1.")
        return list(map(lambda x: 1/(1 + math.exp(-(x-mean)/std)), data))
    return [0 for d in data]

################################################################################
### Meat & Potatoes
################################################################################

def read_info(infile):
    """
    Load JSON file given path.
    :Parameters:
     infile : path to json file
    :Returns:
     info : loaded json file
    """
    with open(infile, 'r') as infofile:
        info = json.load(infofile)
    return info


def setup_raw_data(animal):
    """
    Store the raw data values from the data file location of the animal object
    into the animal object itself.
    :Parameters:
     animal : animal object
    """
    # pylint: disable=too-many-locals
    # Function is complicated, the local variables are necessary.
    with open(animal.get_data_file_location(), 'r') as infile:
        print("LOG: Extracting coordinates for Animal %s..." % animal.get_name())
        header = infile.readline()#.replace('\r','').replace('\n','')
        if '\t' in header:
            delim = '\t'
        elif ',' in header:
            delim = ','
        else:
            _throw_error("invalid data format")
        header = list(map(lambda x: x.strip(), header.split(delim)))
        try: # verify the file can be parsed
            reader = csv.reader(infile, delimiter=delim)
        except FileNotFoundError:
            _throw_error("invalid data format")

        x_ind = find_col_index(header, 'X')
        y_ind = find_col_index(header, 'Y')
        x_vals, y_vals = [], []
        start, end = animal.get_exp_times()
        start_frame = calculate_frame_num(animal, start)
        end_frame = calculate_frame_num(animal, end)

        for line, row in enumerate(reader):
            if line < start_frame:
                continue
            if line == end_frame:
                break
            x_val = row[x_ind]
            y_val = row[y_ind]
            if len(x_val) == 0 or x_val == ' ' or len(y_val) == 0 or y_val == ' ':
                print(row)
                _throw_error("possible truncated data")
            x_vals.append(float(x_val)/animal.get_pixel_density()) #scaling for pixel density
            y_vals.append(float(y_val)/animal.get_pixel_density())
            #DEFN: baseline norm is where we take the stats from the first two minutes of the
            #      exp to get the "baseline normal" numbers
            #DEFN: exp norm is where we take the stats from the whole exp duration and take all
            #      'local data' into consideration
    animal.add_raw_vals('X', x_vals)
    animal.add_raw_vals('Y', y_vals)

    baseline_start, baseline_end = animal.get_baseline_times()
    baseline_start_frame = calculate_frame_num(animal, baseline_start)
    baseline_end_frame = calculate_frame_num(animal, baseline_end)
    animal.add_stats('X', 'baseline', baseline_start_frame, baseline_end_frame)
    animal.add_stats('Y', 'baseline', baseline_start_frame, baseline_end_frame)


def setup_animal_objs(infofile, name_list=None):
    """
    Given a json file, generate and return the animal object files.
    If name_list is given, only generate the animal objects for animals in name_list
    :Parameters:
     infofile : json file. Should contain an entry for each animal.
     name_list : list of str. Names of animals to be generated.
    :Returns:
     list of animal objects
    """
    info = read_info(infofile)
    if name_list is not None:
        objs = [_init_animal(item) for item in info if item["name"] in name_list]
        return objs
    return [_init_animal(item) for item in info]


def _init_animal(json_item):
    """
    Given a json entry, extracts the relevant information and returns an initialized animal object
    :Parameters:
     json_item : json. Corresponds to animal.
    :Returns:
     animal : Animal object.
    """
    animal = Animal(json_item)
    setup_raw_data(animal)
    return animal
