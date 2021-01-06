"""
Copyright Mechanisms Underlying Behavior Lab, Singapore
https://mechunderlyingbehavior.wordpress.com/

animal.py is part of the locomotion python package for analyzing locomotory animal
behaviors via the techniques presented in the paper "Computational geometric tools
for quantitative comparison of locomotory behavior" by MT Stamps, S Go, and AS Mathuru
(https://doi.org/10.1038/s41598-019-52300-8).

This python module defines the Animal class object used throughout the package to store
and process the tracked data of an animal subject. On initialization, the animal object
extracts various pieces of information from JSON files, such as experimental parameters
and coordinate data of the subjects, and prepares them for analysis.
"""
import os
import csv
import re
import json
import warnings
from math import ceil
import numpy as np
from scipy.special import expit

####################
### Animal class ###
####################

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
        """ Initiates the Animal() object.

        Parameters
        ----------
        json_item : dict
            Deserialised JSON file, with the necessary animal information.
        """
        self.__animal_id = json_item["animal_attributes"]["ID"]
        self.__animal_type = json_item["animal_attributes"]["species"]
        self.__baseline_start = json_item["capture_attributes"]["baseline_start_time"] # In Minutes
        self.__baseline_end = json_item["capture_attributes"]["baseline_end_time"]     # In Minutes
        self.__data_file = os.path.abspath(json_item["data_file_location"])
        self.__exp_type = json_item["animal_attributes"]["exp_type"]
        self.__filename = os.path.basename(self.__data_file)
        self.__frame_rate = json_item["capture_attributes"]["frames_per_sec"] # Frames per Second
        self.__group = None
        self.__name = json_item["name"]
        self.__pix = json_item["capture_attributes"]["pixels_per_mm"]         # Pixels per MM
        self.__start = json_item["capture_attributes"]["start_time"] # In Minutes
        self.__end = json_item["capture_attributes"]["end_time"]         # In Minutes
        self.__info = json_item["additional_info"]
        self.__x_min = json_item["capture_attributes"]["x_min"] # Pixels
        self.__x_max = json_item["capture_attributes"]["x_max"] # Pixels
        self.__y_min = json_item["capture_attributes"]["y_min"] # Pixels
        self.__y_max = json_item["capture_attributes"]["y_max"] # Pixels
        self.__dim_x = self.__x_max - self.__x_min
        self.__dim_y = self.__y_max - self.__y_min
        self.__raw_vals = {}
        self.__means = {}
        self.__stds = {}
        self.__boundary_vertices = None
        self.__boundary_edges = None
        self.__central_vertex = None
        self.__colors = None
        self.__flat_coords = None
        self.__grid_size = None
        self.__num_verts = None
        self.__num_triangles = None
        self.__num_x_grid = None
        self.__num_y_grid = None
        self.__reg_coords = None
        self.__triangulation = None
        self.__triangle_triangle_adjacency = None
        self.__vertex_bfs = None

    ############################
    ### Population Functions ###
    ############################

    def add_info(self, info_key, info_value, replace=True):
        """ Updates dictionary self.__info with new data.

        Create a new entry in the dictionary, with key info_key and value info_value.

        Parameters
        ----------
        info_key : str
            Hashable key that will point to info_value in self.__info.
        info_value : any
            Any value to be stored into self.__info.
        replace : bool, optional
            If false, the function will not replace the value if there info_key is already
            pointing to a value. Default value : True.
        """
        if info_key in self.__info and not(replace):
            print("WARNING: %s is already in %s. Since replace is False, will not update."
                  % (info_key, self.get_name()))
        else:
            self.__info.update({info_key:info_value})

    def add_raw_vals(self, var_name, val_list):
        """ Updates dictionary self.__raw_vals with new data.

        Function creates a new entry in the dictionary, with key var_name and value val_list.

        Parameters
        ----------
        var_name : str
            Hashable key that will be used to point to variable in self.__raw_vals.
        val_list : list of floats
            List of data values corresponding to var_name to be stored in self.__raw_vals.
        """
        self.__raw_vals.update({var_name:val_list})

    def add_stats(self, var_name, scope, start_frame, end_frame):
        """ Calculates and updates self.__means and self.__stds for raw value var_name.

        Calculates statistics (means and standard deviation) of var_name over a specific scope,
        as defined by start_frame and end_frame. Uses the norm() method to do so. The function
        then initializes (if not already initialized) and updates the dictionary entries for
        var_name in self.__means and self.__stds.

        Parameters
        ----------
        var_name : str
            Hashable key pointing to variables stored in dict self.__raw_vals.
        scope : str
            Hashable key that will be used to point to scope defined by start_frame and end_frame.
        start_frame : int
            Starting frame of scope.
        end_frame : int
            Ending frame of scope.
        """
        if var_name not in self.__means:
            self.init_stats(var_name)
        means, stds = norm(self.__raw_vals[var_name][start_frame:end_frame])
        self.__means[var_name].update({scope:means})
        self.__stds[var_name].update({scope:stds})

    def init_stats(self, var_name):
        """ Set up empty dict entry for var_name in self.__means and self.__stds.

        Utility function for setting up an empty dictionary entry in self.__means and
        self.__stds for key var_name.

        Parameters
        ----------
        var_name : str
            Hashable key pointing to variable stored in dict self.__raw_vals.
        """
        self.__means.update({var_name:{}})
        self.__stds.update({var_name:{}})

    def get_animal_type(self):
        """Getter function for self.__name."""
        return self.__animal_type

    def get_baseline_times(self):
        """ Getter function for both self.__baseline_start and self.__baseline_end.

        Returns
        -------
        tuple
            First entry of tuple is the baseline start time (in minutes), and the second entry
            is the baseline end time (in minutes).
        """
        return (self.__baseline_start, self.__baseline_end)

    def get_baseline_start_time(self):
        """Getter function for self.__baseline_start."""
        return self.__baseline_start

    def get_baseline_end_time(self):
        """Getter function for self.__baseline_end."""
        return self.__baseline_end

    def get_group(self):
        """Getter function for self.__group."""
        return self.__group

    def get_data_file_location(self):
        """Getter function for self.__data_file."""
        return self.__data_file

    def get_data_file_name(self):
        """Getter function for self.__filename."""
        return self.__filename

    def get_dims(self):
        """Getter function for the x and y dimensions."""
        return self.__dim_x, self.__dim_y

    def get_exp_type(self):
        """Getter function for self.__exp_type."""
        return self.__exp_type

    def get_exp_times(self):
        """ Getter function for both self.__start and self.__end.

        Returns
        -------
        tuple
            First entry of tuple is the experiment start time (in minutes), and the second entry
            is the experiment end time (in minutes).
        """
        return (self.__start, self.__end)

    def get_exp_start_time(self):
        """Getter function for self.__start."""
        return self.__start

    def get_exp_end_time(self):
        """Getter function for self.__end."""
        return self.__end

    def get_frame_rate(self):
        """Getter function for self.__frame_rate."""
        return self.__frame_rate

    def get_id(self):
        """Getter function for self.__animal_id."""
        return self.__animal_id

    def get_name(self):
        """Getter function for self.__name."""
        return self.__name

    def get_pixel_density(self):
        """Getter function for self.__pix."""
        return self.__pix

    def set_group(self, group_no):
        """Setter function for self.__group."""
        self.__group = group_no

    ######################################
    ### Functions for modifying values ###
    ######################################

    def get_info(self, info_key):
        """ Retrieve information stored in Animal object.

        Parameters
        ----------
        info_key : str, hashable key
            Key pointing to information stored in self.__info.
        """
        try:
            value = self.__info[info_key]
        except KeyError:
            raise KeyError("get_info : %s not an entry in animal object %s."
                           % {info_key, self.__name})
        return value

    def get_mult_raw_vals(self, var_names, start_frame=None, end_frame=None):
        """ Retrieve multiple raw values stored in Animal object.

        Runs self.get_raw_vals for all the variables in var_names stored in Animal object.
        If no start_frame or end_frame is given, then the experiment start / end time is used.

        Parameters
        ----------
        var_names : list of strs
            List of hashable keys pointing to variables stored in self.__raw_vals.
        start_frame : int, optional
            Starting frame of portion to extract. Default value : None.
        end_frame : int, optional
            Ending frame of portion to extract. Default value : None.
        """
        return [self.get_raw_vals(v, start_frame, end_frame) for v in var_names]

    def get_raw_vals(self, var_name, start_frame=None, end_frame=None):
        """ Return the raw values with key var_name stored in Animal object.

        Retrieves the raw values stored in self.__raw_vals of the Animal object. If no start_frame
        or end_frame is given, then the experiment start / end time is used.

        Parameters
        ----------
        var_name : str
            Hashable key pointing to variables stored in self.__raw_vals.
        start_frame : int, optional
            Starting frame of portion to extract. Default value : None.
        end_frame : int, optional
            Ending frame of portion to extract. Default value : None.
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

    def get_stats(self, var_name, scope):
        """ Returns the calculated statistics for var_name over previously defined scope.

        Retrieve statistics stored in self.__means and self.__vars for var_name over the scope
        period. The statistics must have been previously calculated with the add_stats method.

        Parameters
        ----------
        var_name : str
            Hashable key pointing to variables stored in self.__raw_vals.
        scope : str
            Hashable key pointing to predefined scope in self.__means and self.__stds.

        Returns
        -------
        float
            Mean of var_name over scope period.
        float
            Standard deviation of var_name over scope period.
        """
        try:
            means = self.__means[var_name][scope]
            stds = self.__stds[var_name][scope]
        except KeyError as wrong_key:
            raise KeyError("get_stats : %s is not a valid variable name or scope."
                           % wrong_key)
        return means, stds

    ################################
    ### Functions for heatmap.py ###
    ################################

    def get_boundary_edges(self):
        """Getter functions for self.__boundary_edges"""
        return self.__boundary_edges

    def get_boundary_vertices(self):
        """Getter functions for self.__boundary_vertices"""
        return self.__boundary_vertices

    def get_central_vertex(self):
        """Getter functions for self.__central_vertex"""
        return self.__central_vertex

    def get_colors(self):
        """Getter functions for self.__colors"""
        return self.__colors

    def get_flattened_coordinates(self):
        """Getter functions for self.__flat_coords"""
        return self.__flat_coords

    def get_grid_size(self):
        """Getter functions for self.__num_x_grid and self.__num_y_grid"""
        return self.__grid_size

    def get_interior_vertex_bfs(self):
        """Getter functions for self.__vertex_bfs"""
        return self.__vertex_bfs

    def get_num_grids(self):
        """Getter functions for self.__num_x_grid and self.__num_y_grid"""
        return self.__num_x_grid, self.__num_y_grid

    def get_num_triangles(self):
        """Getter functions for self.__num_triangles"""
        return self.__num_triangles

    def get_num_verts(self):
        """Getter functions for self.__num_verts"""
        return self.__num_verts

    def get_regular_coordinates(self):
        """Getter functions for self.__reg_coords"""
        return self.__reg_coords

    def get_triangulation(self):
        """Getter functions for self.__triangulation"""
        return self.__triangulation

    def get_triangle_triangle_adjacency(self):
        """Getter functions for self.__triangle_triangle_adjacency"""
        return self.__triangle_triangle_adjacency

    def set_boundary_edges(self, edges):
        """ Setter functions for self.__boundary_edges.

        Parameters
        ----------
        edges : list of int tuple pairs
            The edges of the boundary loop in counter-clockwise order, where each edge
            is a tuple of the two vertices it connects.
        """
        self.__boundary_edges = edges

    def set_boundary_vertices(self, vertices):
        """ Setter functions for self.__boundary_vertices.

        Parameters
        ----------
        vertices : numpy array of ints
            The indices of the vertices that are on the boundary of this animal in
            counter-clockwise order.
        """
        self.__boundary_vertices = vertices

    def set_central_vertex(self, central_vertex):
        """ Setter functions for self.__central_vertex.

        Parameters
        ----------
        central_vertex : int.
            The index of the vertex at the topological centre of the animal's heatmap
            in the x-y plane.
        """
        self.__central_vertex = central_vertex

    def set_colors(self, colors):
        """ Setter functions for self.__colors.

        Parameters
        ----------
        colors : list of triples of floats.
            The RGB coordinates for each triangle in the triangulation associated to
            an animal's heat map.
        """
        self.__colors = colors

    def set_flattened_coordinates(self, coordinates):
        """ Setter functions for self.__flat_coords.

        Parameters
        ----------
        coordinates : list of pairs of floats.
            The x- and y-coordinates of the vertices of a triangulation that have been
            conformally flattened to the unit disk.
        """
        self.__flat_coords = coordinates

    def set_grid_size(self, grid_size):
        """ Setter function for self.__grid_size, self.__num_x_grid, self.__num_y_grid.

        Setter function for self.__grid_size, self.__num_x_grid, and self.__num_y_grid.
        Number of x and y grids is calculated by dividing self.__dim_x and self.__dim_y
        by grid_size.

        Parameters
        ----------
        grid_size : int.
            Size of each grid. Must divide self.__dim_x and self.__dim_y.
        """
        if self.__dim_x % grid_size != 0 or self.__dim_y % grid_size != 0:
            raise Exception("grid_size does not divide dim x or dim y.")
        self.__grid_size = grid_size
        self.__num_x_grid = int(ceil(self.__dim_x/grid_size))
        self.__num_y_grid = int(ceil(self.__dim_y/grid_size))

    def set_interior_vertex_bfs(self, vertex_bfs):
        """ Setter function for self.__vertex_bfs.

        Parameters
        ----------
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

    def set_num_triangles(self, num_triangles):
        """ Setter functions for self.__num_triangles.

        Parameters
        ----------
        num_triangles : int.
            The number of triangles in the triangulation associated with an animal's
            heat map.
        """
        self.__num_triangles = num_triangles

    def set_num_verts(self, num_verts):
        """ Setter functions for self.__num_verts.

        Parameters
        ----------
        num_verts : int.
            The number of vertices in an animal's heat map.
        """
        self.__num_verts = num_verts

    def set_regular_coordinates(self, coordinates):
        """ Setter functions for self.__reg_coords.

        Parameters
        ----------
        coordinates : list of triples of floats.
            The the x-, y-, and z-coordinates of the vertices for a triangulation of
            the animal's heat map.
        """
        self.__reg_coords = coordinates

    def set_triangulation(self, triangles):
        """ Setter functions for self.__triangulation.

        Parameters
        ----------
        triangles : list of triples of ints.
            The indices of the vertices for each triangle in the triangulation of a
            surface.
        """
        self.__triangulation = triangles

    def set_triangle_triangle_adjacency(self, triangle_triangle_adjacency):
        """ Setter functions for self.__triangle_triangle_adjacency.

        Parameters
        ----------
        triangle_triangle_adjacency : num_triangles x 3 numpy array of ints.
            Each 3 X 1 element of triangle_triangle_adjacency[i] corresponds to the
            indices of the triangle in the triangulation of the heat map that is
            adjacent to the three edges of the triangle with index i. -1 indicates
            that no triangles are adjacent to that particular edge of the the triangle.
        """
        self.__triangle_triangle_adjacency = triangle_triangle_adjacency

#######################
### Basic Functions ###
#######################

def norm(data, rm_outliers=True):
    """ Calculates the mean and standard deviation of data.

    Given data, find the mean and standard deviation. If rm_outliers is True, the
    method will remove outliers using _remove_outliers() before the calculation.

    Parameters
    ----------
    data : list of floats
        List of data values for on which to calculate the mean and standard deviation.
    rm_outliers : bool.
        If True, function removes outliers. Default value : True.

    Returns
    -------
    mean : float
        Mean of data, calculated using numpy.
    std : float
        Standard deviation of data, calculated using numpy.
    """
    data_array = np.array(data, dtype=np.float)
    if rm_outliers:
        data_array = _remove_outliers(data_array)
    mean = np.mean(data_array)
    std = np.std(data_array)
    return mean, std


def normalize(data, mean, std):
    """ Normalize data given mean and standard deviation.

    Shifts and scales data so the range is between 0 and 1.

    Parameters
    ----------
    data : list of floats
        List of data values to normalize.
    mean : float
        Mean of data values.
    std : float
        Standard deviation of data values

    Returns
    -------
    np.array of floats
        Array of normalized data. If std == 0, then returns an array of 0.
    """
    data = np.array(data)
    if std != 0:
        if std < 1:
            std = 1
            # print("WARNING: Normalization attempted for data with std < 1. " +
            #       "Normalization done with std set to 1.")
        norm_data = (data - mean)/std
        return expit(norm_data)
    return np.zeros_like(data)

################################
### Initialization Functions ###
################################

def read_info(infile_path):
    """ Load JSON file given path.

    Utility function to read the JSON file from infile_path.

    Parameters
    ----------
    infile_path : str
        Absolute path to json file.

    Returns
    -------
    info : dict
        Deserialised json file through the json.load() function.
    """
    with open(infile_path, 'r') as infofile:
        info = json.load(infofile)
    return info


def setup_animal_objs(infofiles, name_list=None):
    """ Generates and initializes Animal objects from a list of JSONs.

    Given a list of JSON files, generate and return the Animal object. If name_list is
    given, only generate the Animal object whose names are in the name list.

    Parameters
    ----------
    infofile : list of dicts
        Each group of animals should have a corresponding deserialized .json file, stored
        as a dict, which should contain an entry for each animal in the group.
    name_list : list of strs, optional
        Names of animals to be generated.

    Returns
    -------
    objs : list of Animal() objects
        Regardless of the number of groups of animals, will only return one compiled list.
        Groupings are reflected in the Animal() object itself.
    """
    # check if infofiles is a list:
    if not isinstance(infofiles, list):
        raise Exception("setup_animal_objs: infofiles variable needs to be a list.")
    objs = []
    for group_no, infofile in enumerate(infofiles):
        info = read_info(infofile)
        if name_list is not None:
            objs += [_init_animal(item, group_no) for item in info
                     if item["name"] in name_list]
        else:
            objs += [_init_animal(item, group_no) for item in info]
    return objs


def setup_raw_data(animal):
    """ Extracts the raw data from the data file linked in the Animal() object.

    Store the raw data values from the data file location of the animal object
    into the animal object itself. Currently only sets up X and Y coordinates.

    Parameters
    ----------
    animal : Animal() object
        The Animal() object that the raw data is to be set up for.
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
            raise Exception("setup_raw_data : Incorrect type of Data file in animal object.")
        header = list(map(lambda x: x.strip(), header.split(delim)))
        try: # verify the file can be parsed
            reader = csv.reader(infile, delimiter=delim)
        except FileNotFoundError:
            raise Exception("setup_raw_data : Data file not found in animal object.")

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
                raise Exception("Data file in animal object might be truncated.")
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


def _init_animal(json_item, group_no):
    """ Initializes the Animal() object.

    Given a json entry, extracts the relevant information and returns an initialized
    Animal() object.

    Parameters
    ----------
    json_item : dict
        Deserialized JSON item stored as a python dictionary. Corresponds to the Animal()
        object.
    group_no : int
        Group number that the Animal() object is a part of.

    Returns
    -------
    animal : Animal() object
        Initialized Animal() object.
    """
    animal = Animal(json_item)
    setup_raw_data(animal)
    animal.set_group(group_no)
    return animal


#######################
### Other Functions ###
#######################

def calculate_frame_num(animal, time_in_minutes):
    """ Convert time_in_minutes to frame number with stored framerate.

    Calculate the frame number given the time in minutes using the frame rate
    stored in the animal object. Framerate is in frames per second, and the time
    in minutes is converted to time in seconds for the conversion.

    Parameters
    ----------
    time_in_minutes : float.
        Time to be converted in minutes.
    """
    return int(animal.get_frame_rate() * time_in_minutes * 60)


def find_col_index(header, col_name):
    """ Extracts the column index of the given variable in the data.

    Given a list of headers, searches the list for one that first one that contains
    the col_name. Uses regex match to search through each header.

    Parameters
    ----------
    header : list of strs
        List of headers in the dataset.
    col_name : str
        Name of column to be indexed.

    Returns
    -------
    int
        If a match is found, returns the index. If no match, it raises an Exception.
    """
    pat = re.compile('^(")*%s(")*$' % col_name.lower())
    for i, _ in enumerate(header):
        if re.match(pat, header[i].lower()):
            return i
    raise Exception("Column name not found: %s" % col_name)


def _remove_outliers(data):
    """ Remove outliers from data.

    Given a numpy array, removes outliers using 1.5 Interquartile Range standard.

    Parameters
    ----------
    data : numpy array
        Data to be modified.

    Returns
    -------
    numpy array
        Modified data with outliers removed.
    """
    first_quart = np.percentile(data, 25)
    third_quart = np.percentile(data, 75)
    iqr = third_quart - first_quart
    idx = (data > first_quart - 1.5 * iqr) & (data < third_quart + 1.5 * iqr)
    return data[idx]
