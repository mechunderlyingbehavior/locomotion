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
# pylint:disable=too-many-lines

import os
import csv
import re
import json
import warnings
from math import ceil
import numpy as np
from scipy.special import expit
from scipy.signal import savgol_filter
from numpy.polynomial import polynomial as P

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
        self.__baseline_start = json_item["capture_attributes"]["baseline_start_time"] # In Seconds
        self.__baseline_end = json_item["capture_attributes"]["baseline_end_time"]     # In Seconds
        self.__data_file = os.path.abspath(json_item["data_file_location"])
        self.__exp_type = json_item["animal_attributes"]["exp_type"]
        self.__filename = os.path.basename(self.__data_file)
        self.__frame_rate = json_item["capture_attributes"]["frames_per_sec"] # Frames per Second
        self.__group = None
        self.__name = json_item["name"]
        self.__input_unit = json_item["capture_attributes"]["input_unit"]
        self.__output_unit = json_item["capture_attributes"]["output_unit"]
        self.__unit_conversion = json_item["capture_attributes"]["input_per_output_unit"]
        self.__start = json_item["capture_attributes"]["start_time"] # In Seconds
        self.__end = json_item["capture_attributes"]["end_time"]         # In Seconds
        self.__info = json_item["additional_info"]
        self.__x_lims = json_item["capture_attributes"]["x_lims"] # Tuple of Pixels
        self.__y_lims = json_item["capture_attributes"]["y_lims"] # Tuple of Pixels
        self.__dim_x = self.__x_lims[1] - self.__x_lims[0]
        self.__dim_y = self.__y_lims[1] - self.__y_lims[0]
        self.__raw_vals = {}
        self.__norm_info = {}
        self.__boundary_vertices = None
        self.__boundary_edges = None
        self.__central_vertex = None
        self.__colors = None
        self.__flat_coords = None
        self.__frequencies = None
        self.__num_verts = None
        self.__num_triangles = None
        self.__reg_coords = None
        self.__triangulation = None
        self.__triangle_triangle_adjacency = None
        self.__vertex_bfs = None
        self.__x_grid_count = None
        self.__y_grid_count = None
        self.__x_grid_len = None
        self.__y_grid_len = None

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
        if info_key in self.__info and not replace:
            print("WARNING: %s is already in %s. Since replace is False, will not update."
                  % (info_key, self.get_name()))
        else:
            self.__info.update({info_key:info_value})

    def add_norm_bounded(self, var_name, scope, lower, upper):
        """ Updates dictionary self.__norm_info with lower and upper bounds for var_name.

        Updates var_name entry in dictionary self.__norm_info with lower and upper bounds
        for normalization method defined by scope. Use when scope is using bounded
        normalization.

        Parameters
        ----------
        var_name : str
            Hashable key pointing to variables stored in dict self.__raw_vals.
        scope : str
            Hashable key used to define new scope in self.__norm_info.
        lower : float
            Manually defined lower bound for var_name over scope.
        upper : float
            Manually defined upper bound for var_name over scope.
        """
        if var_name not in self.__norm_info:
            self.init_norm_dict(var_name)
        self.__norm_info[var_name].update({scope:{"lower" : lower,
                                                  "upper" : upper}})

    def add_norm_standard(self, var_name, scope, mean, std):
        """ Updates dictionary self.__norm_info with mean and std for var_name.

        Updates var_name entry in dictionary self.__norm_info with new mean and std for
        normalization method defined by scope. Use when scope is using Standardization for
        normalization.

        Parameters
        ----------
        var_name : str
            Hashable key pointing to variables stored in dict self.__raw_vals.
        scope : str
            Hashable key used to define new scope in self.__norm_info.
        mean : float
            Manually defined mean for var_name over scope.
        std : float
            Manually defined standard deviation for var_name over scope.
        """
        if var_name not in self.__norm_info:
            self.init_norm_dict(var_name)
        self.__norm_info[var_name].update({scope:{"mean": mean,
                                                  "std" : std}})

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

    def check_existing_scope(self, var_name, scope):
        """Checks if scope is already defined for var_name.

        Parameters
        ----------
        var_name : str
            Hashable key pointing to variable stored in self.__raw_vals.
        scope : str
            Hashable key to check if exists in self.__norm_info[var_name].

        Returns
        -------
        bool
            Function returns True if scope is already defined for var_name.
        """
        # check if keys match up
        try:
            norm_info = self.__norm_info[var_name]
        except KeyError as wrong_key:
            raise ValueError("check_existing_scope() : %s is not a valid variable name."
                             % wrong_key) from None
        return scope in norm_info

    def check_if_standard(self, var_name, scope):
        """ Checks if normalization method scope defined for var_name is standard.

        The function returns true if the normalization method scope defined for var_name
        in self.__norm_info is for standard normalization (i.e. defined by a mean and std)
        or if it is for bounded normalization (i.e. defined by upper and lower bound).

        Parameters
        ----------
        var_name : str
            Hashable key pointing to variable stored in self.__raw_vals.
        scope : str
            Hashable key pointing to scope stored in self.__norm_info[var_name].

        Returns
        -------
        bool
            Function returns True if scope is defined as a standard normalization method,
            and False if scope is defined as a bounded normalization method.
        """
        try:
            norm_info = self.__norm_info[var_name]
        except KeyError as wrong_key:
            raise ValueError("check_existing_scope() : %s is not a valid variable name."
                             % wrong_key) from None
        try:
            scope_dict = norm_info[scope]
        except KeyError as wrong_scope:
            raise ValueError("check_existing_scope() : %s is not a valid scope for %s."
                             % (wrong_scope, var_name)) from None
        return ("mean" in scope_dict) and ("std" in scope_dict)

    def init_norm_dict(self, var_name):
        """ Set up empty dict entry for var_name in self.__norm_info.

        Utility function for setting up an empty dictionary entry in self.__norm_info for
        key var_name.

        Parameters
        ----------
        var_name : str
            Hashable key pointing to variable stored in dict self.__raw_vals.
        """
        self.__norm_info.update({var_name:{}})

    def get_animal_type(self):
        """Getter function for self.__name."""
        return self.__animal_type

    def get_baseline_times(self):
        """ Getter function for both self.__baseline_start and self.__baseline_end.

        Returns
        -------
        tuple
            First entry of tuple is the baseline start time (in seconds), and the second entry
            is the baseline end time (in seconds).
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
            First entry of tuple is the experiment start time (in seconds), and the second entry
            is the experiment end time (in seconds).
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

    def get_input_unit(self):
        """Getter function for self.__input_unit"""
        return self.__input_unit

    def get_lims(self):
        """Getter function for self.__x_lims and self.__y_lims."""
        return self.__x_lims, self.__y_lims

    def get_name(self):
        """Getter function for self.__name."""
        return self.__name

    def get_output_unit(self):
        """Getter function for self.__output_unit"""
        return self.__output_unit

    def get_unit_conversion(self):
        """Getter function for self.__unit_conversion."""
        return self.__unit_conversion

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
            raise ValueError("get_info : %s not found in animal object %s."
                             % (info_key, self.__name)) from None
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

    def get_norm_bounds(self, var_name, scope):
        """ Returns the stored lower and upper bounds for var_name over scope period.

        Retrieve lower and upper bounds previously defined and stored in self.__norm_info
        for var_name over the scope period.

        Parameters
        ----------
        var_name : str
            Hashable key pointing to variable stored in self.__raw_vals.
        scope : str
            Hashable key pointing to predefined scope in self.__norm_info.

        Returns
        -------
        float
            Lower bound of var_name over scope period.
        float
            Upper bound of var_name over scope period.
        """
        try:
            lower = self.__norm_info[var_name][scope]["lower"]
            upper = self.__norm_info[var_name][scope]["upper"]
        except KeyError as wrong_key:
            raise ValueError("get_norm_bounds : %s is not a valid variable name or scope."
                             % wrong_key) from None
        return lower, upper

    def get_norm_stats(self, var_name, scope):
        """ Returns the calculated statistics for var_name over previously defined scope.

        Retrieve statistics stored in self_norm_info for var_name over the scope period.
        The statistics must have been previously calculated with the calculate_norm_stats
        method.

        Parameters
        ----------
        var_name : str
            Hashable key pointing to variable stored in self.__raw_vals.
        scope : str
            Hashable key pointing to predefined scope in self.__norm_info.

        Returns
        -------
        float
            Mean of var_name over scope period.
        float
            Standard deviation of var_name over scope period.
        """
        try:
            mean = self.__norm_info[var_name][scope]["mean"]
            std = self.__norm_info[var_name][scope]["std"]
        except KeyError as wrong_key:
            raise ValueError("get_norm_stats : %s is not a valid variable name or scope."
                             % wrong_key) from None
        return mean, std

    def get_stat_keys(self, var_name):
        """Returns the existing keys in self.__norm_info for given var_name.

        Parameters
        ----------
        var_name : str
            Hashable key pointing to variable stored in self.__raw_vals.

        Returns
        -------
        dict_keys
            The keys of the self.__norm_info dictionary.
        """
        # check if keys match up
        try:
            norm_info = self.__norm_info[var_name]
        except KeyError as wrong_key:
            raise ValueError("get_stat_keys : %s is not a valid variable name."
                             % wrong_key) from None
        return norm_info.keys()

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
            start_frame = self.__start*self.__frame_rate
        if end_frame is None:
            end_frame = self.__end*self.__frame_rate
        # logic check
        try:
            values = self.__raw_vals[var_name]
        except KeyError:
            raise ValueError("get_raw_vals: %s not found in animal object."
                             % (var_name)) from None
        if start_frame > end_frame:
            raise ValueError("get_raw_vals: Start frame is after End frame.")
        if start_frame > len(values):
            raise ValueError("get_raw_vals: Start frame comes after existing frames.")
        if end_frame > len(values):
            warnings.warn("get_raw_vals: End frame comes after existing frames. "
                          "Defaulting to the final frame stored.")
        return values[start_frame:end_frame]

    def get_stats(self, var_name, scope):
        """ DEPRECIATED: Returns the calculated statistics for var_name over previously
        defined scope.

        Retrieve statistics stored in self_norm_info for var_name over the scope period.
        The statistics must have been previously calculated with the calculate_norm_stats
        method.

        Parameters
        ----------
        var_name : str
            Hashable key pointing to variable stored in self.__raw_vals.
        scope : str
            Hashable key pointing to predefined scope in self.__norm_info.

        Returns
        -------
        float
            Mean of var_name over scope period.
        float
            Standard deviation of var_name over scope period.
        """
        print("WARNING: get_stats() is depreciated. Use get_norm_stats() instead.")
        try:
            means = self.__norm_info[var_name][scope]["mean"]
            stds = self.__norm_info[var_name][scope]["std"]
        except KeyError as wrong_key:
            raise ValueError("get_stats : %s is not a valid variable name or scope."
                             % wrong_key) from None
        return means, stds

    def populate_stats(self, var_name, scope, start_frame, end_frame):
        """ Calculates and updates self.__norm_info for raw value var_name.

        Calculates statistics (means and standard deviation) of var_name over a specific scope,
        as defined by start_frame and end_frame. Uses the norm() method to do so. The function
        then initializes (if not already initialized) and updates the dictionary entries for
        var_name in self.__norm_info.

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
        if var_name not in self.__norm_info:
            self.init_norm_dict(var_name)
        mean, std = calculate_norm_stats(self.__raw_vals[var_name][start_frame:end_frame])
        self.__norm_info[var_name].update({scope:{"mean": mean,
                                                  "std" : std}})

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

    def get_frequencies(self):
        """Getter function for self.__frequencies"""
        return self.__frequencies

    def get_grid_counts(self):
        """Getter functions for self.__x_grid_count and self.__y_grid_count"""
        return self.__x_grid_count, self.__y_grid_count

    def get_grid_lens(self):
        """Getter functions for self.__x_grid_len and self.__y_grid_len"""
        return self.__x_grid_len, self.__y_grid_len

    def get_interior_vertex_bfs(self):
        """Getter functions for self.__vertex_bfs"""
        return self.__vertex_bfs

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

    def set_frequencies(self, frequencies):
        """ Setter function for self.__frequencies.

        Parameters
        ----------
        frequencies : list of lists of floats.
            self.__x_grid_count by self.__y_grid_count histrogram of animal trajectory.
        """
        self.__frequencies = frequencies

    def set_grid_counts(self, x_grid_count, y_grid_count):
        """ Setter function for self.__x_grid_count and self.__y_grid_count.

        Updates self.__x_grid_len and self.__y_grid_len as necessary to ensure grid lengths 
        and counts match the overall dimensions (self.__dim_x and self.__dim_y).

        Parameters
        ----------
        x_grid_count, y_grid_count : int.
            Numbers of columns and rows in grid.
        """
        self.__x_grid_count = x_grid_count
        self.__y_grid_count = y_grid_count
        if self.__x_grid_len == None or self.__x_grid_len * self.__x_grid_count != self.__dim_x:
            self.__x_grid_len = self.__dim_x / self.__x_grid_count
            print("LOG: %s horizontal grid length updated to match specified number of columns." % self.__name)
        if self.__y_grid_len == None or self.__y_grid_len * self.__y_grid_count != self.__dim_y:
            self.__y_grid_len = self.__dim_y / self.__y_grid_count
            print("LOG: %s vertical grid length updated to match specified number of rows." % self.__name)

    def set_grid_lens(self, x_grid_len, y_grid_len):
        """ Setter function for self.__x_grid_len and self.__y_grid_len.

        Updates self.__x_grid_count and self.__y_grid_count as necessary to ensure grid lengths 
        and counts match the overall dimensions (self.__dim_x and self.__dim_y).

        Parameters
        ----------
        x_grid_len, y_grid_len : float.
            Horizontal and vertical length of grid.
        """
        self.__x_grid_len = x_grid_len
        self.__y_grid_len = y_grid_len
        if self.__x_grid_count == None or self.__x_grid_len * self.__x_grid_count != self.__dim_x:
            self.__x_grid_count = self.__dim_x // self.__x_grid_len
            print("LOG: %s horizontal grid count updated to match specified length." % self.__name)
        if self.__y_grid_count == None or self.__y_grid_len * self.__y_grid_count != self.__dim_y:
            self.__y_grid_count = self.__dim_y // self.__y_grid_len
            print("LOG: %s vertical grid count updated to match specified length." % self.__name)

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

def calculate_norm_stats(data, rm_outliers=True):
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


def normalize_bounded(data, lower, upper):
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
    print(data[:10])
    transformed = (data - lower)/(upper - lower)
    too_low_ind, too_high_ind = transformed < 0, transformed > 1
    if any(too_low_ind):
        print("WARNING: Values below bounds during normalization. Mapping to 0.")
        transformed[too_low_ind] = 0.0
    if any(too_high_ind):
        print("WARNING: Values above bounds during normalization. Mapping to 1.")
        transformed[too_high_ind] = 1.0
    print(transformed[:10])
    return transformed


def normalize_standard(data, mean, std):
    """ Standard Normalization on data given mean and standard deviation.

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


def setup_animal_objs(infofiles, name_list=None, smooth_order=3,
                      smooth_window=20,smooth_method="savgol"):
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
    smooth_order : int
        Order of the polynomial used in the smoothening function. Default value : 3.
    smooth_window : int
        Half-window of frames used for smoothening. Default value : 20.
    smooth_method : str
        Method for smoothing. Should be either "savgol" or "lowess".
        Default value: "savgol"
    Returns
    -------
    objs : list of Animal() objects
        Regardless of the number of groups of animals, will only return one compiled list.
        Groupings are reflected in the Animal() object itself.
    """
    # check if infofiles is a list:
    if not isinstance(infofiles, list):
        raise TypeError("setup_animal_objs: infofiles variable needs to be a list.")
    if not isinstance(smooth_order, int):
        raise TypeError("setup_animal_objs : smooth_order must be an int.")
    if not isinstance(smooth_window, int):
        raise TypeError("setup_animal_objs : smooth_window must be an int.")
    objs = []
    for group_no, infofile in enumerate(infofiles):
        info = read_info(infofile)
        if name_list is not None:
            objs += [_init_animal(item, group_no, smooth_order=smooth_order,
                                  smooth_window=smooth_window,
                                  smooth_method=smooth_method) for item in info if
                     item["name"] in name_list]
        else:
            objs += [_init_animal(item, group_no, smooth_order=smooth_order,
                                  smooth_window=smooth_window,
                                  smooth_method=smooth_method) for item in info]
    return objs


def setup_raw_data(animal, smooth_order, smooth_window, smooth_method):
    """ Extracts the raw data from the data file linked in the Animal() object.

    Store the raw data values from the data file location of the animal object
    into the animal object itself. Currently only sets up X and Y coordinates.

    Parameters
    ----------
    animal : Animal() object
        The Animal() object that the raw data is to be set up for.
    smooth_order : int
        Order of the polynomial used in the smoothening function.
    smooth_window : int
        Half-window of frames used for smoothening.
    smooth_method : str
        Method for smoothing. Should be either "savgol" or "lowess".
    """
    # pylint: disable=too-many-locals
    # Function is complicated, the local variables are necessary.
    # Argument Checks & Validations
    with open(animal.get_data_file_location(), 'r') as infile:
        print("LOG: Extracting coordinates for Animal %s, converting from %s to %s."
              % (animal.get_name(),
                 animal.get_input_unit(),
                 animal.get_output_unit()))
        header = infile.readline()#.replace('\r','').replace('\n','')
        if '\t' in header:
            delim = '\t'
        elif ',' in header:
            delim = ','
        else:
            raise TypeError("setup_raw_data : Incorrect type of Data file in animal object.")
        header = list(map(lambda x: x.strip(), header.split(delim)))
        try: # verify the file can be parsed
            reader = csv.reader(infile, delimiter=delim)
        except FileNotFoundError:
            raise ValueError("setup_raw_data : Data file not found in animal object.") \
                from None

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
            x_vals.append(float(x_val)/animal.get_unit_conversion()) #scaling for pixel density
            y_vals.append(float(y_val)/animal.get_unit_conversion())

    smooth_x = _smooth(np.array(x_vals), smooth_order, smooth_window, smooth_method)
    smooth_y = _smooth(np.array(y_vals), smooth_order, smooth_window, smooth_method)

    mse = (np.linalg.norm(np.array([smooth_x, smooth_y]) -
                          np.array([x_vals, y_vals]))**2).mean()
    print(f"MSE of Smoothed Data for {animal.get_name()}: {mse}")

    animal.add_raw_vals('X', smooth_x)
    animal.add_raw_vals('Y', smooth_y)

    baseline_start, baseline_end = animal.get_baseline_times()
    baseline_start_frame = calculate_frame_num(animal, baseline_start)
    baseline_end_frame = calculate_frame_num(animal, baseline_end)
    animal.populate_stats('X', 'baseline', baseline_start_frame, baseline_end_frame)
    animal.populate_stats('Y', 'baseline', baseline_start_frame, baseline_end_frame)


def _init_animal(json_item, group_no, smooth_order, smooth_window, smooth_method):
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
    smooth_order : int
        Order of the polynomial used in the smoothening function.
    smooth_window : int
        Window of frames used for smoothening. Must be odd.
    smooth_method : str
        Method for smoothing. Should be either "savgol" or "lowess".

    Returns
    -------
    animal : Animal() object
        Initialized Animal() object.
    """
    animal = Animal(json_item)
    setup_raw_data(animal,
                   smooth_order=smooth_order,
                   smooth_window=smooth_window,
                   smooth_method=smooth_method)
    animal.set_group(group_no)
    return animal


#######################
### Other Functions ###
#######################

def calculate_frame_num(animal, time_in_seconds):
    """ Convert time_in_minutes to frame number with stored framerate.

    Calculate the frame number given the time in seconds using the frame rate
    stored in the animal object. Framerate is in frames per second, and the time
    is in seconds.

    Parameters
    ----------
    time_in_seconds : float.
        Time to be converted.
    """
    return int(animal.get_frame_rate() * time_in_seconds)


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
    raise ValueError("Column name not found: %s" % col_name)


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


def _smooth(sequence, degree, half_window, smooth_method = "savgol"):
    """ Smooths sequence by applying Savitzky-Golay smoothing.

    Parameters
    ----------
    sequence : list of floats
        Sequence to be smoothed.
    degree : int
        Degree of polynomials to be used for fitting
    half_window : int
        Used to determine window length. Window = (2 * half_window) + 1
    smooth_method : str
        Method for smoothing. Should be either "savgol" or "lowess".
        Default value: "savgol".

    Returns
    -------
    smoothed : list of floats
        Smoothed sequence.
    """
    # Check smooth_method
    if smooth_method not in ["savgol", "lowess"]:
        raise ValueError(f"{smooth_method} not a valid smoothening method.")

    n = len(sequence)
    window = (half_window * 2) + 1

    if smooth_method == "savgol":
        smoothed = savgol_filter(sequence, window, degree)

    if smooth_method == "lowess":
        fitted = np.array([0.0 for _ in sequence])
        xarr = np.arange(-half_window, half_window+1)

        # Step 1: Weighted Polyfit
        weight = (1 - (np.abs(xarr)/half_window) ** 3) ** 3
        for i in range(half_window, n-half_window):
            z = P.polyfit(xarr, sequence[i-half_window:i+half_window+1],
                          deg=degree, w=weight)
            fitted[i] = P.polyval(0.0, z)

        z = P.polyfit(xarr, sequence[:window], deg=degree, w=weight)
        for i in range(half_window):
            fitted[i] = P.polyval(xarr[i], z)

        z = P.polyfit(xarr, sequence[-window:], deg=degree, w=weight)
        for i in range(half_window):
            fitted[n-half_window+i] = P.polyval(xarr[i+half_window+1], z)

        smoothed = fitted
        # # Step 2: Running Median
        res = sequence - fitted
        def new_weights(i):
            res_interval = res[i-half_window:i+half_window+1]
            med = np.median(np.abs(res_interval))
            deltas = (1 - (res_interval/(6 * med)) ** 2) ** 2
            deltas[np.abs(res_interval) > 6 * med] = 0
            return deltas * weight

        smoothed = np.array([0.0 for _ in sequence])
        for i in range(half_window, n-half_window):
            z = P.polyfit(xarr, sequence[i-half_window:i+half_window+1],
                          deg=degree, w=new_weights(i))
            smoothed[i] = P.polyval(0.0, z)

        z = P.polyfit(xarr, sequence[:window], deg=degree, w=new_weights(half_window))
        for i in range(half_window):
            smoothed[i] = P.polyval(xarr[i], z)

        z = P.polyfit(xarr, sequence[-window:], deg=degree, w=new_weights(n-half_window-1))
        for i in range(half_window):
            smoothed[n-half_window+i] = P.polyval(xarr[i+half_window+1], z)

    return smoothed
