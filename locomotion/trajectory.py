"""Copyright Mechanisms Underlying Behavior Lab, Singapore
https://mechunderlyingbehavior.wordpress.com/

trajectory.py is part of the locomotion package comparing
animal behaviours, developed to support the work discussed
in paper Computational Geometric Tools for Modeling Inherent
Variability in Animal Behavior (MT Stamps, S Go, and AS Mathuru)

This python script contains methods for running Dynamic Time Warping on pairs of
animal trajectories. The DTW implementation used in this package is the one
provided in the dtw-python package (T. Giorgino. Computing and Visualizing
Dynamic Time Warping Alignments in R: The dtw Package. J. Stat. Soft.,
doi:10.18637/jss.v031.i07.).
"""

import os
import random
import numpy as np
import dtw
from scipy.signal import savgol_filter
import locomotion.write as write
import locomotion.animal as animal

#Static Variables
SMOOTH_RANGE_MIN = 5 #minimum length of smoothing window in _smooth()
WINDOW_SCALAR = 2.5 #scalar for smoothing window calculation in _smooth()
ORDER = 5 #order of smoothing curve used in _smooth()

######################
### Main Functions ###
######################

def populate_curve_data(animal_obj, col_names=None):
    """ Computes the behavioural curve data such as Velocity and Curvature.

    This function computes and stores the Velocity and Curvature given the coordinate
    data stored in animal_obj. The data used to calculate this is given by col_names,
    the list of column names.

    Currently only works in 2 or 3 dimensions.

    Parameters
    ----------
    animal_obj : Animal() object
        Initialized Animal() object, which should already contain coordinate data.
    col_names : list of strs, optional
        Names of data columns used for calculations. Must coincide with data stored in
        animal_obj.__raw_vals. If not given, defaults to ['X', 'Y']

    Returns
    -------
    first_deriv : list of numpy arrays
        Each numpy array corresponds to the first derivative of the respective coordinate
        data as ordered by col_names.
    second_deriv : list of numpy arrays
        Each numpy array corresponds to the second derivative of the respective coordinate
        data as ordered by col_names.
    velocity : numpy array
        The computed velocity at each frame.
    curvature: numpy array
        The computed curvature at each frame.
    """
    # Extract and smoothens coordinate data
    if col_names is None:
        col_names = ['X', 'Y']
    n_dims = len(col_names)
    if n_dims < 2 or n_dims > 3:
        raise Exception("length of col_names is {}, but it should be 2 or 3.".format(n_dims))
    coords = []
    for col in col_names:
        try:
            coords.append(_smooth(animal_obj.get_raw_vals(col), animal_obj.get_frame_rate()))
        except KeyError:
            raise Exception("column name {} does not exist in animal dataset".format(col))

    # Calculate derivatives and adjust units
    coords = np.array(coords) # MM
    first_deriv = _calculate_derivatives(coords, axis=1) # MM per frame
    first_deriv = first_deriv * animal_obj.get_frame_rate() # MM per second
    second_deriv = _calculate_derivatives(first_deriv, axis=1) # MM per second per frame
    second_deriv = second_deriv * animal_obj.get_frame_rate() # MM per second per second

    # Calculate velocity and curvature and adds to Animal() object
    velocity = _calculate_velocity(first_deriv)
    curvature = _calculate_signed_curvature(first_deriv, second_deriv, velocity)

    start_time, end_time = animal_obj.get_baseline_times()
    animal_obj.add_raw_vals('Velocity', velocity)
    animal_obj.add_stats('Velocity', 'baseline', start_time, end_time)
    animal_obj.add_raw_vals('Curvature', curvature)
    animal_obj.add_stats('Curvature', 'baseline', start_time, end_time)
    return first_deriv, second_deriv, velocity, curvature


def compute_one_bdd(animal_obj_0, animal_obj_1, varnames,
                    seg_start_time_0, seg_end_time_0, seg_start_time_1, seg_end_time_1,
                    norm_mode, fullmode=False, outdir=None):
    """ Compute the Behavioural Distortion Distance (BDD) between two Animal() objects.

    Computes the Behavioural Distortion Distance (BDD) between two animal trajectories
    by applying Dynamic Time Warping (DTW), each starting and ending at the respective
    time frame given in the function call.

    Parameters
    ----------
    animal_obj_0/1 : Animal() object
        The initialized Animal() objects to be compared.
    varnames : list of strs
        List of hashable keys pointing to values stored in the Animal() objects to be used
        to calculate the BDD.
    seg_start/end_time_0/1 : int or float
        Segment start / end time in minutes.
    norm_mode : str, either 'baseline' or 'spec'
        Baseline mode uses the mean and standard deviation from the baseline observation
        time to normalize each variable data, whereas the spec mode uses the mean and
        standard deivation from the time period specified for this comparison.
    fullmode : bool, optional
        If True, the method first obtains the full suite of returns from dtw_ext and
        writes several path graphs. Default value : False.
    outdir : str, optional
        Path to the output directory. If fullmode is True, outdir must given. Default
        value : None.

    Returns
    -------
    bdd : float
        behavioural distortion distance
    """
    # pylint: disable=too-many-arguments
    # required for now, consider using tuples for time pairs?
    # pylint: disable=too-many-locals
    # function is long, requires many local variables

    # Argument Validation
    if fullmode and outdir is None:
        raise Exception("compute_one_bdd : Full mode requires the path to output directory.")
    if seg_end_time_0 - seg_start_time_0 != seg_end_time_1 - seg_start_time_1:
        raise Exception("compute_one_bdd : segments need to be of the same length.")

    # Extract relevant information from Animal() objects
    seg_start_frame_0 = animal.calculate_frame_num(animal_obj_0, seg_start_time_0)
    seg_end_frame_0 = animal.calculate_frame_num(animal_obj_0, seg_end_time_0)
    data_0 = animal_obj_0.get_mult_raw_vals(varnames, seg_start_frame_0, seg_end_frame_0)

    seg_start_frame_1 = animal.calculate_frame_num(animal_obj_1, seg_start_time_1)
    seg_end_frame_1 = animal.calculate_frame_num(animal_obj_1, seg_end_time_1)
    data_1 = animal_obj_1.get_mult_raw_vals(varnames, seg_start_frame_1, seg_end_frame_1)

    print("LOG: Applying DTW to the data from files %s and %s..." % (animal_obj_0.get_name(),
                                                                     animal_obj_1.get_name()))

    # Normalize and convert data to fit the dtw function
    num_vars = len(varnames)
    if norm_mode == 'baseline':
        for i in range(num_vars):
            means, stds = animal_obj_0.get_stats(varnames[i], 'baseline')
            data_0[i] = animal.normalize(data_0[i], means, stds)
            means, stds = animal_obj_1.get_stats(varnames[i], 'baseline')
            data_1[i] = animal.normalize(data_1[i], means, stds)
    elif norm_mode == 'spec':
        for i in range(num_vars):
            means, stds = animal.norm(data_0[i])
            data_0[i] = animal.normalize(data_0[i], means, stds)
            means, stds = animal.norm(data_1[i])
            data_1[i] = animal.normalize(data_1[i], means, stds)
    for i in range(num_vars):
        if varnames[i] == "Curvature": # Convert signed curvature to curvature
            data_0[i] = np.absolute(data_0[i])
            data_1[i] = np.absolute(data_1[i])
    data_0_t = np.array(data_0).T.tolist()
    data_1_t = np.array(data_1).T.tolist()

    # Calculate dtw
    dtw_obj = dtw.dtw(x=data_0_t, y=data_1_t, dist_method='euclidean', distance_only=(not fullmode))
    bdd = dtw_obj.normalizedDistance
    print("LOG: distance between %s and %s: %.5f" % (animal_obj_0.get_name(),
                                                     animal_obj_1.get_name(), bdd))

    # Fullmode to render alignment between the two animal objects
    if fullmode:
        #save alignment graphs in directory specified
        alignment = (dtw_obj.index1, dtw_obj.index2)
        write.render_alignment(alignment, animal_obj_0, animal_obj_1, varnames, outdir)
        seg_len = seg_end_time_0 - seg_start_time_0
        for i in range(num_vars):
            var = varnames[i]
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            write.render_aligned_graphs(data_0[i], data_1[i], alignment,
                                        animal_obj_0, animal_obj_1, seg_len, var, outdir)
            #For individual plots, enable the following two lines
            #write.render_single_animal_graph(data_0[i], animal_obj_0, var, outdir)
            #write.render_single_animal_graph(data_1[i], animal_obj_1, var, outdir)

    return bdd


def compute_all_bdd(animal_list, varnames, seg_start_time, seg_end_time, norm_mode):
    """ Computes pairwise BDDs given a list of Animal() objects.

    Computes the BDD of each pair of trajectories in animal_list using compute_one_bdd(),
    all starting and ending at the respective time frame given in the function call.

    Parameters
    ----------
    animal_list : list of Animal() objects
        List of initialized Animal() objects to be compared.
    varnames : list of strs
        List of hashable keys pointing to values stored in the Animal() objects to be used
        to calculate the BDD.
    seg_start/end_time : int or float
        Segment start / end ime in minutes.
    norm_mode : str, either 'baseline' or 'spec'
        Baseline mode uses the mean and standard deviation from the baseline observation
        time to normalize each variable data, whereas the spec mode uses the mean and
        standard deivation from the time period specified for this comparison.

    Returns
    -------
    bdds : 2D array of float (upper-triangular, empty diagonal)
        i,j-th entry bdds[i][j] is the bdd between trajectories of animal[i] and animal[j].
    """
    # Runs compute_one_bdd() for each pair of Animal() objects
    num_animals = len(animal_list)
    bdds = [['' for i in range(num_animals)] for j in range(num_animals)]
    for i in range(num_animals):
        for j in range(i+1, num_animals):
            bdd = compute_one_bdd(animal_list[i], animal_list[j], varnames,
                                  seg_start_time, seg_end_time,
                                  seg_start_time, seg_end_time, norm_mode)
            bdds[i][j] = bdd
    return bdds


def compute_one_iibdd(animal_obj, varnames, norm_mode,
                      interval_length=None, start_time=None, end_time=None):
    """ Computes the BDD between two intervals of an animal's trajectory.

    Computes the Behavioural Distortion Distance (BDD) of an animal's trajectory to
    itself over a random pair of non-overlapping intervals.

    Parameters
    ----------
    animal_obj : Animal() object
        Initialized Animal() object to be compared.
    varnames : list of strs
        List of hashable keys pointing to values stored in the Animal() objects to be used
        to calculate the BDD.
    norm_mode : str, either 'baseline' or 'spec'
        Baseline mode uses the mean and standard deviation from the baseline observation
        time to normalize each variable data, whereas the spec mode uses the mean and
        standard deivation from the time period specified for this comparison.
    interval_legth : int or float
        length of the interval to use. If unspecified, it will be chosen at random.
    start_time : float
        Time in minutes where the intervals can start. If omitted, exp start time is used.
    end_time : float
        Time in minutes where the intervals can end. If omitted, exp end time is used.

    Returns
    -------
    list : [interval_length, bdd]
        interval_length : int or float
            Length of time interval used for comparison.
        bdd : float
            Computed BDD.
    """
    # pylint: disable=too-many-arguments
    # all arguments are necessary

    # Extract experiment times if not given
    if start_time is None:
        start_time = animal_obj.get_exp_start_time()
    if end_time is None:
        end_time = animal_obj.get_exp_end_time()

    # if no interval lengths are specified, generate random interval lengths
    if interval_length is None:
        intervals = sorted([random.uniform(start_time, end_time) for i in range(3)])
        interval_length = (intervals[0] - start_time) / 2
    else:
        intervals = sorted([2 * interval_length + start_time] +
                           [random.uniform(2 * interval_length + start_time, end_time)
                            for i in range(2)])

    # with generated interval lengths, produce 2 non-overlapping intervals
    interval_start_time_0 = intervals[1] - 2 * interval_length
    interval_end_time_0 = intervals[1] - interval_length
    interval_start_time_1 = intervals[2] - interval_length
    interval_end_time_1 = intervals[2]

    # compute bdd between 2 intervals
    bdd = compute_one_bdd(animal_obj, animal_obj, varnames,
                          interval_start_time_0, interval_end_time_0,
                          interval_start_time_1, interval_end_time_1, norm_mode)

    return [interval_length, bdd]


def compute_all_iibdd(animal_list, varnames, norm_mode, num_exps,
                      interval_lengths=None, outdir=None, outfilename=None,
                      start_time=None, end_time=None):
    """ Computes the intra-individual BDD for each trajectory in list of Animal() objects.

    Computes the intra-individual Behavioural Distortion Distance (IIBDD) for each
    trajectory in a list of Animal() objects, all starting and ending at the respective
    time frame given in the function call. Computes the IIBDD using compute_one_iibdd()
    function.

    Parameters
    ----------
    animal_list : list of Animal() objects
        List of initialized Animal() objects to be compared.
    varnames : list of strs
        List of hashable keys pointing to values stored in the Animal() objects to be used
        to calculate the BDD.
    norm_mode : str, either 'baseline' or 'spec'
        Baseline mode uses the mean and standard deviation from the baseline observation
        time to normalize each variable data, whereas the spec mode uses the mean and
        standard deivation from the time period specified for this comparison.
    num_exps : int
        Number of times to repeat the experiments for each Animal() objects.
    interval_legths : List of ints or floats, or None. Optional.
        list - list of length of the time interval length to use.
               num_exps comparisons will be made for each length in the list
        None - a interval length for each test will be chosen at random to
               be between 0.01 and half the total time
        Default value : None.
    outdir : str, optional
        The output directory path. If specified, the results will be written to a file
        (outfilename). Default value : None.
    outfilename : str, optional
        For output file name. Default value : None.
    start_time : float
        Time in minutes where the intervals can start. If omitted, exp start time will be
        used as start time.
    end_time : float
        Time in minutes where the intervals can end. If omitted, exp end time will be used
        as end time.

    Returns
    -------
    exp_table : 2D-array of tuples of floats
        i,j-th entry exp_table[i][j] is the j-th interval comparison for the i-th animal
        in animal_list each entry is a double [interval lenth, distance].

    If interval_lengths is given, the function additionally returns:
    mean_table : 2D-array of pairs
        mean_table[i][j][0] is the j-th interval_length, and mean_table[i][j][1] is the
        mean of the distances from tests using the j-th interval_length and i-th animal.
    std_table : 2D-array of pairs
        std_table[i][j][0] is the j-th interval_length, and std_table[i][j][1] is the std
        of the distances from tests using the j-th interval_length and i-th animal.
    """
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    # function is long, requires many arguments and local variables

    exp_table = []

    # If interval_length is not specified, run random segment tests for each animal
    if interval_lengths is None:
        for animal_obj in animal_list:
            exp_list = []
            for _ in range(num_exps):
                res = compute_one_iibdd(animal_obj, varnames, norm_mode, None, start_time, end_time)
                exp_list.append(res)
            exp_table.append(exp_list)
        if outdir:
            write.write_segment_exps_to_csv(animal_list, exp_table, None, None, outdir, outfilename)
        return exp_table

    # Else, run num_exps comparisons per length
    mean_table = []
    std_table = []

    for animal_obj in animal_list:
        exp_list = []
        mean_list = []
        std_list = []

        # If timings aren't given, use experiment timings
        if start_time is None:
            start_time = animal_obj.get_exp_start_time()
        if end_time is None:
            end_time = animal_obj.get_exp_end_time()

        # Generate intervals
        slice_areas = [0.5 * ((end_time - start_time) - 2 * length) ** 2
                       for length in interval_lengths]
        total_area = sum(slice_areas)
        num_exps_per_slice = [int(num_exps*slice_area / total_area)
                              for slice_area in slice_areas]
        cum_exps_per_slice = [0] + [sum(num_exps_per_slice[:i+1])
                                    for i, _ in enumerate(interval_lengths)]

        for j, interval_length in enumerate(interval_lengths):
            num_slice_exps = num_exps_per_slice[j]
            for _ in range(num_slice_exps):
                res = compute_one_iibdd(animal_obj, varnames, norm_mode,
                                        interval_length, start_time, end_time)
                exp_list.append(res)
                mean = np.mean(map(lambda x: x[1],
                                   exp_list[cum_exps_per_slice[j]: cum_exps_per_slice[j+1]]))
                std = np.std(map(lambda x: x[1],
                                 exp_list[cum_exps_per_slice[j]: cum_exps_per_slice[j+1]]))
                mean_list.append([interval_length, mean])
                std_list.append([interval_length, std])
                exp_table.append(exp_list)
                mean_table.append(mean_list)
                std_table.append(std_list)

    if outdir:
        # Write results to directory if out directory is given
        write.write_segment_exps_to_csv(animal_list, exp_table, mean_table,
                                        std_table, outdir, outfilename)

    return exp_table, mean_table, std_table


########################
### Helper Functions ###
########################

def _calculate_derivatives(series, axis=0):
    """
    Computes the derivative of the series. Returns a numpy array
    """
    derivatives = np.gradient(series, axis=axis)
    return derivatives


def _calculate_signed_curvature(first_deriv, second_deriv, velocity):
    """
    Given a list of first and second derivatives, return curvature.
    Note: Currently only works for up to 2 / 3 dimensions.

    Parameters
    ----------
    first_deriv : list of numpy arrays
        Each numpy array corresponds to the first derivative of the respective coordinate
        data as ordered by col_names.
    second_deriv : list of numpy arrays
        Each numpy array corresponds to the second derivative of the respective coordinate
        data as ordered by col_names.
    velocity : numpy array
        The computed velocity at each frame.

    Returns
    -------
    curvatures : numpy array
        The computed curvature at each frame.
    """
    if first_deriv.shape != second_deriv.shape:
        raise Exception("first_deriv and second_deriv should be of the same shape.")
    n_dims = first_deriv.shape[0]
    if n_dims == 2:
        mats = np.transpose(np.array([first_deriv, second_deriv]), (2, 0, 1))
    elif n_dims == 3:
        ones = np.ones_like(first_deriv)
        mats = np.transpose(np.array([ones, first_deriv, second_deriv]), (2, 0, 1))
    numer = np.linalg.det(mats)
    denom = np.power(velocity, 3)
    curvatures = []
    for i, _ in enumerate(numer):
        if denom[i] < 0.000125:
            curve = 0
        else:
            curve = numer[i] / denom[i]
        curvatures.append(curve)
    return curvatures


def _calculate_velocity(coordinates):
    """ Calculates the velocity.

    Parameters
    ----------
    coordinates : numpy array
        Numpy matrix where each row corresponds to the coordinates in one axis.
    Returns
    -------
    velocity : numpy array
        The computed velocity at each frame.
    """
    velocity = np.sqrt(np.sum(np.power(coordinates, 2), axis=0))
    return velocity


def _smooth(sequence, frame_rate):
    """ Smooths sequence by applying Savitzky-Golay smoothing.

    Note: this function makes use of global variables SMOOTH_RANGE_MIN and WINDOW_SCALAR.

    Parameters
    ----------
    sequence : list of floats
        Sequence to be smoothed.

    Returns
    -------
    smoothed : list of floats
        Smoothed sequence.
    """
    smooth_range = max(SMOOTH_RANGE_MIN, int(np.ceil(frame_rate * WINDOW_SCALAR)))
    smooth_range_odd = smooth_range + 1 if smooth_range % 2 == 0 else smooth_range
    smoothed = savgol_filter(sequence, smooth_range_odd, ORDER)
    return smoothed
