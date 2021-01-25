"""Copyright Mechanisms Underlying Behavior Lab, Singapore
https://mechunderlyingbehavior.wordpress.com/

trajectory.py is part of the locomotion python package for analyzing locomotory animal
behaviors via the techniques presented in the paper "Computational geometric tools
for quantitative comparison of locomotory behavior" by MT Stamps, S Go, and AS Mathuru
(https://doi.org/10.1038/s41598-019-52300-8).

This python script contains methods for computing behavioral distortion distances
(BDD). The DTW implementation used in this package is the one provided in the
dtw-python package (T. Giorgino. Computing and Visualizing Dynamic Time Warping
Alignments in R: The dtw Package. J. Stat. Soft., doi:10.18637/jss.v031.i07.).
"""

import os
import random
import numpy as np
import dtw
from scipy.signal import savgol_filter
import locomotion.write as write
import locomotion.animal as animal

#Static Variables
EPSILON = 0.0001
SMOOTH_RANGE_MIN = 5 #minimum length of smoothing window in _smooth()
WINDOW_SCALAR = 2.5 #scalar for smoothing window calculation in _smooth()
ORDER = 5 #order of smoothing curve used in _smooth()

######################
### Main Functions ###
######################


def populate_velocity(animal_obj, col_names=None):
    """Computes the velocity given coordinates stored in animal_obj.

    This function computes and stores the Velocity of the coordinate data
    stored in animal_obj. The data used to calculate this is given by
    col_names, the list of column names.

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
    velocity : numpy array
        The computed velocity at each frame.
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

    # Calculate first derivative and adjust units
    coords = np.array(coords) # MM
    first_deriv = _calculate_derivatives(coords, axis=1) # MM per frame
    first_deriv = first_deriv * animal_obj.get_frame_rate() # MM per second

    # Calculate velocity and adds to Animal() object
    velocity = _calculate_velocity(first_deriv)
    start_time, end_time = animal_obj.get_baseline_times()
    animal_obj.add_raw_vals('Velocity', velocity)
    animal_obj.add_stats('Velocity', 'baseline', start_time, end_time)
    return first_deriv, velocity

def populate_curvature(animal_obj, col_names=None, first_deriv=None, velocity=None):
    """Computes the curvature given coordinates stored in animal_obj.

    This function computes and stores the Curvature of the coordinate data
    stored in animal_obj. The data used to calculate this is given by
    col_names, the list of column names.

    Currently only works in 2 or 3 dimensions.

    Parameters
    ----------
    animal_obj : Animal() object
        Initialized Animal() object, which should already contain coordinate data.
    col_names : list of strs, optional
        Names of data columns used for calculations. Must coincide with data stored in
        animal_obj.__raw_vals. Will only be used if first_deriv and velocity is None.
        If not given, defaults to ['X', 'Y'].
    first_deriv : numpy arrays of floats, optional
        Previously calculated first derivatives. If not given, the first derivative would
        be calculated using col_names if velocity is also None.
    velocity : numpy array of floats, optional
        Previously calculated velocity. If not given, the velocity will be calculated with
        col_names if first_deriv is also None.

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
    if first_deriv is None and velocity is None:
        # Calculate first_deriv and velocity
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
        coords = np.array(coords) # MM

        first_deriv = _calculate_derivatives(coords, axis=1) # MM per frame
        first_deriv = first_deriv * animal_obj.get_frame_rate() # MM per second
        velocity = _calculate_velocity(first_deriv)
    else:
        # Quick sense check of given first_deriv and velocity
        if first_deriv is None or velocity is None:
            raise Exception("populate_curvature: both first_deriv and velocity must be given.")
        if len(velocity) != first_deriv.shape[1]:
            raise Exception("populate_curvature: shape of first_deriv and velocity don't match.")

    second_deriv = _calculate_derivatives(first_deriv, axis=1) # MM per second per frame
    second_deriv = second_deriv * animal_obj.get_frame_rate() # MM per second per second
    curvature = _calculate_signed_curvature(first_deriv, second_deriv, velocity)

    # Add curvature data to animal_obj
    start_time, end_time = animal_obj.get_baseline_times()
    animal_obj.add_raw_vals('Curvature', curvature)
    animal_obj.add_stats('Curvature', 'baseline', start_time, end_time)
    return first_deriv, second_deriv, velocity, curvature

def populate_distance_from_point(animal_obj, point_key, param_key, col_names=None):
    """Calculates distance of animal from a point and add to object.

    Given the position coordinates of an animal, calculates the euclidean
    distance of the animal from a point, pre-defined with coordinates stored in
    animal_obj.__info with key point_key. Then, update the dictionary
    animal_obj.__raw_vals with the new distance data, with key param_key.
    Finally, calculates and updates animal_obj.__means and animal_obj.__stds
    for param_key.

    Parameters
    ----------
    animal_obj : Animal() object
        Initialized Animal() object, with coordinate data and with coordinates
        stored in animal_obj.__info, with key point_key.
    point_key : str
        Hashable key pointing to goal coordinate data stored in animal_obj.__info.
    param_key : str
        Hashable key that will be used to point to distance data stored in self.__raw_vals.
    col_names : list of str, optional
        Names of data columns used for calculations. Must coincide with data stored in
        animal_obj.__raw_vals, and must be ordered in the same order as the coordinates of
        the goal. If not given, defaults to ['X', 'Y'].

    Returns
    -------
    distances : list of floats
        Computed distances from goal point at each frame.

    """
    if col_names is None:
        col_names = ['X', 'Y']
    n_dims = len(col_names)
    coords = []
    for col in col_names:
        try:
            coords.append(_smooth(animal_obj.get_raw_vals(col),
                                  animal_obj.get_frame_rate()))
        except KeyError:
            raise KeyError("column name {} does not exist in animal dataset".format(col))

    # extract goal coordinates
    try:
        point = animal_obj.get_info(point_key)
    except KeyError:
        raise KeyError("%s is not a valid key stored in Animal Object %s"
                       % (point_key, animal_obj.get_name()))
    if not(isinstance(point, (list, tuple, np.ndarray))):
        raise Exception("Point is of type %s, but it should be a list, tuple, or numpy array"
                        % type(point))
    elif len(point) != n_dims:
        raise Exception("Dimension of point is not the same as dimension of coordinates.")

    # euclidean distance calculation
    goal = np.array(point)
    coords = np.array(coords).T
    distances = np.sqrt(np.sum((coords - point)**2, axis=1))

    start_time, end_time = animal_obj.get_baseline_times()
    animal_obj.add_raw_vals(param_key, distances)
    animal_obj.add_stats(param_key, 'baseline', start_time, end_time)
    return distances

def populate_curve_data(animal_obj, col_names=None):
    """ OUTDATED: Computes the behavioural curve data such as Velocity and Curvature.

    This function computes and stores the Velocity and Curvature of the coordinate
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
    print("OUTDATED: This method has been split into populate_velocity and populate_curvature.")
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
    """ Computes the BDD between a pair of animal trajectories.

    Computes the Behavioral Distortion Distance (BDD) between two animal trajectories
    for a prescribed set of variables over a specified pair of time intervals and
    normalization mode.

    Both distance-only and full alignment options are available.

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
    seg_diff = abs((seg_end_time_0 - seg_start_time_0) - (seg_end_time_1 - seg_start_time_1))
    if seg_diff >= EPSILON:
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
    """ Computes all pairwise BDDs given a list of animal trajectories.

    Computes the BDD of each pair of trajectories in animal_list using compute_one_bdd()
    with a prescribed set of variables and normalization mode over a common time interval
    given in the function call.

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


def compute_all_to_one_bdd(animal_list, target_animal, varnames,
                           seg_start_time, seg_end_time, norm_mode):
    """ Computes BDDs between a list of animals to a target animal.

    Computes the BDDs of each animal in animal_list to target_animal using
    compute_one_bdd() with a prescribed set of variables and normalization mode
    over a common time interval given in the function call.

    Parameters
    ----------
    animal_list : list of Animal() objects
        List of initialized Animal() objects to be compared.
    target_animal : Animal() object
        Single Animal() object that all animals in animal_list will be compared to.
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
    bdds : list of floats
        i-th entry bdds[i] is the bdd between trajectories of animal_list[i] and
        target_animal.
    """
    bdds = ['' for _ in animal_list]
    for i, a in enumerate(animal_list):
        bdd = compute_one_bdd(a, target_animal, varnames, seg_start_time, seg_end_time,
                              seg_start_time, seg_end_time, norm_mode)
        bdds[i] = bdd
    return bdds


def compute_one_iibdd(animal_obj, varnames, norm_mode, num_samples,
                      interval_length=None, start_time=None, end_time=None):
    """ Computes the IIBDD for an animal trajectory.

    Computes the Intra-Individual Behavioral Distortion Distance (IIBDD) from an animal
    trajectory to itself for a prescribed set of variables and normalization mode over
    a pair of randomly generated non-overlapping time intervals.

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
    num_samples : int
        Number of samples generated and used in calculating the average bdd.
    interval_legth : int or float, optional
        Length of the interval to use, in minutes. If unspecified, generate at random.
    start_time : float, optional
        Time in minutes where the intervals can start. If omitted, exp start time is used.
    end_time : float, optional
        Time in minutes where the intervals can end. If omitted, exp end time is used.

    Returns
    -------
    float
        Average bdd calculated across all samples.
    """
    # pylint: disable=too-many-arguments
    # all arguments are necessary

    # Extract experiment times if not given
    if start_time is None:
        start_time = animal_obj.get_exp_start_time()
    if end_time is None:
        end_time = animal_obj.get_exp_end_time()

    # Run compute_one_bdd() num_samples times
    bdds = []
    for _ in range(num_samples):
        # if no interval lengths are specified, generate random interval lengths
        if interval_length is None:
            intervals = sorted([random.uniform(start_time, end_time) for i in range(3)])
            interval_length = (intervals[0] - start_time) / 2
        else:
            # sanity check for interval_length
            if 2 * interval_length > end_time - start_time:
                raise Exception("compute_one_iibdd : interval length too long.")
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
        bdds.append(bdd)

    return np.mean(bdds)


def compute_all_iibdd(animal_list, varnames, norm_mode, num_samples,
                      interval_length=None, start_time=None, end_time=None):
    """ Computes all IIBDDs given a list of animal trajectories.

    Computes the average IIBDD of each trajectory in animal_list using compute_one_iibdd()
    with a prescribed set of variables, normalization mode, and number of randomly
    generated non-overlapping time intervals.

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
    num_samples : int
        Number of samples generated and used in calculating the average bdd.
    interval_legth : int or float, optional
        Length of the interval to use, in minutes. If unspecified, generate at random.
    start_time : float, optional
        Time in minutes where the intervals can start. If omitted, exp start time is used.
    end_time : float, optional
        Time in minutes where the intervals can end. If omitted, exp end time is used.

    Returns
    -------
    list of floats
        List of average iibdds calculated for each animal in animal_list. i-th entry is
        the iibdd of the i-th animal in animal_list.
    """
    # pylint: disable=too-many-arguments
    bdds = []
    for anim in animal_list:
        bdd = compute_one_iibdd(anim, varnames, norm_mode, num_samples,
                                interval_length, start_time, end_time)
        bdds.append(bdd)
    return bdds


########################
### Helper Functions ###
########################

def _calculate_derivatives(series, axis=0):
    """
    Computes the derivative of the series. Returns a numpy array.
    """
    derivatives = np.gradient(series, axis=axis)
    return derivatives


def _calculate_signed_curvature(first_deriv, second_deriv, velocity):
    """
    Given a list of first and second derivatives, return curvature.
    Note: Currently only works for 2 or 3 dimensions.

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

    Note: This function makes use of global variables SMOOTH_RANGE_MIN and WINDOW_SCALAR.

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
