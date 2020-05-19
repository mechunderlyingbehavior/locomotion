import os
import sys
import csv
import json
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import locomotion

PATH_TO_DATA_DIRECTORY = os.getcwd() + "/data"
try: # Safety check to ensure that folder exists, and makes it otherwise.
    os.mkdir(PATH_TO_DATA_DIRECTORY)
except FileExistsError:
    pass

PATH_TO_RES_DIRECTORY = os.getcwd() + "/results"
try: # Safety check to ensure that folder exists, and makes it otherwise.
    os.mkdir(PATH_TO_RES_DIRECTORY)
except FileExistsError:
    pass


# Constants
NUM_CURVES = 11 # This must match the number of curves.
ZFILL_LEN = int(np.ceil(np.log10(NUM_CURVES)))
NUM_SAMPLES = 100 # Number of samples being tested
SAMP_FILL = int(np.ceil(np.log10(NUM_SAMPLES)))
DEFAULT_START = 0 # Start Time in Minutes
DEFAULT_STOP = 1 # Stop Time in Minutes

########################################################################
#### Utility Functions ####
########################################################################

def genTrigFun(a_k, b_k):
    """
    Input:
    >> int list a_k, b_k: Sequences of length K
    Output:
    >> fun f(t): Desired trig function with coefficients corresponding to sequences a_k, b_k
    """
    def fun_t (t):
        sum = 0
        for i in range(len(a_k)):
            sum += a_k[i]*np.sin(i*t) + b_k[i]*np.cos(i*t)
        return sum
    return (fun_t)


def euclidDist(a, b):
    return np.sqrt(np.power(a, 2) + np.power(b, 2))


def speed(x_t, y_t):
    """
    Input: 2 arrays of length of TIME_T
    Output: 1 array of length of TIME_T representing the speed
    """
    return euclidDist(np.gradient(x_t), np.gradient(y_t))


def curvature(x_t, y_t):
    """
    Input: 2 arrays of length of TIME_T
    Output: 1 array of length of TIME_T representing the curvature
    """
    num = abs(np.gradient(x_t)*np.gradient(np.gradient(y_t))
             - np.gradient(y_t)*np.gradient(np.gradient(x_t)))
    denom = np.power(speed(x_t, y_t), 3)
    return (num/denom)


def changePixDensity(num, density):
    if num == 0 or density == 0 or math.isnan(num) or math.isnan(density):
        return 0
    return math.floor(num * density)


def genVariables(low, high, n):
    """
    Uniformly samples from given interval of values.
    Input:
    >> float low  : lower bound of interval
    >> float high : upper bound of interval
    >> int n : number of samples
    Output:
    >> numpy array of variables
    """
    return list(np.random.uniform(low, high, n))

def cameraFunc(coeff_path, time_start, time_stop, frame_rate, density, plot=False):
    """
    Inputs:
    >> str coeff_path:                 Path to coefficients_xx.csv
    >> float time_start, time_stop:   Beginning and end times to generate time step increments
    >> int frame_rate:                How often to sample the time step between time_start and time_stop
    >> string file_out:               Name of output file
    >> bool plot:                     Plots the captured curves if set to true

    Outputs:
    >> dataframe df:                  Dataframe with columns [X, Y]
    """
    # Read in data from coefficients csv
    # Coefficients - 1 x K
    data = pd.read_csv(coeff_path)

    a_k = data['a_k'].values
    b_k = data['b_k'].values
    c_k = data['c_k'].values
    d_k = data['d_k'].values

    theta = data['extras'][0]
    size = data['extras'][1]
    x_min = data['extras'][2]
    x_max = data['extras'][3]
    x_dif = x_max - x_min
    y_min = data['extras'][4]
    y_max = data['extras'][5]
    y_dif = y_max - y_min

    time_t = np.arange(time_start, time_stop, 1/frame_rate)
    p_time_t = theta * (time_t - time_start) / (time_stop - time_start)

    x_fun = genTrigFun(a_k, b_k)
    y_fun = genTrigFun(c_k, d_k)

    # Get minimum and maximum x,y coordinates of the graph 
    x_og = x_fun(p_time_t) # 1 x len(TIME_T)
    y_og = y_fun(p_time_t) # 1 x len(TIME_T)
    lower_xlim = min(x_og)
    upper_xlim = max(x_og)
    lower_ylim = min(y_og)
    upper_ylim = max(y_og)

    # Coordinates - 1 x len(TIME_T)
    x_enlarged = []
    y_enlarged = []
    for i in range(0, len(time_t)):
        x_enlarged.append((x_dif / (upper_xlim - lower_xlim) * (x_og[i] - lower_xlim) + x_min))
        y_enlarged.append((y_dif / (upper_ylim - lower_ylim) * (y_og[i] - lower_ylim) + y_min))

    # Transform coordinates based on density
    x = []
    y = []
    for i in range(0, len(time_t)):
        x.append(np.array(changePixDensity(x_enlarged[i], density)))
        y.append(np.array(changePixDensity(y_enlarged[i], density)))

    if(plot):
        # Plots
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
        plt.subplot(121)
        plt.plot(x, y)
        plt.title("Curve no. " + str(file_no) + ", m = " + str(len(a_k)))
        plt.axis([0,size,0,size])
        plt.subplot(122)
        kSeq = np.arange(0, len(a_k), 1)
        plt.plot(kSeq, a_k)
        plt.plot(kSeq, b_k)
        plt.plot(kSeq, c_k)
        plt.plot(kSeq, d_k)
        plt.title("Coefficients")
        plt.axis([0,20,-1,1])
        plt.show()

    # Transform data into dataframe
    data = np.transpose(np.array((x, y)))
    df = pd.DataFrame(data, columns = ['X', 'Y'])
    summaryStats = df.describe()
    return df, summaryStats

########################################################################
#### Robustness Testing Setup ####
########################################################################

def captureOneCurve(dat_path, curve_str, test_str, coeff_path, frame_rate, density, control = "False"):
    # Generate Capture Data
    df, _ = cameraFunc(coeff_path, DEFAULT_START * 60, DEFAULT_STOP * 60, frame_rate, density, plot = False)
    # Save Capture Data to CSV
    df.to_csv(dat_path)
    jsonItem = {
        "name": "CRV_{}_TEST_{}".format(curve_str, test_str),
        "data_file_location": dat_path,
        "animal_attributes":
            {
                "species": "Magic Scoliosis Fish",
                "exp_type": "MCS",
                "ID": curve_str,
                "control_group": control
            },
            "capture_attributes":
            {
                "dim_x": 100,
                "dim_y": 100,
                "pixels_per_mm": density,
                "frames_per_sec": frame_rate,
                "start_time": DEFAULT_START,
                "end_time": DEFAULT_STOP,
                "baseline_start_time": DEFAULT_START,
                "baseline_end_time": DEFAULT_STOP
            }
    }
    return jsonItem

def captureAllCurves(test_key):
    # Check / Create directory
    resultPath = PATH_TO_RES_DIRECTORY + "/" + test_key
    try:
        os.mkdir(resultPath)
    except FileExistsError:
        pass
    for curve_no in range(NUM_CURVES):
        curve_str = str(curve_no).zfill(ZFILL_LEN)
        jsonItems = []
        coeff_path = PATH_TO_DATA_DIRECTORY + "/curve_data/coefficients_{}.csv".format(curve_str)
        # Capture Control + Key Check
        try:
            control_fr, control_dens = testData[test_key]["control"]
        except KeyError:
            raise Exception("test_key not in testData")
        control_dat = resultPath + "/CRV_{}_TEST_CTRL.dat".format(curve_str)
        control_json = captureOneCurve(control_dat, curve_str, "CTRL", coeff_path, control_fr, control_dens, "True")
        jsonItems.append(control_json)
        # Capture test curves
        i = 0
        for fr in testData[test_key]["framerates"]:
            for dens in testData[test_key]["densities"]:
                test_str = str(i).zfill(SAMP_FILL)
                dat_path = resultPath + "/CRV_{}_TEST_{}.dat".format(curve_str, test_str)
                jsonItem = captureOneCurve(dat_path, curve_str, test_str, coeff_path, fr, dens)
                jsonItems.append(jsonItem)
        outfilename = resultPath + "/CRV_{}.json".format(curve_str)
        jsonstr = json.dumps(jsonItems, indent = 4)
        with open(outfilename, "w") as outfile:
            outfile.write(jsonstr)
        print("Wrote the information into %s" % outfilename)
    # Save Frame Rate data and Density data
    with open(resultPath + "/Results_variables.json", "w") as outfile:
        varJson = json.dumps(testData[test_key])
        outfile.write(varJson)


def runRobustnessTest(test_key, variables, norm_mode, start_min, end_min):
    results = np.zeros([NUM_CURVES, NUM_SAMPLES])
    for curve_no in range(NUM_CURVES):
        curve_str = str(curve_no).zfill(ZFILL_LEN)
        json_path = PATH_TO_RES_DIRECTORY + "/{}/CRV_{}.json".format(test_key, curve_str)
        # Load all animals
        animals = locomotion.getAnimalObjs(json_path)
        for a in animals:
            locomotion.trajectory.getCurveData(a)
        # Run BDD against control animal (index 0)
        control = animals[0]
        for a_no, a in enumerate(animals[1:]):
            bdd = locomotion.trajectory.computeOneBDD(a, control, variables,
                                                      start_min, end_min,
                                                      start_min, end_min,
                                                      norm_mode)
            results[curve_no][a_no] = bdd
    output = PATH_TO_RES_DIRECTORY + "/{}/Results_BDD.csv".format(test_key)
    pd.DataFrame(results).to_csv(output, index = False)

################################################################################
### Testing Space
################################################################################


testData = {
    "FR_test_lower" : {
        "framerates" : list(range(6,24)),
        "densities" : [2],
        "control" : (24, 2)
    },
    "FR_test_higher" : {
        "framerates" : genVariables(24, 120, NUM_SAMPLES),
        "densities" : [2],
        "control" : (24, 2)
    },
    "density_test_lower" : {
        "framerates" : [24],
        "densities" : genVariables(0.5, 2, NUM_SAMPLES),
        "control" : (24, 2)
    },
    "density_test_higher" : {
        "framerates" : [24],
        "densities" : genVariables(2, 8, NUM_SAMPLES),
        "control" : (24, 2)
    }
}

# Change these variables
test_name = "FR_test_lower"
variables = ['Velocity', 'Curvature']
norm_mode = 'spec'

captureAllCurves(test_name) # Uncomment to recapture curves
runRobustnessTest(test_name, variables, norm_mode, DEFAULT_START, DEFAULT_STOP)
