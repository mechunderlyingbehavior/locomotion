import os
import math
import random
import numpy as np
import pandas as pd

PATH_TO_DATA_DIRECTORY = os.getcwd() + "/data"
try: # Safety check to ensure that folder exists, and makes it otherwise.
    os.mkdir(PATH_TO_DATA_DIRECTORY)
except FileExistsError:
    pass

# Curve Generator Constants
NUM_CURVES = 50
ZFILL_LEN = int(np.ceil(np.log10(NUM_CURVES)))
NUM_TERMS = 50
POWER_OF_TERMS = 2.1
SIZE = 100

########################################################################
#### Utility Functions ####
########################################################################

def coeffGen(p,k):
    """
    Input: Lower limit and higher limit (int)
    Output: List of length K with random values between low and high
    """
    coeffSeq = []
    for i in range(k):
        coeffSeq.append(np.random.normal(0, 1.0/np.power(i+1, p)))
    return coeffSeq


########################################################################
#### Curve Generating ####
########################################################################

def genCurve(k, p, size, file_no):
    """
    Writes the data to Curves/coefficients_(file_no) in a csv file.

    Inputs:
    >> int k: Number of terms in trig series
    >> int p: Power of denominator when sampled from normal distribution
    >> int s: size - boundary box size
    >> int file_no: Index, used for naming files

    Outputs:
    >> Dataframe with columns [a_k, b_k, c_k, d_k], coefficients used to generate parametric curve
    """

    # Coefficients - 1 x K
    a_k = coeffGen(p, k)
    b_k = coeffGen(p, k)
    c_k = coeffGen(p, k)
    d_k = coeffGen(p, k)

    # Theta
    theta = random.random()**0.5 * (2*math.pi)
    x_min = size/2.0*random.random()**2
    x_max = size - size/2.0*random.random()**2
    y_min = size/2.0*random.random()**2
    y_max = size - size/2.0*random.random()**2

    # Transform data into dataframe
    data = np.transpose(np.array((a_k, b_k, c_k, d_k)))
    df1 = pd.DataFrame(data, columns = ['a_k', 'b_k', 'c_k', 'd_k'])
    extras = np.transpose(np.array([theta, size, x_min, x_max, y_min, y_max]))
    df2 = pd.DataFrame(extras, columns = ['extras'])
    df = pd.concat([df1,df2], axis=1)

    # Export csv
    dirPath = PATH_TO_DATA_DIRECTORY + "/curve_data"
    try: # Safety check to ensure that folder exists, and makes it otherwise.
        os.mkdir(dirPath)
    except FileExistsError:
        pass
    curve_file_csv = dirPath + "/coefficients_{}.csv".format(str(file_no).zfill(ZFILL_LEN))
    df.to_csv(curve_file_csv, index=False)
    return df

def genNCurves(n, k, p):
    """
    Inputs:
    >> int n: Number of curves we want to generate
    >> int k: Number of terms in trig series 
    >> int p: Power of denominator when sampled from normal distribution
    Writes n csv files with desired curve data.
    """
    for i in range(n):
        # Uniformly sample k from a range of 2 to 100 to get a bigger variety of curves
        k = random.randrange(2,100)
        # Use grid size 100x100
        genCurve(k, p, 100, i)

genNCurves(NUM_CURVES, NUM_TERMS, POWER_OF_TERMS)
