## Copyright Mechanisms Underlying Behavior Lab, Singapore
## https://mechunderlyingbehavior.wordpress.com/

## curve_gen.py is part of the locomotion package comparing animal behaviours, developed
## to support the work discussed in the paper "Computational geometric tools for
## modeling inherent variability in animal behavior" by MT Stamps, S Go, and AS Mathuru.

## This python script contains methods for generating random curves in R^2. The closed
## of these curves will then be used to capture these curves in different frame rates
## and resolutions in the accompanying script "capture_and_compare.py" for the purpose of
## testing the robustness of our package. The mathematical basis for this curve generation
## described in the paper "Random space and plane curves" by Igor Rivin, which can be
## accessed here: https://arxiv.org/pdf/1607.05239.pdf. The main idea is to use the fourier
## series described in the paper to generate a random sequence of numbers in each coordinate
## of the plane. The resulting parametric curve of the plane f(theta) = [x(theta), y(theta)] in 
## R^2 is guaranteed to be smooth, random (in the sense that the probability of getting one curve is
## the same as getting any other) and closed (it loops back to itself when the theta = 2pi). This
## gives us random trajectories of animals that we can use to test the robustness of the package.
## To avoid getting closed-loop trajectories, we pick domains that are proper subsets of [0, 2pi].

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

#static variables for curve generation
NUM_CURVES = 50
NUM_TERMS = 50
POWER_OF_TERMS = 2.1
SIZE = 100

#prescribed number of digits for file-naming
ZFILL_LEN = int(np.ceil(np.log10(NUM_CURVES)))


########################################################################
#### Utility Functions ####
########################################################################


def coeffGen(k, p):
    """ Generates the sequence of coefficients a_k/b_k for k = [0, no_of_terms] 
        for the fourier series in each variable. This is done by sampling from a 
        Gaussian distribution with mean 0 and a decaying standard deviation.

        :Parameters:
            k : int. The number of terms we want to sum up in the fourier series
            p : float. The power of denominator when sampled from the Gaussian distribution

        :Returns:
            list of floats. The sequence of coefficients for the sine/cosine terms
            in the fourier series for each coordinate variable.
    """
    
    #initialise return list
    coeffSeq = []

    #append a random sample from the Gaussian distribution with mean 0 and decaying standard deviation k times
    for i in range(k):
        #the power of the denominator in the is offset by 1 since range(k) is 0-indexed
        coeffSeq.append(np.random.normal(0, 1.0 / np.power(i + 1, p)))

    return coeffSeq


########################################################################
#### Curve Generating ####
########################################################################


def genCurve(k, p, size, file_no):
    """ Writes the generated coefficients and other relevant data for the curve into a 
        csv file, whose filename corresponds to the file number provided.

        :Parameters:
            k : int. The number of terms we want to sum up in the fourier series
            p : float. The power of denominator when sampled from the Gaussian distribution
            size : int. The size of the boundary box that the resulting generated curve will be contained in
            file_no: int. Index used for file naming

        :Returns:
            dataframe with columns [a_k, b_k, c_k, d_k, extras]
            The first 4 columns will have k rows - these are the the coefficients used to generate parametric plane curve. 
            The extras column will have 6 rows corresponding to theta, size, x_min, x_max, y_min and y_max.
            This is extra data that will help with curve capturing in capture_and_compare.py.

            This dataframe will be written into the file coefficients_(file_number).csv in the curve_data folder.
    """

    #generate coefficients for the fourier series. a_k and b_k are used for the x coordinate,
    #whereas c_k and d_k are used for the y-coordinate. The dimension of each sequence is 1 x K.
    a_k = coeffGen(k, p)
    b_k = coeffGen(k, p)
    c_k = coeffGen(k, p)
    d_k = coeffGen(k, p)

    #generate a random value of theta that is less than 2pi in radians
    theta = random.random()**0.5 * (2*math.pi)
    #create bounding box that is a reasonably-sized proper subset of the given size
    x_min = size / 2.0 * random.random()**2
    x_max = size - size / 2.0 * random.random()**2
    y_min = size / 2.0 * random.random()**2
    y_max = size - size / 2.0 * random.random()**2

    #transform the coefficient and extras data into a dataframe
    data = np.transpose(np.array((a_k, b_k, c_k, d_k)))
    df1 = pd.DataFrame(data, columns = ['a_k', 'b_k', 'c_k', 'd_k'])
    extras = np.transpose(np.array([theta, size, x_min, x_max, y_min, y_max]))
    df2 = pd.DataFrame(extras, columns = ['extras'])
    df = pd.concat([df1,df2], axis = 1)

    #write the dataframe to a csv file in the curve_data directory
    dirPath = PATH_TO_DATA_DIRECTORY + "/curve_data"
    try: #safety check to ensure that folder exists, and makes it otherwise.
        os.mkdir(dirPath)
    except FileExistsError:
        pass
    curve_file_csv = dirPath + "/coefficients_{}.csv".format(str(file_no).zfill(ZFILL_LEN))
    df.to_csv(curve_file_csv, index = False)
    return df

def genNCurves(n, k, p):
    """ Write multiple curves, with their corresponding coefficients and extras, into multiple csv files.

        :Parameters:
            n : int. The number of curves we want to generate.
            k : int. The maximum number of terms in the trig series for any of the curves generated.
            p : float. The power of denominator when sampled from the Gaussian distribution.

        :Returns:
            None.
            Writes the files coefficients_00.csv, ..., coefficients_(n-1).csv into the data/curve_data folder.
    """

    for i in range(n):
        #uniformly sample k from a range of 2 to 100 to get a bigger variety of curves
        k = random.randrange(2,100)
        #use grid size 100x100
        genCurve(k, p, 100, i)

#generate N curves using the static variables defined above
genNCurves(NUM_CURVES, NUM_TERMS, POWER_OF_TERMS)
