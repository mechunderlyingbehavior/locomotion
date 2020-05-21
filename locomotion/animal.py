import os
import sys
import csv
import re
import math
import numpy as np
import json
import warnings
from math import ceil, exp, log, sin, asin, pi, acosh, cosh, sinh, cos, acos, atanh, tanh
from numpy import min, mean, std, array, linalg, dot, cross
from scipy.optimize import minimize_scalar

SMOOTH_RANGE = 5 #technically (range-1)/2

################################################################################
#### Animal class ####
################################################################################

class Animal(object):

  def __init__(self, json_item):
    self.name = json_item["name"]
    self.data_file = os.path.abspath(json_item["data_file_location"])
    self.filename = os.path.basename(self.data_file)
    self.animal_type = json_item["animal_attributes"]["species"]
    self.exp_type = json_item["animal_attributes"]["exp_type"]
    self.ID = json_item["animal_attributes"]["ID"]
    self.isControl = eval(json_item["animal_attributes"]["control_group"])
    self.dim_x = json_item["capture_attributes"]["dim_x"] # Pixels
    self.dim_y = json_item["capture_attributes"]["dim_y"] # Pixels
    self.pix = json_item["capture_attributes"]["pixels_per_mm"]         # Pixels per MM
    self.frame_rate = json_item["capture_attributes"]["frames_per_sec"] # Frames per Second
    self.start = json_item["capture_attributes"]["start_time"] # In Minutes
    self.end = json_item["capture_attributes"]["end_time"]     # In Minutes
    self.baseline_start = json_item["capture_attributes"]["baseline_start_time"] # In Minutes
    self.baseline_end = json_item["capture_attributes"]["baseline_end_time"]     # In Minutes
    self.rawvals = {}
    #self.vals = {}
    self.means = {}
    self.stds = {}

  def getName(self):
    return self.name

  def getDataFileLocation(self):
    return self.data_file

  def getDataFileName(self):
    return self.filename

  def getAnimalType(self):
    return self.animal_type

  def getExpType(self):
    return self.exp_type

  def getID(self):
    return self.ID

  def getExpTimes(self):
    return (self.start, self.end)

  def getExpStartTime(self):
    return self.start

  def getExpEndTime(self):
    return self.end

  def getBaselineTimes(self):
    return (self.baseline_start, self.baseline_end)

  def getBaselineStartTime(self):
    return self.baseline_start

  def getBaselineEndTime(self):
    return self.baseline_end

  def inControlGroup(self):
    return self.isControl

  def getDims(self):
    return self.dim_x, self.dim_y

  def getPixelDensity(self):
    return self.pix

  def getFrameRate(self):
    return self.frame_rate

  #def addVals(self, varname, valList):
  #  self.vals.update({varname:valList})

  #def getVals(self, varname):
  #  return self.vals[varname]

  #def getMultVals(self, varnames):
  #  return [self.vals[v] for v in varnames]

  def addRawVals(self, varname, valList):
    self.rawvals.update({varname:valList})

  def getRawVals(self, varname, start_frame=None, end_frame=None):
        """
    Return the raw vals stored in animal object.
    :Parameters:
     varname       : hashable key pointing to variables in animal object
     start_frame   : starting frame of portion to extract
     end_frame     : ending frame of portion to extract
    """
    if start_frame == None:
      start_frame =self.start*60*self.frame_rate
    if end_frame == None:
      end_frame = self.end*60*self.frame_rate
    # logic check
    try:
      values = self.rawvals[varname]
    except KeyError:
      raise KeyError("getRawVals: {} not found in animal object.".format(varname))
    if start_frame > end_frame:
      raise ValueError("getRawVals: Start frame is after End frame.")
    if start_frame > len(values):
      raise ValueError("getRawVals: Start frame comes after existing frames.")
    if end_frame > len(values):
      warnings.warn("getRawVals: End frame comes after existing frames. Defaulting to the final frame stored.")
    return values[start_frame:end_frame]

  def getMultRawVals(self, varnames, start_frame=None, end_frame=None):
    return [self.getRawVals(v,start_frame,end_frame) for v in varnames]

  def initStats(self, varname):
    self.means.update({varname:{}})
    self.stds.update({varname:{}})

  def addStats(self, varname, scope, start_frame, end_frame):
    if varname not in self.means:
        self.initStats(varname)
    m, s = norm(self.rawvals[varname][start_frame:end_frame])
    self.means[varname].update({scope:m})
    self.stds[varname].update({scope:s})

  def getStats(self, varname, scope):
    return self.means[varname][scope], self.stds[varname][scope]

  def setGridSize(self, grid_size):
    self.grid_size = grid_size
    num_x_grid = int(ceil(self.dim_x/grid_size))
    num_y_grid = int(ceil(self.dim_y/grid_size))
    self.setNumGrids(num_x_grid,num_y_grid)

  def getGridSize(self):
    return self.grid_size

  def setNumGrids(self, num_x_grid, num_y_grid):
    self.num_x_grid = num_x_grid
    self.num_y_grid = num_y_grid

  def getNumGrids(self):
    return self.num_x_grid, self.num_y_grid

  def setPerturbation(self, perturbation):
    self.perturbation = perturbation

  def getPerturbation(self):
    return self.perturbation

  def setConformalFactor(self, conformal_factor):
    self.conformal_factor = conformal_factor

  def getConformalFactor(self):
    return self.conformal_factor

  def setTolerance(self, tolerance):
    self.tolerance = tolerance

  def getTolerance(self):
    return self.tolerance

  def setNumVerts(self, n):
    self.numVerts = n

  def getNumVerts(self):
    return self.numVerts

  def setColors(self, colors):
    self.colors=colors

  def getColors(self):
    return self.colors

  def setRegularCoordinates(self, coordinates):
    self.regCoords = coordinates

  def getRegularCoordinates(self):
    return self.regCoords

  def setFlattenedCoordinates(self, coordinates):
    self.flatCoords = coordinates

  def getFlattenedCoordinates(self):
    return self.flatCoords

  def setTriangulation(self, triangles):
    self.triangulation = triangles

  def getTriangulation(self):
    return self.triangulation

  def setFlattenedBoundaryVertices(self, vertices):
    self.flattenedBoundaryVertices = vertices

  def getFlattenedBoundaryVertices(self):
    return self.flattenedBoundaryVertices

  def setCentralVertex(self, central_vertex):
    self.centralVertex = central_vertex

  def getCentralVertex(self, central_vertex):
    return self.centralVertex

  def setVertexBFS(self, vertex_bfs):
    self.vertexBFS = vertex_bfs

  def getVertexBFS(self):
    return self.vertexBFS

  def setTriangleTriangleAdjacency(self, triangle_triangle_adjacency):
    self.triangleTriangleAdjacency = triangle_triangle_adjacency
  
  def getTriangleTriangleAdjacency(self):
    return self.triangleTriangleAdjacency

  
################################################################################
### Basic Functions
################################################################################

# Sure, I suppose I could use the actual error handling, but...
def throwError(errmsg):
  print("ERROR: %s" % errmsg)
  exit(1)


def getFrameNum(animal, t):
#t is in minutes
  return int(animal.getFrameRate() * t * 60)


def findColIndex(header, colName):
# Finds the column index of the given variable in the data
# TO-DO: make this case insensitive
  pat = re.compile('^(")*%s(")*$' % colName)
  for i in range(len(header)):
    if re.match(pat, header[i]): return i
  # if we didn't find the index, the column name input is incorrect
  throwError("invalid column name: %s" % colName)


def norm(data):
  dArr = np.array(data, dtype=np.float)
  m = np.mean(dArr)
  sd = np.std(dArr)
  return m, sd


def normalize(data, m, s):
  if s != 0: return list(map(lambda x: 1/(1 + math.exp(-(x-m)/s)), data))
  else: return [0 for d in data]

################################################################################
### Meat & Potatoes
################################################################################

def readInfo(infile):
  with open(infile, 'r') as infofile:
    info = json.load(infofile)
  return info


def getRawData(animal, varnames = ['X','Y']):
  #read in X and Y values from the data file
  with open(animal.getDataFileLocation(), 'r') as infile:
    print("LOG: Extracting coordinates for Animal %s..." % animal.getName())
    header = infile.readline()#.replace('\r','').replace('\n','')
    if '\t' in header: delim = '\t'
    elif ',' in header: delim = ','
    else: throwError("invalid data format")

    header = list(map(lambda x: x.strip(), header.split(delim)))
    try: # verify the file can be parsed
      reader = csv.reader(infile, delimiter = delim)
    except:
      throwError("invalid data format")

    XInd = findColIndex(header, 'X')
    YInd = findColIndex(header, 'Y')
    X, Y = [], []
    start, end = animal.getExpTimes()
    start_frame = getFrameNum(animal, start)
    end_frame = getFrameNum(animal, end)

    for line, row in enumerate(reader):
      if line < start_frame: continue
      if line == end_frame: break 
      x = row[XInd]
      y = row[YInd]
      if len(x)==0 or x==' ' or len(y)==0 or y ==' ':
        print(row)
        throwError("possible truncated data")
      X.append(float(x)/animal.getPixelDensity()) #scaling for pixel density while we are at it
      Y.append(float(y)/animal.getPixelDensity())  

      #DEFN: baseline norm is where we take the stats from the first two minutes of the exp to get the "baseline normal" numbers
      #DEFN: exp norm is where we take the stats from the whole exp duration and take all 'local data' into consideration

  animal.addRawVals('X', X)
  animal.addRawVals('Y', Y)

  baseline_start, baseline_end = animal.getBaselineTimes()
  baseline_start_frame = getFrameNum(animal, baseline_start)
  baseline_end_frame = getFrameNum(animal, baseline_end)

  animal.addStats('X', 'baseline', baseline_start_frame, baseline_end_frame)
  animal.addStats('Y', 'baseline', baseline_start_frame, baseline_end_frame)


def getAnimalObjs(infofile, name_list = None):
  info = readInfo(infofile)
  if name_list != None:
    objs = [initAnimal(item) for item in info if item["name"] in name_list]
    return objs
  else:
    return [initAnimal(item) for item in info]


def initAnimal(json_item):
# Given a json entry, extracts the relevant information and returns an initialized animal object
  a = Animal(json_item)
  getRawData(a, ['X','Y'])
  return a
