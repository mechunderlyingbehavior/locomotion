## Copyright Mechanisms Underlying Behavior Lab, Singapore
## https://mechunderlyingbehavior.wordpress.com/

## heatmap.py is part of the locomotion package comparing animal behaviours, developed
## to support the work discussed in the paper "Computational geometric tools for
## modeling inherent variability in animal behavior" by MT Stamps, S Go, and AS Mathuru.

## This python script contains methods for computing the Conformal Spatiotemporal
## Distance (CSD) between the heatmaps of two animals representing the amount of time
## each animal spends in a given location. Specifically, this script contains methods
## for constructing the heatmap of an animal give its raw coordinate and timestamp data,
## an approximately uniform/regular triangulation of its corresponding surface in R^3,
## a conformal "flattening" of the that surface to the unit disk.  There are also
## methods for finding the alignment between two surfaces given for a given rotation
## between their respective flattenings in the unit disk and methods for measuring the
## symmetric distortion energy of such an alignment.

from math import sin, cos, pi
from numpy import mean, std, array, linalg
from scipy.optimize import minimize_scalar
import locomotion.write as write
import locomotion.animal as animal
from locomotion.animal import throwError
from igl import boundary_loop, map_vertices_to_circle, harmonic_weights, adjacency_matrix, bfs, triangle_triangle_adjacency

#Static Variables
PERTURBATION = 0.000000001
TOLERANCE = 0.00001

################################################################################  
### METHOD FOR INITIALIZING HEAT MAP AND SURFACE DATA FOR EACH ANIMAL OBJECT ###
################################################################################


def getSurfaceData(animal_obj, grid_size, start_time=None, end_time=None):
  """ Computes the heatmap for a given animal representing the amount
      of time the animal spent in each location during a specified time
      interval, an approximately regular Delaunay triangulation of
      the corresponding surface, and a conformal flattening of that
      triangulation to the unit disk.

   :Parameters:
    animal_obj : animal object 
      from animal.py, initialized
    grid_size : float or int
      specifies the bin size for calculating the heatmap
      must divide both X_DIM and Y_DIM
      smaller values yield finer triangulations and larger values yield coarser triangulations
    start/end_time : float or int
      time in minutes. If unspecified, start/end time for the experiment will be used
  """
  #Check if start_time or end_time need to be set:
  if start_time == None:
    start_time = animal_obj.getExpStartTime()
  if end_time == None:
    end_time = animal_obj.getExpEndTime()

  #store given parameters
  animal_obj.setGridSize(grid_size)
  animal_obj.setPerturbation(PERTURBATION)
  animal_obj.setTolerance(TOLERANCE)
  
  print("Calculating heatmap for %s..." % animal_obj.getName())
  
  #calculate heatmap
  frequencies = getFrequencies(animal_obj, start_time, end_time)

  print("Calculating triangulation for %s..." % animal_obj.getName())
  
  #get and record vertices
  original_coordinates = getVertexCoordinates(animal_obj, frequencies)
  animal_obj.setNumVerts(len(original_coordinates))
  animal_obj.setRegularCoordinates(original_coordinates)
  
  #get and record triangles
  triangles = getTriangles(animal_obj)
  animal_obj.setNumTriangles(len(triangles))
  animal_obj.setTriangulation(triangles)

  #calculate and store colors for output file
  colors = getColors(animal_obj)
  animal_obj.setColors(colors)

  print("Calculating flattened coordinates for %s..." % animal_obj.getName())

  #calculate and record boundary vertices
  boundary_vertices = getBoundaryLoop(animal_obj)
  animal_obj.setBoundaryVertices(boundary_vertices)

  #calculate and record boundary edges
  boundary_edges = getBoundaryEdges(animal_obj)
  animal_obj.setBoundaryEdges(boundary_edges)
  
  #calculate and record flattened coordinates of triangulation
  flattened_coordinates = getFlatCoordinates(animal_obj)
  animal_obj.setFlattenedCoordinates(flattened_coordinates)

  print("Calculating vertex BFS and triangle adjacency for %s..." % animal_obj.getName())

  #calculate and record central vertex and BFS from the centre
  central_vertex = getCentralVertex(animal_obj)
  animal_obj.setCentralVertex(central_vertex)

  #for each animal, we want a BFS of just the interior vertices, not the boundary vertices
  #as a first step, we need to get all triangles that do not contain a boundary vertex (No-Boundary-Vertex/NBV triangles)
  #this is because the edge information between interior vertices is stored within the NBV-triangles
  nbv_triangles = getNBVTriangles(animal_obj)

  #find the adjacency matrix of the interior vertices using NBV triangles to calculate and record the BFS
  interior_vertex_adjacency_matrix = adjacency_matrix(array(nbv_triangles))
  interior_vertex_bfs = bfs(interior_vertex_adjacency_matrix, central_vertex)
  animal_obj.setInteriorVertexBFS(interior_vertex_bfs)

  #calculate and record triangle-triangle adjacency matrix
  triangle_adjacency_matrix = triangle_triangle_adjacency(array(triangles))[0]
  animal_obj.setTriangleTriangleAdjacency(triangle_adjacency_matrix)

  
#######################################################################################  
### METHODS FOR CALCULATING HEAT MAPS AND THEIR CORRESPONDING TRIANGULATED SURFACES ###
#######################################################################################  


def getFrequencies(animal_obj, start_time, end_time):
  """ Gathers the frequency data for approximating the heat map representing
      the amount of time an animal spent in each location of the assay 
      chamber over a specified time interval.

   :Parameters:
    animal_obj : animal object, initialized
    start_time : float, time in minutes
    end_time : float, time in minutes

   :Returns:
     two-dimensional array of ints counting the number of frames the animal
     spent in each square chamber of the bounding rectangle during the 
     specified time interval
  """

  #set or get relevant parameters
  start_frame = animal.getFrameNum(animal_obj, start_time)
  end_frame = animal.getFrameNum(animal_obj, end_time)
  perturb = animal_obj.getPerturbation()
  grid_size = animal_obj.getGridSize()
  x_dim, y_dim = animal_obj.getDims()
  num_x_grid, num_y_grid = animal_obj.getNumGrids()
  X = animal_obj.getRawVals('X', start_frame, end_frame)
  Y = animal_obj.getRawVals('Y', start_frame, end_frame)

  #initialize frequency matrix
  freqency_matrix = [[0 for j in range(num_y_grid)] for i in range(num_x_grid)]
  
  #check that coordinate data is within the specified bounds
  x_max = max(X)
  x_offset = max(x_max - x_dim, 0) + perturb
  y_max = max(Y)
  y_offset = max(y_max - y_dim, 0) + perturb
  
  #iterate through each frame, adjust out-of-bounds data, and update frequency matrix
  for i in range(len(X)):
    x = X[i] - x_offset
    if x < 0:
      print("WARNING: X data is out of bounds. Frame #%d, x=%f" % (i+1, X[i]))
      x = 0
    x_index = int(x/grid_size)
    y = Y[i] - y_offset
    if y < 0:
      print("WARNING: Y data is out of bounds. Frame #%d, x=%f" % (i+1, Y[i]))
      y = 0
    y_index = int(y/grid_size)
    freqency_matrix[x_index][y_index] += 1

  return freqency_matrix

def getZDim(animal_obj):
  """ Calculates the vertical bound for a heatmap surface
    We set it to be the smaller of the two horizontal dimensions, but it
    can be set to specified value depending on the context.

   :Parameter:
     animal_obj : animal object, initialized

   :Returns:
     int, value of vertical dimension
  """
  
  return min(animal_obj.getDims())

def getVertexCoordinates(animal_obj, freqs):
  """ Calculates the vertex coordinates for a triangulation of the surface 
      corresponding to a heat map.

    :Parameters:
      animal_obj : animal object, initialized
      freqs : 2D array of ints
        Frequency data for heatmap

    :Returns:
      list of triples of floats, specifying the x-, y-, and z-coordinates of the vertices
      for a triangulation of the surface corresponding to a heat map
  """

  #gather relevant parameters
  grid_size = animal_obj.getGridSize()
  num_x_grid, num_y_grid = animal_obj.getNumGrids()

  #normalize the values to floats between 0 and a specified z-dimension
  m = mean(freqs)
  s = std(freqs)
  z_dim = getZDim(animal_obj)
  for i in range(len(freqs)):
    freqs[i] = animal.normalize(freqs[i],m,s)
    freqs[i] = list(map(lambda x : z_dim*x, freqs[i]))

  #initialize list of coordinates to return
  coordinates = []

  #append coordinates for the lower left corner of each square in the heat map grid
  for i in range(num_x_grid):
    for j in range(num_y_grid):
      coordinates.append([i*grid_size, j*grid_size, freqs[i][j]])

  return coordinates

def getTriangles(animal_obj):
  """ Computes a basic triangulation on the regular coordinates of an animal

    :Parameters:
      animal_obj : animal object, initialized with regular coordinates set/updated

    :Returns:
      list of triples of ints, specifying the indices of the vertices for each triangle in
      in the triangulation of a surface
  """
  #store relevant parameters
  num_x_grid, num_y_grid = animal_obj.getNumGrids()

  #initialize triangle list
  triangles = []
 
  #iterate through lower left corners of grid and append canonical triangles
  for i in range(num_x_grid-1):
    for j in range(num_y_grid-1):
      triangles.append([i*num_y_grid+j, (i+1)*num_y_grid+j, (i+1)*num_y_grid+(j+1)])
      triangles.append([i*num_y_grid+j, (i+1)*num_y_grid+(j+1), i*num_y_grid+(j+1)])

  return triangles
  
def getBoundaryLoop(animal_obj):
  """ Given an animal object, get its boundary vertices in counter-clockwise order. This method is a wrapper for the corresponding IGL function.

    :Parameters:
        animal_obj : animal object, initialized with regular coordinates and triangulation set/updated

      :Returns:
        array of ints. The indices of the vertices that are on the boundary of this animal.
  """
  #convert triangulation to array for IGL 
  triangulation = array(animal_obj.getTriangulation())
  return boundary_loop(triangulation)

def getBoundaryEdges(animal_obj):
  """ Given an animal object, get its ordered boundary edges in counter-clockwise order.

    :Parameters:
        animal_obj : animal object, initialized with regular coordinates and triangulation set/updated

      :Returns:
        list of int tuple pairs: list of edges ordered as in the boundary loop, where each edge is a tuple of the two vertices it connects
  """
  boundary_vertices = list(animal_obj.getBoundaryVertices())
  #zip the boundary vertices with itself with an offset of 1 and its head appended at the back (so it goes full circle), then cast to a list
  boundary_edges = list(zip(boundary_vertices, boundary_vertices[1:] + [boundary_vertices[0]]))
  return boundary_edges

def getColors(animal_obj):
  """ Calculates color for rendering each triangle in the triangulation of an animal according 
    to the average height of the regular coordinates of its vertices 

    :Parameters:
      animal_obj : animal object, initialized with regular coordinates and triangulation set/updated

    :Returns:
      list of triples of floats, specifying the RGB coordinates for each triangle in
      in the triangulation associated to an animals heat map
  """

  #gather relevant parameters
  coordinates = animal_obj.getRegularCoordinates()
  triangles = animal_obj.getTriangulation()

  #initialize return list
  colors = []

  #extract the heights (z-coordinates) of each vertex in the triangulation
  heights = [c[2] for c in coordinates]

  #gather basic statistics
  min_height = min(heights) 
  max_height = max(heights)
  mid_height = (min_height+max_height)/2

  #assign a color to each triangle based on the average height of the regular coordinates of its vertices
  for triangle in triangles:
    color = [1.0,1.0,0]
    height = mean([heights[v] for v in triangle])
    if height > mid_height:
      color[1] -= (height-mid_height)/(max_height-mid_height)
    else:
      color[0] -= (mid_height-height)/(mid_height-min_height)
      color[1] -= (mid_height-height)/(mid_height-min_height)
      color[2] += (mid_height-height)/(mid_height-min_height)
    colors.append(color)
    
  return colors

#######################################################################################  
###    METHODS NEEDED FOR TRIANGLE-TRIANGLE AND VERTEX-VERTEX ADJACENCIES AND BFS   ###
#######################################################################################  

def getCentralVertex(animal_obj):
  """ 
  Finds the index of the vertex coordinate for the triangulation of an animal that is closest to its topological centre.

    :Parameters:
      animal_obj : animal objects, initialized with regular/flattened coordinates and triangulation set/updated

    :Returns:
      integer index of the vertex at the the central coordinate. We know that it will be there because of our triangulation method.
  """
  #get the regular coordinates in the x, y dimension to find the central vertex in that plane
  x_y_coordinates = [coord[:2] for coord in animal_obj.getRegularCoordinates()]

  #get the central coordinate in the grid. It must be a multiple of the grid size.
  mid_x_coordinate = (animal_obj.num_x_grid // 2) * animal_obj.grid_size
  mid_y_coordinate = (animal_obj.num_y_grid // 2) * animal_obj.grid_size

  #find the index of this central coordinate
  central_vertex = x_y_coordinates.index([mid_x_coordinate, mid_y_coordinate])

  return central_vertex

def getNBVTriangles(animal_obj):
  """ 
  Finds all triangles in an animal that do not include its boundary vertices.

    :Parameters:
      animal_obj : animal objects, initialized with regular/flattened coordinates and triangulation set/updated

    :Returns:
      list of list triples. list of all triangles in the animal that do not contain boundary vertices.
  """
  #get relevant parameters
  triangles = animal_obj.getTriangulation()
  boundary_vertices = set(animal_obj.getBoundaryVertices())
  interior_triangles = []
  
  #for each triangle, check if each vertex is a boundary vertex. If it does not contain a boundary vertex, add it to interior_triangles
  for triangle in triangles:
    contains_boundary_vertex = False
    for vertex in triangle:
      if vertex in boundary_vertices:
        contains_boundary_vertex = True
        break
    #either all vertices in this triangle are not boundary vertices, or we found a boundary vertex at this point
    if not contains_boundary_vertex:
      interior_triangles.append(triangle)

  return interior_triangles

####################################################################################  
### METHODS FOR CALCULATING CONFORMAL FLATTENINGS OF TRIANGULATIONS TO UNIT DISK ###
#################################################################################### 


def mobius(u, v, a, b):
  #this is a helper method for the getFlatCoordinates method below
  return [((u-a)*(a*u+b*v-1)+(v-b)*(a*v-b*u))/((a*u+b*v-1)**2+(a*v-b*u)**2), ((v-b)*(a*u+b*v-1)-(u-a)*(a*v-b*u))/((a*u+b*v-1)**2+(a*v-b*u)**2)]

def getFlatCoordinates(animal_obj):
  """ Calculates the vertex coordinates for the triangulation of an animal from its corresponding circle packing in the unit disk

    :Parameters:
      animal_obj : animal object, initialized with regular coordinates and triangulation set/updated

    :Returns:
      list of pairs of floats, specifying the x- and y-coordinates of the vertices of a triangulation that has been conformally flattened
      to the unit disk
  """

  # store relevant parameters and convert to arrays
  tolerance = animal_obj.getTolerance()
  regular_coordinates, triangles = array(animal_obj.getRegularCoordinates()), array(animal_obj.getTriangulation())
    
  # get boundary vertice indices (already an array) from the animal
  boundary_vertices = animal_obj.getBoundaryVertices()

  # map boundary vertices to unit circle, preserving edge proportions, to get the flattened boundary coordinates
  flattened_boundary_coordinates = map_vertices_to_circle(regular_coordinates, boundary_vertices)

  # map internal vertices to unit circle
  flat_coordinates = harmonic_weights(regular_coordinates, triangles, boundary_vertices, flattened_boundary_coordinates, 1)
  flat_coordinates = list(flat_coordinates)

  # apply a conformal automorphism (Mobius transformation) of the unit disk that moves the center of mass of the flattened coordinates to the origin
  p = mean([c[0] for c in flat_coordinates])
  q = mean([c[1] for c in flat_coordinates])

  while p**2+q**2 > tolerance:
    print("LOG: Distance of original centroid to origin is %f. Moving closer to origin." % (p**2+q**2))
    for i in range(len(flat_coordinates)):
      x = flat_coordinates[i][0]
      y = flat_coordinates[i][1]
      flat_coordinates[i] = mobius(x,y,p,q)
    p = mean([c[0] for c in flat_coordinates])
    q = mean([c[1] for c in flat_coordinates])

  return flat_coordinates


#########################################################################  
### METHODS FOR ALIGNING TWO SURFACES VIA THEIR CONFORMAL FLATTENINGS ###
######################################################################### 


# Given a point and a theta, map the point to the rotation by theta
def rotation(p, theta):
  #this is a helper method for the method getAlignedCoordinates below.  It rotates a given point in the plane about the origin by a given angle.
  return [cos(theta)*p[0]-sin(theta)*p[1], sin(theta)*p[0]+cos(theta)*p[1]]

def inUnitInterval(x):
  return x >= 0 and x <= 1

def isInside(barycentric_coords):
  """ Given a list of any number of barycentric coordinates, check if each value is between 0 and 1.

    :Parameters:
      barycentric_coords: list of n lambda values for this point in barycentric coordinates.

    :Returns:
      boolean: true only if all lambda values are between 0 and 1
  """    
  return all(map(inUnitInterval, barycentric_coords))

def getBarycentricCoordinates(point, simplex, coordinates):
  """ Given a 2D point inside a simplex (a line segment or a triangle), find out its barycentric coordinates.
      In the case of the line (1-simplex), this would be the point expressed as a linear combination of the two endpoints.
      In the case of the triangle (2-simplex), this would be the point expressed as a linear combination of three corner coordinates.

      NOTE: This method will not work when finding barycentric coordinates of points within a triangle or line segment in R^3!
      It is only meant for finding barycentric coordinates of 2D points within 2D line segments or triangles. 

    :Parameters:
      point: float pair list. The 2D coordinates of the flattened vertex of Animal 1. The z-component should be 0.
      simplex: int pair or triple list. The indices corresponding to coordinates making up the line segment/triangle.
      coordinates: list of float pairs. The 2D coordinates.

    :Returns:
        if input point and coordinates are both in R^2 and simplex is valid:
          list pair/triple of floats corresponding to the barycentric coordinates of the point in the simplex.
          These are the lambda values i.e the weights used in the linear combination.
          If all these values are between 0 and 1, the point is in the simplex. Otherwise, it is not.
        else:
          empty list.
  """

  if not (len(point) == len(coordinates[0]) == 2):
    print("WARNING: Invalid coordinate dimensions. This method is only defined to get the barycentric coordinates of a 2D point within a 2D simplex.")
    #return empty list to standardise output and avoid computation. Otherwise, the code may still run without throwing an error.
    return []

  #initialise result
  result = []

  #if the simplex is a triangle, calculate the barycentric coordinates of the point in the triangle
  if len(simplex) == 3:
    #get coordinates from vertices of simplex
    triangle_coordinates = [coordinates[i] for i in simplex]
    (x_0, y_0), (x_1, y_1), (x_2, y_2) = triangle_coordinates
  
    #find each of the three weights
    lambda_0 = ((y_1-y_2)*(point[0]-x_2)+(x_2-x_1)*(point[1]-y_2)) / \
              ((y_1-y_2)*(x_0-x_2)+(x_2-x_1)*(y_0-y_2))
    lambda_1 = ((y_2-y_0)*(point[0]-x_2)+(x_0-x_2)*(point[1]-y_2)) / \
              ((y_1-y_2)*(x_0-x_2)+(x_2-x_1)*(y_0-y_2))
    lambda_2 = 1 - lambda_0 - lambda_1
    result = [lambda_0, lambda_1, lambda_2]

  #if the simplex is a line segment, find the proportions of each point in the line segment
  elif len(simplex) == 2:
    #since it's linear interpolation, the proportions are the same for both x and y components, so we just use one of them
    x_0, x_1 = coordinates[simplex[0]] [0], coordinates[simplex[1]] [0]

    #find the two weights
    lambda_1 = (point[0] - x_0) / (x_1 - x_0)
    lambda_0 = 1 - lambda_1
    result = [lambda_0, lambda_1]

  else: 
    print("WARNING: Invalid input simplex. This method is only defined for triangles and edges.")
  return result

def fromBarycentricToCoordinates(barycentric_coords, simplex, coordinates):
  """ Given barycentric coordinates, a list of coordinates and a simplex (triangle or line segment), return the
      actual coordinates in R^3 corresponding to the barycentric coordinates. 

      NOTE: This method will not work when finding the corresponding coordinates in R^2! We will be trying to access
      the z-component, which will cause an index error. If this method is needed in such a case, assign 0 as the third coordinate.

    :Parameters:
      barycentric_coords: float triple or pair list. The barycentric coordinates of a point in the triangle or line segment.
      simplex: int triple or pair list. The indices corresponding to vertices making up the triangle or line segment.
      coordinates: list of float triples. If they are flattened coordinates, the third element should be 0.

    :Returns:
       if input coordinates are in R^3 and the barycentric coordinates match up with the simplex:
        list of float triple. The corresponding converted coordinates in R^3.
      else:
        empty list.
  """

  if len(coordinates[0]) != 3:
    print("WARNING: Invalid coordinate dimensions. This method is only defined to find the coordinates of a point in 3D.")
    #return empty list to standardise output and avoid errors thrown later on
    return []

  #initialise return value
  result = []

  #if the simplex is a triangle, get the values of the corresponding coordinates in R^3 componentwise
  if len(barycentric_coords) == len(simplex) == 3:   
    x = barycentric_coords[0] * coordinates[simplex[0]] [0] + \
        barycentric_coords[1] * coordinates[simplex[1]] [0] + \
        barycentric_coords[2] * coordinates[simplex[2]] [0]
    y = barycentric_coords[0] * coordinates[simplex[0]] [1] + \
        barycentric_coords[1] * coordinates[simplex[1]] [1] + \
        barycentric_coords[2] * coordinates[simplex[2]] [1]
    z = barycentric_coords[0] * coordinates[simplex[0]] [2] + \
        barycentric_coords[1] * coordinates[simplex[1]] [2] + \
        barycentric_coords[2] * coordinates[simplex[2]] [2]
    result = [x, y, z]

  #if the simplex is a line segment, get the values of the corresponding coordinates using the equation of a line
  elif len(barycentric_coords) == len(simplex) == 2:
    #extract the two coordinates of the line segment and just one of the barycentric coordinate weights (the gradient of the line)
    (x_0, y_0, z_0), (x_1, y_1, z_1) = coordinates[simplex[0]], coordinates[simplex[1]]
    lambda_1 = barycentric_coords[1]

    x = x_0 + lambda_1 * (x_1 - x_0)
    y = y_0 + lambda_1 * (y_1 - y_0)
    z = z_0 + lambda_1 * (z_1 - z_0)
    result = [x, y, z]
  
  else:
    print("WARNING: Invalid barycentric coordinates and/or simplex dimensions. They must both be of length 2 or 3, since this method is only defined for triangles and edges.")

  return result

def searchForAlignedCoordinate(point, simplices, simplex_indices, input_coordinates, output_coordinates):
  """ Given a point in the 2D input coordinate system, search through the given simplices (either triangle or edges) in the 
      input coordinate system to check if it is inside one of them. If it is, convert the point into barycentric coordinates
      corresponding to the simplex, and use those barycentric coordinates to return the point in the 3D output coordinate system.
      Otherwise, return an empty list.

    :Parameters:
      point: float pair list. A point in the input coordinate system.
      simplices: list of list of float triples or pairs. A list of the triangles or edges we want to search (in indices of vertices)
      simplex_indices : int list or set. Indices of the simplices above in the input coordinate system.
      input_coordinates: list of float pairs. The 2D input coordinate system where the point and simplices lie.
      output_coordinates: list of float triples. The 3D output coordinate system we want to align the point to.

    :Returns:
      if the point is found in one of the simplices:
        list of an int and a list of float triple. 
        The int is the index of the simplex we found the point inside.
        The list of float triple is the point's aligned coordinate in the output coordinate system.
      else:
        empty list.
  """

  #initialise the result
  result = []

  for simplex_i, simplex in zip(simplex_indices, simplices):
    #get the barycentric coordinates of this point in this simplex in the input coordinate system
    barycentric_coords = getBarycentricCoordinates(point, simplex, input_coordinates)
    if isInside(barycentric_coords):
      #set the result as the regular coordinates corresponding to the barycentric coordinates
      result = [simplex_i, fromBarycentricToCoordinates(barycentric_coords, simplex, output_coordinates)]
      break

  return result

def findClosestVertex(point, vertices, input_coordinates, output_coordinates):
  """ Given a point in the input coordinate system, the vertices in the input coordinate system to search through, the 2D input coordinates and
      the 3D output coordinates, return the coordinates corresponding to the vertex in the vertices we searched through that is closest to the vertex we input.

      NOTE: This method is used only for emergencies when we cannot find a corresponding boundary edge or triangle when aligning vertices.
      It should not be called often.

    :Parameters:
      point: list of float pair. The 2D coordinates of the point whose closest vertex coordinate we want to find.
      vertices: range object from 0 to the total number of vertices. The vertices (in indices) that we want to search through.
      input_coordinates: list of float pairs. The 2D input coordinate system where the point and vertices lie.
      output_coordinates: list of float triples. The 3D output coordinate system we want to align the point to.

    :Returns:
      [closest_vertex, [x, y, z]]: list of int and float triple list. 
      [x, y, z] corresponds to the coordinates of the vertex in the output coordinates closest to this point.
  """
  closest_vertex = 0
  for candidate_vertex in vertices:
    if linalg.norm(array(point) - array(input_coordinates[candidate_vertex])) < linalg.norm(array(point)-array(input_coordinates[closest_vertex])):
      closest_vertex = candidate_vertex
  x, y, z = output_coordinates[closest_vertex][0], output_coordinates[closest_vertex][1], output_coordinates[closest_vertex][2]
  return [closest_vertex, [x, y, z]]

def getTriangleContainingVertex(vertex, triangles):
  """ Given a vertex and the corresponding triangulation it belongs to, return the index of the first triangle that contains the vertex.

    :Parameters:
      vertex : int. A vertex within a triangulation.
      triangles : list of list of int triples. The list of all triangles.

    :Returns:
      int. The index of the triangle that contains the vertex.
  """
  for triangle_i, triangle in enumerate(triangles):
    if vertex in triangle:
      return triangle_i

def getNextNeighbourhood(animal_obj, current_triangles, traversed_triangles):
  """ Given an animal object, a set of triangles whose neighbours we want to get and a set of inner triangles or edges 
      we have already traversed, find the next layer of neighbour triangles that we have not yet traversed.

    :Parameters:
      animal_obj: the animal object we are looking at.
      current_triangles: int set. The set of the indices of triangles whose neighbours we want to find.
      traversed_triangles: int set. The set of indices of triangles which we have already traversed.

    :Returns:
      int set. the set of all triangles that are in the outer neighbourhood
  """
  #initialise return set
  all_adjacent_triangles = set()

  #use the triangle-triangle adjacency array to find neighbouring triangles
  triangle_triangle_adjacency_array = animal_obj.getTriangleTriangleAdjacency()

  for triangle_i in current_triangles:
    #update all adjacent triangles with the triangles adjacent to each triangle
    adjacent_triangles = triangle_triangle_adjacency_array[triangle_i]
    all_adjacent_triangles.update(adjacent_triangles)

  #remove -1 (indicating that no triangle is adjacent to that edge) and traversed triangles from all the adjacent triangles we've found
  all_adjacent_triangles.difference_update(traversed_triangles)
  all_adjacent_triangles.discard(-1)

  return all_adjacent_triangles
  
def getAlignedCoordinates(animal_obj_0, animal_obj_1, theta):
  """ Calculates the vertex coordinates for the triangulation of Animal 1 aligned to the triangulation of Animal 0 by factoring
    through their respective conformal flattenings and applyling a rotation of angle theta.

    :Parameters:
      animal_obj : animal object, initialized with regular/flattened coordinates and triangulation set/updated
      theta : float with value between 0 and pi, an angle of rotation

    :Returns:
      list of triples of floats, specifying the x-, y-, and z-coordinates of the vertices of the triangulation of Animal 1 aligned to
      the triangulation of Animal 0
  """
  
  #store relevant parameters
  num_verts_0 = animal_obj_0.getNumVerts()
  num_verts_1 = animal_obj_1.getNumVerts()
  regular_coordinates_0 = animal_obj_0.getRegularCoordinates()
  flat_coordinates_0 = animal_obj_0.getFlattenedCoordinates()
  flat_coordinates_1 = animal_obj_1.getFlattenedCoordinates()
  triangles_0 = animal_obj_0.getTriangulation()
  num_triangles_0 = animal_obj_0.getNumTriangles() 
  boundary_vertices_1 = list(animal_obj_1.getBoundaryVertices())
  boundary_edges_0 = animal_obj_0.getBoundaryEdges()
  num_edges_0 = len(boundary_edges_0)
  
  #given the bfs ordering of vertices, store the first vertex and the rest of the list separately
  bfs_ordering, bfs_ancestors = animal_obj_1.getInteriorVertexBFS()
  first_vertex, *v_traversal_1 = bfs_ordering
  #initialize return list with triples of -1
  aligned_coordinates_1 = [[-1,-1,-1]] * num_verts_1
  #initialise dictionary that maps each vertex index of Animal 1 to the triangle index of Animal 0
  vertex_to_triangle_map = {}

  # ================= 1. FIND THE COORDINATES FOR THE FIRST INTERIOR VERTEX VIA BRUTE FORCE TRIANGLE SEARCH ====================== 
  #rotate the flattened coordinates of the first vertex
  first_rotated_coordinate = rotation(flat_coordinates_1[first_vertex], theta)

  #search through all the triangles in the triangulation of Animal 0 for one whose flattened coordinates contain the first vertex
  triangle_coordinate_pair = searchForAlignedCoordinate(first_rotated_coordinate, triangles_0, range(num_triangles_0), flat_coordinates_0, regular_coordinates_0)
  #if we can't find a triangle for the first vertex, set the triangle-coordinate-pair as the closest vertex and the first triangle containing that vertex
  if triangle_coordinate_pair == []:
    print("WARNING: Central vertex in Animal 1 is not contained in any triangle in Animal 0.")
    closest_vertex, closest_vertex_coordinate = findClosestVertex(first_rotated_coordinate, range(num_verts_0), flat_coordinates_0, regular_coordinates_0)
    triangle_i = getTriangleContainingVertex(closest_vertex, triangles_0)
    triangle_coordinate_pair = [triangle_i, closest_vertex_coordinate]

  #add the index of the triangle we found to our vertex-to-triangle map and add the aligned coordinate to return list
  vertex_to_triangle_map[first_vertex] = triangle_coordinate_pair[0]
  aligned_coordinates_1[first_vertex]  = triangle_coordinate_pair[1]

  # ================ 2. FIND THE CORRESPONDING COORDINATES FOR THE REST OF THE INTERIOR VERTICES VIA TRIANGLE BFS ================
  for vertex in v_traversal_1:
    #rotate the flattened coordinates of this vertex and get the parent of this vertex from our BFS of interior vertices
    rotated_coordinate = rotation(flat_coordinates_1[vertex], theta)
    parent_vertex = bfs_ancestors[vertex]
    triangle_coordinate_pair = []

    #initialize what we need to kickstart the while loop - traversed triangles and current list of triangles to search
    #we start by searching the triangle corresponding to this vertex's parent
    traversed_triangles = set()
    current_triangle_indices = {vertex_to_triangle_map[parent_vertex]}
    current_triangles = [triangles_0[vertex_to_triangle_map[parent_vertex]]]

    while triangle_coordinate_pair == []:
      #if we haven't found a matching triangle after searching all of them, assign the closest vertex and the first triangle containing that vertex
      if len(traversed_triangles) == num_triangles_0:
        print("WARNING: no triangle found for interior vertex " + str(vertex) + ". Assigning closest vertex instead.")
        closest_vertex, closest_vertex_coordinate = findClosestVertex(rotated_coordinate, range(num_verts_0), flat_coordinates_0, regular_coordinates_0)
        triangle_i = getTriangleContainingVertex(closest_vertex, triangles_0)
        triangle_coordinate_pair = [triangle_i, closest_vertex_coordinate]
        break

      #check if our rotated coordinate is in the current triangles
      triangle_coordinate_pair = searchForAlignedCoordinate(rotated_coordinate, current_triangles, current_triangle_indices, flat_coordinates_0, regular_coordinates_0)

      #update values for next iteration - add the triangles we just traversed, and set the current triangles to their neighbours
      traversed_triangles = traversed_triangles.union(current_triangle_indices)
      current_triangle_indices = getNextNeighbourhood(animal_obj_0, current_triangle_indices, traversed_triangles)
      current_triangles = [triangles_0[i] for i in current_triangle_indices]

    #add the index of the triangle we found to our vertex-to-triangle map, and add the aligned coordinate to return list
    vertex_to_triangle_map[vertex] = triangle_coordinate_pair[0]
    aligned_coordinates_1[vertex] = triangle_coordinate_pair[1]

  # ========================= 3. FIND THE THE CORRESPONDING COORDINATES FOR THE BOUNDARY VERTICES ==============================
  #initialise the root edge
  root_edge = 0

  for vertex in boundary_vertices_1:
    #initialize the ordering of boundary edges to search through: [root_edge, root_edge+1, ... , num_edges-1, 0, 1, ... , root_edge-1]
    boundary_edge_indices = list(range(num_edges_0))
    edge_search_ordering = boundary_edge_indices[root_edge:] + boundary_edge_indices[:root_edge]
    edges_searched = 0

    #rotate the flattened coordinates of this vertex by theta
    rotated_coordinate = rotation(flat_coordinates_1[vertex],theta)
    edge_coordinate_pair = []

    while edge_coordinate_pair == []:
      #if we haven't found an edge after searching all the edges, assign the same root edge and the closest vertex coordinate
      if edges_searched == num_edges_0:
        print("WARNING: BOUNDARY FAILURE: Could not find boundary edge for boundary vertex " + str(vertex) + ". Assigning closest vertex instead.")
        closest_vertex_coordinate = findClosestVertex(rotated_coordinate, range(num_verts_0), flat_coordinates_0, regular_coordinates_0)[1]
        edge_coordinate_pair = [root_edge, closest_vertex_coordinate]
        break

      #assign the edge to search based on how many edges we've searched so far
      edge_index_to_search = edge_search_ordering[edges_searched]
      edge_to_search = boundary_edges_0[edge_index_to_search]
      #search through one boundary edge at a time to find the boundary edge that the this boundary vertex is mapped to
      edge_coordinate_pair = searchForAlignedCoordinate(rotated_coordinate, [edge_to_search], [edge_index_to_search], flat_coordinates_0, regular_coordinates_0)
      #update the edges searched for the next iteration
      edges_searched += 1

    #update root edge with the index of the edge we found and assign aligned coordinates to return list
    root_edge = edge_coordinate_pair[0]
    aligned_coordinates_1[vertex] = edge_coordinate_pair[1]

  return aligned_coordinates_1

def area(p, q, r):
  #this is a helper method for the distortionEnergy and computeOneCSD methods below. It calculates the
  #area of the triangle spanned by three points in R^2 or R^3.
  if len(p) == 2:
    p.append(0)
    q.append(0)
    r.append(0)
  x = []
  y = []
  for i in range(3):
    x.append(q[i]-p[i])
    y.append(r[i]-p[i])
  return 0.5*((x[1]*y[2]-x[2]*y[1])**2+(x[2]*y[0]-x[0]*y[2])**2+(x[0]*y[1]-x[1]*y[0])**2)**0.5

def distortionEnergy(animal_obj_0, animal_obj_1, rho):
  """ Calculates the elastic energy required to stretch the triangulation of Animal 0 onto the triangulation of Animal 1 
    via the conformal mapping obtained by factoring through their respective conformal flattenings and applyling a rotation 
    of angle rho.

    :Parameters:
      animal_obj_0/1 : animal objects, initialized with regular/flattened coordinates and triangulation set/updated
      rho : float with value between 0 and pi, an angle of rotation

    :Returns:
      float, specifying the elastic energy required to align the triangulation of Animal 1 that of Animal 0
  """

  #store relevant parameters
  num_verts = animal_obj_0.getNumVerts()
  regular_coordinates = animal_obj_0.getRegularCoordinates()
  aligned_coordinates = getAlignedCoordinates(animal_obj_1,animal_obj_0,rho)
  triangles = animal_obj_0.getTriangulation()

  #calculate four matrices whose entries correspond to pairs of vertices in the triangulation of Animal 0
  #with values given by (1) the number of triangles containing that pair of vertices, (2) the length of the
  #edge between them (if one exists) in the regular triangulation of Animal 0, (3) the length of the edge
  #between them (if one exists) in the triangulation of Animal 0 aligned to that of Animal 1 via the rotation
  #rho, and (4) the sum of the areas of the triangles in the regular triangulation of Animal 0 containing the
  #pair of vertices.
  incidence_matrix = [[[0 for k in range(4)] for j in range(num_verts)] for i in range(num_verts)]

  for triangle in triangles:
    sorted_triangle = sorted(triangle)
    u = sorted_triangle[0]
    v = sorted_triangle[1]
    w = sorted_triangle[2]

    incidence_matrix[v][u][0] += 1
    incidence_matrix[v][u][1] = linalg.norm(array(regular_coordinates[v])-array(regular_coordinates[u]))
    incidence_matrix[v][u][2] = linalg.norm(array(aligned_coordinates[v])-array(aligned_coordinates[u]))
    incidence_matrix[v][u][3] += area(regular_coordinates[u],regular_coordinates[v],regular_coordinates[w])

    incidence_matrix[w][u][0] += 1
    incidence_matrix[w][u][1] = linalg.norm(array(regular_coordinates[w])-array(regular_coordinates[u]))
    incidence_matrix[w][u][2] = linalg.norm(array(aligned_coordinates[w])-array(aligned_coordinates[u]))
    incidence_matrix[w][u][3] += area(regular_coordinates[u],regular_coordinates[v],regular_coordinates[w])

    incidence_matrix[w][v][0] += 1
    incidence_matrix[w][v][1] = linalg.norm(array(regular_coordinates[w])-array(regular_coordinates[v]))
    incidence_matrix[w][v][2] = linalg.norm(array(aligned_coordinates[w])-array(aligned_coordinates[v]))
    incidence_matrix[w][v][3] += area(regular_coordinates[u],regular_coordinates[v],regular_coordinates[w])

  #initialize the return value
  alignment_value = 0

  #sum the squares of the conformal stretching factors of the alignment over each edge in the triangulation
  for i in range(num_verts):
    for j in range(i):
      if incidence_matrix[i][j][0] == 2:
        alignment_value += (incidence_matrix[i][j][3]/3.0)*(incidence_matrix[i][j][2]/incidence_matrix[i][j][1]-1.0)**2

  return alignment_value**0.5

def symmetricDistortionEnergy(animal_obj_0, animal_obj_1, rho):
  """ Calculates the symmetric distortion energy required to stretch the triangulation of Animal 0 onto the 
    triangulation of Animal 1 and vice versa via the conformal mapping obtained by factoring through their 
    respective conformal flattenings and applyling a rotation of angle rho.

    :Parameters:
      animal_obj_0/1 : animal objects, initialized with regular/flattened coordinates and triangulation set/updated
      rho : float with value between 0 and pi, an angle of rotation

    :Returns:
      float, specifying the symmetric distortion energy required to align the triangulations of Animals 0 and 1
  """
  return distortionEnergy(animal_obj_0, animal_obj_1, rho) + distortionEnergy(animal_obj_1, animal_obj_0, -rho)

def optimalRotation(animal_obj_0,animal_obj_1):
  """ Calculates the optimal rotation of the unit disk that minimizes the symmetric distortion energy between 
    the triangulations of two animals

    :Parameters:
      animal_obj_0/1 : animal objects, initialized with regular/flattened coordinates and triangulation set/updated

    :Returns:
      float, specifying an angle between 0 and pi
  """

  #define a single variable function for a fixed pair of animals that takes an angle as input and outputs the
  #corresponding symmetric distortion energy
  def optimization_function(x):
    return symmetricDistortionEnergy(animal_obj_0,animal_obj_1,x)

  return minimize_scalar(optimization_function,bounds=(0,pi),method='Brent',tol=1.0).x


####################################################################################  
### METHODS FOR CALCULATING CONFORMAL SPATIOTEMPORAL DISTANCES BETWEEN HEAT MAPS ###
#################################################################################### 

def computeOneCSD(animal_obj_0, animal_obj_1, fullmode=False, outdir=None):
  """ Computes the Conformal Spatiotemporal Distance between the heatmaps of two animals

    :Parameters:
      animal_obj_0/1 : animal objects, initialized with regular/flattened coordinates and triangulation set/updated
      fullmode : Boolean, writes triangulations and their corresponding flattenings and alignments to .OFF files if True
      outdir : string, specifying directory to save .OFF files if fullmode is True

    :Returns:
      float, specifying the Conformal Spatiotemporal Distance between the heatmaps of two animals
  """

  #check that a directory is specified if fullmode is true
  if fullmode and outdir == None:
    throwError("Full mode requiers the path to output direcotry")

  #notify user of progress
  print("Measuring conformal spatiotemporal distance between heat maps of %s and %s..." % (animal_obj_0.getName(),animal_obj_1.getName()))

  #calculate the optimal rotation for aligning the triangulations of the two animals
  #theta = optimalRotation(animal_obj_0,animal_obj_1)
  theta = 0

  #store relevant parameters.  Note that we assume both animal observations have the same dimensions
  x_dim, y_dim = animal_obj_0.getDims()
  z_dim = getZDim(animal_obj_0)
  num_verts_0 = animal_obj_0.getNumVerts()
  regular_coordinates_0 = animal_obj_0.getRegularCoordinates()
  aligned_coordinates_0 = getAlignedCoordinates(animal_obj_1,animal_obj_0,theta)
  triangles_0 = animal_obj_0.getTriangulation()
  num_verts_1 = animal_obj_1.getNumVerts()
  regular_coordinates_1 = animal_obj_1.getRegularCoordinates()
  aligned_coordinates_1 = getAlignedCoordinates(animal_obj_0,animal_obj_1,-theta)
  triangles_1 = animal_obj_1.getTriangulation()

  #Save the triangulation data in .OFF files if fullmode is True
  if fullmode:
    write.writeOFF(animal_obj_0, regular_coordinates_0, outdir, "heatmap_%s_regular.off" % animal_obj_0.getName())
    write.writeOFF(animal_obj_1, regular_coordinates_1, outdir, "heatmap_%s_regular.off" % animal_obj_1.getName())
    flattened_coordinates_0 = [coord + [0] for coord in animal_obj_0.getFlattenedCoordinates()]
    flattened_coordinates_1 = [coord + [0] for coord in animal_obj_1.getFlattenedCoordinates()]
    write.writeOFF(animal_obj_0, flattened_coordinates_0, outdir, "heatmap_%s_flat.off" % animal_obj_0.getName())
    write.writeOFF(animal_obj_1, flattened_coordinates_1, outdir, "heatmap_%s_flat.off" % animal_obj_1.getName())
    write.writeOFF(animal_obj_0, aligned_coordinates_0, outdir, "heatmap_%s_aligned_to_%s.off" % (animal_obj_0.getName(),animal_obj_1.getName()))
    write.writeOFF(animal_obj_1, aligned_coordinates_1, outdir, "heatmap_%s_aligned_to_%s.off" % (animal_obj_1.getName(),animal_obj_0.getName()))

  #calculate the distance between the aligned surfaces 
  difference_val_0 = 0
  dA = 0
  for vertex in range(num_verts_1):
    for triangle in triangles_1:
      if vertex in triangle:
        dA += area(regular_coordinates_1[triangle[0]][0:2],regular_coordinates_1[triangle[1]][0:2],regular_coordinates_1[triangle[2]][0:2])/3.0
    difference_val_0 += dA*(aligned_coordinates_1[vertex][2]-regular_coordinates_1[vertex][2])**2

  difference_val_1 = 0
  dA = 0
  for vertex in range(num_verts_0):
    for triangle in triangles_0:
      if vertex in triangle:
        dA += area(regular_coordinates_0[triangle[0]][0:2],regular_coordinates_0[triangle[1]][0:2],regular_coordinates_0[triangle[2]][0:2])/3.0
    difference_val_1 += dA*(aligned_coordinates_0[vertex][2]-regular_coordinates_0[vertex][2])**2

  distance = (difference_val_0**0.5+difference_val_1**0.5)/(2*z_dim*x_dim*y_dim)

  #record distance in terminal
  print("LOG: distance  between aligned surfaces of %s and %s: %.3f" % (animal_obj_0.getName(), animal_obj_1.getName(), distance))

  return distance

def computeAllCSD(animal_list):
  """ Computes the Conformal Spatiotemporal Distances between the heatmaps of all pairs in list of animals

    :Parameters:
      animal_list : list of animal objects, initialized with regular/flattened coordinates and triangulation set/updated

    :Returns:
      2D array of floats, specifying the Conformal Spatiotemporal Distance between the heatmaps of each pair of animals in the input list
  """

  #initialize return array
  num_animals = len(animal_list)
  Dists = [['' for i in range(num_animals)] for j in range(num_animals)]

  #calculate the CSD between each pair of animals in the input list
  for i in range(num_animals):
    for j in range(i+1, num_animals):
      Dists[i][j] = computeOneCSD(animal_list[i],animal_list[j])
      
  return Dists