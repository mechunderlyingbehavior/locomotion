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

from math import ceil, exp, log, sin, asin, pi, acosh, cosh, sinh, cos, acos, atanh, tanh
from numpy import min, mean, std, array, linalg, dot, cross
from scipy.optimize import minimize_scalar
import locomotion.write as write
import locomotion.animal as animal
from locomotion.animal import throwError

#Static Variables
PERTURBATION = 0.000000001
CONFORMAL_FACTOR = 1.45
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
  animal_obj.setConformalFactor(CONFORMAL_FACTOR)
  animal_obj.setTolerance(TOLERANCE)
  
  print("Calculating heatmap for %s..." % animal_obj.getName())
  
  #calculuate heatmap
  frequencies = getFrequencies(animal_obj, start_time, end_time)

  print("Calculating triangulation for %s..." % animal_obj.getName())
  
  #get and record vertices
  original_coordinates = getVertexCoordinates(animal_obj, frequencies)
  animal_obj.setNumVerts(len(original_coordinates))
  animal_obj.setRegularCoordinates(original_coordinates)
  
  #record boundary vertices
  boundary_vertices = getBoundaryVertices(animal_obj)
  
  #get and record triangles
  triangles = getTriangles(animal_obj)
  while hasHoles(animal_obj, triangles, boundary_vertices): #check for and patch any holes
    triangles = patchHoles(animal_obj, triangles, boundary_vertices)
  animal_obj.setTriangulation(triangles)

  #calculate and store colors for output file
  colors = getColors(animal_obj)
  animal_obj.setColors(colors)

  print("Calculating flattened coordinates for %s..." % animal_obj.getName())
  
  #calculate conformal flattening of triangulation to unit disk
  flowers = getFlowers(animal_obj, boundary_vertices) #find the neighbors of each vertex and arrange them in counterclockwise order
  initial_radii = initializeRadii(animal_obj, boundary_vertices) #initialize circle packing of triangulation
  radii = getCirclePacking(animal_obj, initial_radii, flowers, boundary_vertices) #calculate maximal circle packing of triangulation in unit disk
  
  #calculate and record flattened coordinates of triangulation
  center_vertex = getCenterVertex(animal_obj) #identify center vertex of triangulation
  flattened_coordinates = getFlatCoordinates(animal_obj, radii, flowers, center_vertex, boundary_vertices) 
  animal_obj.setFlattenedCoordinates(flattened_coordinates)

  
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
  x_dim, y_dim = animal_obj.getDims()
  num_x_grid,num_y_grid = animal_obj.getNumGrids()
  conformal_factor = animal_obj.getConformalFactor()

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

  #check edge lengths in the induced triangulation and subdivide as necessary
  for i in range(num_x_grid-1): #all vertices except those on the right and upper boundaries
    for j in range(num_y_grid-1):
      dist = linalg.norm(array([i*grid_size, j*grid_size, freqs[i][j]])-array([i*grid_size, (j+1)*grid_size, freqs[i][j+1]])) #vertical edges
      if dist > conformal_factor*grid_size:
        num_segments = int(ceil(dist/grid_size))
        for k in range(1,num_segments):
          coordinates.append([i*grid_size, (j+float(k)/num_segments)*grid_size, freqs[i][j] + float(k)/num_segments*(freqs[i][j+1]-freqs[i][j])])
      dist = linalg.norm(array([i*grid_size,j*grid_size,freqs[i][j]])-array([(i+1)*grid_size, j*grid_size, freqs[i+1][j]])) #horizontal edges
      if dist > conformal_factor*grid_size:
        num_segments = int(ceil(dist/grid_size))
        for k in range(1,num_segments):
          coordinates.append([(i+float(k)/num_segments)*grid_size,j*grid_size,freqs[i][j]+float(k)/num_segments*(freqs[i+1][j]-freqs[i][j])])
      dist = linalg.norm(array([i*grid_size, j*grid_size,freqs[i][j]])-array([(i+1)*grid_size,(j+1)*grid_size, freqs[i+1][j+1]])) #diagonal edges
      if dist > conformal_factor*grid_size:
        num_segments = int(ceil(dist/grid_size))
        for k in range(1,num_segments):
          coordinates.append([(i+float(k)/num_segments)*grid_size,(j+float(k)/num_segments)*grid_size,freqs[i][j]+float(k)/num_segments*(freqs[i+1][j+1]-freqs[i][j])])  
  for i in range(num_x_grid-1): #upper boundary vertices
    j = num_y_grid-1
    dist = linalg.norm(array([i*grid_size, j*grid_size, freqs[i][j]])-array([(i+1)*grid_size, j*grid_size, freqs[i+1][j]])) #horizontal edges
    if dist > conformal_factor*grid_size:
      num_segments = int(ceil(dist/grid_size))
      for k in range(1,num_segments):
        coordinates.append([(i+float(k)/num_segments)*grid_size,j*grid_size,freqs[i][j]+float(k)/num_segments*(freqs[i+1][j]-freqs[i][j])])
  for j in range (num_y_grid-1): #right boundary vertices
    i = num_x_grid-1
    dist = linalg.norm(array([i*grid_size, j*grid_size, freqs[i][j]])-array([i*grid_size, (j+1)*grid_size, freqs[i][j+1]])) #vertical edges
    if dist > conformal_factor*grid_size:
      num_segments = int(ceil(dist/grid_size))
      for k in range(1,num_segments):
        coordinates.append([i*grid_size,(j+float(k)/num_segments)*grid_size,freqs[i][j]+float(k)/num_segments*(freqs[i][j+1]-freqs[i][j])])

  # sort the coordinates by z-value (third entry)
  coordinates.sort(key=lambda c: -c[2]) 

  return coordinates


def getBoundaryVertices(animal_obj):
  """ Returns the subset of boundary vertices from a list of vertex coordinates

    :Parameters:
      animal_obj : animal object, initialized with regular coordinates updated

    :Returns:
      list of ints specifying the indices of the boundary vertices within the 
      regular coordinates of the animal in counterclockwise order starting with
      the bottom left corner
  """
  
  #gather relevant parameters
  grid_size = animal_obj.getGridSize()
  x_dim, y_dim = animal_obj.getDims()
  coordinates = animal_obj.getRegularCoordinates()

  #initialize lists for each edge of the boundary rectangle
  lower_edge, upper_edge, left_edge, right_edge = [], [], [], []

  #iterate through list of vertex coordinates and sort boundary vertices into their respective edge lists
  for c in coordinates:
    if c[0] == 0.0:
      left_edge.append(c)
    if c[0] == x_dim-grid_size:
      right_edge.append(c)
    if c[1] == 0.0:
      lower_edge.append(c)
    if c[1] == y_dim-grid_size:
      upper_edge.append(c)

  #initialize return list
  boundary_vertices = []  

  #arrange boundary vertices in counter-clockwise order
  lower_edge.sort(key=lambda c: c[0])
  upper_edge.sort(key=lambda c: c[0])
  left_edge.sort(key=lambda c: c[1])
  right_edge.sort(key=lambda c: c[1])
  for i in range(len(lower_edge)-1):
    boundary_vertices.append(coordinates.index(lower_edge[i]))
  for i in range(len(right_edge)-1):
    boundary_vertices.append(coordinates.index(right_edge[i]))
  for i in range(len(upper_edge)-1):
    boundary_vertices.append(coordinates.index(upper_edge[-i-1]))
  for i in range(len(left_edge)-1):
    boundary_vertices.append(coordinates.index(left_edge[-i-1]))

  return boundary_vertices


def getCircumcircle(a, b, c, tolerance):
  """ Helper method for getTriangles method below. Returns the circumcenter 
    and circumradius of three points on the plane.

    :Parameters:
      a,b,c : three pairs of floats
        coordinates of three points in the plane
      tolerance : float
        small positive number used to avoid division by zero

    :Returns:
      a list of length two (coordinates of a point in the plane, namely the circumcenter 
      of the three input points) and a float (the circumradius of the three input points)
  """
  
  d = 2.0*(a[0]*(b[1]-c[1])+b[0]*(c[1]-a[1])+c[0]*(a[1]-b[1]))
  if d < tolerance:
    d = tolerance
  center = [1.0/d*((a[0]**2+a[1]**2)*(b[1]-c[1])+(b[0]**2+b[1]**2)*(c[1]-a[1])+(c[0]**2+c[1]**2)*(a[1]-b[1])), \
            1.0/d*((a[0]**2+a[1]**2)*(c[0]-b[0])+(b[0]**2+b[1]**2)*(a[0]-c[0])+(c[0]**2+c[1]**2)*(b[0]-a[0])),0]
  radius = linalg.norm(array([a[0],a[1],0])-array(center))
  return center, radius


def getTriangles(animal_obj):
  """ Computes a Delaunay triangulation on the regular coordinates of an animal using a version 
    of the Bowyer-Watson algorithm

    :Parameters:
      animal_obj : animal object, initialized with regular coordinates set/updated

    :Returns:
      list of triples of ints, specifying the indices of the vertices for each triangle in
      in the triangulation of a surface
  """
  #store relevant parameters
  x_dim, y_dim = animal_obj.getDims()
  coordinates = animal_obj.getRegularCoordinates()
  tolerance = animal_obj.getTolerance()
  num_verts = animal_obj.getNumVerts()

  #append bounding triangle (three points in the z=0 plane whose convex hull contains the bounding rectangle of x- and y-coordinates
  coordinates.append([-5,-5,0])
  coordinates.append([x_dim+y_dim+2,-1,0])
  coordinates.append([-1,x_dim+y_dim+2,0])

  #initialize triangle list with bounding triangle
  triangles = [[num_verts,num_verts+1,num_verts+2]]

  #increment through the list of vertices 
  for vertex in range(num_verts):

    #for each new vertex, v, find all the current triangles whose circumcircles contain v 
    bad_triangles = []
    for triangle in triangles:
      triangle_center, triangle_radius = getCircumcircle(coordinates[triangle[0]], coordinates[triangle[1]], coordinates[triangle[2]], tolerance)
      if linalg.norm(array([coordinates[vertex][0],coordinates[vertex][1],0])-array(triangle_center)) < triangle_radius:
        bad_triangles.append(triangle)

    #find the boundary polygon of the "bad" triangles whose circumcircles contain v
    polygon = []
    edge_count = [[0 for j in range(i)] for i in range(num_verts+3)]
    for triangle in bad_triangles:
      sorted_triangle = sorted(triangle)
      edge_count[sorted_triangle[2]][sorted_triangle[1]] += 1
      edge_count[sorted_triangle[2]][sorted_triangle[0]] += 1
      edge_count[sorted_triangle[1]][sorted_triangle[0]] += 1
    for v in range(num_verts+3):
      for w in range(v):
        if edge_count[v][w] == 1:
          polygon.append([v,w])

    #remove all the "bad" triangles in the triangulation and replace them with a triangulation of the boundary polygon centered at v
    for triangle in bad_triangles:
      triangles.remove(triangle)
    for edge in polygon:
      if cross(array(coordinates[edge[0]])-array(coordinates[vertex]),array(coordinates[edge[1]])-array(coordinates[vertex]))[2] > -tolerance:
        triangle = [vertex,edge[0],edge[1]]
      else:
        triangle = [vertex,edge[1],edge[0]]
      triangles.append(triangle)

  #remove all the triangles that contain one of the three bounding triangle vertices
  removal = []
  for triangle in triangles:
    if num_verts in triangle or num_verts+1 in triangle or num_verts+2 in triangle:
      removal.append(triangle)
  for triangle in removal:
    triangles.remove(triangle)
    
  #remove the three bounding triangle vertices
  coordinates.remove([-5,-5,0])
  coordinates.remove([x_dim+y_dim+2,-1,0])
  coordinates.remove([-1,x_dim+y_dim+2,0])

  return triangles


def hasHoles(animal_obj, triangles, boundary_vertices):
  """ All of the triangulations in this project are assumed to be simply connected.  This method 
    is used identify any holes that may have arisen from calculation error.  Specifically, it 
    checks a triangulation for holes by searching for "boundary" edges that are not along a boundary 
    edge of the bounding rectangle.

    :Parameters:
      animal_obj : animal object, initialized with regular coordinates set/updated
      triangles : a list of triples of ints whose values are between 0 and one less than the number of regular vertices stored in the animal object
      boundary_vertices : list of ints whose values are between 0 and one less than the number of regular vertices stored in the animal object

    :Returns:
      bool (true if the method found a non-boundary "boundary" edge and false otherwise)
  """

  #store relevant parameter
  num_verts = animal_obj.getNumVerts()

  #initial return value
  answer = False

  #calculate incidence matrix between edges and triangles in input
  check_matrix = [[0 for j in range(i)] for i in range(num_verts)]
  for triangle in triangles:
    sorted_triangle = sorted(triangle)
    check_matrix[sorted_triangle[2]][sorted_triangle[1]] += 1
    check_matrix[sorted_triangle[2]][sorted_triangle[0]] += 1
    check_matrix[sorted_triangle[1]][sorted_triangle[0]] += 1

  #count the number of interior edges that belong to exactly one triangle 
  count = 0
  for v in range(num_verts):
    for w in range(v):
      if check_matrix[v][w] == 1 and v not in boundary_vertices and w not in boundary_vertices:
        print("!!!Warning: Triangulation has holes!!! (%d,%d)" % (v,w))
        count += 1
        
  #update return value based on counting result
  if count > 1:
    answer = True
    
  return answer


def patchHoles(animal_obj, triangles, boundary_vertices):
  """ All of the triangulations in this project are assumed to be simply connected.  This method 
    is used patch any holes that may arise from calculation error.  Specifically, it identifies 
    adjacent pairs of non-boundary edges that each belong to exactly one triangle in a propsed 
    triangulation and glues in a triangle to fill in or "patch" the corresponding hole. 

    :Parameters:
      animal_obj : animal object, initialized with regular coordinates set/updated
      triangles : a list of triples of ints whose values are between 0 and one less than the number of regular vertices stored in the animal object
      boundary_vertices : list of ints whose values are between 0 and one less than the number of regular vertices stored in the animal object

    :Returns:
      list of triples of ints, specifying the indices of the vertices for each triangle in
      in the triangulation of a surface
  """

  #store relevant paramters
  num_verts = animal_obj.getNumVerts()
  coordinates = animal_obj.getRegularCoordinates()
  tolerance = animal_obj.getTolerance()

  #calculate incidence matrix between edges and triangles in input
  check_matrix = [[0 for j in range(i)] for i in range(num_verts)]
  for triangle in triangles:
    sorted_triangle = sorted(triangle)
    check_matrix[sorted_triangle[2]][sorted_triangle[1]] += 1
    check_matrix[sorted_triangle[2]][sorted_triangle[0]] += 1
    check_matrix[sorted_triangle[1]][sorted_triangle[0]] += 1

  #gather the interior edges that belong to exactly one triangle
  bad_edges = []
  for w in range(num_verts-1):
    for v in range(w+1,num_verts):
      if check_matrix[v][w] == 1 and v not in boundary_vertices and w not in boundary_vertices:
        bad_edges.append([w,v])
        
  #if there are more than two "bad" edges, append a new triangle with the first pair of adjacent "bad" edges using the correct orientation
  if len(bad_edges) > 1:
    if cross(array(coordinates[bad_edges[0][1]])-array(coordinates[bad_edges[0][0]]),array(coordinates[bad_edges[1][1]])-array(coordinates[bad_edges[0][0]]))[2] > -tolerance:
      new_triangle = [bad_edges[0][0],bad_edges[0][1],bad_edges[1][1]]
    else:
      new_triangle = [bad_edges[0][0],bad_edges[1][1],bad_edges[0][1]]
    triangles.append(new_triangle)

    #notify user that a patch was made
    print("Added patch: %s" % str(new_triangle))
    
  return triangles
  

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


####################################################################################  
### METHODS FOR CALCULATING CONFORMAL FLATTENINGS OF TRIANGULATIONS TO UNIT DISK ###
#################################################################################### 


def getFlowers(animal_obj, boundary_vertices):
  """ Calculuates the flower (list of neighbors), in counter-clockwise order, of each vertex in the triangulation of an animal

    :Parameters:
      animal_obj : animal object, initialized with regular coordinates and triangulation set/updated
      boundary_vertices : list of ints whose values are between 0 and one less than the number of regular vertices stored in the animal object

    :Returns:
      list of lists of ints, specifying the indices of the vertices neighboring each vertex, in counter-clockwise order, of the triangulation
      associated to an animal
  """

  #gather relevant parameters
  num_verts = animal_obj.getNumVerts()
  triangles = animal_obj.getTriangulation()

  #initialize return list
  flowers = [[] for vertex in range(num_verts)]

  #iterate through the vertices of the animal object. Flowers of interior vertices will be cycles while
  #flowers of boundary vertices will be ordered lists with distinct start/end vertices.
  for vertex in range(num_verts):

    #initialize first and last neighbors of the current vertex
    first_neighbor, last_neighbor = 0, 0

    #the first and last neighbors of each boundary_vertex is already known since the boundary_vertices are stored in counter-clockwise order
    if vertex in boundary_vertices: 
      first_neighbor = boundary_vertices[(boundary_vertices.index(vertex)+1) % len(boundary_vertices)]
      last_neighbor = boundary_vertices[(boundary_vertices.index(vertex)-1) % len(boundary_vertices)]

    #for interior vertices, set an arbitrary first neighbor by finding a triangle that contains the current vertex
    else:
      for triangle in triangles:
        if vertex in triangle:
          first_neighbor = triangle[(triangle.index(vertex)+2) % 3]
          last_neighbor = triangle[(triangle.index(vertex)+1) % 3]
          flowers[vertex].append(last_neighbor) #the last vertex is listed at the beginning and end of the return list for each interior vertex
          break 

    #append neighbors to return list sequentially starting with the first neighbor found above
    flowers[vertex].append(first_neighbor)

    #search for all the neighbors in counter-clockwise order by searching through the triangles
    #that contain the current vertex and the most recent "next" neighbor.  Keep going until the
    #next neighbor is the last neighbor.
    empty_search = False
    next_neighbor = first_neighbor #initialize next neighbor 
    while next_neighbor != last_neighbor and empty_search == False: 
      empty_search = True
      
      #search for a triangle that contains the current vertex and the next vertex
      for t in range(len(triangles)):
        if vertex in triangles[t] and next_neighbor in triangles[t]: 
          for k in range(3):
            
            #if the third vertex of the triangle has not already by appended to the return list, add it and update the next neighbor
            if triangles[t][k] != vertex and triangles[t][k] not in flowers[vertex]: 
              flowers[vertex].append(triangles[t][k])
              next_neighbor = triangles[t][k] #update the next neighbor to the most recent neighbor added
              empty_search = False
              
    #append the last neighbor to the return list
    if vertex not in boundary_vertices: 
      flowers[vertex].append(last_neighbor)
      
  return flowers


def initializeRadii(animal_obj,boundary_vertices):
  """ Initializes the radii for the circle packing of the triangulation associated to an animal.
    Different radii are assigned to interior and boundary vertices.

    :Parameters:
      animal_obj : animal object, initialized with regular coordinates and triangulation set/updated
      boundary_vertices : list of ints whose values are between 0 and one less than the number of regular vertices stored in the animal object

    :Returns:
      list of floats, specifying the initial radii to begin the circle packing algorithm for the triangulation associated to an animal 
  """

  #store relevant parameters
  num_verts = animal_obj.getNumVerts()

  #initialize return list
  radii = []

  #assign radius 0.001 to each boundary vertex and radius 0.5 to each interior vertex
  for vertex in range(num_verts):
    if vertex in boundary_vertices:
      radii.append(0.001)
    else:
      radii.append(0.5)
  
  return radii


def getAlpha(i , j , k , radii):
  #this is a helper method for the circle packing algorithm below 
  return 2*asin((radii[i]*(1-radii[j])/(1-radii[i]*radii[j])*(1-radii[k])/(1-radii[i]*radii[k]))**0.5)


def getAngle(a, b, c):
  #this is a helper method for the circle pacing algorithm below
  val = (cosh(b)*cosh(c)-cosh(a))/(sinh(b)*sinh(c))
  if val > 1:
    val = 1
  elif val < -1:
    val = -1
  return acos(val)


def getTheta(i , flower, radii):
  #this is a helper method for the circle pacing algorithm below
  theta = 0
  for k in range(len(flower)-1):
    theta += getAlpha(i , flower[k], flower[k+1], radii)
  return theta


def getCirclePacking(animal_obj, radii, flowers, boundary_vertices):
  """ Calculuates a maximal circle packing of the triangulation associated to an animal in the hyperbolic disk
    according to the algorithm presented in "A Circle Packing Algorithm" by Collins and Stephenson

    :Parameters:
      animal_obj : animal object, initialized with regular coordinates and triangulation set/updated
      radii : list of floats containing the initial radii for the circle packing algorithm
      flowers : list of lists of ints specifying the flower (neighbors in counter-clockwise order) of each vertex
      boundary_vertices : list of ints whose values are between 0 and one less than the number of regular vertices stored in the animal object

    :Returns:
      list of floats specifying the radii of the circles in a maximal circle packing of the triangulation associated to an animal
  """

  #store relevant parameters
  tolerance = animal_obj.getTolerance()
  perturb = animal_obj.getPerturbation()

  #run circle packing algorithm (see paper referenced above)
  error = tolerance+1
  while error > tolerance:
    error = 0
    for vertex in range(len(radii)):
      if vertex not in boundary_vertices:
        k = len(flowers[vertex])-1
        theta = getTheta(vertex,flowers[vertex],radii)
        beta = sin(theta/(2*k))
        delta = sin(pi/k)
        v_hat = (beta-radii[vertex]**0.5)/(beta*radii[vertex]-radii[vertex]**0.5)
        if v_hat < 0:
          v_hat = 0.0
        radii[vertex] = ((2*delta)/(((1-v_hat)**2+4*(delta**2)*v_hat)**0.5+(1-v_hat)))**2
        error += abs(theta-2*pi)
        
  return radii


def getCenterVertex(animal_obj):
  #this is a helper method for the getFlatCoordinates method below
  center_vertex = -1
  x_dim, y_dim = animal_obj.getDims()
  coordinates = animal_obj.getRegularCoordinates()
  for c in coordinates:
    if c[0] == x_dim/2.0 and c[1] == y_dim/2.0:
      center_vertex = coordinates.index(c)
  return center_vertex


def mobius(u, v, a, b):
  #this is a helper method for the getFlatCoordinates method below
  return [((u-a)*(a*u+b*v-1)+(v-b)*(a*v-b*u))/((a*u+b*v-1)**2+(a*v-b*u)**2), ((v-b)*(a*u+b*v-1)-(u-a)*(a*v-b*u))/((a*u+b*v-1)**2+(a*v-b*u)**2)]


def getFlatCoordinates(animal_obj, radii, flowers, center, boundary_vertices):
  """ Calculates the vertex coordinates for the triangulation of an animal from its corresponding circle packing in the unit disk

    :Parameters:
      animal_obj : animal object, initialized with regular coordinates and triangulation set/updated
      radii : list of floats containing the initial radii for the circle packing algorithm
      flowers : list of lists of ints specifying the flower (neighbors in counter-clockwise order) of each vertex
      center : int, index of the most central vertex of the triangulation
      boundary_vertices : list of ints whose values are between 0 and one less than the number of regular vertices stored in the animal object

    :Returns:
      list of pairs of floats, specifying the x- and y-coordinates of the vertices of a triangulation that has been conformally flattened
      to the unit disk
  """

  #store relevant parameters
  triangles = animal_obj.getTriangulation()
  tolerance = animal_obj.getTolerance()

  #initialize return list
  coordinates = [[] for r in radii]

  #initialize list of booleans to keep track of which circles have and have not been placed
  placed = [False for r in radii]

  #convert the radii from hyperbolic to euclidean distances
  adjusted_radii = []
  for radius in range(len(radii)):
    adjusted_radii.append(-0.5*log(radii[radius]))

  #place center vertex at the origin
  coordinates[center].append(0.0)
  coordinates[center].append(0.0)
  placed[center] = True

  #place first neighbor of center vertex on the x-axis 
  coordinates[flowers[center][0]].append(adjusted_radii[center]+adjusted_radii[flowers[center][0]])
  coordinates[flowers[center][0]].append(0.0)
  placed[flowers[center][0]] = True

  #place remaining neighbors of center vertex
  for neighbor in range(1,len(flowers[center])):
    if not placed[flowers[center][neighbor]]:
      coordinates[flowers[center][neighbor]].append(adjusted_radii[center]+adjusted_radii[flowers[center][neighbor]])
      coordinates[flowers[center][neighbor]].append(coordinates[flowers[center][neighbor-1]][1]+getAlpha(center,flowers[center][neighbor-1],flowers[center][neighbor],radii))
      placed[flowers[center][neighbor]] = True

  #place remaining vertices by searching through the triangles with exactly two vertices already placed and placing the third
  while False in placed:
    for triangle in triangles:
      count = 0
      A = 0
      for i in range(3):
        if placed[triangle[i]]:
          count += 1
        else:
          A = triangle[i]
      if count == 2:
        B = triangle[(triangle.index(A)+1)%3]
        C = triangle[(triangle.index(A)+2)%3]
        r1 = coordinates[C][0]
        r2 = coordinates[B][0]
        a = adjusted_radii[B]+adjusted_radii[C]
        b = adjusted_radii[A]+adjusted_radii[C]
        c = adjusted_radii[A]+adjusted_radii[B]
        if (coordinates[B][1]-coordinates[C][1])%(2*pi) < pi:
          alpha = getAngle(c,a,b) + getAngle(r2,a,r1)
        else:
          alpha = getAngle(c,a,b) - getAngle(r2,a,r1)
        if alpha > pi:
          delta = getAngle(r1,a,r2)+getAngle(b,a,c)
          r3 = acosh(cosh(r2)*cosh(c)-sinh(r2)*sinh(c)*cos(delta))
          beta = coordinates[C][1]-getAngle(b,r1,r3)
        elif alpha < 0:
          alpha = -alpha
          r3 = acosh(cosh(r1)*cosh(b)-sinh(r1)*sinh(b)*cos(alpha))
          beta = coordinates[C][1]-getAngle(b,r1,r3)
        else:
          r3 = acosh(cosh(r1)*cosh(b)-sinh(r1)*sinh(b)*cos(alpha))
          beta = coordinates[C][1]+getAngle(b,r1,r3)
        coordinates[A].append(r3)
        coordinates[A].append(beta)
        placed[A] = True
        break
      
  #convert return list from polar coordinates to cartesian coordinates
  coordinates = [[tanh(c[0])*cos(c[1]),tanh(c[0])*sin(c[1])] for c in coordinates]
  coordinates = [[c[0]/(1+(1-c[0]**2-c[1]**2)**0.5),c[1]/(1+(1-c[0]**2-c[1]**2)**0.5)] for c in coordinates]

  #apply a conformal automorphism (Mobius transformation) of the unit disk that moves the center of mass of the flattened coordinates to the origin
  p = mean([c[0] for c in coordinates])
  q = mean([c[1] for c in coordinates])
  while p**2+q**2 > tolerance:
    for i in range(len(coordinates)):
      x = coordinates[i][0]
      y = coordinates[i][1]
      coordinates[i] = mobius(x,y,p,q)
      coordinates[i].append(0)
    p = mean([c[0] for c in coordinates])
    q = mean([c[1] for c in coordinates])
  
  return coordinates


#########################################################################  
### METHODS FOR ALIGNING TWO SURFACES VIA THEIR CONFORMAL FLATTENINGS ###
######################################################################### 


def rotation(p, theta):
  #this is a helper method for the method getAlignedCoordinates below.  It rotates a given point in the plane about the origin by a given angle.
  return [cos(theta)*p[0]-sin(theta)*p[1],sin(theta)*p[0]+cos(theta)*p[1]]


def getAlignedCoordinates(animal_obj_0, animal_obj_1, theta):
  """ Calculates the vertex coordinates for the triangulation of Animal 1 aligned to the triangulation of Animal 0 by factoring
    through their respective conformal flattenings and applyling a rotation of angle theta.

    :Parameters:
      animal_obj_0/1 : animal objects, initialized with regular/flattened coordinates and triangulation set/updated
      theta : float with value between 0 and pi, an angle of rotation

    :Returns:
      list of triples of floats, specifying the x-, y-, and z-coordinates of the vertices of the triangulation of Animal 1 aligned to
      the triangulation of Animal 0
  """

  #store relevant parameters
  num_verts_0 = animal_obj_0.getNumVerts()
  regular_coordinates_0 = animal_obj_0.getRegularCoordinates()
  flat_coordinates_0 = animal_obj_0.getFlattenedCoordinates()
  flat_coordinates_0 = [f[:2] for f in flat_coordinates_0]
  triangles_0 = animal_obj_0.getTriangulation()
  num_verts_1 = animal_obj_1.getNumVerts()
  flat_coordinates_1 = animal_obj_1.getFlattenedCoordinates()
  flat_coordinates_1 = [f[:2] for f in flat_coordinates_1]

  #initialize return list
  aligned_coordinates_1 = []

  #iterate through the vertices of the triangulation of Animal 1
  for vertex in range(num_verts_1):

    #rotate the flattened coordinates of each such vertex by theta
    rotated_coordinate = rotation(flat_coordinates_1[vertex],theta)

    #initialize individual return values
    x, y, z = 0, 0, 0
    success = False

    #search through all the triangles in the triangulation of Animal 0 for one whose flattened coordinates contain
    #the rotated flattened coordinates of the current vertex in the triangulation of Animal 1
    for triangle in triangles_0:

      #extract flattened coordinates of the vertices of the given triangle
      x_0 = flat_coordinates_0[triangle[0]][0]
      x_1 = flat_coordinates_0[triangle[1]][0]
      x_2 = flat_coordinates_0[triangle[2]][0]
      y_0 = flat_coordinates_0[triangle[0]][1]
      y_1 = flat_coordinates_0[triangle[1]][1]
      y_2 = flat_coordinates_0[triangle[2]][1]

      #calculate barycentric coordinates for current vertex in current triangle
      lambda_0 = ((y_1-y_2)*(rotated_coordinate[0]-x_2)+(x_2-x_1)*(rotated_coordinate[1]-y_2)) / \
                ((y_1-y_2)*(x_0-x_2)+(x_2-x_1)*(y_0-y_2))
      lambda_1 = ((y_2-y_0)*(rotated_coordinate[0]-x_2)+(x_0-x_2)*(rotated_coordinate[1]-y_2)) / \
                ((y_1-y_2)*(x_0-x_2)+(x_2-x_1)*(y_0-y_2))
      lambda_2 = 1 - lambda_0 - lambda_1

      #if current triangle contains rotated flattened coordinates of current vertex, update return values using the barycentric
      #coordinates above and the regular coordinates of Animal 0
      if lambda_0 >= 0 and lambda_0 <= 1 and lambda_1 >=0 and lambda_1 <=1 and lambda_2 >= 0 and lambda_2 <= 1:
        location = triangle
        success = True
        x = lambda_0*regular_coordinates_0[location[0]][0] + \
            lambda_1*regular_coordinates_0[location[1]][0] + \
            lambda_2*regular_coordinates_0[location[2]][0]
        y = lambda_0*regular_coordinates_0[location[0]][1] + \
            lambda_1*regular_coordinates_0[location[1]][1] + \
            lambda_2*regular_coordinates_0[location[2]][1]
        z = lambda_0*regular_coordinates_0[location[0]][2] + \
            lambda_1*regular_coordinates_0[location[1]][2] + \
            lambda_2*regular_coordinates_0[location[2]][2]
        break

    #if no such triangle is found, update the return values with the coordinates of the closest vertex in Animal 0 to the current vertex
    if not success:
      closest_vertex = 0
      for candidate_vertex in range(num_verts_0):
        if linalg.norm(array(rotated_coordinate)-array(flat_coordinates_0[candidate_vertex])) < linalg.norm(array(rotated_coordinate)-array(flat_coordinates_0[closest_vertex])):
          closest_vertex = candidate_vertex
      x = regular_coordinates_0[closest_vertex][0]
      y = regular_coordinates_0[closest_vertex][1]
      z = regular_coordinates_0[closest_vertex][2]

    #append aligned coordinates to return list
    aligned_coordinates_1.append([x,y,z])

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
    write.writeOFF(animal_obj_0, animal_obj_0.getFlattenedCoordinates(), outdir, "heatmap_%s_flat.off" % animal_obj_0.getName())
    write.writeOFF(animal_obj_1, animal_obj_1.getFlattenedCoordinates(), outdir, "heatmap_%s_flat.off" % animal_obj_1.getName())
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
    
