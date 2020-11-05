"""
Copyright Mechanisms Underlying Behavior Lab, Singapore
https://mechunderlyingbehavior.wordpress.com/

heatmap.py is part of the locomotion python package for analyzing locomotory animal 
behaviors via the techniques presented in the paper "Computational geometric tools  
for quantitative comparison of locomotory behavior" by MT Stamps, S Go, and AS Mathuru 
(https://doi.org/10.1038/s41598-019-52300-8).

This python script contains methods for computing conformal spatiotemporal distances
(CSD) between heatmaps of animal trajectories representing the amount of time each
subject spends in a given location. The heatmaps are modeled as triangular meshes 
that are conformally flattened to the unit disk for the purpose of shape alignment
and analysis. The implementation for conformal flattening used in this package is 
the one provided in the libigl package (https://libigl.github.io/).
"""
# pylint:disable=too-many-lines

from math import sin, cos, pi
from numpy import mean, std, array, linalg
from scipy.optimize import minimize
from igl import boundary_loop, map_vertices_to_circle, harmonic_weights, \
    adjacency_matrix, bfs, triangle_triangle_adjacency
import locomotion.write as write
import locomotion.animal as animal

#Static Variables
PERTURBATION = 0.000000001
TOLERANCE = 0.00001

#Enable warnings for boundary mappings
ENABLE_BOUNDARY_WARNINGS = False

######################
### Main Functions ###
######################

def populate_surface_data(animal_obj, grid_size, start_time=None, end_time=None):
    """ Computes the heatmap representation of an animal's movement.

    Computes the heatmap for a given animal trajectory representing the amount of time
    the subject spent in each location during a specified time interval as a histogram 
    with a specified grid size along with a triangular mesh of the corresponding 2D
    surface, and a conformal flattening of the mesh to the unit disk.

    Parameters
    ----------
    animal_obj : animal object
        Initialized Animal() object.
    grid_size : float or int
        Specifies the bin size for calculating the heatmap. Value must divide both x_dim
        and y_dim stored in animal_obj, where smaller values yield finer triangulations
        and larger values yield coarser triangulations
    start/end_time : float or int, optional.
        Time in minutes. If unspecified, start/end time for the experiment will be used.
        Default value: None.
    """
    # pylint:disable=too-many-locals

    #Check if start_time or end_time need to be set:
    if start_time is None:
        start_time = animal_obj.get_exp_start_time()
    if end_time is None:
        end_time = animal_obj.get_exp_end_time()

    #store given parameters
    animal_obj.set_grid_size(grid_size)

    print("Calculating heatmap for %s..." % animal_obj.get_name())

    #calculate heatmap
    frequencies = _assemble_frequencies(animal_obj, start_time, end_time)

    print("Calculating triangulation for %s..." % animal_obj.get_name())

    #get and record vertices
    original_coordinates = _assemble_vertex_coordinates(animal_obj, frequencies)
    animal_obj.set_num_verts(len(original_coordinates))
    animal_obj.set_regular_coordinates(original_coordinates)

    #get and record triangles
    triangles = _assemble_triangles(animal_obj)
    animal_obj.set_num_triangles(len(triangles))
    animal_obj.set_triangulation(triangles)

    #calculate and store colors for output file
    colors = _calculate_triangle_colors(animal_obj)
    animal_obj.set_colors(colors)

    print("Calculating flattened coordinates for %s..." % animal_obj.get_name())

    #calculate and record boundary vertices
    boundary_vertices = _find_boundary_loop(animal_obj)
    animal_obj.set_boundary_vertices(boundary_vertices)

    #calculate and record boundary edges
    boundary_edges = _find_boundary_edges(animal_obj)
    animal_obj.set_boundary_edges(boundary_edges)

    #calculate and record flattened coordinates of triangulation
    flattened_coordinates = _determine_flat_coordinates(animal_obj)
    animal_obj.set_flattened_coordinates(flattened_coordinates)

    print("Calculating vertex BFS and triangle adjacency for %s..." % animal_obj.get_name())

    #calculate and record central vertex and BFS from the centre
    central_vertex = _find_central_vertex(animal_obj)
    animal_obj.set_central_vertex(central_vertex)

    #identify the triangules that contain no boundary vertices
    nbv_triangles = _find_nbv_triangles(animal_obj)

    #find the adjacency matrix and BFS for the interior vertices
    interior_vertex_adjacency_matrix = adjacency_matrix(array(nbv_triangles))
    interior_vertex_bfs = bfs(interior_vertex_adjacency_matrix, central_vertex)
    animal_obj.set_interior_vertex_bfs(interior_vertex_bfs)

    #calculate and record triangle-triangle adjacency matrix
    triangle_adjacency_matrix = triangle_triangle_adjacency(array(triangles))[0]
    animal_obj.set_triangle_triangle_adjacency(triangle_adjacency_matrix)


def compute_one_csd(animal_0, animal_1, fullmode=False, outdir=None):
    """ Computes the CSD between a pair of animal heatmaps.

    Computes the Conformal Spatiotemporal Distance (CSD) between two animal heatmaps.

    Parameters
    ----------
    animal_0/1 : Animal() object
        Initialized Animal() object with regular/ flattened coordinates and triangulation.
    fullmode : bool, optional
        If True, writes triangulations and their corresponding flattenings and alignments
        to .OFF files. Default value : False.
    outdir : str, optional
        Specifying directory to save .OFF files. Must be provided if fullmode is True.
        Default value : None.

    Returns
    -------
    float
        Computed Conformal Spatiotemporal Distance between the heatmaps of two animals.
    """
    # pylint:disable=too-many-locals
    #check that a directory is specified if fullmode is true
    if fullmode and outdir is None:
        raise Exception("Full mode requires a path to output directory.")

    #notify user of progress
    print("Measuring conformal spatiotemporal distance between heat maps of" \
          " %s and %s..." % (animal_0.get_name(), animal_1.get_name()))

    #calculate the optimal mapping between both animals
    theta, rho = _find_optimal_mapping(animal_0, animal_1)

    #store relevant parameters, assuming both animal observations have the same dimensions
    x_dim, y_dim = animal_0.get_dims()
    z_dim = _calculate_z_dim(animal_0)
    num_verts_0 = animal_0.get_num_verts()
    reg_coordinates_0 = animal_0.get_regular_coordinates()
    aligned_coordinates_0 = _determine_aligned_coordinates(animal_1, animal_0, theta, rho)
    triangles_0 = animal_0.get_triangulation()
    num_verts_1 = animal_1.get_num_verts()
    regular_coordinates_1 = animal_1.get_regular_coordinates()
    aligned_coordinates_1 = _determine_aligned_coordinates(animal_0, animal_1, -theta, rho)
    triangles_1 = animal_1.get_triangulation()

    #save the triangulation data in .OFF files if fullmode is True
    if fullmode:
        write.write_off(animal_0, reg_coordinates_0, outdir,
                        "heatmap_%s_regular.off" % animal_0.get_name())
        write.write_off(animal_1, regular_coordinates_1, outdir,
                        "heatmap_%s_regular.off" % animal_1.get_name())
        flat_coordinates_0 = [coord + [0] for coord in animal_0.get_flattened_coordinates()]
        flat_coordinates_1 = [coord + [0] for coord in animal_1.get_flattened_coordinates()]
        write.write_off(animal_0, flat_coordinates_0, outdir,
                        "heatmap_%s_flat.off" % animal_0.get_name())
        write.write_off(animal_1, flat_coordinates_1, outdir,
                        "heatmap_%s_flat.off" % animal_1.get_name())
        write.write_off(animal_0, aligned_coordinates_0, outdir,
                        "heatmap_%s_aligned_to_%s.off" % (animal_0.get_name(), animal_1.get_name()))
        write.write_off(animal_1, aligned_coordinates_1, outdir,
                        "heatmap_%s_aligned_to_%s.off" % (animal_1.get_name(), animal_0.get_name()))

    #calculate the distance between the aligned surfaces
    difference_val_0 = 0
    change_in_area = 0
    for vertex in range(num_verts_1):
        for triangle in triangles_1:
            if vertex in triangle:
                change_in_area += _calculate_area(regular_coordinates_1[triangle[0]][0:2],
                                                  regular_coordinates_1[triangle[1]][0:2],
                                                  regular_coordinates_1[triangle[2]][0:2])/3.0
        difference_val_0 += change_in_area * \
            (aligned_coordinates_1[vertex][2]-regular_coordinates_1[vertex][2])**2

    difference_val_1 = 0
    change_in_area = 0
    for vertex in range(num_verts_0):
        for triangle in triangles_0:
            if vertex in triangle:
                change_in_area += _calculate_area(reg_coordinates_0[triangle[0]][0:2],
                                                  reg_coordinates_0[triangle[1]][0:2],
                                                  reg_coordinates_0[triangle[2]][0:2])/3.0
        difference_val_1 += change_in_area * \
            (aligned_coordinates_0[vertex][2]-reg_coordinates_0[vertex][2])**2

    distance = (difference_val_0**0.5+difference_val_1**0.5)/(2*z_dim*x_dim*y_dim)

    #record distance in terminal
    print("LOG: distance between aligned surfaces of" \
          " %s and %s: %.3f" % (animal_0.get_name(), animal_1.get_name(), distance))

    return distance

def compute_all_csd(animal_list):
    """ Computes all pairwise CSDs given a list of animal heatmaps.

    Computes the CSD between each pair of heatmaps in animal_list using compute_one_csd().

    Parameters
    ----------
    animal_list : list of Animal() objects All initialized with regular/flattened coordinates and triangulation set/updated.
        Order will determine the order of calculations.

    Returns
    -------
    2D array of floats (upper-triangular, empty diagonal)
        Matrix that captures the Conformal Spatiotemporal Distance between the heatmaps of
        each pair of animals in the input list. i,j-th entry is the CSD between the i-th
        and j-th animal.
    """
    #initialize return array
    num_animals = len(animal_list)
    dists = [['' for i in range(num_animals)] for j in range(num_animals)]

    #calculate the CSD between each pair of animals in the input list
    for i in range(num_animals):
        for j in range(i+1, num_animals):
            dists[i][j] = compute_one_csd(animal_list[i], animal_list[j])
    return dists

##########################
### Assembly Functions ###
##########################

def _assemble_frequencies(animal_obj, start_time, end_time):
    """ Converts the coordinates of the animal trajectory into frequency data over the grid.

    Gathers the frequency data for approximating the heatmap representing the amount of
    time an animal spent in each location over a specified time interval.

    Parameters
    ----------
    animal_obj : Animal() object
        Initialized Animal() object.
    start_time : float
        Start time (in minutes) of time period.
    end_time : float
        End time (in minutes) of time period.

    Returns
    -------
    2D array of int
        Two-dimensional array of ints counting the number of frames the animal spent in
        each square chamber of the bounding rectangle during the specified time interval.
    """
    # pylint:disable=too-many-locals

    #set or get relevant parameters
    perturb = PERTURBATION
    start_frame = animal.calculate_frame_num(animal_obj, start_time)
    end_frame = animal.calculate_frame_num(animal_obj, end_time)
    grid_size = animal_obj.get_grid_size()
    x_dim, y_dim = animal_obj.get_dims()
    num_x_grid, num_y_grid = animal_obj.get_num_grids()
    x_vals = animal_obj.get_raw_vals('X', start_frame, end_frame)
    y_vals = animal_obj.get_raw_vals('Y', start_frame, end_frame)

    #initialize frequency matrix
    freqency_matrix = [[0 for j in range(num_y_grid)] for i in range(num_x_grid)]

    #check that coordinate data is within the specified bounds
    x_max = max(x_vals)
    x_offset = max(x_max - x_dim, 0) + perturb
    y_max = max(y_vals)
    y_offset = max(y_max - y_dim, 0) + perturb

    #iterate through each frame, adjust out-of-bounds data, and update frequency matrix
    for i, _ in enumerate(x_vals):
        x_val = x_vals[i] - x_offset
        if x_val < 0:
            print("WARNING: X data is out of bounds. Frame #%d, x=%f" % (i+1, x_vals[i]))
            x_val = 0
        x_index = int(x_val/grid_size)
        y_val = y_vals[i] - y_offset
        if y_val < 0:
            print("WARNING: Y data is out of bounds. Frame #%d, x=%f" % (i+1, y_vals[i]))
            y_val = 0
        y_index = int(y_val/grid_size)
        freqency_matrix[x_index][y_index] += 1
    return freqency_matrix


def _assemble_triangles(animal_obj):
    """ Computes a basic triangulation on the regular coordinates of an animal.

    Parameters
    ----------
    animal_obj : Animal() object
        Initialized with regular coordinates set/updated

    Returns
    -------
    list of triples of ints
        Specifying the indices of the vertices for each triangle in the triangulation of
        a surface.
    """
    #store relevant parameters
    num_x_grid, num_y_grid = animal_obj.get_num_grids()

    #initialize triangle list
    triangles = []

    #iterate through lower left corners of grid and append canonical triangles
    for i in range(num_x_grid-1):
        for j in range(num_y_grid-1):
            triangles.append([i*num_y_grid+j, (i+1)*num_y_grid+j, (i+1)*num_y_grid+(j+1)])
            triangles.append([i*num_y_grid+j, (i+1)*num_y_grid+(j+1), i*num_y_grid+(j+1)])

    return triangles


def _assemble_vertex_coordinates(animal_obj, freqs):
    """ Calculates the vertex coordinates for a triangulation of the heatmap surface.

    Parameters
    ----------
    animal_obj : Animal() object
        Initialized with regular coordinates set/updated.
    freqs : 2D array of ints
        Frequency data for heatmap, generated by _assemble_frequencies().

    Returns
    -------
    list of triples of floats
        Each triple specifies the x-, y-, and z-coordinates of the vertices for a
        triangulation of the surface corresponding to a heat map
    """
    #gather relevant parameters
    grid_size = animal_obj.get_grid_size()
    num_x_grid, num_y_grid = animal_obj.get_num_grids()

    #normalize the values to floats between 0 and a specified z-dimension
    f_mean = mean(freqs)
    f_std = std(freqs)
    z_dim = _calculate_z_dim(animal_obj)
    for i, _ in enumerate(freqs):
        freqs[i] = animal.normalize(freqs[i], f_mean, f_std)
        freqs[i] = list(map(lambda x: z_dim*x, freqs[i]))

    #initialize list of coordinates to return
    coordinates = []

    #append coordinates for the lower left corner of each square in the heat map grid
    for i in range(num_x_grid):
        for j in range(num_y_grid):
            coordinates.append([i*grid_size, j*grid_size, freqs[i][j]])
    return coordinates


def _calculate_triangle_colors(animal_obj):
    """ Converts the average height of each triangle to colors for rendering.

    Generates color for rendering each triangle in the triangulation of an animal
    according to the average height of the regular coordinates of its vertices.

    Parameters
    ----------
    animal_obj : Animal() object
        Initialized with regular coordinates and triangulation set/updated.

    Returns
    -------
    list of triples of floats
        Each triple specifies the RGB values for each triangle in in the triangulation
        associated to an animal's heatmap.
    """

    #gather relevant parameters
    coordinates = animal_obj.get_regular_coordinates()
    triangles = animal_obj.get_triangulation()

    #initialize return list
    colors = []

    #extract the heights (z-coordinates) of each vertex in the triangulation
    heights = [c[2] for c in coordinates]

    #gather basic statistics
    min_height = min(heights)
    max_height = max(heights)
    mid_height = (min_height+max_height)/2

    #assign a color to each triangle based on the average height of the regular
    #coordinates of its vertices
    for triangle in triangles:
        color = [1.0, 1.0, 0]
        height = mean([heights[v] for v in triangle])
        if height > mid_height:
            color[1] -= (height-mid_height)/(max_height-mid_height)
        else:
            color[0] -= (mid_height-height)/(mid_height-min_height)
            color[1] -= (mid_height-height)/(mid_height-min_height)
            color[2] += (mid_height-height)/(mid_height-min_height)
        colors.append(color)
    return colors


def _find_nbv_triangles(animal_obj):
    """ Returns all triangles in an animal mesh that contain no boundary vertices.

    Parameters
    -----------
    animal_obj : Animal() object
        Initialized with regular coordinates and triangulation set/updated.

    Returns
    -------
    list of triples of floats
        List of all triangles in the animal that do not contain boundary vertices.
    """
    #get relevant parameters
    triangles = animal_obj.get_triangulation()
    boundary_vertices = set(animal_obj.get_boundary_vertices())
    interior_triangles = []

    #for each triangle, check if each vertex is a boundary vertex. If it does
    #not contain a boundary vertex, add it to interior_triangles
    for triangle in triangles:
        contains_boundary_vertex = False
        for vertex in triangle:
            if vertex in boundary_vertices:
                contains_boundary_vertex = True
                break
        #either all vertices in this triangle are not boundary vertices, or we
        #found a boundary vertex at this point
        if not contains_boundary_vertex:
            interior_triangles.append(triangle)

    return interior_triangles


#############################
### Measurement Functions ###
#############################

def _calculate_area(p, q, r):
    """ Calculates the area of the triangle by its vertex coordinates.

    Helper method for the _calculate_distortion_energy() and compute_one_csd() methods.
    It calculates the area of the triangle spanned by three points in R^2 or R^3.

    Parameters
    ----------
    p/q/r : triple of floats
        Coordinates of the 3 points that make up the triangle.

    Returns
    -------
    float
        The area of the triangle.
    """
    # pylint:disable=invalid-name
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


def _calculate_distortion_energy(animal_0, animal_1, theta, rho):
    """ Calculates elastic energy required to stretch one animal mesh onto another.

    Calculates the elastic energy required to stretch the mesh of Animal 0 onto the 
    mesh of Animal 1 via the conformal mapping obtained by factoring through their
    respective conformal flattenings and applyling a Mobius transformation.

    Parameters
    ----------
    animal_0/1 : Animal() object
        Initialized with regular coordinates and triangulation set/updated.
    theta : float
        Angle of rotation, between 0 and pi.
    rho : float
        Distance of the reference point to be mapped to the origin by the mobius function,
        with value between 0 and 1.

    Notes
    -----
    The mobius function maps the the point (rho*cos(theta), rho*sin(theta)) to the origin.

    Returns
    -------
    float
        Elastic energy required to align the triangulation of Animal 1 that of Animal 0.
    """
    # pylint:disable=too-many-locals

    #store relevant parameters
    num_verts = animal_0.get_num_verts()
    reg_coordinates = animal_0.get_regular_coordinates()
    aligned_coordinates = _determine_aligned_coordinates(animal_1, animal_0, theta, rho)
    triangles = animal_0.get_triangulation()

    #initialize four matrices whose entries correspond to pairs of vertices in
    #the triangulation of Animal 0
    
    #the number of triangles containing each pair of vertices
    triangles_per_edge = [[0 for j in range(num_verts)] for i in range(num_verts)]
    #the distance between each pair of vertices with the regular coordinates
    original_edge_lens = [[0 for j in range(num_verts)] for i in range(num_verts)]
    #the distance between the images of each pair of vertices following an alignment
    aligned_edge_lens = [[0 for j in range(num_verts)] for i in range(num_verts)]
    #the sum of the areas of the triangles that contain pair of vertices
    area_sum = [[0 for j in range(num_verts)] for i in range(num_verts)]

    #loop through the triangulation to fill in the values of each matrix
    for triangle in triangles:
        first, second, third = sorted(triangle)
        edge_ordering = [(second, first), (third, first), (third, second)]

        for vert_0, vert_1 in edge_ordering:
            triangles_per_edge[vert_0][vert_1] += 1
            original_edge_lens[vert_0][vert_1] = linalg.norm(array(reg_coordinates[vert_0]) - \
                                                             array(reg_coordinates[vert_1]))
            aligned_edge_lens[vert_0][vert_1] = linalg.norm(array(aligned_coordinates[vert_0]) - \
                                                            array(aligned_coordinates[vert_1]))
            area_sum[vert_0][vert_1] += _calculate_area(reg_coordinates[first],
                                                        reg_coordinates[second],
                                                        reg_coordinates[third])
    #initialize the return value
    alignment_value = 0

    #sum the squares of the conformal stretching factors of the alignment over
    #each distinct edge in the triangulation
    for i in range(num_verts):
        for j in range(i):
            #only get the alignment value for interior edges - there must be
            #exactly two triangles containing the edge
            if triangles_per_edge[i][j] == 2:
                alignment_value += (area_sum[i][j] / 3.0) * \
                    (aligned_edge_lens[i][j] / original_edge_lens[i][j] - 1.0)**2
    return alignment_value**0.5


def _calculate_symmetric_distortion_energy(animal_0, animal_1, theta, rho):
    """ Calculates symmetric distortion energy required to stretch one animal mesh onto another.

    Calculates the symmetric distortion energy required to stretch the mesh of Animal 0 
    onto the mesh of Animal 1 via the conformal mapping obtained by factoring through 
    their respective conformal flattenings and applyling a Mobius transformation.

    Parameters
    ----------
    animal_0/1 : Animal() object
        Initialized with regular coordinates and triangulation set/updated.
    theta : float
        Angle of rotation, between 0 and pi.
    rho : float
        Distance of the reference point to be mapped to the origin by the Mobius function,
        with value between 0 and 1.

    Notes
    -----
    The mobius function maps the the point (rho*cos(theta), rho*sin(theta)) to the origin.

    Returns
    -------
    float
        Symmetric distortion energy required to align the triangulation of Animal 1 that
        of Animal 0.
    """
    return _calculate_distortion_energy(animal_0, animal_1, theta, rho) + \
        _calculate_distortion_energy(animal_1, animal_0, -theta, rho)


def _calculate_z_dim(animal_obj):
    """ Generates a value for the vertical bound of a heatmap surface.

    Generates the vertical bound for a heatmap surface, which is set to be the smaller of
    the two horizontal dimensions, but it can be set to specified value depending on the
    context.

    Parameters
    ----------
    animal_org : Animal() object
        Initialized Animal() object.

    Returns
    -------
    int
        Vertical bound of vertical dimension, i.e. lower of the 2 horizontal dimensions.
    """
    return min(animal_obj.get_dims())


########################
### Search Functions ###
########################

def _find_aligned_coordinate(point, simplices, simplex_indices,
                             input_coordinates, output_coordinates):
    """ Converts the coordinates of a point on one flattened mesh to the coordinates of 
    the preimage of its position in an aligned mesh.

    Given a point in the 2D input coordinate system, search through the given simplices
    (either triangle or edges) in the input coordinate system to check if it is inside one
    of them. If it is, convert the point into barycentric coordinates corresponding to the
    simplex, and use those barycentric coordinates to return the point in the 3D output
    coordinate system. Otherwise, return an empty list.

    Parameters
    ----------
    point: 2-tuple of float
        A point in the input coordinate system.
    simplices: list of triples or pairs of ints
        A list of the triangles or edges we want to search (in indices of vertices).
    simplex_indices : list or set of int
        Indices corresponding to the simplices above in the input coordinate system.
    input_coordinates: list of pairs of floats
        The 2D input coordinate system where the point and simplices lie.
    output_coordinates: list of triples of floats
        The 3D output coordinate system we want to align the point to.

    Returns
    -------
    list
        If point is found in one of the simplices, then the list will contain 2 entries,
        the first being the index of the simplex and the second be a triple of floats
        corresponding to the point's aligned coordinate in the output coordinate system.
        Else, the list is empty.
    """
    # Define helper functions
    def in_unit_interval(x_val):
        return 0 <= x_val <= 1

    def is_inside(barycentric_coords):
        """Given a list of any number of barycentric coordinates, check if each value
        is between 0 and 1. Returns true only if all are in the unit interval."""
        return all(map(in_unit_interval, barycentric_coords))

    #initialise the result
    result = []

    for simplex_i, simplex in zip(simplex_indices, simplices):
        #get the barycentric coordinates of this point in this simplex in the
        #input coordinate system
        barycentric_coords = _convert_to_barycentric(point, simplex,
                                                     input_coordinates)
        if is_inside(barycentric_coords):
            #set the result as the regular coordinates corresponding to the
            #barycentric coordinates
            result = [simplex_i,
                      _convert_from_barycentric(barycentric_coords, simplex,
                                                output_coordinates)]
            break
    return result


def _find_boundary_edges(animal_obj):
    """Find the boundary edges of an animal mesh in counter-clockwise order.

    Parameters
    ----------
    animal_obj : Animal() object
        Initialized with regular coordinates and triangulation set/updated.

    Returns
    -------
    list of 2-tuples of ints
        List of edges ordered as in the boundary loop, where each edge is a tuple of the
        two indices of the adjacent vertices.
    """
    boundary_vertices = list(animal_obj.get_boundary_vertices())
    #zip the boundary vertices with itself with an offset of 1 and its head
    #appended at the back (so it goes full circle), then cast to a list
    boundary_edges = list(zip(boundary_vertices, boundary_vertices[1:] +
                              [boundary_vertices[0]]))
    return boundary_edges


def _find_boundary_loop(animal_obj):
    """Find the boundary vertices of an animal mesh in counter-clockwise
    order. This method is a wrapper for the corresponding IGL function.

    Parameters
    ----------
    animal_obj : Animal() object
        Initialized with regular coordinates and triangulation set/updated.

    Returns
    -------
    array of int
        The indices of the vertices that are on the boundary of this animal in counter
        clock-wise order.
    """
    #convert triangulation to array for IGL
    triangulation = array(animal_obj.get_triangulation())
    return boundary_loop(triangulation)


def _find_central_vertex(animal_obj):
    """Finds the index of the vertex closest to the topological center of an animal mesh.

    Parameters
    ----------
    animal_obj : Animal() object
        Initialized with regular coordinates and triangulation set/updated.

    Returns
    -------
    int
        Index of the vertex at the the central coordinate. We know that it exists because
        of our triangulation method.
    """
    #get the regular coordinates in the x, y dimension to find the central vertex in that plane
    x_y_coordinates = [coord[:2] for coord in animal_obj.get_regular_coordinates()]
    num_x_grid, num_y_grid = animal_obj.get_num_grids()
    grid_size = animal_obj.get_grid_size()

    #get the central coordinate in the grid. It must be a multiple of the grid size.
    mid_x_coordinate = (num_x_grid // 2) * grid_size
    mid_y_coordinate = (num_y_grid // 2) * grid_size

    #find the index of this central coordinate
    central_vertex = x_y_coordinates.index([mid_x_coordinate, mid_y_coordinate])

    return central_vertex


def _find_closest_vertex(point, vertices, input_coordinates, output_coordinates):
    """ Find the coordinates of the vertex closest to a prescribed point.

    Given a point in the input coordinate system, the vertices in the input coordinate
    system to search through, the 2D input coordinates and the 3D output coordinates,
    return the coordinates corresponding to the vertex in the vertices we searched through
    that is closest to the vertex we input.

    Notes
    -----
    This method is used only for emergencies when we cannot find a corresponding boundary
    edge or triangle when aligning vertices. It should not be called often.

    Parameters
    ----------
    point: list of pairs of floats
        The 2D coordinates of the point whose closest vertex coordinate we want to find.
    vertices: iterator
        Range from 0 to the total number of vertices. The vertices (in indices) that we
        want to search through.
    input_coordinates: list of pairs of floats
        The 2D input coordinate system where the point and vertices lie.
    output_coordinates: list of triples of floats
        The 3D output coordinate system we want to align the point to.

    Returns
    -------
    [int, triple of floats]
        int corresponds to the index of the closest vertex, and triple of floats
        corresponds to the coordinates of the vertex in the output coordinates closest
        to this point.
    """
    closest_vertex = 0
    closest_dist = linalg.norm(array(point)-array(input_coordinates[closest_vertex]))
    for candidate_vertex in vertices:
        candidate_dist = linalg.norm(array(point)-array(input_coordinates[candidate_vertex]))
        if candidate_dist < closest_dist:
            closest_vertex = candidate_vertex
            closest_dist = candidate_dist
    return [closest_vertex, output_coordinates[closest_vertex][:3]]


def _find_next_neighbourhood(animal_obj, current_triangles, traversed_triangles):
    """ Find the triangles adjacent to a given list of triangles in an animal mesh, 
    disregarding previously traversed triangles.

    Given an animal object, a set of triangles whose neighbours we want to get and a set of
    inner triangles or edges we have already traversed, find the next layer of adjacent
    triangles that we have not yet traversed.

    Parameters
    ----------
    animal_obj : Animal() object
        Initialized with regular coordinates and triangulation set/updated.
    current_triangles: set of int
        The set containing the indices of triangles whose neighbours we want to find.
    traversed_triangles: set of int
        The set containing the ndices of triangles which we have already traversed.

    Returns
    -------
    set of int
        The set of all triangles not in traversed_triangles and are adjacent to any
        triangle in current_triangles.
    """
    #initialise return set
    all_adjacent_triangles = set()

    #use the triangle-triangle adjacency array to find neighbouring triangles
    triangle_triangle_adjacency_array = animal_obj.get_triangle_triangle_adjacency()

    for triangle_i in current_triangles:
        #update all adjacent triangles with the triangles adjacent to each triangle
        adjacent_triangles = triangle_triangle_adjacency_array[triangle_i]
        all_adjacent_triangles.update(adjacent_triangles)

    #remove -1 (indicating that no triangle is adjacent to that edge) and
    #traversed triangles from all the adjacent triangles we've found
    all_adjacent_triangles.difference_update(traversed_triangles)
    all_adjacent_triangles.discard(-1)

    return all_adjacent_triangles


def _find_optimal_mapping(animal_0, animal_1):
    """ Finds the Mobius transformation of the unit disk that minimizes the symmetric
    distortion energy between two animal meshes.

    Parameters
    ----------
    animal_0/1 : Animal() object
        Initialized with regular coordinates and triangulation set/updated.

    Returns
    -------
    float
        Optimal rotation, an angle in radians between 0 and pi.
    """
    #define a two-variable function that, for a fixed pair of animals, takes an
    #angle in [0, 2*pi] and rho value in [0, 1) as input and outputs the
    #corresponding symmetric distortion energy
    def optimization_function(theta_rho_pair):
        return _calculate_symmetric_distortion_energy(animal_0, animal_1,
                                                      theta_rho_pair[0], theta_rho_pair[1])
    #find the optimal theta and rho values that minimize the symmetric distortion energy
    #set 'disp' to True to print convergence messages.
    #This will show the number of iterations and function evaluations
    #for faster conversion, reduce the values of 'maxiter' and 'maxfev'
    res = minimize(optimization_function, [0., 0.], method='Powell',
                   bounds=((0, 2*pi), (0, 1)),
                   options={'maxiter': 2, 'maxfev': 40,
                            'disp': False, 'direc':[[0, 0.9], [0.9, 0]]})
    print("LOG: Found an optimal (theta, rho) mapping of " + str(res.x) + ". ")
    return res.x


def _find_triangle_containing_vertex(vertex, triangles):
    """ Finds a triangle in a list that contains a prescribed point.

    Given a vertex and the corresponding triangulation it belongs to, return the index of
    the first triangle that contains the vertex.

    Parameters
    ----------
    vertex : int
        An index of the vertex of interest. Index is in relation to the corresponding
        coordinates stored in the animal object.
    triangles : list of triples of ints
        List of all the triangles to be searched. Each triple represents the indices of
        the points of a triangle.

    Returns
    -------
    int or None
        If there exists a triangle containing the vertex, it returns the index of the
        first triangle that contains it. Otherwise, returns None.
    """
    triangle_index = None
    for triangle_i, triangle in enumerate(triangles):
        if vertex in triangle:
            triangle_index = triangle_i
    return triangle_index


################################
### Transformation Functions ###
################################

def _convert_to_barycentric(point, simplex, coordinates):
    """ Converts the coordinates of a point into barycentric coordinates given a simplex.

    Given a 2D point inside a simplex (a line segment or a triangle), find out its
    barycentric coordinates. In the case of the line (1-simplex), this would be the point
    expressed as a linear combination of the two endpoints. In the case of the triangle
    (2-simplex), this would be the point expressed as a linear combination of three corner
    coordinates.

    This method will not work when finding barycentric coordinates of points within a
    triangle or line segment in R^3. It is only meant for finding barycentric coordinates
    of 2D points within 2D line segments or triangles.

    Parameters
    ----------
    point: list of floats, length 2
        The 2D coordinates of the flattened vertex. The z-component should be 0.
    simplex: list of ints, length 2 or 3
        The indices corresponding to coordinates making up the line segment/triangle.
    coordinates: list of pairs of floats
        The 2D coordinate system in which the point and simplex lie.

    Returns
    -------
    list of floats, length 2 or 3
        The lambda values (i.e. the weights used in the linear combination) corresponding
        to the barycentric coordinates of the point in the simplex. Length depends on the
        type of simplex - 2 if a line, 3 if a triangle. If all values are between 0 and 1,
        the point is in the simplex.
    """

    if not len(point) == len(coordinates[0]) == 2:
        raise Exception("_convert_to_barycentric: Invalid coordinate dimensions. " \
                        "This method only accepts coordinates in 2D.")
    #initialise result
    result = []

    #if the simplex is a triangle, calculate the barycentric coordinates of the
    #point in the triangle
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
        #since it's linear interpolation, the proportions are the same for both
        #x and y components, so we just use one of them
        x_0, x_1 = coordinates[simplex[0]][0], coordinates[simplex[1]][0]

        #find the two weights
        lambda_1 = (point[0] - x_0) / (x_1 - x_0)
        lambda_0 = 1 - lambda_1
        result = [lambda_0, lambda_1]

    else:
        raise Exception("_convert_to_barycentric: Invalid input simplex. " \
                        "This method is only defined for triangles and edges")
    return result


def _convert_from_barycentric(barycentric_coords, simplex, coordinates):
    """ Converts barycentric coordinates for a given the simplex to Cartesian coordinates.

    Given barycentric coordinates, a list of coordinates and a simplex (triangle or line
    segment), return the actual coordinates in R^3 corresponding to the barycentric
    coordinates.

    This method will not work when finding the corresponding coordinates in R^2. We will be
    trying to access the z-component, which will cause an index error. If this method is
    needed in such a case, assign 0 as the third coordinate.

    Parameters
    ----------
    barycentric_coords : list of floats, length 2 or 3
        The barycentric coordinates of a point in the triangle or line segment.
    simplex: list of ints, length 2 or 3
        The indices corresponding to vertices making up the triangle or line segment.
    coordinates: list of triples of floats
        Coordinate system in which the vertices exist in. If they are flattened
        coordinates, the third element should be 0.

    Returns
    -------
    list of floats, length 3
        Converted coordinates of the point defined by the barycentric coordinates and the
        given simplex.
    """

    if len(coordinates[0]) != 3:
        raise Exception("_convert_from_barycentric: Invalid coordinate dimensions. " \
                        "This method requires the coordinates to be in 3D.")

    #initialise return value
    result = []

    #if the simplex is a triangle, get the values of the corresponding
    #coordinates in R^3 componentwise
    if len(barycentric_coords) == len(simplex) == 3:
        x_val = barycentric_coords[0] * coordinates[simplex[0]][0] + \
            barycentric_coords[1] * coordinates[simplex[1]][0] + \
            barycentric_coords[2] * coordinates[simplex[2]][0]
        y_val = barycentric_coords[0] * coordinates[simplex[0]][1] + \
            barycentric_coords[1] * coordinates[simplex[1]][1] + \
            barycentric_coords[2] * coordinates[simplex[2]][1]
        z_val = barycentric_coords[0] * coordinates[simplex[0]][2] + \
            barycentric_coords[1] * coordinates[simplex[1]][2] + \
            barycentric_coords[2] * coordinates[simplex[2]][2]
        result = [x_val, y_val, z_val]

    #if the simplex is a line segment, get the values of the corresponding
    #coordinates using the equation of a line
    elif len(barycentric_coords) == len(simplex) == 2:
        #extract the two coordinates of the line segment and just one of the
        #barycentric coordinate weights (the gradient of the line)
        (x_0, y_0, z_0) = coordinates[simplex[0]]
        (x_1, y_1, z_1) = coordinates[simplex[1]]
        lambda_1 = barycentric_coords[1]

        x_val = x_0 + lambda_1 * (x_1 - x_0)
        y_val = y_0 + lambda_1 * (y_1 - y_0)
        z_val = z_0 + lambda_1 * (z_1 - z_0)
        result = [x_val, y_val, z_val]

    else:
        raise Exception("_convert_from_barycentric: Invalid dimensions for barycentric " \
                        "coordinates and/or simplex. They must both be of either length " \
                        "2 or 3, since the method is only defined for triangles and edges.")
    return result


def _determine_aligned_coordinates(animal_obj_0, animal_obj_1, theta, rho):
    """ Determines the coordinates for the image of an animal mesh aligned to another 
    via a prescribed Mobius transformation.

    Calculates the vertex coordinates for the triangulation of Animal 1 aligned to the
    triangulation of Animal 0 by factoring through their respective conformal flattenings
    and applyling a Mobius transformation that moves the point
    (rho*cos(theta), rho*sin(theta)) to the origin.

    Parameters
    ----------
    animal_obj : Animal() object
        Initialized with regular coordinates and triangulation set/updated.
    theta : float
        An angle of rotation (in radians) with value between 0 and pi.
    rho : float
        The magnitude of the reference point to be mapped to the origin by the mobius
        function, with value between 0 and 1.

    Returns
    -------
    list of triples of floats
        Corresponds to the x-, y-, and z-coordinates of the vertices of the triangulation
        of Animal 1 aligned to the triangulation of Animal 0
    """
    # pylint:disable=too-many-locals
    # pylint:disable=too-many-statements

    #store relevant parameters
    num_verts_0 = animal_obj_0.get_num_verts()
    num_verts_1 = animal_obj_1.get_num_verts()
    reg_coordinates_0 = animal_obj_0.get_regular_coordinates()
    flat_coordinates_0 = animal_obj_0.get_flattened_coordinates()
    flat_coordinates_1 = animal_obj_1.get_flattened_coordinates()
    triangles_0 = animal_obj_0.get_triangulation()
    num_triangles_0 = animal_obj_0.get_num_triangles()
    boundary_vertices_1 = list(animal_obj_1.get_boundary_vertices())
    boundary_edges_0 = animal_obj_0.get_boundary_edges()
    num_edges_0 = len(boundary_edges_0)

    #calculate a BFS for the interior vertices of Animal 1
    bfs_ordering, bfs_ancestors = animal_obj_1.get_interior_vertex_bfs()
    #store the first vertex of the BFS seperately
    first_vertex, *v_traversal_1 = bfs_ordering
    #initialize return list with triples of -1
    aligned_coordinates_1 = [[-1, -1, -1]] * num_verts_1
    #initialise dictionary that maps each vertex index of Animal 1 to its
    #corresponding triangle index of Animal 0
    vertex_to_triangle_map = {}

    #1. FIND THE COORDINATES FOR THE FIRST INTERIOR VERTEX VIA BRUTE FORCE
    # transform the flattened coordinates of the first vertex
    first_transformed_coord = _mobius(flat_coordinates_1[first_vertex],
                                      [rho*cos(theta), rho*sin(theta)])

    #search through the triangles in the triangulation of Animal 0 for one
    #whose flattened coordinates contain the first vertex
    triangle_coord_pair = _find_aligned_coordinate(first_transformed_coord,
                                                   triangles_0,
                                                   range(num_triangles_0),
                                                   flat_coordinates_0,
                                                   reg_coordinates_0)
    #if no such triangle is found, set the triangle-coordinate pair as
    #the closest vertex and the first triangle containing that vertex
    if triangle_coord_pair == []:
        print("WARNING: Central vertex in Animal 1 is not contained in any triangle in Animal 0.")
        closest_vertex, closest_vertex_coord = _find_closest_vertex(first_transformed_coord,
                                                                    range(num_verts_0),
                                                                    flat_coordinates_0,
                                                                    reg_coordinates_0)
        triangle_i = _find_triangle_containing_vertex(closest_vertex, triangles_0)
        if triangle_i is None:
            print("WARNING: No triangle associated to the closest vertex " + \
                  str(closest_vertex) + \
                  ". Not updating vertex-to-triangle map for this vertex.")
        triangle_coord_pair = [triangle_i, closest_vertex_coord]

    #add the index of the triangle we found to our vertex-to-triangle map and
    #add the aligned coordinate to return list
    vertex_to_triangle_map[first_vertex] = triangle_coord_pair[0]
    aligned_coordinates_1[first_vertex] = triangle_coord_pair[1]

    # 2. FIND THE CORRESPONDING COORDINATES FOR THE REST OF THE INTERIOR
    #    VERTICES VIA TRIANGLE BFS
    for vertex in v_traversal_1:
        #transform the flattened coordinates of this vertex and get the parent of
        #this vertex from our BFS of interior vertices
        transformed_coordinate = _mobius(flat_coordinates_1[vertex],
                                         [rho*cos(theta), rho*sin(theta)])
        parent_vertex = bfs_ancestors[vertex]
        triangle_coord_pair = []

        #initialize a set of traversed triangles and current list of triangles
        #to search, starting with the triangle corresponding to this vertex's parent
        traversed_triangles = set()
        current_triangle_indices = {vertex_to_triangle_map[parent_vertex]}
        current_triangles = [triangles_0[vertex_to_triangle_map[parent_vertex]]]

        while triangle_coord_pair == []:
            #if no matching triangle is found, set the triangle-coordinate pair as
            #the closest vertex and the first triangle containing that vertex
            if len(traversed_triangles) == num_triangles_0:
                print("WARNING: no triangle found for interior vertex " + \
                      str(vertex) + ". Assigning closest vertex instead.")
                closest_vertex, closest_vertex_coord = _find_closest_vertex(transformed_coordinate,
                                                                            range(num_verts_0),
                                                                            flat_coordinates_0,
                                                                            reg_coordinates_0)
                triangle_i = _find_triangle_containing_vertex(closest_vertex, triangles_0)
                if triangle_i is None:
                    print("WARNING: No triangle associated to the closest vertex " + \
                          str(closest_vertex) + \
                          ". Not updating vertex-to-triangle map for this vertex.")
                triangle_coord_pair = [triangle_i, closest_vertex_coord]
                break

            #check if the transformed coordinate is contained in the current triangles
            triangle_coord_pair = _find_aligned_coordinate(transformed_coordinate,
                                                           current_triangles,
                                                           current_triangle_indices,
                                                           flat_coordinates_0,
                                                           reg_coordinates_0)

            #update values for next iteration - add the triangles that were just
            #traversed and set the current triangles to their neighbours
            traversed_triangles = traversed_triangles.union(current_triangle_indices)
            current_triangle_indices = _find_next_neighbourhood(animal_obj_0,
                                                                current_triangle_indices,
                                                                traversed_triangles)
            current_triangles = [triangles_0[i] for i in current_triangle_indices]

        #add the index of the triangle found to the vertex-to-triangle map
        #and add the aligned coordinate to return list
        vertex_to_triangle_map[vertex] = triangle_coord_pair[0]
        aligned_coordinates_1[vertex] = triangle_coord_pair[1]

    # 3. FIND THE THE CORRESPONDING COORDINATES FOR THE BOUNDARY VERTICES
    # initialise the root edge
    root_edge = 0

    for vertex in boundary_vertices_1:
        #initialize the ordering of boundary edges to search through
        boundary_edge_indices = list(range(num_edges_0))
        edge_search_ordering = boundary_edge_indices[root_edge:] + \
            boundary_edge_indices[:root_edge]
        edges_searched = 0

        #transform the flattened coordinates of this vertex
        transformed_coordinate = _mobius(flat_coordinates_1[vertex],
                                         [rho*cos(theta), rho*sin(theta)])
        #initialise the edge-coordinate-pair and search through each edge
        #corresponding to the transformed coordinate
        edge_coordinate_pair = []
        while edge_coordinate_pair == []:
            #if no edge is found after searching all the edges, assign
            #the same root edge and the closest vertex coordinate
            if edges_searched == num_edges_0:
                if ENABLE_BOUNDARY_WARNINGS:
                    print("WARNING: BOUNDARY FAILURE: " \
                    "Could not find boundary edge for boundary vertex " + \
                        str(vertex) + ". Assigning closest vertex instead.")
                closest_vertex_coord = _find_closest_vertex(transformed_coordinate,
                                                            range(num_verts_0),
                                                            flat_coordinates_0,
                                                            reg_coordinates_0)[1]
                edge_coordinate_pair = [root_edge, closest_vertex_coord]
                break

            #update current edge in the search
            edge_index_to_search = edge_search_ordering[edges_searched]
            edge_to_search = boundary_edges_0[edge_index_to_search]
            #search through remaining boundary edges
            edge_coordinate_pair = _find_aligned_coordinate(transformed_coordinate,
                                                            [edge_to_search],
                                                            [edge_index_to_search],
                                                            flat_coordinates_0,
                                                            reg_coordinates_0)
            #update the edges searched for the next iteration
            edges_searched += 1

        #update root edge and assign aligned coordinates to return list
        root_edge = edge_coordinate_pair[0]
        aligned_coordinates_1[vertex] = edge_coordinate_pair[1]

    return aligned_coordinates_1


def _determine_flat_coordinates(animal_obj):
    """Determines the coordinates for the conformal flattening of an animal mesh.

    Parameters
    ----------
    animal_obj : Animal() object
        Initialized with regular coordinates and triangulation set/updated.

    Returns
    -------
    list of pairs of floats
        Corresponds to the x- and y-coordinates of the vertices of a triangulation that
        has been conformally flattened to the unit disk.
    """

    # store relevant parameters and convert to arrays
    reg_coordinates = array(animal_obj.get_regular_coordinates())
    triangles = array(animal_obj.get_triangulation())

    # get boundary vertice indices (already an array) from the animal
    boundary_vertices = animal_obj.get_boundary_vertices()

    # map boundary vertices to unit circle, preserving edge proportions, to get
    # the flattened boundary coordinates
    flattened_boundary_coordinates = map_vertices_to_circle(reg_coordinates,
                                                            boundary_vertices)

    # map internal vertices to unit circle
    flat_coordinates = harmonic_weights(reg_coordinates, triangles,
                                        boundary_vertices, flattened_boundary_coordinates, 1)
    flat_coordinates = list(flat_coordinates)

    # apply a conformal automorphism (Mobius transformation) of the unit disk
    # that moves the center of mass of the flattened coordinates to the origin
    p_val = mean([c[0] for c in flat_coordinates])
    q_val = mean([c[1] for c in flat_coordinates])

    while p_val**2+q_val**2 > TOLERANCE:
        print(f"LOG: Distance of original centroid to origin is {(p_val**2+q_val**2)}. " \
              "Moving closer to origin.")
        for i, _ in enumerate(flat_coordinates):
            flat_coordinates[i] = _mobius(flat_coordinates[i],
                                          [p_val, q_val])
        p_val = mean([c[0] for c in flat_coordinates])
        q_val = mean([c[1] for c in flat_coordinates])
    return flat_coordinates


def _mobius(p, q):
    """ Applies the mobius transformation that fixes the line from q to the origin
    and sends q to the origin to the point p.

    Parameters
    ----------
    p : list of floats, length 2
        The coordinate we are applying this transformation to.
    q : list of floats, length 2
        The coordinate that gets sent to the origin by this map.

    Returns
    -------
    list of floats, length 2
        The transformed coordinate p after applying the transformation that moves q to the
        origin.
    """
    # pylint:disable=invalid-name
    # pure math formula
    u, v = p
    a, b = q
    return [-1 * ((u-a)*(a*u+b*v-1)+(v-b)*(a*v-b*u))/((a*u+b*v-1)**2+(a*v-b*u)**2),
            -1 * ((v-b)*(a*u+b*v-1)-(u-a)*(a*v-b*u))/((a*u+b*v-1)**2+(a*v-b*u)**2)]
