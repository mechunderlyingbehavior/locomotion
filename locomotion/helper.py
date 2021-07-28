# Reference: https://github.com/onbrian/kirkpatrick-point-location
import numpy as np

class Piece:
    """
    Class representing a face on the original graph, as well as a vertex in the final
    point locator Directed Acyclic Graph (DAG).
    """
    def __init__(self, triangle, children=None, is_leaf=False, is_inside=False):
        # Check Input
        assert (len(triangle) == 3)

        # Indexes of points
        triangle.sort()
        self.p1 = triangle[0]
        self.p2 = triangle[1]
        self.p3 = triangle[2]
        # List of children (in DAG construction)
        self.children = children
        # Is this piece a leaf (in DAG)
        self.is_leaf = is_leaf

    def set_children(children):
        """ Set children for Piece """
        self.children = children

    def equals(self, other):
        """ Check if this Piece represents the same triangle as another Piece. """
        if (self.p1 != other.p1) or (self.p2 != other.p2) or (self.p3 != other.p3):
            return False
        return True

    def leaf(self):
        """ Check if Piece is a leaf in DAG """
        return self.is_leaf

    def inside(self):
        """ Check if Piece is inside the original convex hull of points """
        assert(self.leaf())
        return self.is_inside

    def to_list(self):
        """ Returns triangle's point in a list. """
        return [self.p1, self.p2, self.p3]

    def to_string(self):
        """ Returns a string representation of Piece i.e. its triangle indices """
        return f"piece: {self.p1}, {self.p2}, {self.p3}"


class MyGraph:
    """
    Class to represent the graph formed by the original points.
    """
    def __init__(self, points, pieces):
        # Points in the Graph
        self.points = points
        self.N = len(points)
        self.current_N = len(points)

        # Triangles in the current layer
        self.pieces = pieces

        # Array of active points (i.e. not removed by Kirkpatrick)
        self.active_points = np.ones(self.N, dtype=bool)

        # Adjacency matrix with edge to face information
        self.edge_to_face = np.empty((self.N, self.N, 2), dtype=object)
        self.edge_to_face.fill(None)

        # Add Pieces to adjacency matrix
        for piece in self.pieces:
            self.add_piece(piece)

    def add_piece(self, piece):
        """ Adds Piece object into MyGraph datastructure. """
        a, b, c = piece.p1, piece.p2, piece.p3

        # Add face (piece) to edge
        self._add_face_to_edge(a, b, piece)
        self._add_face_to_edge(a, c, piece)
        self._add_face_to_edge(b, c, piece)

    def degree(self, vertex):
        """ Returns the degree of a given vertex. """
        # Ensure boundary vertex removed, and index is not out of bounds
        assert (vertex >= 3) and (vertex < self.N)
        # Ensure that vertex to be removed was not already removed
        assert self.active_points[vertex]

        count = 0
        for i in range(0, self.N):
            if self.edge_to_face[vertex][i][0] or self.edge_to_face[vertex][i][1]:
                count += 1
        return count

    def get_faces_at_vertex(self, vertex):
        """ Returns all adjacent faces to given vertex. """
        all_faces = []
        neighbors = self.get_neighbors(vertex)

        for neighbor in neighbors:
            for face in self.edge_to_face[vertex][neighbor]:
                all_faces.append(face)
        return set(all_faces)

    def get_intersecting_pieces(self, vertex, triangle):
        """ Find all pieces around vertex that intersects triangle. """

        a = self.points[triangle[0]]
        b = self.points[triangle[1]]
        c = self.points[triangle[2]]
        v = self.points[vertex]

        # If vertex is in the triangle, it intersects all old triangles.
        if inside_triangle(a, b, c, v):
            return self.get_faces_at_vertex(vertex)

        # Check edges adjacent to vertex to see if they intersect the triangle.
        intersections = []
        segments = [[a, b], [b, c], [c, a]]

        for neighbor in self.get_neighbors(vertex):
            edge = [v, self.points[neighbor]]
            for segment in segments:
                if segment_intersect(edge, segment):
                    intersections.append(self.edge_to_face[vertex][neighbor][0])
                    intersections.append(self.edge_to_face[vertex][neighbor][1])
                    break
        return set(intersections)

    def get_neighbors(self, vertex):
        """ Get neighbors of given vertex. """
        # Ensure boundary vertex removed, and index is not out of bounds
        assert (vertex >= 3) and (vertex < self.N)
        # Ensure that vertex to be removed was not already removed
        assert self.active_points[vertex]

        neighbors = []
        for i in range(0, self.N):
            if self.edge_to_face[vertex][i][0] or self.edge_to_face[vertex][i][1]:
                neighbors.append(i)
        return neighbors

    def get_surrounding_polygon(self, vertex):
        """ Given a vertex, returns the polygon created if it was removed.

        Points are returned ordered in an arbitrary rotation, starting from the
        point with the lowest index.
        """
        # Ensure boundary vertex removed, and index is not out of bounds
        assert (vertex >= 3) and (vertex < self.N)
        # Ensure that vertex to be removed was not already removed
        assert self.active_points[vertex]

        polygon = []
        neighbors = self.get_neighbors(vertex)

        # Start with first neighbor
        curr_point = neighbors[0]
        polygon.append(curr_point)
        faces = self.edge_to_face[vertex][curr_point]
        curr_face, ending_face = self._get_ccw_face([vertex, curr_point], faces)

        while curr_face != ending_face:
            # Advance point in direction of curr_face
            curr_point = self._get_other_point(curr_face, vertex, curr_point)
            # Get next face adjacent to [vertex, curr_point] edge.
            curr_face = self._get_other_face(self.edge_to_face[vertex][curr_point],
                                             curr_face)
            polygon.append(curr_point)

        return polygon

    def remove_vertex(self, vertex):
        """ Remove vertex from MyGraph datastructure. """
        # Ensure boundary vertex removed, and index is not out of bounds
        assert (vertex >= 3) and (vertex < self.N)
        # Ensure that vertex to be removed was not already removed
        assert self.active_points[vertex]

        faces = self.get_faces_at_vertex(vertex)
        for face in faces:
            self.remove_face(face)

        self.active_points[vertex] = False
        self.current_N -= 1

    def remove_face(self, piece):
        """ Remove piece from self.edge_to_face matrix. """
        self._remove_face_at_edge(piece.p1, piece.p2, piece)
        self._remove_face_at_edge(piece.p1, piece.p3, piece)
        self._remove_face_at_edge(piece.p2, piece.p3, piece)

    def _add_face_to_edge(self, a, b, piece):
        """ Associates Piece to vertices a and b. """
        if self.edge_to_face[a][b][0] is None:
            self.edge_to_face[a][b][0] = piece
        else:
            if not self.edge_to_face[a][b][0].equals(piece):
                self.edge_to_face[a][b][1] = piece
        # Both sides
        if self.edge_to_face[b][a][0] is None:
            self.edge_to_face[b][a][0] = piece
        else:
            if not self.edge_to_face[b][a][0].equals(piece):
                self.edge_to_face[b][a][1] = piece

    def _get_ccw_face(self, edge, faces):
        """ Checks faces and returns ccw face from perspective of edge. """
        other_point = self._get_other_point(faces[0], edge[0], edge[1])
        if ccw(self.points[edge[0]], self.points[edge[1]], self.points[other_point]):
            return faces[0], faces[1]
        return faces[1], faces[0]

    def _get_other_point(self, piece, a, b):
        """ Given 2 points of a Piece, return the remaining point. """
        if piece.p1 != a and piece.p1 != b:
            return piece.p1
        elif piece.p2 != a and piece.p2 != b:
            return piece.p2
        assert(piece.p3 != a and piece.p3 != b)
        return piece.p3

    def _get_other_face(self, face_pair, face):
        """ Given a pair of Pieces and a piece in the pair, return the other piece."""
        if face_pair[0].equals(face):
            return face_pair[1]
        assert (face_pair[1].equals(face))
        return face_pair[0]

    def _remove_face_at_edge(self, a, b, piece):
        """ Removes face from verices a and b. """
        curr_faces = self.edge_to_face[a][b]
        for i, poss in enumerate(curr_faces):
            if poss and poss.equals(piece):
                self.edge_to_face[a][b][i] = None

        # Both sides, since graph is undirected.
        curr_faces = self.edge_to_face[b][a]
        for i, poss in enumerate(curr_faces):
            if poss and poss.equals(piece):
                self.edge_to_face[b][a][i] = None

# Other functions

def area2(a, b, c):
    """ Calculates size of triangle formed by points a, b, c. """
    return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])

def ccw(a, b, c):
    """ Checks if points a, b, c are in counter-clockwise order. """
    return (b[0] - a[0]) * (c[1] - a[1]) > (b[1] - a[1]) * (c[0] - a[0])

def inside_triangle(a, b, c, p):
    """ Checks if point p is inside triangle formed by a, b, and c"""
    # Check if p lies on the same side of all 3 lines.
    side1 = ccw(a, b, p)
    side2 = ccw(b, c, p)
    side3 = ccw(c, a, p)

    if (side1 == side2) and (side2 == side3):
        return True
    return False

def is_collinear(a, b, c):
    EPSILON = 0.0001
    area = area2(a, b, c)
    return -EPSILON <= area <= EPSILON

def segment_intersect(a, b):
    """ Check if line segments a and b intersect. """
    assert (len(a) == 2)
    assert (len(b) == 2)

    if (is_collinear(a[0], a[1], b[0]) and is_collinear(a[0], a[1], b[1])):
        max_X = max(a[0][0], a[1][0])
        min_X = min(a[0][0], a[1][0])

        if (min_X <= b[0][0] <= max_X) or (min_X <= b[1][0] <= max_X):
            return True
        return False

    # Use line a as anchor
    b_left_of_a = on_left(a[0], a[1], b[0]) and on_left(a[0], a[1], b[1])
    b_right_of_a = on_right(a[0], a[1], b[0]) and on_right(a[0], a[1], b[1])

    if b_left_of_a or b_right_of_a:
        return False

    # Use line b as anchor
    a_left_of_b = on_left(b[0], b[1], a[0]) and on_left(b[0], b[1], a[1])
    a_right_of_b = on_right(b[0], b[1], a[0]) and on_right(b[0], b[1], a[1])

    if a_left_of_b or a_right_of_b:
        return False

    return True

def on_left(a, b, c):
    """ Checks if c is on the left of line segment ab. """
    return area2(a, b, c) > 0

def on_right(a, b, c):
    """ Checks if c is on the right of line segment ab. """
    return area2(a, b, c) < 0

