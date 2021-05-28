"""Copyright Mechanisms Underlying Behavior Lab, Singapore
https://mechunderlyingbehavior.wordpress.com/

write.py is part of the locomotion python package for analyzing locomotory animal
behaviors via the techniques presented in the paper "Computational geometric tools
for quantitative comparison of locomotory behavior" by MT Stamps, S Go, and AS Mathuru
(https://doi.org/10.1038/s41598-019-52300-8).

This python script contains methods for writing outputs to file and rendering the
results from the functions defined in trajectory.py and heatmap.py. The current
implementation makes use of plotly to render graphs, which produces a .html that
enables the user to look interact with the resulting plots.
"""
import os
import csv
import operator
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

COLORS = plotly.colors.qualitative.Plotly

def post_process(animal_list, dists, outdir, outfilename, sort_table,
                 square_table, color_min=0.0, color_max=1.0):
    """ Writes and renders the outputs of computeAllBDD() or ComputeAllCSD().

    Parameters
    ----------
    animal_list : list of Animal() objects
        List of initialized Animal() objects to be compared. Should coincide with the
        Animal() objects used to calculate dists, and the order should match dists.
    dists : 2D array of floats (upper-triangular, empty diagonal)
        dists[i][j] is the distances between trajectories of animal[i] and animal[j].
    outdir : str
        Absolute path to the output directory for the .csv and the .html files exported by
        the function.
    outfilename : str
        Name that will be given to the files printed by this function.
    sort_table : bool
        If True, dists will be sorted according to average row/column values.
    square_table: bool
        If True, dists will be mirrored along the diagonal to fill up the empty entries.
    color_min : float, optional
        Defines the minimum value on the color scale. Default value: 0.0.
    color_max : float, optional
        Defines the maximum value on the color scale. Default value: 1.0.
    """
    # pylint:disable=too-many-arguments
    # pylint:disable=too-many-locals
    # pylint:disable=too-many-branches
    num_animals = len(animal_list)
    if square_table:
        for i in range(num_animals):
            for j in range(i):
                dists[i][j] = dists[j][i]
        write_dist_table_to_csv(animal_list, dists, outdir, outfilename+".csv")
        write_dist_table_to_heatmap(animal_list, dists, outdir,
                                    outfilename+".html", color_min, color_max)
    else:
        write_dist_table_to_csv(animal_list, dists, outdir, outfilename+".csv")
        write_dist_table_to_heatmap(animal_list, dists, outdir,
                                    outfilename+".html", color_min, color_max)
    if sort_table:
        dist_means = {}
        for i in range(num_animals):
            dlist = [dists[j][i] for j in range(i)] + \
                [dists[i][j] for j in range(i+1, num_animals)]
            dist_means.update({animal_list[i]:np.mean(dlist)})
        sorted_dists = sorted(dist_means.items(), key=operator.itemgetter(1))
        sorted_indices = [animal_list.index(sorted_dists[i][0])
                          for i in range(num_animals)]
        new_dists = [['' for i in range(num_animals)] for j in range(num_animals)]
        for i in range(num_animals):
            for j in range(i+1, num_animals):
                if sorted_indices[j] > sorted_indices[i]:
                    new_dists[i][j] = dists[sorted_indices[i]][sorted_indices[j]]
                else:
                    new_dists[i][j] = dists[sorted_indices[j]][sorted_indices[i]]
        dists = new_dists
        animal_list = [animal_list[sorted_indices[i]] for i in range(num_animals)]

        if square_table:
            for i in range(num_animals):
                for j in range(i):
                    dists[i][j] = dists[j][i]
        write_dist_table_to_csv(animal_list, dists, outdir,
                                "%s" % outfilename+"_sorted.csv")
        write_dist_table_to_heatmap(animal_list, dists, outdir,
                                    "%s" % outfilename+"_sorted.html",
                                    color_min, color_max)


def plot_heatmap(animal, outdir, add_path=True):
    """ Plot heatmap representing the frequencies used for CSD.

    Given a 2D matrix of frequencies where frequencies[x][y] represents the number of
    frames that animal spends in the grid indexed by x and y, plot and save a heatmap
    that represents the frequencies in the matrix.

    Parameters
    ----------
    animal: Animal() object
        Animal object containing the coordinate data to be plotted.
        heatmap.populate_surface_data(animal) should be called before
        running plot_heatmap.
    frequencies: 2-dimensional array of int
        Frequency matrix corresponding to animal's frequency in each grid.
    outdir: str
        File path to output directory where the output is to be saved.
    """
    animal_name = animal.get_name()
    freqs = animal.get_frequencies()
    filename = "plot_%s_frequency_heatmap" % (animal_name)
    html_outpath = os.path.join(outdir, filename + '.html').replace(' ', '')
    png_outpath = os.path.join(outdir, filename + '.png').replace(' ', '')
    x_count, y_count = animal.get_grid_counts()
    x_len, y_len = animal.get_grid_lens()
    x_lims, y_lims = animal.get_lims()
    z = np.array([[np.log(freqs[i][j]+1) for i in range(x_count)]
                            for j in range(y_count)])
    z = z/np.max(z)
    x = np.arange(x_lims[0], x_lims[1]+x_len, x_len)
    y = np.arange(y_lims[0], y_lims[1]+y_len, y_len)

    figure = {'data': [], 'layout': {}}
    trace = go.Heatmap(
        z=z, x=x, y=y,
        name='Heatmap',
        colorscale='plasma',
        showscale=True,
        visible=True,
        zmin=0,
        zmax=1
    )
    figure['data'].append(trace)

    if add_path:
        x_vals = animal.get_raw_vals('X')
        y_vals = animal.get_raw_vals('Y')
        path = go.Scatter(x=x_vals, y=y_vals,
                          mode='lines', showlegend=False,
                          marker={'color':'rgba(0,255,255,0.7)'},
                          line={'width':2},
                          name=animal_name)
        figure['data'].append(path)

    figure['layout'] = dict(
        height=500, width=int(400*(x_lims[1]-x_lims[0])/(y_lims[1]-y_lims[0])),
        title="Frequency Heatmap for %s" % animal_name,
        xaxis={'title': "X axis (%s)" % animal.get_output_unit(),
               'range': x_lims},
        yaxis={'title': "Y axis (%s)" % animal.get_output_unit(),
               'range': y_lims}
    )
    fig = go.Figure(figure)
    fig.write_image(png_outpath)
    plotly.offline.plot(figure, filename=html_outpath, auto_open=False)
    print("Saved heatmap in %s" % html_outpath)


def plot_path(animal, outdir):
    """ Plot path of animal.

    Given the X and Y coordinates stored in animal, plot the path of the animal,
    using x_lims and y_lims to determine the size of the canvas.

    Parameters
    ----------
    animal: Animal() object
        Animal object containing the coordinate data to be plotted.
    outdir: str
        File path to output directory where the output is to be saved.
    """
    animal_name = animal.get_name()
    filename = "plot_%s_smoothened_path" % (animal_name)
    html_outpath = os.path.join(outdir, filename + '.html').replace(' ', '')
    png_outpath = os.path.join(outdir, filename + '.png').replace(' ', '')

    x_lims, y_lims = animal.get_lims()
    width, height = x_lims[1]-x_lims[0], y_lims[1]-y_lims[0]
    x_vals = animal.get_raw_vals('X')
    y_vals = animal.get_raw_vals('Y')

    data = [
        go.Scatter(x=x_vals, y=y_vals,
                   mode='lines', showlegend=False,
                   marker={'color':'cyan'},
                   line={'width':3},
                   name=animal_name)
    ]
    figure = {
        'data': data,
        'layout': {'height': 500, 'width': int(400*width/height),
                   'title': "Smoothened Coordinates for %s" % animal_name,
                   'plot_bgcolor' : 'white',
                   'xaxis': {'title': "X axis (%s)" % animal.get_output_unit(),
                             'range': x_lims},
                   'yaxis': {'title': "Y axis (%s)" % animal.get_output_unit(),
                             'range': y_lims, 'scaleanchor': 'x', 'scaleratio': 1}}
    }
    fig = go.Figure(figure)
    fig.write_image(png_outpath)
    plotly.offline.plot(figure, filename=html_outpath, auto_open=False)
    print("Saved path plot in %s" % html_outpath)


def render_alignment(alignment, animal_obj_0, animal_obj_1, varnames, outdir):
    """ Prints the alignment plot between 2 animal objects.

    Given the alignment arrays and the respective animal objects, generates and
    exports the alignment graph between the two objects. Outputs a .html file of the
    Plotly plot.

    Parameters
    ----------
    alignment : 2-tuple of numpy arrays
        Contains the arrays of indices for the alignment. Each array should be of the same
        length, and should correspond to the respective animal.
    animal_obj_0/1 : Animal() object
        For each respective animal.
    varnames : list of strs
        Each string should be linked to the variable names in the animal objects.
    outdir : str
        Absolute path to the output directory for the .html file.
    """
    # pylint: disable=too-many-locals
    filename = "figure_%s-%s_%s_alignment" % (animal_obj_0.get_name(),
                                              animal_obj_1.get_name(),
                                              '-'.join(varnames))
    html_outpath = os.path.join(outdir, filename + '.html').replace(' ', '')
    png_outpath = os.path.join(outdir, filename + '.png').replace(' ', '')
    fr_0 = animal_obj_0.get_frame_rate()
    fr_1 = animal_obj_1.get_frame_rate()
    nums = len(alignment[0])
    data = []
    for i in range(nums):
        data.append(go.Scatter(x=[0, alignment[0][i]/fr_0, alignment[0][i]/fr_0],
                               y=[alignment[1][i]/fr_1, alignment[1][i]/fr_1, 0],
                               mode='lines',
                               marker={'color':'black'},
                               showlegend=False, opacity=0.1))
    trace = go.Scatter(x=alignment[0]/fr_0, y=alignment[1]/fr_1, mode='lines',
                       showlegend=False, line={'width':4},
                       marker={'color':'yellow'})
    data.append(trace)
    figure = {'data':data,
              'layout':{'height':500, 'width':500,
                        'title' : '%s - %s Alignment Graph' % (animal_obj_0.get_name(),
                                                               animal_obj_1.get_name()),
                        'plot_bgcolor' : 'white',
                        'xaxis':{'title': '%s Time (s)' % animal_obj_0.get_name()},
                        'yaxis': {'title': '%s Time (s)' % animal_obj_1.get_name()}}}
    fig = go.Figure(figure)
    fig.write_image(png_outpath)
    plotly.offline.plot(figure, filename=html_outpath, auto_open=False)
    print("Saved alignment graph in %s" % html_outpath)


def render_aligned_graphs(points_0, points_1, alignment,
                          animal_obj_0, animal_obj_1, seg_len, varname, outdir):
    """ Prints the aligned graphs based on a certain variable between both animals.

    Exports a series of Plotly plots as .html files corresponding to various comparison
    plots for two Animal() objects on a given variable, considering the original values
    and the distorted values obtained by using the alignment.

    Parameters
    ---------
    points_0/1 : numpy array of floats
        Correspond to the values of the given variable on each frame for each
        respective animal.
    alignment : 2-tuple of numpy arrays
        Contains the arrays of indices for the alignment. Each array should be of the same
        length, and should correspond to the respective animal.
    animal_obj_0/1 : Animal() object
        For each respective animal.
    seg_len : float
        Length of the segment (in minutes).
    varname : str
        The name of the variable to be plotted.
    outdir : str
        Absolute path to the output directory for the .html files.
    """
    # pylint:disable=too-many-arguments
    # pylint:disable=too-many-locals
    # pylint:disable=too-many-statements
    filename = "figure_%s-%s_%s_plots" % (animal_obj_0.get_name(),
                                          animal_obj_1.get_name(),
                                          varname)
    html_outpath = os.path.join(outdir, filename + '.html').replace(' ', '')
    png_outpath = os.path.join(outdir, filename + '.png').replace(' ', '')
    nums = len(alignment[0])

    fr_0 = animal_obj_0.get_frame_rate()
    fr_1 = animal_obj_1.get_frame_rate()
    fr_param = nums/(seg_len * 60)

    original_trace_0 = go.Scatter(x=[alignment[0][k]/fr_0 for k in range(nums)],
                                  y=[points_0[alignment[0][k]] for k in range(nums)],
                                  mode='lines', showlegend=False, marker={'color':'cyan'},
                                  line={'width':3}, name=animal_obj_0.get_name())
    original_trace_1 = go.Scatter(x=[alignment[1][k]/fr_1 for k in range(nums)],
                                  y=[points_1[alignment[1][k]] for k in range(nums)],
                                  mode='lines', showlegend=False, marker={'color':'magenta'},
                                  line={'width':3}, name=animal_obj_1.get_name())
    reparam_trace_0 = go.Scatter(x=[k/fr_param for k in range(nums)],
                                 y=[points_0[alignment[0][k]] for k in range(nums)],
                                 mode='lines', showlegend=False, marker={'color':'cyan'},
                                 line={'width':3}, name=animal_obj_0.get_name())
    reparam_trace_1 = go.Scatter(x=[k/fr_param for k in range(nums)],
                                 y=[points_1[alignment[1][k]] for k in range(nums)],
                                 mode='lines', showlegend=False, marker={'color':'magenta'},
                                 line={'width':3}, name=animal_obj_1.get_name())
    original_data_pair = []
    reparam_data_pair = []
    original_data_0 = []
    original_data_1 = []
    reparam_data_0 = []
    reparam_data_1 = []

    for i in range(nums):
        original_data_pair.append(go.Scatter(x=[alignment[0][i]/fr_0, alignment[1][i]/fr_1],
                                             y=[points_0[alignment[0][i]],
                                                points_1[alignment[1][i]]],
                                             mode='lines', marker={'color':'black'},
                                             showlegend=False, opacity=0.1))
        reparam_data_pair.append(go.Scatter(x=[i/fr_param, i/fr_param],
                                            y=[points_0[alignment[0][i]],
                                               points_1[alignment[1][i]]],
                                            mode='lines', marker={'color':'black'},
                                            showlegend=False, opacity=0.1))
        original_data_0.append(go.Scatter(x=[alignment[0][i]/fr_0, alignment[0][i]/fr_0],
                                          y=[0, points_0[alignment[0][i]]],
                                          mode='lines', marker={'color':'black'},
                                          showlegend=False, opacity=0.1))
        original_data_1.append(go.Scatter(x=[alignment[1][i]/fr_1, alignment[1][i]/fr_1],
                                          y=[0, points_1[alignment[1][i]]],
                                          mode='lines', marker={'color':'black'},
                                          showlegend=False, opacity=0.1))
        reparam_data_0.append(go.Scatter(x=[i/fr_param, i/fr_param],
                                         y=[0, points_0[alignment[0][i]]],
                                         mode='lines', marker={'color':'black'},
                                         showlegend=False, opacity=0.1))
        reparam_data_1.append(go.Scatter(x=[i/fr_param, i/fr_param],
                                         y=[0, points_1[alignment[1][i]]],
                                         mode='lines', marker={'color':'black'},
                                         showlegend=False, opacity=0.1))

    original_data_pair.append(original_trace_0)
    original_data_pair.append(original_trace_1)
    reparam_data_pair.append(reparam_trace_0)
    reparam_data_pair.append(reparam_trace_1)
    original_data_0.append(original_trace_0)
    original_data_1.append(original_trace_1)
    reparam_data_0.append(reparam_trace_0)
    reparam_data_1.append(reparam_trace_1)

    fig = make_subplots(rows=6, cols=1,
                        subplot_titles=["Matched %s" % varname,
                                        "Aligned %s" % varname,
                                        "%s - %s" % (animal_obj_0.get_name(), varname),
                                        "%s - Warped %s" % (animal_obj_0.get_name(), varname),
                                        "%s - %s" % (animal_obj_1.get_name(), varname),
                                        "%s - Warped %s" % (animal_obj_1.get_name(), varname)])

    # Matched Plots
    for trace in original_data_pair:
        fig.add_trace(trace, row=1, col=1)
    fig.update_xaxes(title_text='Real Time (s)', row=1, col=1,
                     linecolor='#555', gridcolor='#DDD', gridwidth=0.5)
    fig.update_yaxes(title_text='Normalized %s (au)' % varname, range=[0, 1], row=1, col=1,
                     linecolor='#555', gridcolor='#DDD', gridwidth=0.5)

    # Aligned Plots
    for trace in reparam_data_pair:
        fig.add_trace(trace, row=2, col=1)
    fig.update_xaxes(title_text='Warped Time (s)', row=2, col=1,
                     linecolor='#555', gridcolor='#DDD', gridwidth=0.5)
    fig.update_yaxes(title_text='Normalized %s (au)' % varname, range=[0, 1], row=2, col=1,
                     linecolor='#555', gridcolor='#DDD', gridwidth=0.5)

    # Animal 0 Regular Plot
    for trace in original_data_0:
        fig.add_trace(trace, row=3, col=1)
    fig.update_xaxes(title_text='%s Time (s)' % animal_obj_0.get_name(), row=3, col=1,
                     linecolor='#555', gridcolor='#DDD', gridwidth=0.5)
    fig.update_yaxes(title_text='Normalized %s (au)' % varname, range=[0, 1], row=3, col=1,
                     linecolor='#555', gridcolor='#DDD', gridwidth=0.5)

    # Animal 0 Warped Plot
    for trace in reparam_data_0:
        fig.add_trace(trace, row=4, col=1)
    fig.update_xaxes(title_text='Warped Time (s)', row=4, col=1,
                     linecolor='#555', gridcolor='#DDD', gridwidth=0.5)
    fig.update_yaxes(title_text='Normalized %s (au)' % varname, range=[0, 1], row=4, col=1,
                     linecolor='#555', gridcolor='#DDD', gridwidth=0.5)

    # Animal 1 Regular Plot
    for trace in original_data_1:
        fig.add_trace(trace, row=5, col=1)
    fig.update_xaxes(title_text='%s Time (s)' % animal_obj_1.get_name(), row=5, col=1,
                     linecolor='#555', gridcolor='#DDD', gridwidth=0.5)
    fig.update_yaxes(title_text='Normalized %s (au)' % varname, range=[0, 1], row=5, col=1,
                     linecolor='#555', gridcolor='#DDD', gridwidth=0.5)

    # Animal 0 Warped Plot
    for trace in reparam_data_1:
        fig.add_trace(trace, row=6, col=1)
    fig.update_xaxes(title_text='Warped Time (s)', row=6, col=1,
                     linecolor='#555', gridcolor='#DDD', gridwidth=0.5)
    fig.update_yaxes(title_text='Normalized %s (au)' % varname, range=[0, 1], row=6, col=1,
                     linecolor='#555', gridcolor='#DDD', gridwidth=0.5)

    fig.update_layout(height=2400, width=1000,
                      title='%s-%s %s Alignment Figures' % (animal_obj_0.get_name(),
                                                            animal_obj_1.get_name(),
                                                            varname),
                      title_font_size=24,
                      plot_bgcolor='white',
                      showlegend=False)

    fig.write_image(png_outpath)
    plotly.offline.plot(fig, filename=html_outpath, auto_open=False)
    print("Saved the alignment graphs in directory %s" % outdir)


def render_single_animal_graph(points, animal_obj, varname, outdir):
    """ Renders the time-series graph for the variables of a given Animal() object.

    Exports an offline Plotly plot of variable over frame for the given Animal() object.

    Parameters
    ----------
    points : np.array of floats
        Correspond to the values of the given variable on each frame for the animal.
    animal_obj : Animal() object
        The Animal() object corresponding to the given animal.
    varname : str
        The name of the variable to be plotted.
    outdir : str
        Absolute path to the output directory for the .html file.
    """
    filename = "figure_%s_%s" % (animal_obj.get_name(), varname)
    html_outpath = os.path.join(outdir, filename + '.html').replace(' ', '')
    png_outpath = os.path.join(outdir, filename + '.png').replace(' ', '')
    num_points = len(points)
    trace = go.Scatter(x=range(num_points)/animal_obj.get_frame_rate(), y=points,
                       mode='lines', showlegend=False, line={'width':4})
    data = [trace]
    fig = go.Figure()
    fig.add_trace(trace)
    fig.write_image(png_outpath)
    plotly.offline.plot(data, filename=html_outpath, auto_open=False)
    print("Saved single animal graph in %s" % html_outpath)


def write_dist_table_to_csv(animal_list, results, outdir, outfilename):
    """ Exports the matrix of distances to a .csv file.

    Takes in a matrix of pair-wise distances between a list of animals and writes the
    results in the specified outfile name (expects absolute/full path).

    Parameters
    ----------
    animal_list : list of Animal() objects
        Corresponds to the animals that the pair-wise distances were calculated for.
        Order is assumed to match the order of the results.
    results : 2D array of floats (upper-triangular, empty diagonal)
        results[i][j] is the distances between trajectories of animal[i] and animal[j].
    outdir : str
        Absolute path to the output directory for the .csv files exported by the function.
    outfilename : str
        Name that will be given to the files printed by this function.
    """
    num_animals = len(animal_list)
    outpath = os.path.join(outdir, outfilename)
    with open(outpath, 'w') as outfile:
        csvwriter = csv.writer(outfile, delimiter=',')
        csvwriter.writerow([''] + [animal_obj.get_name()
                                   for animal_obj in animal_list])
        for i in range(num_animals):
            csvwriter.writerow([animal_list[i].get_name()] +
                               ['' if results[i][j] == ''
                                else'%.5f' % results[i][j] for j in range(num_animals)])
    print("LOG: Wrote the results in %s" % outpath)


def write_dist_table_to_heatmap(animal_list, results, outdir,
                                outfilename, color_min=0.0, color_max=1.0):
    """ Exports the matrix of distances to a Plotly heatmap.

    Takes in a matrix of pair-wise distances between a list of animals and produces a
    Plotly heatmap that visually represents the distance between each animal.

    Parameters
    ----------
    animal_list : list of Animal() objects
        Corresponds to the animals that the pair-wise distances were calculated for.
        Order is assumed to match the order of the results.
    results : 2D array of floats (upper-triangular, empty diagonal)
        results[i][j] is the distances between trajectories of animal[i] and animal[j].
    outdir : str
        Absolute path to the output directory for the .csv files exported by the function.
    outfilename : str
        Name that will be given to the files printed by this function.
    color_min : float, optional
        Defines the minimum value on the color scale. Default value: 0.0.
    color_max : float, optional
        Defines the maximum value on the color scale. Default value: 1.0.
    """
    # pylint:disable=too-many-arguments
    nums = len(animal_list)
    outpath = os.path.join(outdir, outfilename)
    figure = {'data':[], 'layout':{}}
    trace = go.Heatmap(z=[[results[nums-i-1][j] for j in range(nums)]
                          for i in range(nums)],
                       name='Heat Map',
                       x=[animal_list[j].get_name() for j in range(nums)],
                       y=[animal_list[nums-i-1].get_name() for i in range(nums)],
                       colorscale='Viridis',
                       showscale=True,
                       visible=True,
                       zmin=color_min,
                       zmax=color_max)
    figure['data'].append(trace)
    figure['layout'] = dict(height=600, width=630,
                            margin=go.layout.Margin(l=100, r=100, b=100, t=100),
                            showlegend=False,
                            xaxis={'showticklabels':False, 'showgrid':False, 'ticks':''},
                            yaxis={'showticklabels':False, 'showgrid':False, 'ticks':''},
                            annotations=[dict(x=j+0.0,
                                              y=nums+1.0,
                                              text=animal_list[j].get_name(),
                                              font={'color':COLORS[animal_list[j].get_group()],
                                                    'size':7},
                                              textangle=-45, showarrow=False)
                                         for j in range(nums)]
                            +[dict(x=nums+1.0, y=i+0.0,
                                   text=animal_list[nums-i-1].get_name(),
                                   font={'color':COLORS[animal_list[nums-i-1].get_group()],
                                         'size':7},
                                   textangle=0, showarrow=False)
                              for i in range(nums)])
    plotly.offline.plot(figure, filename=outpath)
    print("LOG: Plot the heatmap in %s" % outpath)


def write_off(animal_obj, coordinates, outdir, filename):
    """ Exports triangular mesh to an .off file.

    Records the vertex coordinates and list of triangles for the triangular mesh
    of an Animal() object in an .off file for visualization purposes.

    Parameters
    ----------
    animal_obj : Animal() object
        Corresponds to the animals that the pair-wise distances were calculated for.
        Order is assumed to match the order of the results.
    coordinates : list of triples of floats
        Regular (3D) coordinates that the animal's triangulation is defined for.
    outdir : str
        Absolute path to the output directory for the .off file exported by the function.
    outfilename : str
        Name that will be given to the file printed by this function.
    """
    outpath = os.path.join(outdir, filename).replace(' ', '')
    triangles = animal_obj.get_triangulation()
    colors = animal_obj.get_colors()
    print("Writing triangulation to file %s..." % outpath)
    with open(outpath, 'w') as outfile:
        outfile.write("OFF\n")
        outfile.write("%d %d %d\n" % (len(coordinates), len(triangles), 0))
        for coord in coordinates:
            outfile.write("%f %f %f\n" % (coord[0], coord[1], coord[2]))
        for triangle in triangles:
            color = colors[triangles.index(triangle)]
            outfile.write("%d %d %d %d %f %f %f\n" % (3, triangle[0],
                                                      triangle[1], triangle[2],
                                                      color[0], color[1], color[2]))


def write_iibdd_to_csv(animal_list, results, outdir, outfilename):
    """ Exports IIBDD results into a .csv file.

    Given the results from running compute_all_iibdd() over animal_list, exports the
    results into a .csv file.

    Parameters
    ----------
    animal_list : list of Animal() objects
        Corresponds to the animals that the intra-individual BDD was calculated for.
    results : list of floats
        List of IIBDD calculated for each animal in animal_list. Should have the same
        length as animal_list, and the order should also match.
    outdir : str
        Absolute path to the output directory for the .csv files exported by the function.
    outfilename : str
        Name that will be given to the files printed by this function.
    """
    # ensure animal_list and results have same length
    if len(animal_list) != len(results):
        raise Exception("write_iibdd_to_csv : animal_list and results must have same length.")
    header = [animal_obj.get_name() for animal_obj in animal_list]
    with open(os.path.join(outdir, outfilename), 'w') as outfile:
        csvwriter = csv.writer(outfile, delimiter=',')
        csvwriter.writerow(header)
        csvwriter.writerow(results)
    print("Saved results into %s" % outfilename)
