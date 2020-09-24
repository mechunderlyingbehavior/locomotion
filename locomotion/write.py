"""Copyright Mechanisms Underlying Behavior Lab, Singapore
https://mechunderlyingbehavior.wordpress.com/

write.py is part of the locomotion package comparing
animal behaviours, developed to support the work discussed
in paper Computational Geometric Tools for Modeling Inherent
Variability in Animal Behavior (MT Stamps, S Go, and AS Mathuru)

This python script contains methods for displaying the results from the
functions defined in trajectory.py and heatmap.pt. Current implementation makes
use of plotly to plot the results, and produces a .html that allows for the user
to look into the results with greater detail.
"""
import os
import csv
import itertools
import operator
import plotly
import plotly.graph_objs as go
import numpy as np

def write_dist_table_to_csv(animal_list, results, outdir, outfilename):
    """
    Takes in a matrix of DTW results in order specified in fileList
    Writes the results in the specified outfile name (expects absolute/full path)
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
    """
    Takes in a matrix of DTW results in order specified in fileList
    Writes the results in the specified outfile name (expects absolute/full path)
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
                            margin=go.Margin(l=100, r=100, b=100, t=100),
                            showlegend=False,
                            xaxis={'showticklabels':False, 'showgrid':False, 'ticks':''},
                            yaxis={'showticklabels':False, 'showgrid':False, 'ticks':''},
                            annotations=[dict(x=j+0.5,
                                              y=nums+1.0,
                                              text=animal_list[j].get_name()[4:]
                                              if animal_list[j].get_control_boolean()
                                              else animal_list[j].get_name()[4:]+' ',
                                              font={'color':'cyan'
                                                            if animal_list[j].get_control_boolean()
                                                            else 'magenta',
                                                    'size':7},
                                              textangle=-45, showarrow=False)
                                         for j in range(nums)]
                            +[dict(x=nums+1.0, y=i+0.0,
                                   text=animal_list[nums-i-1].get_name()[4:]
                                   if animal_list[nums-i-1].get_control_boolean()
                                   else animal_list[nums-i-1].get_name()[4:]+' ',
                                   font={'color':'cyan'
                                                 if animal_list[nums-i-1].get_control_boolean()
                                                 else 'magenta', 'size':7},
                                   textangle=0, showarrow=False)
                              for i in range(nums)])
    plotly.offline.plot(figure, filename=outpath)
    print("LOG: Plot the heatmap in %s" % outpath)


def write_segment_exps_to_csv(animal_list, results, means, stds, outdir, outfilename):
    """
    Writes the results from trajectory.runRandomSegmentComparisons() to CSV
    results is a table where results[i][j] is the j-th segment comparison for
    the i-th animal in animal_list. Note that the entry is a double, (segment
    lenth, distance).
    """
    # pylint:disable=too-many-arguments
    num_animals = len(animal_list)
    if means is None:
        header_top = list(itertools.chain.from_iterable([[animal_obj.get_name(), ""]
                                                         for animal_obj in animal_list]))
        header_bottom = ["Segment Length", "Distance"] * num_animals
        num_exps = len(results[0])
        with open(os.path.join(outdir, outfilename), 'w') as outfile:
            csvwriter = csv.writer(outfile, delimiter=',')
            csvwriter.writerow(header_top)
            csvwriter.writerow(header_bottom)
            for i in range(num_exps):
                row = list(itertools.chain.from_iterable([results[j][i]
                                                          for j in range(num_animals)]))
                csvwriter.writerow(row)

    else: #if means is not None:
        header_top = [""]+[animal_obj.get_name() for animal_obj in animal_list]
        header_bottom = ["Segment Length"] + ["Distance"]*num_animals
        num_exps = len(results[0])
        with open(os.path.join(outdir, outfilename), 'w') as outfile:
            csvwriter = csv.writer(outfile, delimiter=',')
            csvwriter.writerow(header_top)
            csvwriter.writerow(header_bottom)
            for i in range(num_exps):
                row = [results[0][i][0]] + [results[j][i][1] for j in range(num_animals)]
                csvwriter.writerow(row)
                num_segs = len(means[0])

            #write the mean information
            csvwriter.writerow([""])
            csvwriter.writerow(["Means"])
            csvwriter.writerow(header_top)
            csvwriter.writerow(header_bottom)
            for i in range(num_segs):
                row = [means[0][i][0]] + [means[j][i][1] for j in range(num_animals)]
                csvwriter.writerow(row)
            #write the std information
            csvwriter.writerow([""])
            csvwriter.writerow(["Standard Deviations"])
            csvwriter.writerow(header_top)
            csvwriter.writerow(header_bottom)
            for i in range(num_segs):
                row = [stds[0][i][0]] + [stds[j][i][1] for j in range(num_animals)]
                csvwriter.writerow(row)
        print("Saved the table in %s" % outfile)


def render_single_animal_graph(points, animal_obj, varname, outdir):
    """
    To be documented.
    """
    filename = "figure_%s_%s.html" % (animal_obj.get_name(), varname)
    outpath = os.path.join(outdir, filename).replace(' ', '')
    num_points = len(points)
    trace = go.Scatter(x=range(num_points), y=points, mode='lines',
                       showlegend=False, line={'width':4})
    data = [trace]
    plotly.offline.plot(data, filename=outpath, auto_open=False)

    print("Saved single animal graph in %s" % outpath)


def render_aligned_graphs(points_0, points_1, alignment,
                          animal_obj_0, animal_obj_1, varname, outdir):
    """
    To be documented.
    """
    # pylint:disable=too-many-arguments
    # pylint:disable=too-many-locals
    # pylint:disable=too-many-statements
    matched_filename = "figure_%s-%s_matched_%s.html" % (animal_obj_0.get_name(),
                                                         animal_obj_1.get_name(),
                                                         varname)
    aligned_filename = "figure_%s-%s_aligned_%s.html" % (animal_obj_0.get_name(),
                                                         animal_obj_1.get_name(),
                                                         varname)
    original_filename_0 = "figure_%s_%s.html" % (animal_obj_0.get_name(), varname)
    original_filename_1 = "figure_%s_%s.html" % (animal_obj_1.get_name(), varname)
    reparam_filename_0 = "figure_%s_%s_warped_by_%s.html" % (animal_obj_0.get_name(),
                                                             varname,
                                                             animal_obj_1.get_name())
    reparam_filename_1 = "figure_%s_%s_warped_by_%s.html" % (animal_obj_1.get_name(),
                                                             varname,
                                                             animal_obj_0.get_name())
    fulloutpath_matched = os.path.join(outdir, matched_filename).replace(' ', '')
    fulloutpath_aligned = os.path.join(outdir, aligned_filename).replace(' ', '')
    fulloutpath_original_0 = os.path.join(outdir, original_filename_0).replace(' ', '')
    fulloutpath_original_1 = os.path.join(outdir, original_filename_1).replace(' ', '')
    fulloutpath_reparam_0 = os.path.join(outdir, reparam_filename_0).replace(' ', '')
    fulloutpath_reparam_1 = os.path.join(outdir, reparam_filename_1).replace(' ', '')
    nums = len(alignment[0])
    original_trace_0 = go.Scatter(x=[alignment[0][k] for k in range(nums)],
                                  y=[points_0[alignment[0][k]] for k in range(nums)],
                                  mode='lines', showlegend=False,
                                  line={'width':3}, name=animal_obj_0.get_name())
    original_trace_1 = go.Scatter(x=[alignment[1][k] for k in range(nums)],
                                  y=[points_1[alignment[1][k]] for k in range(nums)],
                                  mode='lines', showlegend=False,
                                  line={'width':3}, name=animal_obj_1.get_name())
    reparam_trace_0 = go.Scatter(x=[k*alignment[0][-1]/nums for k in range(nums)],
                                 y=[points_0[alignment[0][k]] for k in range(nums)],
                                 mode='lines', showlegend=False,
                                 line={'width':3}, name=animal_obj_0.get_name())
    reparam_trace_1 = go.Scatter(x=[k*alignment[1][-1]/nums for k in range(nums)],
                                 y=[points_1[alignment[1][k]] for k in range(nums)],
                                 mode='lines', showlegend=False,
                                 line={'width':3}, name=animal_obj_1.get_name())
    original_data_pair = []
    reparam_data_pair = []
    original_data_0 = []
    original_data_1 = []
    reparam_data_0 = []
    reparam_data_1 = []

    for i in range(nums):
        original_data_pair.append(go.Scatter(x=[alignment[0][i], alignment[1][i]],
                                             y=[points_0[alignment[0][i]],
                                                points_1[alignment[1][i]]],
                                             mode='lines', marker={'color':'black'},
                                             showlegend=False, opacity=0.1))
        reparam_data_pair.append(go.Scatter(x=[i*alignment[0][-1]/nums, i*alignment[0][-1]/nums],
                                            y=[points_0[alignment[0][i]],
                                               points_1[alignment[1][i]]],
                                            mode='lines', marker={'color':'black'},
                                            showlegend=False, opacity=0.1))
        original_data_0.append(go.Scatter(x=[alignment[0][i], alignment[0][i]],
                                          y=[0, points_0[alignment[0][i]]],
                                          mode='lines', marker={'color':'black'},
                                          showlegend=False, opacity=0.1))
        original_data_1.append(go.Scatter(x=[alignment[1][i], alignment[1][i]],
                                          y=[0, points_1[alignment[1][i]]],
                                          mode='lines', marker={'color':'black'},
                                          showlegend=False, opacity=0.1))
        reparam_data_0.append(go.Scatter(x=[i*alignment[0][-1]/nums, i*alignment[0][-1]/nums],
                                         y=[0, points_0[alignment[0][i]]],
                                         mode='lines', marker={'color':'black'},
                                         showlegend=False, opacity=0.1))
        reparam_data_1.append(go.Scatter(x=[i*alignment[0][-1]/nums, i*alignment[0][-1]/nums],
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

    matched_figure = {'data':original_data_pair,
                      'layout':{'height':350, 'width':1000,
                                'xaxis':{'title': 'Real Time'},
                                'yaxis':{'title': varname, 'range':[0, 1]}},
                      'frames':[]}
    aligned_figure = {'data':reparam_data_pair,
                      'layout':{'height':350, 'width':1000,
                                'xaxis':{'title': 'Warped Time'},
                                'yaxis':{'title': varname, 'range':[0, 1]}},
                      'frames':[]}
    original_figure_0 = {'data':original_data_0,
                         'layout':{'height':350, 'width':1000,
                                   'xaxis':{'title': '%s Time' % animal_obj_0.get_name()},
                                   'yaxis':{'title': varname, 'range':[0, 1]}},
                         'frames':[]}
    original_figure_1 = {'data':original_data_1,
                         'layout':{'height':350, 'width':1000,
                                   'xaxis':{'title': '%s Time' % animal_obj_1.get_name()},
                                   'yaxis':{'title': varname, 'range':[0, 1]}},
                         'frames':[]}
    reparam_figure_0 = {'data':reparam_data_0,
                        'layout':{'height':350, 'width':1000,
                                  'xaxis':{'title': 'Warped Time'},
                                  'yaxis':{'title': varname, 'range':[0, 1]}},
                        'frames':[]}
    reparam_figure_1 = {'data':reparam_data_1,
                        'layout':{'height':350, 'width':1000,
                                  'xaxis':{'title': 'Warped Time'},
                                  'yaxis':{'title': varname, 'range':[0, 1]}},
                        'frames':[]}

    plotly.offline.plot(matched_figure,
                        filename=fulloutpath_matched, auto_open=False)
    plotly.offline.plot(aligned_figure,
                        filename=fulloutpath_aligned, auto_open=False)
    plotly.offline.plot(original_figure_0,
                        filename=fulloutpath_original_0, auto_open=False)
    plotly.offline.plot(original_figure_1,
                        filename=fulloutpath_original_1, auto_open=False)
    plotly.offline.plot(reparam_figure_0,
                        filename=fulloutpath_reparam_0, auto_open=False)
    plotly.offline.plot(reparam_figure_1,
                        filename=fulloutpath_reparam_1, auto_open=False)
    print("Saved the alignment graphs in directory %s" % outdir)


def render_alignment(alignment, animal_obj_0, animal_obj_1, varnames, outdir):
    """
    To be documented.
    """
    filename = "figure_%s-%s_%s_alignment.html" % (animal_obj_0.get_name(),
                                                   animal_obj_1.get_name(),
                                                   '-'.join(varnames))
    outpath = os.path.join(outdir, filename).replace(' ', '')
    nums = len(alignment[0])
    data = []
    for i in range(nums):
        data.append(go.Scatter(x=[0, alignment[0][i], alignment[0][i]],
                               y=[alignment[1][i], alignment[1][i], 0],
                               mode='lines',
                               marker={'color':'black'},
                               showlegend=False, opacity=0.1))
    trace = go.Scatter(x=alignment[0], y=alignment[1], mode='lines',
                       showlegend=False, line={'width':4})
    data.append(trace)
    figure = {'data':data,
              'layout':{'height':500, 'width':500,
                        'xaxis':{'title': '%s Time' % animal_obj_0.get_name()},
                        'yaxis': {'title': '%s Time' % animal_obj_1.get_name()}}}
    plotly.offline.plot(figure, filename=outpath+".html", auto_open=False)
    print("Saved alignment graph in %s" % outpath)


def write_off(animal_obj, coordinates, outdir, filename):
    """
    To be documented.
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


def post_process(animal_list, dists, outdir, outfilename, sort_table,
                 square_table, color_min=0.0, color_max=1.0):
    """
    To be documented.
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
