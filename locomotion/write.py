###################################################################################################
### Record & Render Information
###################################################################################################
import os
import csv
import plotly
import plotly.graph_objs as go
import itertools
import operator
import numpy as np

def writeDistTableToCSV(animal_list, results, outdir, outfilename):
# Takes in a matrix of DTW results in order specified in fileList
# Writes the results in the specified outfile name (expects absolute/full path)
  N = len(animal_list)
  outpath = os.path.join(outdir, outfilename)
  with open(outpath, 'w') as outfile:
    csvwriter = csv.writer(outfile, delimiter=',')
    csvwriter.writerow(['']+[animal_obj.get_name() for animal_obj in animal_list])
    for i in range(N):
      csvwriter.writerow([animal_list[i].get_name()]+['' if results[i][j]=='' else'%.5f' % results[i][j] for j in range(N)])
  print("LOG: Wrote the results in %s" % outpath)

  
def writeDistTableToHeatmap(animal_list, results, outdir, outfilename, color_min=0.0, color_max=1.0):
# Takes in a matrix of DTW results in order specified in fileList
# Writes the results in the specified outfile name (expects absolute/full path)
  N = len(animal_list)
  outpath = os.path.join(outdir, outfilename)
  figure = {'data':[],'layout':{}}
  trace = go.Heatmap(z=[[results[N-i-1][j] for j in range(N)] for i in range(N)], name='Heat Map', 
                      x=[animal_list[j].get_name() for j in range(N)],
                      y=[animal_list[N-i-1].get_name() for i in range(N)],
                      colorscale='Viridis',
                      showscale=True,
                      visible=True,
                      zmin=color_min,
                      zmax=color_max
                     )
  figure['data'].append(trace)
  figure['layout']=dict(height=600,
                        width=630,
                       # margin=go.Margin(l=100,r=100,b=100,t=100),
                        showlegend=False,
                        xaxis={'showticklabels':False,'showgrid':False,'ticks':''},
                        yaxis={'showticklabels':False,'showgrid':False,'ticks':''},
                        annotations=[dict(x=j+0.5,
                                          y=N+1.0,
                                          text=animal_list[j].get_name()[4:] if animal_list[j].in_control_group()
                                          else animal_list[j].get_name()[4:]+' ',
                                          font={'color':'cyan' if animal_list[j].in_control_group()
                                                else 'magenta', 'size':7},
                                          textangle=-45,showarrow=False) for j in range(N)]
                                   +[dict(x=N+1.0,y=i+0.0,text=animal_list[N-i-1].get_name()[4:] if animal_list[N-i-1].in_control_group() else animal_list[N-i-1].get_name()[4:]+' ',
                                          font={'color':'cyan' if animal_list[N-i-1].in_control_group() else 'magenta', 'size':7},
                                          textangle=0,showarrow=False) for i in range(N)])
  plotly.offline.plot(figure, filename=outpath)
  print("LOG: Plot the heatmap in %s" % outpath)


def writeSegmentExpsToCSV(animal_list, results, means, stds, outdir, outfilename):
  # Writes the results from trajectory.runRandomSegmentComparisons() to CSV
  # results is a table where results[i][j] is the j-th segment comparison for the i-th animal in animal_list. Note that the entry is a double, (segment lenth, distance).  
 
  num_animals = len(animal_list)
  if means == None:
    header_top = list(itertools.chain.from_iterable([[animal_obj.get_name(),""] for animal_obj in animal_list]))
    header_bottom = ["Segment Length", "Distance"] * num_animals

    num_exps = len(results[0])

    with open(os.path.join(outdir, outfilename),'w') as outfile:
      csvwriter = csv.writer(outfile, delimiter=',')
      csvwriter.writerow(header_top)
      csvwriter.writerow(header_bottom)
      for i in range(num_exps):
        row = list(itertools.chain.from_iterable([results[j][i] for j in range(num_animals)]))
        csvwriter.writerow(row)
    
  else: #if means != None:
    header_top = [""]+[animal_obj.get_name() for animal_obj in animal_list]
    header_bottom = ["Segment Length"] + ["Distance"]*num_animals

    num_exps = len(results[0])
    with open(os.path.join(outdir, outfilename),'w') as outfile:
      csvwriter = csv.writer(outfile, delimiter=',')
      csvwriter.writerow(header_top)
      csvwriter.writerow(header_bottom)
      for i in range(num_exps):
        #row = list(itertools.chain.from_iterable([results[j][i] for j in range(num_animals)]))
        row = [results[0][i][0]] + [results[j][i][1] for j in range(num_animals)]
        csvwriter.writerow(row)
        num_segs = len(means[0])

      #write the mean information
      csvwriter.writerow([""])
      csvwriter.writerow(["Means"])
      csvwriter.writerow(header_top)
      csvwriter.writerow(header_bottom)
      for i in range(num_segs):
        #row = list(itertools.chain.from_iterable([means[j][i] for j in range(num_animals)]))
        row = [means[0][i][0]] + [means[j][i][1] for j in range(num_animals)]
        csvwriter.writerow(row)
      #write the std information
      csvwriter.writerow([""])
      csvwriter.writerow(["Standard Deviations"])
      csvwriter.writerow(header_top)
      csvwriter.writerow(header_bottom)
      for i in range(num_segs):
        #row = list(itertools.chain.from_iterable([stds[j][i] for j in range(num_animals)]))
        row = [stds[0][i][0]] + [stds[j][i][1] for j in range(num_animals)]
        csvwriter.writerow(row)
    print("Saved the table in %s" % outfile )


def renderSingleAnimalGraph(points, animal_obj, varname, outdir):

  filename = "figure_%s_%s.html" % (animal_obj.get_name(), varname)
  outpath = os.path.join(outdir,filename).replace(' ','')
  N = len(points)
  trace = go.Scatter(x = range(N), y=points, mode='lines',showlegend=False,line={'width':4})
  data = [trace]
  plotly.offline.plot(data, filename=outpath, auto_open=False)

  print("Saved single animal graph in %s" % outpath)


def renderAlignedGraphs( points_0, points_1, alignment, animal_obj_0, animal_obj_1, varname, outdir ):

  matched_filename = "figure_%s-%s_matched_%s.html" % (animal_obj_0.get_name(), animal_obj_1.get_name(), varname )
  aligned_filename = "figure_%s-%s_aligned_%s.html" % (animal_obj_0.get_name(), animal_obj_1.get_name(), varname )
  original_filename_0 = "figure_%s_%s.html" % (animal_obj_0.get_name(), varname )
  original_filename_1 = "figure_%s_%s.html" % (animal_obj_1.get_name(), varname )
  reparam_filename_0 = "figure_%s_%s_warped_by_%s.html" % (animal_obj_0.get_name(), varname, animal_obj_1.get_name() )
  reparam_filename_1 = "figure_%s_%s_warped_by_%s.html" % (animal_obj_1.get_name(), varname, animal_obj_0.get_name() )
  
  fulloutpath_matched = os.path.join( outdir, matched_filename ).replace(' ','')
  fulloutpath_aligned = os.path.join( outdir, aligned_filename ).replace(' ','')
  fulloutpath_original_0 = os.path.join( outdir, original_filename_0 ).replace(' ','')
  fulloutpath_original_1 = os.path.join( outdir, original_filename_1 ).replace(' ','')
  fulloutpath_reparam_0 = os.path.join( outdir, reparam_filename_0 ).replace(' ','')
  fulloutpath_reparam_1 = os.path.join( outdir, reparam_filename_1 ).replace(' ','')

  N = len(alignment[0])

  original_trace_0 = go.Scatter(x = [alignment[0][k] for k in range(N)], y=[points_0[alignment[0][k]] for k in range(N)], \
                                mode='lines',showlegend=False,line={'width':3},name=animal_obj_0.get_name())
  original_trace_1 = go.Scatter(x = [alignment[1][k] for k in range(N)], y=[points_1[alignment[1][k]] for k in range(N)], \
                                mode='lines',showlegend=False,line={'width':3},name=animal_obj_1.get_name())
  reparameterized_trace_0 = go.Scatter(x = [k*alignment[0][-1]/N for k in range(N)], y=[points_0[alignment[0][k]] for k in range(N)], \
                                       mode='lines',showlegend=False,line={'width':3},name=animal_obj_0.get_name())
  reparameterized_trace_1 = go.Scatter(x = [k*alignment[1][-1]/N for k in range(N)], y=[points_1[alignment[1][k]] for k in range(N)], \
                                       mode='lines',showlegend=False,line={'width':3},name=animal_obj_1.get_name()) 
  original_data_pair = []
  reparameterized_data_pair = []
  original_data_0 = []
  original_data_1 = []
  reparameterized_data_0 = []
  reparameterized_data_1 = []

  for i in range(N):
    original_data_pair.append(go.Scatter(x=[alignment[0][i],alignment[1][i]],y=[points_0[alignment[0][i]],points_1[alignment[1][i]]], \
                                         mode='lines',marker={'color':'black'},showlegend=False,opacity=0.1))
    reparameterized_data_pair.append(go.Scatter(x=[i*alignment[0][-1]/N,i*alignment[0][-1]/N],y=[points_0[alignment[0][i]],points_1[alignment[1][i]]], \
                                                mode='lines',marker={'color':'black'},showlegend=False,opacity=0.1))
    original_data_0.append(go.Scatter(x=[alignment[0][i],alignment[0][i]],y=[0,points_0[alignment[0][i]]], \
                                      mode='lines',marker={'color':'black'},showlegend=False,opacity=0.1))
    original_data_1.append(go.Scatter(x=[alignment[1][i],alignment[1][i]],y=[0,points_1[alignment[1][i]]], \
                                      mode='lines',marker={'color':'black'},showlegend=False,opacity=0.1))
    reparameterized_data_0.append(go.Scatter(x=[i*alignment[0][-1]/N,i*alignment[0][-1]/N],y=[0,points_0[alignment[0][i]]], \
                                             mode='lines',marker={'color':'black'},showlegend=False,opacity=0.1))
    reparameterized_data_1.append(go.Scatter(x=[i*alignment[0][-1]/N,i*alignment[0][-1]/N],y=[0,points_1[alignment[1][i]]], \
                                             mode='lines',marker={'color':'black'},showlegend=False,opacity=0.1))

  original_data_pair.append(original_trace_0)
  original_data_pair.append(original_trace_1)
  reparameterized_data_pair.append(reparameterized_trace_0)
  reparameterized_data_pair.append(reparameterized_trace_1)
  original_data_0.append(original_trace_0)
  original_data_1.append(original_trace_1)
  reparameterized_data_0.append(reparameterized_trace_0)
  reparameterized_data_1.append(reparameterized_trace_1)

  matched_figure = {'data':original_data_pair,'layout':{'height':350,'width':1000,'xaxis':{'title': 'Real Time'},'yaxis':{'title': varname,'range':[0,1]}},'frames':[]}
  aligned_figure = {'data':reparameterized_data_pair,'layout':{'height':350,'width':1000,'xaxis':{'title': 'Warped Time'},'yaxis':{'title': varname,'range':[0,1]}},'frames':[]}
  original_figure_0 = {'data':original_data_0,'layout':{'height':350,'width':1000,'xaxis':{'title': '%s Time' % animal_obj_0.get_name()},'yaxis':{'title': varname,'range':[0,1]}},'frames':[]}
  original_figure_1 = {'data':original_data_1,'layout':{'height':350,'width':1000,'xaxis':{'title': '%s Time' % animal_obj_1.get_name()},'yaxis':{'title': varname,'range':[0,1]}},'frames':[]}
  reparam_figure_0 = {'data':reparameterized_data_0,'layout':{'height':350,'width':1000,'xaxis':{'title': 'Warped Time'},'yaxis':{'title': varname,'range':[0,1]}},'frames':[]}
  reparam_figure_1 = {'data':reparameterized_data_1,'layout':{'height':350,'width':1000,'xaxis':{'title': 'Warped Time'},'yaxis':{'title': varname,'range':[0,1]}},'frames':[]}
  
  plotly.offline.plot(matched_figure, filename=fulloutpath_matched, auto_open=False)
  plotly.offline.plot(aligned_figure, filename=fulloutpath_aligned, auto_open=False) 
  plotly.offline.plot(original_figure_0, filename=fulloutpath_original_0, auto_open=False)
  plotly.offline.plot(original_figure_1, filename=fulloutpath_original_1, auto_open=False)
  plotly.offline.plot(reparam_figure_0, filename=fulloutpath_reparam_0, auto_open=False) 
  plotly.offline.plot(reparam_figure_1, filename=fulloutpath_reparam_1, auto_open=False)

  print( "Saved the alignment graphs in directory %s" % outdir )


def renderAlignment(alignment, animal_obj_0, animal_obj_1, varnames, outdir):
  filename = "figure_%s-%s_%s_alignment.html" % (animal_obj_0.get_name(), animal_obj_1.get_name(), '-'.join(varnames))
  outpath = os.path.join(outdir,filename).replace(' ','')
  N = len(alignment[0])
  data = []
  for i in range(N):
    data.append(go.Scatter(x=[0,alignment[0][i],alignment[0][i]],
                           y=[alignment[1][i],alignment[1][i],0],
                           mode='lines',
                           marker={'color':'black'},
                           showlegend=False,opacity=0.1))
  trace = go.Scatter(x = alignment[0], y=alignment[1], mode='lines',showlegend=False,line={'width':4})
  data.append(trace)
  figure = {'data':data,'layout':{'height':500,'width':500,'xaxis':{'title': '%s Time' % animal_obj_0.get_name()},'yaxis': {'title': '%s Time' % animal_obj_1.get_name()}}}
  plotly.offline.plot(figure, filename=outpath+".html", auto_open=False)
  print("Saved alignment graph in %s" % outpath)


def writeOFF(animal_obj, coordinates, outdir, filename):
  outpath = os.path.join(outdir,filename).replace(' ','')
  triangles = animal_obj.get_triangulation()
  colors = animal_obj.get_colors()
  print("Writing triangulation to file %s..." % outpath)
  with open(outpath, 'w') as outfile:
    outfile.write("OFF\n")
    outfile.write("%d %d %d\n" % (len(coordinates), len(triangles), 0))
    for c in coordinates:
      outfile.write("%f %f %f\n" % (c[0], c[1], c[2]))   
    for t in triangles:
      c = colors[triangles.index(t)]
      outfile.write("%d %d %d %d %f %f %f\n" % (3, t[0], t[1], t[2], c[0], c[1], c[2]))


def postProcess(animal_list, dists, outdir, outfilename, sort_table, square_table, color_min=0.0, color_max=1.0):
  num_animals = len(animal_list)
  if square_table:
    for i in range(num_animals):
      for j in range(i):
        dists[i][j] = dists[j][i]
    writeDistTableToCSV(animal_list, dists, outdir, outfilename+".csv")
    writeDistTableToHeatmap(animal_list, dists, outdir, outfilename+".html", color_min, color_max)
  else:
    writeDistTableToCSV(animal_list, dists, outdir, outfilename+".csv")
    writeDistTableToHeatmap(animal_list, dists, outdir, outfilename+".html", color_min, color_max)

  if sort_table:
    dist_means = {}
    D = []
    for i in range(num_animals):
      dlist = [dists[j][i] for j in range(i)] + [dists[i][j] for j in range(i+1,num_animals)]
      dist_means.update({animal_list[i]:np.mean(dlist)})
    sorted_dists = sorted(dist_means.items(), key=operator.itemgetter(1))
    sorted_indices = [animal_list.index(sorted_dists[i][0]) for i in range(num_animals)]
    new_dists = [['' for i in range(num_animals)] for j in range(num_animals)]
    for i in range(num_animals):
      for j in range(i+1, num_animals):
        new_dists[i][j] = dists[sorted_indices[i]][sorted_indices[j]] if sorted_indices[j] > sorted_indices[i] else dists[sorted_indices[j]][sorted_indices[i]]
    dists = new_dists
    animal_list = [animal_list[sorted_indices[i]] for i in range(num_animals)]

    if square_table:
      for i in range(num_animals):
        for j in range(i):
          dists[i][j] = dists[j][i]
    writeDistTableToCSV(animal_list, dists, outdir, "%s" % outfilename+"_sorted.csv")
    writeDistTableToHeatmap(animal_list, dists, outdir, "%s" % outfilename+"_sorted.html", color_min, color_max)
