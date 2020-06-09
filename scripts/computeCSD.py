import os
import sys
import json
import locomotion

def main():
  # First, get the dossier
  infofile = raw_input(str("Specify the path to the json file with animal information: "))
  infofile = os.path.abspath(infofile)
  if not os.path.isfile(infofile):
    print("Invalid info file! Try again?")
    exit(1)
    
  batch_mode = raw_input(str("Use all entries in the info file? [y/n] ")).lower()
  if batch_mode == 'n':
    name_list = raw_input(str("Specify the names of the animals, space separated.  (e.g., 'NSS_01 SS_01') ")).split()
  elif batch_mode == 'y':
    name_list = None
  else:
    print("Invalid answer.")
    exit(1)

  start_time = input(str("Specify the start time of the segment you want to compare: "))
  end_time = input(str("Specify the end time of the segment you want to compare: "))

  grid_size = input(str("Specify the grid size for the heatmap (in the same units as the x- and y- dimensions): "))
  
  output = raw_input(str("Do you want to write the results into a file? [y/n] "))
  if output == 'y':
    outdir = raw_input(str("Specify the output directory: "))
    outdir = os.path.abspath(outdir)
    outfilename = raw_input(str("Specify the output file name: "))
    sort_table = raw_input(str("Do you want to sort the output? [y/n] "))
    if sort_table == 'y':
      sort_table = True
    elif sort_table == 'n':
      sort_table = False
    else:
      print("Invalid input.")
      exit(1)
    square_table = raw_input(str("Do you want the distance table to be square instead of upper triangular? [y/n] "))
    if square_table == 'y':
      square_table = True
    elif square_table == 'n':
      square_table = False
    else:
      print("Invalid input.")
      exit(1)
    color_min = input(str("What should be the minimum value for the distance table color scale?" ))
    color_max = input(str("What should be the maximum value for the distance table color scale?" ))
      
  elif output != 'n':
    print("That wasn't quite one of the options...")
    exit(1)

  animals = locomotion.get_animal_objs(infofile,name_list)
  for a in animals:
    x_dim,y_dim = a.get_dims()
    if not float(x_dim/grid_size).is_integer() or not float(y_dim/grid_size).is_integer():
      print("The grid size must divide each of the dimensions evenly.")
      exit(1)
  for a in animals:
    locomotion.heatmap.getSurfaceData(a, grid_size, start_time, end_time)
  D = locomotion.heatmap.computeAllCSD(animals)
  if output == 'y':
    locomotion.write.postProcess(animals, D, outdir, outfilename, sort_table, square_table, color_min, color_max)
  
if __name__=="__main__":
  main()
