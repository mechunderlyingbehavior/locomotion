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
  
  variable_names = raw_input(str("Which variables do you want to use? (e.g., 'Y Velocity Curvature') ")).split()

  num_exps = input(str("Specify the number of comparisons you want to run for each individual: "))

  start_time = input(str("Specify the start time of the overall segment in which you want to run the comparisons: "))
  end_time = input(str("Specify the end time of the overall segment in which you want to run the comparisons: "))

  norm_mode = raw_input(str("Which time segment do you want to normalize over: the predetermined baseline or the specific segment [b/s]: "))
  if norm_mode == 'b': norm_mode = 'baseline'
  elif norm_mode == 's': norm_mode = 'spec'
  else:
    print("Invalid norm mode")
    exit(1)

  intervals = raw_input(str("Do you want to specify a set of interval lengths to use? [y/n] "))
  if intervals == 'y':
    interval_lengths = raw_input(str("Which interval lengths would you like to use? (e.g., '0.5 1.0 1.25 2.4') ")).split()
    interval_lengths = [float(l) for l in interval_lengths]
  elif intervals == 'n':
    interval_lengths = None
  else:
    print("That wasn't quite one of the options...")
    exit(1)
  
  output = raw_input(str("Do you want to write the results into a file? [y/n] "))
  if output == 'y':
    outdir = raw_input(str("Specify the output directory: "))
    outdir = os.path.abspath(outdir)
    outfilename = raw_input(str("Specify the output file name: "))
  elif output != 'n':
    print("That wasn't quite one of the options...")
    exit(1)

  animals = locomotion.getAnimalObjs(infofile,name_list)
  for a in animals:
    locomotion.trajectory.getCurveData(a)
  if output == 'y':
    locomotion.trajectory.runIndividualVariabilityTests(animals, variable_names, norm_mode, num_exps, interval_lengths, outdir, outfilename, start_time, end_time)
  else:
    locomotion.trajectory.runIndividualVariabilityTests(animals, variable_names, norm_mode, num_exps, interval_lengths, None, None, start_time, end_time)
  
if __name__=="__main__":
  main()
