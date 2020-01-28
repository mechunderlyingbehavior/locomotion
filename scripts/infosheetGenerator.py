#!/usr/bin/env python

import os
import sys
import json

def getItemInfo(f):
  print("Required information for %s..." % f )
  name = input(str("Name: "))
  species = input(str("Species: "))
  exp_type = input(str("Experiment type: "))
  ID = input(str("ID: "))
  control_group = input(str("Is this the control group? [y/n] ")).lower()
  if control_group == 'y':
    control_group = "True"
  else:
    control_group = "False"
  dim_x = input(str("Horizontal dimension of the capture area (in mm): "))
  dim_y = input(str("Vertical dimension of the capture area (in mm): "))
  pixels_per_mm = input(str("Pixels per mm ratio: "))
  frames_per_sec = input(str("Frame rate (per second): "))
  start_time = input(str("Experiment starts at (in min): "))
  end_time = input(str("Experiment ends at (in min): "))
  baseline_start_time = input(str("Baseline segment starts at (in min): "))
  baseline_end_time = input(str("Baseline segment ends at (in min): "))
  
  item = {
    "name": name,
    "data_file_location": f,
    "animal_attributes":
    {
      "species": species,
      "exp_type": exp_type,
      "ID": ID,
      "control_group": control_group
    },
    "capture_attributes": 
    {
      "dim_x": dim_x,
      "dim_y": dim_y,
      "pixels_per_mm": pixels_per_mm,
      "frames_per_sec": frames_per_sec,
      "start_time": start_time,
      "end_time": end_time,
      "baseline_start_time": baseline_start_time,
      "baseline_end_time": baseline_end_time
    }
  }
  return item

  
def main():
  #Where is the file going to be saved?
  outdir = input(str("Specify the directory the json file will be saved: "))
  outdir = os.path.abspath(outdir)
  filename = input(str("Specify the json file name: "))
  outfilename = os.path.join(outdir,filename)
  write_mode = input(str("Write new file or append to an existing file? [w/a] ")).lower()
  if write_mode != 'w' and write_mode != 'a':
    print("Invalid input")
    exit(1)
  if write_mode == 'w' and os.path.isfile(outfilename): 
    confirmation = input(str("%s already exists. Are you sure you want to overwrite? [y/n] " % outfilename))
    if confirmation == 'n':
      print("Alrighty then, let's have you start over.")
      exit()
  if write_mode == 'a' and not os.path.isfile(os.path.join(outdir,filename)): 
    confirmation = input(str("%s doesn't exist. Do you want to write it as a new file? [y/n] " % outfilename))
    if confirmation == 'n':
      print("Alrighty then, let's have you start over.")
      exit()
    else:
      write_mode = 'w'

  # Ask how we are going to read in the files
  input_mode = input(str("Read in all data files in a directory or individual data files? [d/f] ")).lower()

  jsonItems = []
  if input_mode == 'd':
  #Survey the directory
    dirname = input(str("Specify the path of the directory containing data files: "))
    dirname = os.path.abspath(dirname)
    datext = input(str("Specify the file extension of the data files: "))
    if os.path.isdir(dirname):
      #Edit to filter out the hidden files on Windows based systems
      files = [ os.path.join(dirname,f) for f in os.listdir(dirname) if f.endswith(datext) and not f.startswith(".")] 
      if len(files) == 0:
        print("There is no valid data file in the directory.")
        exit(1)
    else:
      print("Invalid directory name")
      exit(1)
    for f in files:
    #Actually go through files one by one
      item = getItemInfo(f)
      jsonItems.append(item)

  elif input_mode == 'f':
  #Select data files one by one
    while True:
      filename = input(str("Specify the data file to read from: "))
      if os.path.isfile(filename): filename = os.path.abspath(filename)
      else:
        print("Invalid file name %s" % filename)
        exit(1)
      item = getItemInfo(filename)
      jsonItems.append(item)
      cont = input(str("Do you want to add another animal? [y/n] ")).lower()
      if cont == 'y': continue
      if cont == 'n': break
      else:
        print("Invalid answer. Going to assume that was a no. (If you were not done, start again and choose to append to this info file.)")
        break
      
  else: 
    print("Invalid input")
    exit(1)

  #form a json object with the information entered
  jsonstr = json.dumps(jsonItems, indent=4)
  # and now, write the file
  with open(outfilename, write_mode) as outfile:
    outfile.write(jsonstr)
    print("Wrote the information entered into %s" % outfilename)

if __name__=='__main__':
  main()
