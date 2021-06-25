import sys
import os
import numpy as np
PATH_TO_DIRECTORY = os.getcwd()

# If this works, then locomotion has been installed into your system.
import locomotion

# Setup animal objects using 1 .json file per group
group_1_json = PATH_TO_DIRECTORY + "/../samples/sample_check_SS.json"
group_2_json = PATH_TO_DIRECTORY + "/../samples/sample_check_NSS.json"

info_files = [group_1_json, group_2_json]
animals = locomotion.setup_animal_objs(info_files)

# Populate each animal objects with variables
for a in animals:
    # Setup smoothened coordinate data
    mse = locomotion.trajectory.populate_curve_data(a,
                                                    smooth_order=3,# Degree of Polynomial used for Smoothing
                                                    smooth_window=25,# Length of Half-Window
                                                    smooth_method='savgol')

    # Print path to check smoothening
    locomotion.write.plot_path(a, 'results/', 'smooth_X', 'smooth_Y')
    print(f"MSE of smoothening for {a.get_name()} is {mse}.")

    # Setup Velocity and Curvature for BDD
    first_deriv, velocity = locomotion.trajectory.populate_velocity( a )
    locomotion.trajectory.populate_curvature(a, first_deriv=first_deriv, velocity=velocity)

    # Setup Distance to Point using "additional_info" in .json files for BDD
    locomotion.trajectory.populate_distance_from_point(a, "point", 'Dist to Point',
                                                       col_names=['smooth_X', 'smooth_Y'])

    # Setup surface data for CSD and Plot Heatmap for checking
    locomotion.heatmap.populate_surface_data(a, plot_heatmap=True, outdir='results/')

# Defining 'universal' normalization by using mean and std of all animals (For Velocity and Curvature)
raw_vals = {'Velocity':[], 'Curvature':[]}

# Extract all values from all animals
for a in animals:
    for var in ['Velocity', 'Curvature']:
        raw_vals[var].extend(a.get_vals(var))

# Calculate mean and std and use it to define new normalization method
for var in ['Velocity', 'Curvature']:
    mean = np.mean(raw_vals[var])
    std = np.std(raw_vals[var])
    for a in animals:
        a.add_norm_standard(var, 'universal', mean, std)

# Define bounded normalization method using predefined lower and upper bounds for Dist to Point
lower_bound = 0
upper_bound = 100
for a in animals:
    a.add_norm_bounded('Dist to Point', 'bounded', lower_bound, upper_bound)

# Running BDD Experiments
variables = ['Velocity', 'Curvature', 'Dist to Point']
norm_mode = ['universal', 'universal', 'bounded'] # List rather than Str to define different norm methods
start_time, end_time = 0, 120 # in seconds

# EXAMPLE 1: Calculate BDD between all animals
bdds = locomotion.trajectory.compute_all_bdd(animals, variables, start_time, end_time, norm_mode)

# Print out a Dendrogram using caluclated BDD matrix
locomotion.write.render_dendrogram(animals, bdds, 'results/',
                                   f'dendro_{start_time}-{end_time}',
                                   threshold=0.125)
# Plot heatmap of distances
output_directory, outfile_name = "results/", "bdd_heatmap"
try: # Safety check to ensure that data folder exists, and makes it otherwise.
        os.mkdir(output_directory)
except FileExistsError:
   pass

sort_table, square_table = False, False
color_min, color_max = 0.05,0.15
locomotion.write.post_process(animals, bdds, output_directory, outfile_name,
                              sort_table, square_table, color_min, color_max )

# EXAMPLE 2 : Calculate pairwise BDD with fullmode prints
a1 = animals[0]
a2 = animals[1]

bdd_12 = locomotion.trajectory.compute_one_bdd(a1, a2, variables,
                                               start_time, end_time,
                                               start_time, end_time,
                                               norm_mode, fullmode=True, outdir='results/')

# EXAMPLE 3 : Calculate CSD between all animals
csds = locomotion.heatmap.compute_all_csd( animals )

output_directory, outfile_name = "results/", "csd_heatmap"
try: # Safety check to ensure that data folder exists, and makes it otherwise.
        os.mkdir(output_directory)
except FileExistsError:
   pass

sort_table, square_table = False, False
color_min, color_max = 0.002,0.008
locomotion.write.post_process(animals, csds, output_directory, outfile_name,
                              sort_table, square_table, color_min, color_max )
