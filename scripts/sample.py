import sys
import os
import numpy as np
PATH_TO_DIRECTORY = os.getcwd()

# If this works, then locomotion has been installed into your system.
import locomotion

outfile = PATH_TO_DIRECTORY + "/../data/rodent_sample/rodent_JSON.json"
# outfile = PATH_TO_DIRECTORY + "/../samples/sample_check_SS.json"
info_files = [outfile] # 5 animals in this json
animals = locomotion.setup_animal_objs(info_files,
                                       smooth_order=3,
                                       smooth_window=21) # CHANGE THESE TO TEST SMOOTHENING
for a in animals:
    locomotion.write.plot_path(a, 'results/')
    first_deriv, velocity = locomotion.trajectory.populate_velocity( a )
    _, _, _, curvature = locomotion.trajectory.populate_curvature(a, first_deriv=first_deriv, velocity=velocity)
    locomotion.write.render_single_animal_graph(curvature, a, 'Curvature', 'results/')
    #### EVERYTHING BELOW THIS POINT IS NOT NEEDED FOR SMOOTHENING CHECK ####
#     first_deriv, velocity = locomotion.trajectory.populate_velocity( a )
#     locomotion.trajectory.populate_curvature(a, first_deriv=first_deriv, velocity=velocity)
#     locomotion.trajectory.populate_distance_from_point(a, "point", 'Dist to Point', col_names=['X', 'Y'])

# variables = ['Velocity', 'Curvature', 'Dist to Point']
# norm_mode = ['universal','universal', 'bounded']
# start_time, end_time = 0, 1

# # Populating mean and std into animal objects for norm_mode = 'universal'
# raw_vals = {}
# for var in variables:
#     raw_vals.update({var:[]})

# for a in animals:
#     for var in variables:
#         raw_vals[var].extend(a.get_raw_vals(var, start_time, end_time))

# for var in variables[:2]:
#     mean = np.mean(raw_vals[var])
#     std = np.std(raw_vals[var])
#     for a in animals:
#         a.add_norm_standard(var, 'universal', mean, std)

# # NEW BOUNDED NORMALIZATION FOR DISTANCE TYPE METHODS
# # TODO: Find proper bounds for Dist to Point
# lower_bound = 0
# upper_bound = 100
# for a in animals:
#     a.add_norm_bounded('Dist to Point', 'bounded', lower_bound, upper_bound)



# # IF YOU ARE RUNNING PAIRWISE BDD COMPARISONS ON FULL MODE:
# a1 = animals[0]
# a2 = animals[1]
# # a3 = animals[2]


# # a1 and a2
# bdd_12 = locomotion.trajectory.compute_one_bdd(a1, a2, variables,
#                                                start_time, end_time,
#                                                start_time, end_time,
#                                                norm_mode, fullmode=True, outdir='results/')

# a1 and a3
# bdd_13 = locomotion.trajectory.compute_one_bdd(a1, a3, variables,
#                                                start_time, end_time,
#                                                start_time, end_time,
#                                                norm_mode, fullmode=True, outdir='BDD_13/')

# # a2 and a3
# bdd_23 = locomotion.trajectory.compute_one_bdd(a2, a3, variables,
#                                                start_time, end_time,
#                                                start_time, end_time,
#                                                norm_mode, fullmode=True, outdir='BDD_23/')

# # OTHERWISE, just run locomotion.trajectory.compute_all_bdd() and other write stuff.
