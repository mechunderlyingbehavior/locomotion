import sys
import os
PATH_TO_DIRECTORY = os.getcwd()

# If this works, then locomotion has been installed into your system.
import locomotion

outfile = PATH_TO_DIRECTORY + "/../samples/sample.json"
info_files = [outfile] # 3 animals in this json
animals = locomotion.setup_animal_objs( info_files )
for a in animals:
    first_deriv, velocity = locomotion.trajectory.populate_velocity( a )
    locomotion.trajectory.populate_curvature(a, first_deriv=first_deriv, velocity=velocity)
    locomotion.trajectory.populate_distance_from_point(a, "point", 'Dist to Point', col_names=['X', 'Y'])

variables = ['Velocity', 'Curvature', 'Dist to Point']
norm_mode = 'universal'
start_time, end_time = 0, 1

# Populating mean and std into animal objects for norm_mode = 'universal'
raw_vals = {}
for var in variables:
    raw_vals.update({var:[]})

for a in animals:
    for var in variables:
        raw_vals[var].extend(a.get_raw_vals(var, start_time, end_time))

for var in variables:
    mean = np.mean(raw_vals[var])
    std = np.std(raw_vals[var])
    for a in animals:
        a.add_stats(var, norm_mode, mean, std)

# IF YOU ARE RUNNING PAIRWISE BDD COMPARISONS ON FULL MODE:
a1 = animals[0]
a2 = animals[1]
a3 = animals[2]


# a1 and a2
bdd_12 = locomotion.trajectory.compute_one_bdd(a1, a2, variables,
                                               start_time, end_time,
                                               start_time, end_time,
                                               norm_mode, fullmode=True, outdir='BDD_12/')

# a1 and a3
bdd_13 = locomotion.trajectory.compute_one_bdd(a1, a3, variables,
                                               start_time, end_time,
                                               start_time, end_time,
                                               norm_mode, fullmode=True, outdir='BDD_13/')

# a2 and a3
bdd_23 = locomotion.trajectory.compute_one_bdd(a2, a3, variables,
                                               start_time, end_time,
                                               start_time, end_time,
                                               norm_mode, fullmode=True, outdir='BDD_23/')

# OTHERWISE, just run locomotion.trajectory.compute_all_bdd() and other write stuff.
