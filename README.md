# Locomotion - Quantitative Comparisons of Locomotive Behavior
[![Tag](https://img.shields.io/github/v/tag/mechunderlyingbehavior/locomotion?logo=github)](https://github.com/mechunderlyingbehavior/locomotion/tags)
[![PyPI](https://img.shields.io/pypi/v/locomotion)](https://pypi.org/project/locomotion/)
[![Requirements Status](https://requires.io/github/mechunderlyingbehavior/locomotion/requirements.svg?branch=stable)](https://requires.io/github/mechunderlyingbehavior/locomotion/requirements/?branch=stable)
[![License](https://img.shields.io/github/license/mechunderlyingbehavior/locomotion)](https://github.com/mechunderlyingbehavior/locomotion/blob/stable/LICENSE.md)

`locomotion` is a Python package that provides computational geometric tools for
quantitative comparisons of locomotive behaviors. The package makes use of curve
and shape alignment techniques to accurately quantify (dis)similarities in
animal behavior without excluding the inherent variability present between
individuals. For a pair of animals, the package computes 2 metrics, the
Behavioral Distortion Distance (BDD) and the Conformal Spatiotemporal Distance
(CSD).

For more information on the techniques implemented in this repository, please
read our [publication](https://www.nature.com/articles/s41598-019-52300-8).

## Table of Contents
* [Getting Started](#getting-started)
  * [Installation and Requirements](#installation-and-requirements)
* [Data Format](#data-format)
  * [File Format](#file-format)
  * [Header Format](#header-format)
  * [Information File Format](#information-file-format)
* [Using the Package](#using-the-package)
  * [Setting Up](#setting-up)
  * [Executing the Computations](#executing-the-computations)
* [Contributing](#contributing)
* [Versioning](#versioning)
* [Authors](#authors)
* [Citation](#citation)
* [License](#license)
* [Acknowledgments](#acknowledgments)

## Getting Started

To begin using the `locomotion` package, please follow the installation
instructions below. After the package has been successfully installed, please
proceed to the [Data Format](#data-format) section to understand how your data
should be formatted for use.

### Installation and Requirements

As of 24 January 2020, the `locomotion` package has been converted for use on
Python 3.7.3. This module also requires the following python packages:

* numpy (>= 1.16.2)
* scipy (>= 1.5.0)
* plotly (>= 4.4.1)
* dtw-python (>= 1.1.4)
* igl (>=2.2.0)

#### Installation instructions

The `locomotion` package is available for installation on 
[PyPI](https://pypi.org/project/locomotion/), which can be done by running the 
following command on your terminal:
```
pip install locomotion
```

Another installation option is to install it from the source directory directly.
Once you've cloned the directory or  downloaded the source file, run the 
following command while in the main directory.
```
python setup.py install
```

**Note**: One of our dependencies, `igl`, is still in development phase, and is
not currently on PyPI. You will have to install it manually in order to use
`locomotion`. You may find the installation instructions
[here](https://github.com/libigl/libigl-python-bindings).

After installing all the requirements, you should now be able to run `import
locomotion` in your Python shell.

#### Checking the installation

To ensure that all the functions in the package work as intended, run the
jupyter notebook `scripts/example_notebook.ipynb`. Follow through the notebook,
which will run through the basic functions of the package. If the package is
installed properly, it should be able to generate a small sample dataset and run
the BDD and CSD methods on it.

## Data Format

### File Format
The package accepts `.csv` and `.tsv` files. However, because it distinguishes
between tsv and csv by doing a simple check to see if tabs or commas are present
in the first (header) line, make sure to avoid using both delimiters in the
header of your data file. If your data must use both, you will need to edit the
`get_raw_data` function in `animal.py`.

### Header Format 
The computations require X and Y coordinate data with corresponding column
titles in string format, so the coordinate data columns must be labelled "X" and
"Y", including the quotation marks. They can also be single quotes. 

### Information File Format 
The information about the animals are stored in a json file, which is required
to read in relevant data for each computation. The fields and format are as in
this sample entry below. Note that all times are in minutes. In general, avoid
using spaces in the field values.

```javascript
{
        "name": "NSS_01", //Can be anything. Make it unique
        "data_file_location": "/data/medaka/NSS_01.dat", //full path to the data file
        "animal_attributes": {
            "species": "Medaka", //species name
            "exp_type": "NSS", //experiment type
            "ID": "01" //number, but in quotations
        },
        "capture_attributes": {
            "frames_per_sec": 20, //integer
            "input_unit": "px" //string
            "output_unit": "mm" //string
            "input_per_output_unit": 1.6, //float
            "x_lims": [0, 320], //in output unit
            "y_lims": [0, 160], //in output unit
            "start_time": 0, //in seconds
            "end_time": 600, //in seconds
            "baseline_start_time": 0, //in seconds
            "baseline_end_time": 120 //in seconds
        },
        "additional_info": {
            //define any useful information, e.g. a point to calculate distant from
            "point": [10, 10] //in output unit
        }
    }
```

To generate the info file, you can make use of `scripts/json_generator.ipynb`
which contains a step-by-step walkthrough that populates a `.json` file, prompting
you to include the correct data.

## Using the Package

In this section, we outline the typical experimental process that we've designed
the `locomotion` package around. This process is split to roughly 2 stages - the
setup stage, and the computation stage.

### Setting Up 

After importing the `locomotion` package, you will need to do some additional
setup before the computations can be done. This will entail initializing the
`Animal()` objects, and populating the objects with the necessary variables. We've
included methods that will allow you to check certain aspects of your setup,
such as smoothening and grid sizes.

All `locomotion` scripts should start off the same way - initializing the `Animal()`
objects using `locomotion.setup_animal_objs` and the prepared `.json` files.

Sample code:

```python
import locomotion
info_files = ["/path/to/animal_group_1.json", "/path/to/animal_group_2.json"']
animals = locomotion.setup_animal_objs(info_files)
```

#### Setting up for BDD 

When preparing the `Animal()` objects for BDD computations, we want to first
smoothen the raw data. This is done through inbuilt methods, and should be
checked before proceeding. We then populate other relevant variables such as
Velocity and Curvature, and we also define any necessary normalization methods.

Sample code for populating and checking smoothened data. Assumes initialized
`Animal()` objects.

```python
for a in animals:
    # Setup smoothened coordinate data
    mse = locomotion.trajectory.populate_curve_data(
        a,
        smooth_order=3,# Degree of Polynomial used for Smoothing
        smooth_window=25,# Length of Half-Window
        smooth_method='savgol'
    )

    # Print path to check smoothening
    locomotion.write.plot_path(a, '/path/to/output_dir', 'smoothed_path', 'smooth_X', 'smooth_Y')
    print(f"MSE of smoothening for {a.get_name()} is {mse}.")
```

Sample code for populating other variables. Assumes initialized `Animal()` objects
and smoothened cooridnate data.

```python
for a in animals:
    # Setup Velocity and Curvature for BDD
    first_deriv, velocity = locomotion.trajectory.populate_velocity( a )
    locomotion.trajectory.populate_curvature(a, first_deriv=first_deriv, velocity=velocity)

    # Setup Distance to Point using "additional_info" in .json files for BDD
    locomotion.trajectory.populate_distance_from_point(a, "point", 'Dist to Point',
                                                       col_names=['smooth_X', 'smooth_Y'])
```

Sample code for defining special normalization methods. Assumes initialized `Animal()` objects, smoothened coordinate data, and variables `Velocity`, `Curvature`, and `Dist to Point`.

```python
# Defining 'universal' normalization by using mean and std of Velocity and Curvature
# across all animals.

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
```

#### Setting up for CSD

When preparing the data for CSD computations, want to first split the coordinate
data into grids by either using the automatic gridsize calculation or by
defining the number of grids manually. A heatmap can be printed to see how the
resulting 2-d histogram will look like. This is done through the
`locomotion.heatmap.populate_surface_data()` function, which also computes the
surface formed by the heatmap and the associated triangulations needed for the
CSD calculation.

Sample code for setting up the surface data and printing the heatmap for
verification purposes.

```python
for a in animals:
    locomotion.heatmap.populate_surface_data(a,
        a, val_names=['raw_X', 'raw_Y'], # X and Y data used for calculations
        x_grid_count=None, # Number of grids along the X axis 
        y_grid_count=None, # Number of grids along the Y axis
        # Setting both x_grid_count and y_grid_count to None will trigger automatic calculations
        plot_heatmap=True, # Plot heatmap for checking purposes
        outdir='/path/to/output_directory/"
    )
```

### Executing the Computations

Once the `Animal()` objects have been appropriately set up, we can then execute
the computations.

#### Behavioral Distortion Distance (BDD)

To calculate the BDD between all `Animal()` objects in a list, you may use the
following script after setting up the animal object as in [Setting up for
BDD](#setting-up-for-bdd).

```python
variables = ['Velocity', 'Curvature', 'Dist to Point']
norm_mode = ['universal', 'universal', 'bounded'] # List rather than Str to define different norm methods
start_time, end_time = 0, 120 # in seconds

bdds = locomotion.trajectory.compute_all_bdd(animals, variables, start_time, end_time, norm_mode)

# Print out a Dendrogram using caluclated BDD matrix
locomotion.write.render_dendrogram(animals, bdds, '/path/to/output_directory/',
                                   f'dendro_{start_time}-{end_time}',
                                   threshold=0.125)
# Plot heatmap of distances
output_directory, outfile_name = "/path/to/output_directory/", "bdd_results"
try: # Safety check to ensure that data folder exists, and makes it otherwise.
        os.mkdir(output_directory)
except FileExistsError:
   pass

sort_table, square_table = False, False
color_min, color_max = 0.05,0.15
locomotion.write.post_process(animals, bdds, output_directory, outfile_name,
                              sort_table, square_table, color_min, color_max )
```

#### Intra-Individual Behavioral Distortion Distance (IIBDD)

To calculate the IIBDD of all `Animal()` objects in a list, you may use the
following script after setting up the animal object as in [Setting up for
BDD](#setting-up-for-bdd).

```python
variables = ['Velocity','Curvature']
norm_mode = 'spec'
num_samples = 5 
iibdds = locomotion.trajectory.compute_all_iibdd(animals, variables, norm_mode, num_samples)

output_directory, outfile_name = "/path/to/output_directory/", "iibdd_results"
try: # Safety check to ensure that data folder exists, and makes it otherwise.
    os.mkdir(output_directory)
except FileExistsError:
    pass
locomotion.write.write_iibdd_to_csv(animals, iibdds, output_directory, outfile_name)
```

#### Behavioral Distortion Distance (BDD)

To calculate the BDD between all `Animal()` objects in a list, you may use the
following script after setting up the animal object as in [Setting up for
CSD](#setting-up-for-csd).

```python
csds = locomotion.heatmap.compute_all_csd( animals )

output_directory, outfile_name = "/path/to/output_directory/", "csd_results"
try: # Safety check to ensure that data folder exists, and makes it otherwise.
        os.mkdir(output_directory)
except FileExistsError:
   pass

sort_table, square_table = False, False
color_min, color_max = 0.002,0.008
locomotion.write.post_process(animals, csds, output_directory, outfile_name,
                              sort_table, square_table, color_min, color_max )
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of
conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available,
see the [tags on this repository](https://github.com/mechunderlyingbehavior/locomotion/tags).

## Authors

This package was written by Matthew T. Stamps, Zhong Xuan Khwa, Elaine Wijaya,
Soo Go, and Ajay S. Mathuru.

See also the list of
[contributors](https://github.com/mechunderlyingbehavior/locomotion/contributors)
who participated in this project.

## Citation

If you use `locomotion` in a scientific paper, we would appreciate citations to
the following [paper](https://www.nature.com/articles/s41598-019-52300-8):
```
Stamps, M.T., Go, S. & Mathuru, A.S. Computational geometric tools for quantitative comparison of locomotory behavior. Sci Rep 9, 16585 (2019). https://doi.org/10.1038/s41598-019-52300-8
```

Preferred citation format can be found
[here](https://www.nature.com/articles/s41598-019-52300-8#citeas).

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

We would like to acknowledge the work of Alaukik Pant, Haroun Chahed, Karolina
Grzeszkiewicz, Katherine Sun, Lucy Zhu, Saoirse Therese Lightbourne, Sultan Aitzhan, and 
Yanhua Wang, for their contributions to this package.

README template provided by [PurpleBooth](https://github.com/PurpleBooth/a-good-readme-template).
