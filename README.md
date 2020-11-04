# Locomotion Python Package

## Getting Started

These instructions will get you a copy of the project up and running on your
local machine for development and testing purposes. See deployment for notes on
how to deploy the project on a live system.

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
jupyter notebook `scripts/example_notebook.ipynb`, which is located in the main
folder. Follow through the notebook, which will run through the basic functions
of the package. If the package is installed properly, it should be able to
generate a small sample dataset and run the BDD and CSD on it.

## Data format

### File Format
The package accepts csv and tsv files. However, because it distinguishes between
tsv and csv by doing a simple check to see if tabs or commas are present in the
first (header) line, make sure to avoid using both delimiters in the header of
your data file. If your data must use both, you will need to edit the `get_raw_data`
function in `animal.py`. 

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
        "data_file_location": "/data/medaka/NSS_01.dat",
        //full path to the data file
        "animal_attributes": {
            "species": "Medaka", //species name
            "exp_type": "NSS", //experiment type
            "ID": "01" //number, but in quotations
        },
        "capture_attributes": {
            "frames_per_sec": 20, //integer
            "pixels_per_mm": 1.6, //float
            "dim_x": 200, //in mm
            "dim_y": 100, //in mm
            "start_time": 0, //in min
            "end_time": 10, //in min
            "baseline_start_time": 0, //in min
            "baseline_end_time": 2 //in min
        }
    }
```

To generate the info file, you can make use of `scripts/json_generator.ipynb`
which contains a step-by-step walkthrough that populates a .json file, prompting
you to include the correct data.

## Using the Package

Once you import the `locomotion` package, you will need to first initiate animal
objects using the `locomotion.setup_animal_objs` command, which returns a list of
animal objects with basic X and Y data from the data files.

The routines for calculating Behavioral Distortion Distance (BDD) are located in
the `trajectory.py` file and can be called by
`locomotion.trajectory.[routine_name]`.

Example script:

```python
import locomotion
info_files = ["/path/to/animal_info.json"]
animals = locomotion.setup_animal_objs( info_files )
for a in animals:
  locomotion.trajectory.populate_curve_data( a )
variables = ['Y','Velocity','Curvature']
start_time, end_time = 0, 1
norm_mode = 'spec'
distances = locomotion.trajectory.compute_all_bdd( animals, variables, start_time, end_time, norm_mode )
output_directory, outfile_name = "/path/to/outdir", "results"
sort_table, square_table = False, False
color_min, color_max = 0.1, 0.5
locomotion.write.post_process( animals, distances, output_directory, outfile_name, sort_table, square_table, color_min, color_max )
```

To calculate the Intra-Individual Behavioral Distortion Distance (IIBDD) for each animal in a
specified info sheet, one can run a script like the following:

```python
import locomotion
info_files = ["/path/to/animal_info.json"]
animals = locomotion.setup_animal_objs( info_files )
for a in animals:
  locomotion.trajectory.populate_curve_data( a )
variables = ['Y','Velocity','Curvature']
norm_mode = 'spec'
number_of_samples = 100
output_directory, outfile_name = "/path/to/outdir", "results.csv"
start_time, end_time = 0, 1
iibdds = locomotion.trajectory.compute_all_iibdd( animals, variables, norm_mode, number_of_samples, start_time=start_time, end_time=end_time )
locomotion.write.write_iibdd_to_csv( animals, iibdds, output_directory, outfile_name )
```

The routines for calculating Conformal Spatiotemporal Distance (CSD) are located
in the `heatmap.py` file and can be called by `locomotion.heatmap.[routine_name]`.

Example script:

```python
import locomotion
info_files = ["/path/to/animal_info.json"]
animals = locomotion.setup_animal_objs( info_files )
grid_size, start_time, end_time = 10, 0, 2
for a in animals:
  locomotion.heatmap.populate_surface_data( a, grid_size, start_time, end_time )
distances = locomotion.heatmap.compute_all_csd( animals )
output_directory, outfile_name = "/path/to/outdir", "results"
sort_table, square_table = False, False
color_min, color_max = 0, 0.2
locomotion.write.post_process( animals, distances, output_directory, outfile_name, sort_table, square_table, color_min, color_max )
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
Grzeszkiewicz, Katherine Sun, Lucy Zhu, Sultan Aitzhan, and Yanhua Wang, for
their contributions to this package.

README template provided by [PurpleBooth](https://github.com/PurpleBooth/a-good-readme-template).
