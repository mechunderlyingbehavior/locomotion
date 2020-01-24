# Locomotion Python Package

One Paragraph of project description goes here

## Getting Started

These instructions will get you a copy of the project up and running on your
local machine for development and testing purposes. See deployment for notes on
how to deploy the project on a live system.

### Prerequisites

As of 24 January 2020, the `locomotion` package has been converted for use on
Python 3.7.3. This module also requires the following python packages:
* Numpy (>= 1.16.2)
* Scipy (>= 1.2.1)
* Plotly (>= 4.4.1)

### Installing

The `pip` installation is a work in progress. In the meantime, the package can
be manually installed in your system.

#### Step 1: Adding `locomotion` to your Python Path

To access the `locomotion` package, you could either copy the `locomotion`
folder into your working directory, or add the package to your environment. To
do the latter, you may use the following methods in your terminal, replacing
`/PATH/TO/DIRECTORY` with the directory containing the package:

```
export PYTHONPATH=$PYTHONPATH:/PATH/TO/DIRECTORY
export PYTHONPATH=$PYTHONPATH:/PATH/TO/DIRECTORY/locomotion

```

Note that this method will have to be repeated every time a new terminal window
is opened. An alternative to the terminal method would be to add the following
directly to the top of your python script, before any other imports:

```
import sys
sys.path.append('/PATH/TO/DIRECTORY')
sys.path.append('/PATH/TO/DIRECTORY/locomotion')
```

You should now be able to `import locomotion` from your python shell.

#### Step 2: Compiling `extendedDTW.so` file

If `import extendedDTW` results in an `ImportError`, you will need to rebuild
the package to make it compatible to your system. To do so, perform the following
steps:
1. Delete (or move to another folder) the `extendedDTW.so` file in the `locomotion` directory.
1. Move to the `extendedDTW` directory.
1. If a previous version of `extendedDTW.c` exists, remove it.
1. Run `setup.py` with the following command on terminal.
```
python setup.py build_ext --inplace
```

After these steps, you can now copy the `extendDTW.so` generated in the
`extendedDTW` directory and paste it into the `locomotion` directory. If all
goes well, you should be able to run `import extendDTW` in your Python shell
now.

## Data format

### File Format
The package accepts csv and tsv files. However, because it distinguishes between
tsv and csv by doing a simple check to see if tabs or commas are present in the
first (header) line, make sure to avoid using both delimiters in the header of
your data file. If your data must use both, you will need to edit the `getRawData`
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

```
{
        "name": "NSS_01", //Can be anything. Make it unique
        "data_file_location": "/data/medaka/NSS_01.dat",
        //full path to the data file
        "animal_attributes": {
            "species": "Medaka", //species name
            "exp_type": "NSS", //experiment type
            "control_group": "True", //True or False
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

To generate the info file, you can use `infosheetGenerator.py` provided along with
the locomotion package, which will populate a json file in the correct format by
prompting the user for each necessary piece of information. It should run
similarly to the following snippet.

```
Specify the directory the json file will be saved: /path/to/json/files/
Specify the json file name: sample.json
Write new file or append to an existing file? [w/a] w
Read in all data files in a directory or individual data files? [d/f] f
Specify the data file to read from: /path/to/data/files/SS_01.tsv
Required information for /path/to/data/files/SS_01.tsv...
Name: SS_01
Species: medaka
Experiment type: SS
ID: 01
Is this the control group? [y/n] n
Horizontal dimension of the capture area (in mm): 200
Vertical dimension of the capture area (in mm): 100
Pixels to mm ratio: 2.4
Frame rate (per second): 20
Experiment starts at (in min): 0
Experiment ends at (in min): 10
Baseline segment starts at (in min): 0
Baseline segment ends at (in min): 2
Do you want to add another file? [y/n] n
Wrote the information entered into /path/to/json/files/sample.json
```

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Contributing

Please read
[CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for
details on our code of conduct, and the process for submitting pull requests to
us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **AUTHORS** 

See also the list of [contributors](https://github.com/mechunderlyingbehavior/locomotion/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
