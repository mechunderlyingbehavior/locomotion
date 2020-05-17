from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import json
import pandas as pd

import locomotion

app = Flask(__name__)
_key_delimiter = "_"
_host = "127.0.0.1"
_port = 5002
_api_base = "/api"


# Set this to the directory where "locomotion" is present, i.e. the first locomotion folder
#example: "/Users/alaukik/code/webAppLocomotion/locomotion/UI"
PATH_TO_DIRECTORY = "/path/to/locomotion/UI"

# Folder where data will be stored
UPLOAD_FOLDER = PATH_TO_DIRECTORY + "/dataStorage/"

# This variable initializes the number of files the user has loaded
FILE_COUNTER = 0

# Example of variables that can be used for camera information
species = "medaka"
exp_type = "SS"
dim_x = 200
dim_y = 100
frames_per_sec = 20
pixels_per_mm = 1.6
start_time = 0
end_time = 10
baseline_start_time = 0
baseline_end_time = 2

# Example of a data structure that stores the details of an animal in a dictionary
egDS = {
    "name": None,
    "data_file_location": None,
    "animal_attributes":
    {
      "species": None,
      "exp_type": None,
      "ID": None,
      "control_group": None
    },
    "capture_attributes":
    {
      "dim_x": None,
      "dim_y": None,
      "pixels_per_mm": None,
      "frames_per_sec": None,
      "start_time": None,
      "end_time": None,
      "baseline_start_time": None,
      "baseline_end_time": None
    }
}

# A list of the details of of animals stored in a dictionary format
jsonItems = []

# Extensions of allowed data formats
ALLOWED_EXTENSIONS = {'dat', 'json', 'csv'}

# Function to check if the given file is in the allowed file format
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


# Route that guides the user in the process of choosing files to be uploaded
@app.route("/chooseFile", methods=['GET', 'POST'])
def chooseFile():
    if request.method == 'POST':
        # check if the post request has the file part
        if request.method == 'POST':
            # check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            uploaded_files = request.files.getlist("file")
            for file in uploaded_files:
                if file.filename == '':
                    flash('No selected file')
                    return redirect(request.url)
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    # file.save(os.path.join(app.config[UPLOAD_FOLDER], filename))
                    file.save(UPLOAD_FOLDER + filename)
                    global FILE_COUNTER
                    FILE_COUNTER += 1
                    newDS = egDS.copy()
                    newDS["name"] = filename
                    newDS["data_file_location"] = UPLOAD_FOLDER + filename
                    jsonItems.append(newDS)
            # noOfAnimals = len(jsonItems)
        return redirect(url_for('chooseCam'))
    return render_template("chooseFile.html")


# This variable represents the index of the animal in jsonItems whose camera information is being stored
camIndex = 0

# Route to specify camera information about Camera
@app.route("/cameraInfo", methods=['GET', 'POST'])
def chooseCam():
    global camIndex
    if request.method == 'POST':
        animalInfo = jsonItems[camIndex]
        setCam(animalInfo)
        camIndex+= 1
        if camIndex >= len(jsonItems):
            return redirect(url_for('chooseExperiment'))
        else:
            if request.form.get('checkbox') == 'checked':
                while camIndex < len(jsonItems):
                    animalInfo = jsonItems[camIndex]
                    setCam(animalInfo)
                    camIndex += 1
                return redirect(url_for('chooseExperiment'))
            else:
                return redirect(url_for('chooseCam'))
    return render_template("cameraInfo.html", headline=jsonItems[camIndex]["name"])

# Helper function used to store the information of the camera being considered
def setCam(animalInfo):
    animalInfo['capture_attributes']['dim_x'] = int(request.form['dim_x'])
    animalInfo['capture_attributes']['dim_y'] = int(request.form['dim_y'])
    animalInfo['capture_attributes']['pixels_per_mm'] = float(request.form['pixels_per_mm'])
    animalInfo['capture_attributes']['frames_per_sec'] = int(request.form['frames_per_sec'])
    animalInfo['capture_attributes']['start_time'] = int(request.form['start_time'])
    animalInfo['capture_attributes']['end_time'] = int(request.form['end_time'])
    animalInfo['capture_attributes']['baseline_start_time'] = int(request.form['baseline_start_time'])
    animalInfo['capture_attributes']['baseline_end_time'] = int(request.form['baseline_end_time'])

# This variable represents the index of the animal in jsonItems whose expreiment information is being stored
expIndex = 0

# Route to specify camera information about Experiment Information:
@app.route("/experimentInfo", methods=['GET', 'POST'])
def chooseExperiment():
    global expIndex
    if request.method == 'POST':
        if request.form.get('checkbox') == 'checked':
            animalInfo = jsonItems[expIndex]
            getExpInfo(animalInfo)
            expIndex += 1
        if expIndex >= len(jsonItems):
            return redirect(url_for('sendInfo'))
        else:
            if request.form.get('checkbox') == 'checked':
                print("lado")
                while expIndex < len(jsonItems):
                    animalInfo = jsonItems[expIndex]
                    getExpInfo(animalInfo)
                    expIndex += 1
                return redirect(url_for('sendInfo'))
            else:
                return redirect(url_for('chooseExperiment'))
    return render_template("experimentInfo.html", headline=jsonItems[expIndex]["name"])

# Helper function used to store the information of the expreiment being considered
def getExpInfo(animalInfo):
    animalInfo['animal_attributes']['species'] = request.form['species']
    animalInfo['animal_attributes']['exp_type'] = request.form['exp_type']
    animalInfo['animal_attributes']['ID'] = request.form['ID']
    animalInfo['animal_attributes']['control_group'] = request.form['control_group']


# Table that shows all the files with the experiment information
# This section requires more work
@app.route("/experimentInfoTable", methods=['GET', 'POST'])
def getCamTable():
    global expIndex
    lenData = len(jsonItems)
    columns = ['File Name', 'Species', 'Experiment Type', 'Control Group', 'ID', 'Link']
    fileList = []
    speciesList = []
    typeList = []
    controlList = []
    idList = []

    # This line requires work in order to guide the user to the appropriate link
    linkList = ['''<a href="experimentInfo">Edit</a>'''] * lenData

    for i in range(lenData):
        animalInfo = jsonItems[i]
        fileName = animalInfo["name"]
        species = animalInfo["animal_attributes"]["species"]
        type = animalInfo["animal_attributes"]["exp_type"]
        control = animalInfo["animal_attributes"]["control_group"]
        id = animalInfo["animal_attributes"]["ID"]
        fileList.append(fileName)
        speciesList.append(species)
        typeList.append(type)
        controlList.append(control)
        idList.append(id)
    df = pd.DataFrame(list(zip(fileList, speciesList, typeList, controlList, idList, linkList)),
                      columns=columns)
    df_copy = df
    expIndex = 0
    df = df.to_html(escape=False)
    print(df)

    return render_template('experimentInfoTable.html', tables=[df], titles=df_copy.columns.values)


# 4. Route to specify what file will store the information about the files uploaded
@app.route("/sendInfo", methods=['GET', 'POST'])
def sendInfo():
    if request.method == 'POST':
        global jsonItems
        savefileName = request.form['fileName']
        outfilename = UPLOAD_FOLDER + savefileName
        jsonstr = json.dumps(jsonItems, indent=4)
        with open(outfilename, "w") as outfile:
            outfile.write(jsonstr)
            print("Wrote the information entered into %s" % outfilename)
            jsonItems = []
            global camIndex
            global expIndex
            camIndex = 0
            expIndex = 0
        return redirect(url_for("getActions", outfileName=savefileName))
    return render_template("sendInfo.html")


# Route to specify what are the possible actions the user can take
@app.route("/getActions/<outfileName>", methods=['GET', 'POST'])
def getActions(outfileName):
    if request.method == 'POST':

        if request.form['submit_button'] == 'allCSD':
            return redirect(url_for("getAllCsd", outfileName=outfileName))

        if request.form['submit_button'] == 'twoCSD':
            return redirect(url_for("getTwoCsd", outfileName=outfileName))

        if request.form['submit_button'] == 'allBDD':
            return redirect(url_for("getAllBdd", outfileName=outfileName))

        if request.form['submit_button'] == 'twoBDD':
            return redirect(url_for("getOneBdd", outfileName=outfileName))
    return render_template("methods.html")

#  Possible Actions

# Error here- work on this
#a. heatmap- getAllCSD
@app.route("/getAllCsd/<outfileName>", methods=['GET', 'POST'])
def getAllCsd(outfileName):
    info_file = UPLOAD_FOLDER + outfileName
    animals = locomotion.getAnimalObjs(info_file)
    grid_size, start_time, end_time = 10, 0, 2
    grid_size, start_time, end_time = 10, 0, 2
    for a in animals:
        locomotion.heatmap.getSurfaceData(a, grid_size, start_time, end_time)
    distances = locomotion.heatmap.computeAllCSD(animals)
    output_directory, outfile_name = PATH_TO_DIRECTORY + "/results", "results"
    sort_table, square_table = False, False
    color_min, color_max = 0, 0.2
    locomotion.write.postProcess(animals, distances, output_directory, outfile_name, sort_table, square_table,
                                 color_min, color_max)
    return render_template("methods.html")

# Have not worked on this
#b. getTwoCSD
@app.route("/getTwoCsd/<outfileName>", methods=['GET', 'POST'])
def getTwoCsd(outfileName):
    info_file = UPLOAD_FOLDER + outfileName
    animals = locomotion.getAnimalObjs(info_file)
    for a in animals:
        print("here 1")
        locomotion.trajectory.getCurveData(a)
    variables = ['Y', 'Velocity', 'Curvature']
    norm_mode = 'spec'
    number_of_comparisons_per_animal, specified_durations = 100, None
    output_directory, outfile_name = PATH_TO_DIRECTORY, "results"
    start_time, end_time = 0, 1
    distances = locomotion.trajectory.computeAllBDD(animals,
                                                    variables,
                                                    start_time,
                                                    end_time,
                                                    norm_mode)
    print(distances)
    output_directory, outfile_name = PATH_TO_DIRECTORY + "/results", "results"
    sort_table, square_table = False, False
    color_min, color_max = 0.1, 0.5
    locomotion.write.postProcess(animals,
                                 distances,
                                 output_directory,
                                 outfile_name,
                                 sort_table,
                                 square_table,
                                 color_min,
                                 color_max)
    return render_template("methods.html")

#c. getAllBDD
@app.route("/getAllBDD/<outfileName>", methods=['GET', 'POST'])
def getAllBdd(outfileName):
    if request.method == 'POST':
        info_file = UPLOAD_FOLDER + outfileName
        animals = locomotion.getAnimalObjs(info_file)
        for a in animals:
            locomotion.trajectory.getCurveData(a)
        variables = ['Y', 'Velocity', 'Curvature']
        # start_time, end_time = 0, 1
        # norm_mode = 'spec'
        start_time = request.form['seg_start_time']
        end_time = request.form['seg_end_time']
        norm_mode = request.form['norm_mode']
        distances = locomotion.trajectory.computeAllBDD(animals, variables, start_time, end_time, norm_mode)
        output_directory, outfile_name = PATH_TO_DIRECTORY + "/results", "results"
        sort_table, square_table = False, False
        color_min, color_max = 0.1, 0.5
        locomotion.write.postProcess(animals, distances, output_directory, outfile_name, sort_table, square_table,
                                     color_min, color_max)
        df = pd.read_csv(PATH_TO_DIRECTORY + "/results/results.csv")

        def make_clickable(val):
            # target _blank to open new window
            return '<a target="_blank" href="{}">{}</a>'.format(val, val)

        df.style.format({'url': make_clickable})

        return render_template('allBddResults.html', tables=[df.to_html(classes='data')], titles=df.columns.values)
    return render_template("computeAllBDD.html")

#c. getOneBDD - intra-individual variation in BDD
# almost complete, requires more work
@app.route("/getOneBDD/<outfileName>", methods=['GET', 'POST'])
def getOneBdd(outfileName):
    info_file = UPLOAD_FOLDER + outfileName
    animals = locomotion.getAnimalObjs(info_file)
    namesOfAniamls= []
    for a in animals:
        namesOfAniamls.append(a.getName())
    if request.method == 'POST':
        animal1 = animals[request.form['animal1']]
        animal2 = animals[animals[request.form['animal2']]]
        locomotion.trajectory.getCurveData(animal1)
        locomotion.trajectory.getCurveData(animal2)
        norm_mode = request.form['norm_mode']

        distances = locomotion.trajectory.computeOneBDD(animal1, animal2, varnames=['Y', 'Velocity', 'Curvature'], seg_start_time_0=0,
                                            seg_end_time_0=1, seg_start_time_1=0,
                                            seg_end_time_1=1, norm_mode=norm_mode, fullmode=False, outdir=None)

        return render_template('oneBddResults.html', value=str(distances))
    return render_template("computeOneBDD.html", calculations=namesOfAniamls)


app.config.update(
    #Set the secret key to a sufficiently random value
    SECRET_KEY=os.urandom(24),

    #Set the session cookie to be secure
    SESSION_COOKIE_SECURE=True,

    #Set the session cookie for our app to a unique name
    SESSION_COOKIE_NAME='YourAppName-WebSession',

    #Set CSRF tokens to be valid for the duration of the session. This assumes you’re using WTF-CSRF protection
    WTF_CSRF_TIME_LIMIT=None
)

if __name__ == '__main__':
    app.run()
