# Applied DataScience - Ortho Eyes
### Cooperation between LUMC Laboratorium of Kinematics en Neuromechanics (LK&N) and The Hague University of Applied Sciences. 


| # | Chapter | Paragraph | Description |
| --- | --- | --- | --- |
| 1. | [Research](#1-Research) | 
| 1.1 | | [Previous Groups](#11-Previous-groups) | 
| 1.2 | | [Project Management](#12-Project-Management) | 
| 2. | [Data-set](#2-Data-set) | 
| 3. | [Visualisation](#3-Visualisation) | 
| 3.1 | | [Visualising raw data](#31-Visualising-raw-data) |
| 3.2 | | [Visualising converted data as 2D](#32-Visualising-converted-data-as-2D) |
| 3.3 | | [t-SNE](#33-t-SNE) |
| 3.4 | | [Combining raw + converted data](#34-Combining-raw-+-converted-data) | 
| 4. | [Converting data](#4-Converting-data) | 
| 4.1 | | [Combining exercises](#41-Combining-exercises) |
| 4.2 | | [Extracting more exercises](#42-Extracting-more-exercises) |
| 4.3 | | [Occupied euler space](#43-Occupied-euler-space) |
| 4.4 | | [ Images (pictures) from data ](#44-Images-(pictures)-from-data) | 
| 5. | [Machine Learning](#5-Machine-Learning) | 
| 6. | [Coding-Framework](#6-Coding-Framework) | 
| 7. | [Scrum Tasks](#7-Scrum-Tasks) | 
| 8. | [Neural Networks](#8-Neural-Networks) | 
| 9. | [Personal Development](#9-Personal-Development) | 
| 10. | [Presentations](#10-Presentations) | 
| 11. | [Wordlist](#11-Wordlist) | 
| 12. | [Failed Attempts](#12-Failed-Attempts) | 
| 13. | [Feature of the project](#13-Feature-of-the-project) | 
| 14. | [Git Commits](#14-Git-Commits) | 

# 1. Research
We are doing research for the LUMC in a collaboration with the Laboratory of Kinematics and Neuromechanics (LK&N). The LUMC has requested known patient for muscle torment for a special medical recording to the hospital. The patients are pre-selected by specialized physicians in different levels of torment. Every patient was seated into a special recording room where a physician attached multiple sensors from the Flock of Birds (FOB) recording system on bones of the patient. The patient did multiple types of exercises in most cases multiple times. 

This leaves us with a data-set of labeled patient recordings. The labels are created an put on the data by the LUMC physicians and based upon the type of exercise and the amount of torment in the muscles. 

Our research is about using machine learning techniques to classify feature patients based on the previous classification of the LUMC physicians. 


# 1.1 Previous groups 
The previous research group that took an interest in this subject has done allot of work to get us started quickly. [https://github.com/Lukelumia/Applied-Data-Science]. They mainly did research to determent what type of machine learning model would fit the dataset produced by the LUMC the best. They created a way to visualize the data and figure out what parts of the exercise are possibly leading to worse classification of the data. They also created an approach to increase the dataset. 

After reading their full reports the 19/20 project group had some doubts about certain assumptions the group made. Based on this information we contacted the LUMC for clarification. This led to the LUMC sharing more labels on our dataset in order to take the doubts / possible assumptions from the previous group away. 


# 1.2 Project Management
For our research we had to use SCRUMM. This approach is not commonly used for research projects. However in our project group it worked good. After a few weeks reading / understanding the work of the previous group we were able sub questions (issues) building up to a main question. Each issue was built upon multiple tasks that were shared over the project group. All of this was implemented in Azure Dev Ops.  


# 2. Data-set
Physicians requested patients back in 4 groups. 
Each patient was requested to do multiple exercises: 

| Short | Description |
| --- | --- |
| AB[nr.] | Abduction |
| AF[nr.] | Anteflexion |
| RF[nr.] | Retroflexion |
| EH[nr.] | Endo/Exorotation coronal |
| EL[nr.] | Endo/Exorotation humerus |

Every recording was done with the FoB (Flock of Birds) system. 


This system uses sensors attached to the skin of a patient to record the exact location of a bone using electromagnetic fields. For each recording moment of a single sensor the FoB system stores a 3D matrix with the 3D position relative to the ‘black box’ that creates the electromagnetic fields. This 3D position is saved in plain text as a euler angle / rotation matrix. 

_AB1.txt_
```
    2  857.25  -41.08 574.18    // sensorid | x y z
         0.84   -0.54   0.01    // euler rotation matrix
         0.54    0.84   0.06    // euler rotation matrix
        -0.04   -0.04   1.00    // euler rotation matrix
```  

The LUMC created a script that was able to determent the exact position of a bone based upon the sensor location placed on the skin of a patient. 

The ‘raw data’ contains the position in a 3D space for a single bone. This could not be used 1:1 for machine learning. Different examples of why this could be an issue are: 
- Length of limbs could be read by the machine learning model.
- No certainty that every recording environment produces the same 3D space.  

To tackle these issues the LUMC implemented the script to calculate the rotation angle between bones. This would mean movements would be preserved, but the recording environment / bone lengths do not have influence anymore. 

The limbs of a human body are mirrored. So we use the same latin names for each bone and define the side by adding it into the name ```_l_``` or ```_r_```. Each bone is represented by 3 axis to define its rotation in space defined as euler angles. This data is stored as a simple .csv file format. Values are represented as floating point values. 

_Thorax is mirrored (data represented a single sensor)._
_Bones recored by FoB:_
- thorax
- clavicula
- scapula
- humerus
- elbowangle


_example patient exercise format (.csv):_
```
   thorax_r_x_ext  thorax_r_y_ax  thorax_r_z_lat  clavicula_r_y_pro  clavicula_r_z_ele  ...  humerus_l_z_ele  humerus_l_y_ax  elbowangle_l  28  29
0        6.485206      -4.220661       -1.233433          -15.00546           10.47724  ...         14.79337        46.02733        399.8214   0   0
1        6.485206      -4.220661       -1.233433          -15.44328           10.46473  ...         14.77317        45.71592        399.3666   0   0
2        6.485206      -4.220661       -1.233433          -15.42001           10.48047  ...         14.76965        45.32890        399.4807   0   0
3        6.485206      -4.220661       -1.233433          -15.54270           10.43327  ...         14.86782        45.30117        399.7175   0   0
4        6.485206      -4.220661       -1.233433          -15.49114           10.93031  ...         15.45898        34.02906        391.7332   0   0
```

As being said we have 4 patient groups. Each patient did multiple exercises. The following folder structure is used through the whole project: 

_Project file tree, summarized :_
```
.
├── Category_1
│   ├── 1
│   │   ├── AB1.csv
│   │   ├── AB1.txt
│   │   ├── AB2.csv
│   │   ├── AB2.txt
│   │   ├── AF1.csv
│   │   ├── AF1.txt
│   │   ├── AF2.csv
│   │   ├── AF2.txt
│   │   └── ...
│   ├── 2
│   │   ├── AB1.csv
│   │   ├── AB1.txt
│   │   ├── AB2.csv
│   │   ├── AB2.txt
│   │   ├── AF1.csv
│   │   ├── AF1.txt
│   │   ├── AF2.csv
│   │   ├── AF2.txt
│   │   └── ...
│   └── ...
├── Category_2
│   ├── 1
│   │   ├── AB1.csv
│   │   ├── AB1.txt
│   │   └── ...
│   └── ...
├── Category_3
│   ├── 1
│   │   ├── AB1.csv
│   │   ├── AB1.txt
│   │   └── ...
│   └── ...
├── Category_4
│   ├── 1
│   │   ├── AB1.csv
│   │   ├── AB1.txt
│   │   └── ...
│   └── ...
```

Throughout the project reading the dataset very pretty easy. Using pandas we were able to load in the csv with `.read_csv()`. We attached numbers to the columns in order to replace those with the representing bone name.

```python
import pandas as pd
columns = {
        0: "thorax_r_x_ext", 1: "thorax_r_y_ax", 2: "thorax_r_z_lat",
        3: "clavicula_r_y_pro", 4: "clavicula_r_z_ele", 5: "clavicula_r_x_ax",
        6: "scapula_r_y_pro", 7: "scapula_r_z_lat", 8: "scapula_r_x_tilt",
        9: "humerus_r_y_plane", 10: "humerus_r_z_ele", 11: "humerus_r_y_ax",
        12: "ellebooghoek_r",
        15: "thorax_l_x_ext", 16: "thorax_l_y_ax", 17: "thorax_l_z_lat",
        18: "clavicula_l_y_pro", 19: "clavicula_l_z_ele", 20: "clavicula_l_x_ax",
        21: "scapula_l_y_pro", 22: "scapula_l_z_lat", 23: "scapula_l_x_tilt",
        24: "humerus_l_y_plane", 25: "humerus_l_z_ele", 26: "humerus_l_y_ax",
        27: "ellebooghoek_l"
    }
df = pd.read_csv(".../Category_1/1/AF1.csv", names=list(range(30))
df = df.rename(columns=columns)
```

```
TODO: ## new data-set
- Matlab
    - raw + calibration
    - extracting to csv 
    - woo angle system changed to lumc standards 
    - No changes in rotating when a sensor is switched
- Exercise groups are known
- Combining exercises based on labels instead of assumptions
```

# 3. Visualisation
For the project group it was the first few weeks hard to understand what data we could work with. In order to get insights into the data for our own understanding, as for the verification / data cleaning I created multiple scripts that were able to read the data an visualize it in different ways. 


# 3.1 Visualising raw data

![raw gif](https://github.com/v3rslu1s/Applied-Datascience/raw/master/images/raw_visualisation.gif)

The FoB system has multiple receivers who on their own time write the position of a sensor to a text file. In order to visualize this data i first had to split each sensor recording, and read its ID (first value). For all sensors i created a large dictionary with keys for each sensor, and storing all recordings of the sensor in a large list. -> `unpack_values()`

```
9  1267.57  -278.16  471.93  
      0.37     0.92    0.14  
      0.12    -0.19    0.97  
      0.92    -0.35   -0.18  

2  857.25  -41.08  574.18  
     0.84   -0.54    0.01  
     0.54    0.84    0.06  
    -0.04   -0.04    1.00  

3  1259.53  -17.86  35.27  
     -0.58    0.19  -0.80  
     -0.16   -0.98  -0.11  
     -0.80    0.06   0.59  
```

Each recording has 3 parameters i want to use in the visualisation: `X Y Z`. For each FoB recording the sensors were not outputted in a static order. Because of this i recreated the timeline `generate_timeline()`. 
1. **Extracting the sensor data**: From each sensor recording i retreived the first row. Splitted this into 4 chunks, and stored the values as a `NumPy` array with type `Float64`. 
2. **Looping through all sensor recordings**: and storing the float values in a dictionary with the key based on the sensor id. This groups all recordings of each sensor in a list ordered on moment of recording. 
3. **Generating timeline**: The animation function requires to draw single frames. Because of this i choose to generate a timeline from the data in a different format. The `generate_timeline()` creates a list of dictionary's in order of time. Every dictionary contains the `X Y Z` for each of the sensors. 
4. **Working with matplotlib animations**: The matplotlib `FuncAnimation()` requires a list of things to work: 
    - Figure to draw the animation
    - Function that updates the frame
    - Init function   
    - Total count of frames
    - Interval

    The init function initalises the variables used for the animation. 
    The function that updates the frames receives a frame index from the `FuncAnimation()` witch i used to get the correct sensor recordings from the timeline generated before. 

5. **Creating a skeleton**: While attempting to draw a single frame i could add some plots with the sensor index, with this information i was able to create a skeleton (lines between the bones) to form a body shape. 

![skeleton from raw data](https://i.snipboard.io/GHBKQ5.jpg "Skeleton from raw data")

6. **Animating**: My first attempt was to recreate each point in the dataset for every frame drawn in the animation. This resulted to be very slow. This is why i updated the values for each plot/line with the matplotlib `set_data()` and `set_3d_properties()` functions. 

```Python
# Updating locations of points and their labels
for index, enum in enumerate(zip(self.points, self.texts)):
    point, text = enum  
    point.set_data(x[index], y[index])
    point.set_3d_properties(z[index]) 
    text._position3d = [x[index] * 1.10, y[index], z[index]]
    text._text = str(label[index] - 2)

# Updating location of lines between points
for index, line in enumerate(self.lines):
    start_index, end_index = VisualiseRaw.stick_bones[index]
    xdata = [x[start_index], x[end_index]]
    ydata = [y[start_index], y[end_index]]
    zdata = [z[start_index], z[end_index]]
    line.set_data(xdata, ydata)
    line.set_3d_properties(zdata)
```

7. **Beautifying the animation**: In order to get some more detail from the animation i added trajectory lines to each of the bones. After each frame the trajectory point that is plotted is reduced in size what leads to a nice animation. The last object in the list is redrawn into a new trajectory point to save memory. To get a better understanding about the elbow angle i added this too as value in the visualization and added colors / legend. 

```python
# Updating trajectory points
# Last used point is updated with lastest coordinates 
for index, trajectory_list in enumerate(self.trajectory_points): 
    trajectory_list[self.current_trajectory].set_data(x[index], y[index])
    trajectory_list[self.current_trajectory].set_3d_properties(z[index])
    trajectory_list[self.current_trajectory]._color = self.colors[index]
    trajectory_list[self.current_trajectory]._markersize = 6

    # Upon each new frame, decreasing size / color of each trajectory point
    for index, trajectory in enumerate(trajectory_list):
        if index != self.current_trajectory:
            trajectory._color = trajectory._color * 0.85
            trajectory._markersize = trajectory._markersize * 0.9

# Update current_trajectory to last used trajectory point for next frame
if self.current_trajectory == len(self.colors) - 1:
    self.current_trajectory = 0 
else: 
    self.current_trajectory = self.current_trajectory + 1
```

![raw gif](https://github.com/v3rslu1s/Applied-Datascience/raw/master/images/raw-beautifed-gf.gif)

[repository for the first part of the code](https://dev.azure.com/DataScienceMinor/_git/Data%20Science?path=%2F&version=GBRaw-visualisation&_a=contents)


# 3.2 Visualising converted data as 2D 
# 3.3 t-SNE

Our data-set is so large, and has so many features that it was not possible to visualise the different catagory's in one visualisation that is still readable.
For this case i created a t-SNE visualisation on our data-set based on a [tutorial](https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b?gi=4059c9d035b8). The tutorial defines a large dataframe where two new columns were added `'y'` and `'label'`. At that moment all the code that was written was mostly object oriented, and focused upon improving memory usage and one time reading of the data. Because of this I had to rewrite a large part of the importing of the full file structure especially for t-SNE. 

[A new branch](https://dev.azure.com/DataScienceMinor/_git/Data%20Science?path=%2F&version=GBt-SNE&_a=contents) was created, where i modified the `patient.py` and `exercise.py` to read a single exercise from a patient. And store this in a single dataframe with all the additional labels / true values. I especially did not choose to load all data in t-SNE so the results would be cleaner and easier to verify. 

Using `sklearn`'s `PCA` module a new representation of the dataset was created. And by adding some configurations the script was able to view the data-set as 2D/3D with or without neighbouring enabled!

_t-SNE AB1 Thorax_
![t-SNE AB1 Thorax](https://raw.githubusercontent.com/v3rslu1s/Applied-Datascience/master/images/TSNE-Result-AB1-Thorax-l-r.png)

_t-SNE AB1 Thorax left-right_
![t-SNE AB1 Thorax left-right](https://github.com/v3rslu1s/Applied-Datascience/raw/master/images/TSNE-Result-AB1-Thorax.png)

_t-SNE RF1 Thorax_
![t-SNE RF1 Thorax](https://raw.githubusercontent.com/v3rslu1s/Applied-Datascience/master/images/TSNE-Result-RF-Thorax.png)

Seen from the images is clear that different groups are present in the data. There are some outliners in a couple catagory's but nothing special. At the moment of creating these images there was not much data-cleaning done (for example removing double exercises and detecting anomolies.)

Also catagory 4 is missing from the dataset in the visualisations. Known was that the recordings from catagory 4 were not converted from raw to euler rotations correctly. This was clearly visible on the following visualisation. 

_t-SNE AB1 catagory 4_
![t-SNE AB1 catagory 4](https://github.com/v3rslu1s/Applied-Datascience/raw/master/images/TSNE-Result-AB1-cat4.jpeg)

The small center in the middle is a zoomed out version of the first two images. We expected the data from catagory 4 to be somewhat comparable to catagory 1-3 but this result shows otherwise. Based upon this visualisation the project group choose to ignore this data until verification that the data is correct.

 
# 3.4 Combining raw + converted data

One of the ideas that was always present is to combine the information from rawdata with the converted data. The converded data was only readable by visualsing the plots. However this was hard for us to understand. With the data from the LUMC we were able to combine these two data-sets in one visualisation. With a group partner i have attempted to read both raw / converted values into a matplotlib visualisation to get the best understanding of the data-set that we have. 

![combined gif](https://github.com/v3rslu1s/Applied-Datascience/raw/master/images/combined_animation.gif)


# 4. Converting data

We were required to convert the data we received from the LUMC into a new dataset. From the previous group we knew that we should use logistic regression. This gave us a few problems creating a input dataset. 

- Large variation in amount of recorded exercises (per patient)
- Large variation in amount of exercise types (per patient)
- Exercises have different recording length (sample count)

We had to create our own implementation of this data-set in order to get it working with logistic regression. The implementation also had to be reliable, and keep as much original data. 

From this point on we as a project group started to create data-enrichment methods to get the most out of our data set. 

```
The data set that we received from the LUMC was not large in volume, but large in folders and structure. We also had allot of variation / repetition in the amount of recordings and the type of exercise. The research from the previous group pointed us to using logistic regression. Since our data-set did not have a static length, static amount of exercise recordings, static amount of exercise types for each of the patients we had to create our own interpretation for a logistic regression compatable dataset. 

At the same time did we not want to 
```

# 4.1 Combining exercises

Patient data is devided in 5 main exercises (table 1). Physician’s recorded one or more exercises each category from a single patient. 

| Exercise type | Recording 1 | Recording 2 |
| --- | --- | --- |
| Abductie [AB] | _AB1_ | __AB2__ |
| ... [AF] | _AF1_ | __AF2__ |
| ... [EH] | _EH1_ | __EH2__ |
| ... [EL] | _EL1_ | __EL2__ |
| ... [RF] | _RF1_ | __RF2__ |

The goal is to train a logistics regression model with a combination of all exercise types.
To do this we have to solve a time / exercise length problem. Exercises when executed by patients almost never have the same length. A logistics regression model expects the same amount of inputs for every entry in the dataset. We solved this by creating a combination of exercises with a fixed length. 

Timing issue 
From each exercise we have picked n frames (smaller than the smallest exercise in the whole dataset). We stepped through the exercise with a step size of exercise-length / n. This simple approach leaves us with a static number of frames for each exercise. 


In the case of an exercise with 10 frames, we can pick 5 frames from the exercise: 10 / 5 = 2. We pick the following frames from the exercise: 

||&darr;||&darr;||&darr;||&darr;||&darr;|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | __2__ | 3 | __4__ | 5 | __6__ | 7 | __8__ | 9 | __10__ |

Creating a single patient 
As said above we have 5 exercise types for each patient. We appended these combinations together in order to create a single row in our dataset. 

| [n frames] | [n frames]  | [n frames] | [n frames] | [n frames] |
| --- | --- | --- | --- | --- |
| AB1 | AF1	| EH1 | EL1 | RF1 | 

In order to maximize the training dataset, we used a combination of exercise types from a single patient. 

|Combination # | [n frames] | [n frames]  | [n frames] | [n frames] | [n frames] |
| --- | --- | --- | --- | --- | --- |
|1| AB1 | AF1 | EH1 | EL1 | RF1 |
|2| __AB2__ | AF1 | EH1 | EL1 | RF1 | 
|3| AB1 | __AF2__ | EH1 | EL1 | RF1 | 
|4| AB1 | AF1 | __EH2__ | EL1 | RF1 |
|5| AB1 | AF1 | EH1 | __EL2__ | RF1 |
|6| AB1 | AF1 | EH1 | EL1 | __RF2__ |


In the case of 5 frames per exercise, 5 exercise types per patient, 26 features per exercise = 650 features for a single patient exercise combination. 

The a mount of combinations for a single patient =  
[n AB recordings] · [n AF recordings] · [n EH recordings] · [n EL recordings] · [n RF recordings]


In order to implement this we had to reform the data into a dictionary this is sorted by exercise group for every individual patient in the dataset. 
```python
{
    "AB": [<list of exercises>],
    "AF": [<list of exercises>],
    "EH": [<list of exercises>],
    "EL": [<list of exercises>],
    "RF": [<list of exercises>]
}
```

```python
patient_data = {}
for name in Exercise.names: # contains list AB, AF, EH, etc..
    # creating empty array for each of the exercise keys
    patient_data[name] = []

# Looping through all exercises of a patient
for exercise in patient.exercises:
    # Appending the exercise into the list of key
    patient_data[exercise.name].append(exercise)

# returning a list of every single combination possible between the exercise types using itertools.product. 
return list(itertools.product(patient_data['AF'], 
                              patient_data['EL'], 
                              patient_data['AB'], 
                              patient_data['RF'], 
                              patient_data['EH']))
```



# 4.2 Extracting more exercises
In the case of an exercise with 10 frames, we can pick 5 frames from the exercise: 10 / 2 = 5. We pick the following frames from the exercise: 

|||||&darr;|||||&darr;|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 3 | 4 | __5__ | 6 | 7 | 8 | 9 | __10__ |

However this would leave us with unused parts of the exercise. In order to still use all the data for training we created a new method that looks before and after the selected frame (if possible) and extracts these as a new formatted exercise. In the example case its only possible to look before values, this leaves us with two exercise extractions: 

||||&darr;|&darr;||||&darr;|&darr;|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 3 | __4__ | _5_ | 6 | 7 | 8 | __9__ | _10_ |

This method would leave us with more data, and we did not use the same data twice. Since the data consists of movements the values almost always fluctuates. 

# 4.3 Occupied euler space
# 4.4 Images (pictures) from data 


- Five exercises 
- 5 splits of the data
    - Combining multiple exercises from one patient
    - More than 5 splits
- Combining exercises for patients
- 360 euler space
- Creating images from the data 


## enriching methods
- Combining exercises per patient
- Using 5 frames, changing offset

## Cleaning our data-set
- Removing idle 
- Filtering / Ordering
- Splitting test / train (on patient level)
- Splitting exercises 


# 5. Machine Learning
- Logistic Regression
- SVM 
- Implementing in framework

# 6. Coding Framework
- Reading the data-set
- UML 
- Benefits 
- Multi-os support
- Multi-threading
- Memory management
- Configurations
- Configuration loader
- Datalogger
- Statistics
- Visualsing results (tabulate / )
- Configloader – parsing model results (sorting results)

# 7. Scrum Tasks
- Taable here 

# 8. Neural Networks
- Creating an image from our data-set
- Adding a 4th layer
- CNN 

# 9. Personal Development 
- Datacamp
- Lectures
- Udemy 


# 10. Presentations

# 11. Wordlist 

| # | Word | Description |
| --- | --- | --- |
| 1. | FoB (Flock of Birds) | System to record bones of a patient  

# 12. Failed Attempts 
- 

# 13. Feature of the project
- AR 

# 14. Git Commits 


# LINKS TO PROVE : 
- Github commits 
- Github files 

## domain knowlegde
- Used Language

- Questions LUMC 
    - What bodypart is the most important
    - Understanding the axis
    - What information can a physician get from the arm's extention
    - What information can a physician get from the smoothness of movement 
    - Left right
    - Man vrouw