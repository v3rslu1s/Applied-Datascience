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
# 3.1 Visualising raw data
# 3.2 Visualising converted data as 2D 
# 3.3 t-SNE
# 3.4 Combining raw + converted data

- Raw visualistion 2d/3d
- 2d Visualistion
- t-SNE
- Combining raw + converted

# 4. Converting data
- Five exercises
- Time issue
- 5 splits of the data
- More than 5 splits
- Combining exercises for patients
- Enriching methods

## enriching methods
- Combining exercises per patient
- Using 5 frames, changing offset

## Cleaning our data-set
- Filtering / Ordering
- Combining multiple exercises from one patient
- 5 rows implementation
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