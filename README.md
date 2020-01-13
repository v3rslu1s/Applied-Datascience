# Applied DataScience - Ortho Eyes
### Cooperation between LUMC Laboratorium of Kinematics en Neuromechanics (LK&N) and The Hague University of Applied Sciences. 

| # | Chapter | Paragraph | Description |
| --- | --- | --- | --- |
| 1. | [Research](#1-Research) | 
| 1.1 | | [Previous Groups](#11-Previous-groups) | 
| 1.2 | | [Project Management](#12-Project-Management) | 


# 1. Research

### Context
We are doing research for the LUMC in a collaboration with the Laboratory of Kinematics and Neuromechanics (LK&N). The LUMC has requested known patient for muscle torment for a special medical recording to the hospital. The patients are pre-selected by specialized physicians in different levels of torment. Every patient was seated into a special recording room where a physician attached multiple sensors from the Flock of Birds (FOB) recording system on bones of the patient. The patient did multiple types of exercises in most cases multiple times. 

### Data
This leaves us with a data-set of labeled kinematic patient recordings. The labels are created an put on the data by the LUMC physicians and based upon the type of exercise and the amount of torment in the muscles. 

### Research
Our research is about using machine learning techniques to classify feature patients based on the previous classification of the LUMC physicians. 

#### Research question 
> **To what extend and in what way, can different (unsupervised) data science techniques be used on kinematic recordings to contribute to a more valid and more reliable diagnosis, made by a doctor, on shoulder disability.**

#### Subquestions

# 1.1 Previous groups 
The projectgroup of 19/20 is not the first group who contributed to this research. The previous research group that took an interest in this subject has done allot of work to get us started quickly. [https://github.com/Lukelumia/Applied-Data-Science]. They mainly did research to determent what type of machine learning model would fit the data-set produced by the LUMC the best. They created a way to visualize the data and figure out what parts of the exercise are possibly leading to worse classification of the data. They also created an approach to increase the data-set. 

After reading their full reports the 19/20 project group had some doubts about certain assumptions the group made. Based on this information we contacted the LUMC for clarification. This led to the LUMC sharing more labels on our data-set in order to take some doubts / possible assumptions about the labels of the data-set from the previous group away. This still leaves allot of information to processes, and domain knowledge to gain. The verification of their process took almost the full time of the minor. 

# 1.2 Project Management
For our research we had to use SCRUM. This approach is not commonly used for research projects. However in our project group it worked good. After a few weeks reading / understanding the work of the previous group we were able sub questions (issues) building up to a main question. Each issue was built upon multiple tasks that were shared over the project group. All of this was implemented in Azure Dev Ops.  

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
I have created a class that was able to visualise raw data (before convertion to relative euler angels). [Link to full explanation of raw data](/TechincalDocumentation.md#31-Visualising-raw-data)

I first created a skeleton of each of the bones in 3D environment 
![skeleton from raw data](https://i.snipboard.io/GHBKQ5.jpg "Skeleton from raw data")

After this i started putting this code in a animation based on matplotlib. 
![raw gif](images/raw_visualisation.gif)

I added some trajectory lines and more details to the animation to be more clear of what is being visualised. 
![raw gif](images/raw-beautifed-gf.gif)

[repository for the first part of the code](https://dev.azure.com/DataScienceMinor/_git/Data%20Science?path=%2F&version=GBRaw-visualisation&_a=contents)


Because of the animation I created, we as a group were able to determined that the elbow angle is so far off a regular angle that we have skip these in future datasets. 

The number that is shown in the visualisation is the original elbow angle `% 360` witch still results in a too wide of range values to represent a normal angle. 


 
# 3.3 t-SNE
I have created a file that was able to visualise the data of all catagory's in one plot by using t-SNE. [Link to full explanation of t-SNE](/TechincalDocumentation.md#33-t-SNE)


### Results using category 1-3: 
_t-SNE AB1 Thorax_
![t-SNE AB1 Thorax](https://raw.githubusercontent.com/v3rslu1s/Applied-Datascience/master/images/TSNE-Result-AB1-Thorax-l-r.png)

_t-SNE AB1 Thorax left-right_
![t-SNE AB1 Thorax left-right](images/TSNE-Result-AB1-Thorax.png)

_t-SNE RF1 Thorax_
![t-SNE RF1 Thorax](https://raw.githubusercontent.com/v3rslu1s/Applied-Datascience/master/images/TSNE-Result-RF-Thorax.png)

Seen from the images is clear that different groups are present in the data. There are some outliners in a couple catagory's but nothing special. At the moment of creating these images there was not much data-cleaning done (for example removing double exercises and detecting anomolies.)

Also catagory 4 is missing from the dataset in the visualisations. Known was that the recordings from catagory 4 were not converted from raw to euler rotations correctly. This was clearly visible on the following visualisation. 
### Results using category 1-3 + 4 
_t-SNE AB1 catagory 4_
![t-SNE AB1 catagory 4](images/TSNE-Result-AB1-cat4.jpeg)

The small center in the middle is a zoomed out version of the first two images. We expected the data from catagory 4 to be somewhat comparable to catagory 1-3 but this result shows otherwise. Based upon this visualisation the project group choose to ignore this data until verification that the data is correct.

 
# 3.4 Combining raw + converted data

One of the ideas that was always present is to combine the information from rawdata with the converted data. The converded data was only readable by visualsing the plots. However this was hard for us to understand. With the data from the LUMC we were able to combine these two data-sets in one visualisation. With a group partner i have attempted to read both raw / converted values into a matplotlib visualisation to get the best understanding of the data-set that we have. 

![combined gif](images/animationV1.gif)

# 13. Future of the project
## Argumented Reality 
Mobile bodytracking is possible by using Apple's [ARKit](https://developer.apple.com/augmented-reality/arkit/): 
Apple presented a example for their developers on WWDC where they are [bringing people into AR](https://developer.apple.com/videos/play/wwdc2019/607). Implementing this project in wide scale available devices such as the iPhone make the work accesable for normal people. It could also provide large data-sets for physician and datascientist to do research on. Results could also be stored privatly in [HealthKit](https://developer.apple.com/healthkit/)

>![People Occlusion](https://multitudes.github.io/assets/img/arkit3/4.png)
>[_picture from Laurent Brusa - Introducing-ARKit3_](https://multitudes.github.io/2019/07/Introducing-ARKit3.html)

>![3D Motion Capture](https://multitudes.github.io/assets/img/arkit3/10.png)
>[_picture from Laurent Brusa - Introducing-ARKit3_](https://multitudes.github.io/2019/07/Introducing-ARKit3.html)

## Neural Networks
Seeing the power and the avaiablility of high quality pretrained networks i would really like to see what a model could find in a result. 

### Understanding Neural Network Descisions 
If in the feature a research group would be able to create a high functioning neural network. The descisions of the neural network would be really interesting for physician's. Understanding what the difference is in the data-set could help them: 
- Verify their current methodolgy is correct, and works like expected
- Develop new methods to investigate a patient for the disease

There are already developed technology's to get information from a neural network.

>![understanding cnn](https://miro.medium.com/max/960/1*cA9BSngo5Jgzc76CJtKJaA.jpeg)
>[https://towardsdatascience.com/understanding-your-convolution-network-with-visualizations-a4883441533b](https://towardsdatascience.com/understanding-your-convolution-network-with-visualizations-a4883441533b)

>![ai makes decisions](https://cdn-images-1.medium.com/freeze/max/1000/0*y2TVIsjnZ2cBtRdH?q=20)
>[https://mc.ai/learning-how-ai-makes-decisions/](https://mc.ai/learning-how-ai-makes-decisions/)


# 14. Scrum Tasks
- Tasks were not always assniged to a name. 
- Tasks were not always written down in DevOps

|||||||||
| --- | --- | --- | --- | --- |  --- |  --- | --- |
|ID|Work Item Type|Title|State|Area Path|Tags|Comment Count|Changed Date
|178|Task|Data: Combinations|Done|Data Science| |0|7-1-2020 14:06
|77|Task|Read through code of Matlab to see if new information could be added to csv (entropy etc)|Done|Data Science| |0|7-1-2020 13:01
|73|Task|Convert raw patient data to CSV with Matlab|Done|Data Science| |0|7-1-2020 12:55
|151|Task|Running all configs with different ml models|Doing|Data Science| |0|16-12-2019 13:41
|133|Task|Detect abnormal movement|Doing|Data Science| |0|16-12-2019 13:40
|24|Task|Read Paper|Done|Data Science| |0|16-12-2019 13:17
|51|Issue|Split all the data in a new trainset and a testset|Done|Data Science| |1|16-12-2019 13:13
|109|Task|I want to give a personal presentation about the global steps of the project / overfitting / how i solved some of my coding issues|Done|Data Science| |0|16-12-2019 12:40
|38|Issue|As a student I need to understand the basic steps the last group took|Done|Data Science| |0|16-12-2019 12:38
|149|Task|Creating configurations [based upon reached goals]|Done|Data Science| |0|2-12-2019 10:24
|145|Task|ConfigLoader|Done|Data Science| |0|25-11-2019 09:57
|137|Task|Generate more data from one exercise|Done|Data Science| |0|5-11-2019 10:28
|52|Issue|Redo the analysis from last year with the new split dataset|Done|Data Science| |0|11-10-2019 10:40
|78|Task|Converting the exercise data from Brice / Lennart / Rapahel to CSV|Doing|Data Science| |0|11-10-2019 10:20
|76|Task|Downloading course for Deep Neural Networks|Done|Data Science| |0|11-10-2019 10:19
|42|Task|Compare ml results with and without normalized data.|Done|Data Science| |1|30-9-2019 13:01
|30|Task|Comparing [super cleaned data] vs [normal data] in 2d visualisations|Done|Data Science| |0|30-9-2019 10:21
|40|Task|Use the created visualisation tool to verify the cleaned data|Done|Data Science| |1|30-9-2019 10:21
|39|Task|Create tool to visualize the cleaned data|Done|Data Science| |0|30-9-2019 10:18
|28|Task|Create animated visualisation of Raw data file|Done|Data Science| |0|16-9-2019 09:10
|15|Task|Eddie|Done|Data Science| |0|6-9-2019 09:48
 
# 15. Git Commits
![commits](https://ms-vsts.gallerycdn.vsassets.io/extensions/ms-vsts/team/1.161.0/1573137504755/Microsoft.VisualStudio.Services.Icons.Default)

[Git Commits.md](/Gitcommits.md) 