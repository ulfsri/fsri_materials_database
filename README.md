# Materials and Products Database
### Repository of materials and products properties and fire test data to improve fire modeling and fire investigation.

This project was supported in part by Award No. 2019-DU-BX-0018, awarded by the National Institute of Justice, Office of Justice Programs, U.S. Department of Justice. The opinions, findings, and conclusions or recommendations expressed in this publication / program / exhibition are those of the author(s) and do not necessarily reflect those of the Department of Justice.

***
***

__Cite this database:__ "McKinnon, M., Weinschenk, C., Dow, N., DiDomizio, M., and Madrzykowski, D. Materials and Products Database (Version 1.0.0), Fire Safety Research Institute, UL Research Institutes. Columbia, MD 21045, 2023."

__Cite the Technical Reference Guide:__ "McKinnon, M., Weinschenk, C., and Madrzykowski, D. Materials and Products Database - Technical Reference Guide, Fire Safety Research Institute, UL Research Institutes. Columbia, MD 21045, 2023."

__Cite the User Guide:__ "McKinnon, M., and Weinschenk, C. Materials and Products Database - User Guide, Fire Safety Research Institute, UL Research Institutes. Columbia, MD 21045, 2023."

***
***

## Database Structure

## 01_Data/
Raw data generated from each apparatus are included here in plain-text format. The data is organized first by material, next by the short name of test apparatus used, and where applicable additional filtering by test settings.

### Test Apparatus and Data File Structure
- __Cone Calorimeter__ (Cone)
    + Data file structure includes the material, the short name for the apparatus, heat flux setting and scan/scalar, date of test, and replicate number. Scan is for the raw output data and scalar is for the initial conditions and high-level test parameters. 
        * Example: __MDF_Cone_HF25scalar_210323_R1__ stands for the first replicate of scalar data generated from a medium density fiber board test in cone calorimeter exposed to 25 kW/m<sup>2</sup> on March 23, 2021.
    + A _Cone_Notes.csv_ is generated for each material from the __plot_Cone_data.py__ script. For materials whose experiments deviated from typical behavior relevant pictures are included in the respective material directory.
- __Fourier Transform Infrared Spectrometer__ (FTIR)
    + The FTIR directory contains subdirectories for the Attenuated Total Reflectance method (ATR) and the Integrating Sphere (IS)
    + Data file structure for the ATR experiment files includes the material, the abbreviated name for the apparatus, date of test, and replicate number. 
        * Example: __MDF_ATR_210323_R5__ stands for the fifth replicate of data collected with the ATR accessory conducted on June 8, 2021.
    + Data file structure for the IS experiment files includes the material, the abbreviated name for the apparatus, the measurand, whether the test is a sample measurement or standard reference, date of test, and replicate number.
        * Example: __OSB_IS_REFLECT_MEAS_210623_R2__ stands for the second replicate conducted on oriented strand board to measure spectral reflection in the integrating sphere on June 23, 2021.
- __Heat Flow Meter__ (HFM)
    + Data file structure is the material, the abbreviated name for the apparatus, whether test was dry or wet, whether test was for thermal conductivity or heat capacity, date of test, and replicate number. Scan is for the raw output data and scalar is for the initial conditions and high-level test parameters. 
        * Example: __OSB_HFM_Dry_Conductivity_210315_R3.tst__ stands for the third replicate of the data generated from an oriented strand board test in the heat flow meter tested dry for thermal conductivity on March 15, 2021.
- __Micro-scale Combustion Calorimeter__ (MCC)
    + Data file structure is the material, the abbreviated name for the apparatus, the heating rate in Kelvin per minute, date of test, and replicate number. Additionally, the final mass of the sample after the test is included in a separate text file named with the appendix 'FINAL_MASS'. 
        * Example: __Polyester_Fabric_MCC_30K_min_210308_R1.csv__ stands for the first replicate of the data generated from a test on polyester fabric in the micro-scale combustion calorimeter tested with a heating rate of 30 Kelvin per minute on March 8, 2021.
- __Simultaneous Thermal Analyzer__ (STA)
    + Data file structure is the material, the abbreviated name for the apparatus, the atmosphere tested in, the heating rate in Kelvin per minute and data/meta, date of test, and replicate number. Data is for the raw output data and meta is for the initial conditions and high-level test parameters. 
        * Example: __Polyester_Batting_STA_N2_3KData_210215_R1.csv__ stands for the first replicate of the data generated from a polyester batting board test in the simultaneous thermal analyzer tested in nitrogen with a heating rate of 3 Kelvin per minute on March 15, 2021.
- This directory also includes representative photographs of the respective material.

### Additional Data Files
- __material.json__
    Each material has a json file that links the data stored in the repository and data and graphs produced by the accompanying scripts to the respective landing page for the material on the front-end website. The file also contains a brief description of the material and alternate names of the material to facilitate search.
- __Photos__
    A full-size and thumbnail photo are included of materials for visualization on front-end website.

## 02_Scripts/
Python processing scripts exist for analyzing the experimental data to generate derived quantities and to plot the experimental data. The scripts are apparatus specific and cycle through all materials upon execution. Each apparatus has a pair of scripts: __data.py__ and __data_html.py__. 

- The __data.py__ script computes derived quantities, produces reduced data _.csv_ files, and produces _.pdf_ graphs in __03_Charts/Material/Apparatus__. 
    + Derived quantities and/or reduced data files will get dropped into __01_Data/Material/Apparatus__. These files get updated each time the script gets executed.

- The __data_html.py__ script produces interactive _.html_ files that allow interactions such as hover, zoom, and pan and _.html_ tables of relevant processed data. Similar to the __data.py__ script, _.html_ graphs are output to __03_Charts/Material/Apparatus__ and can be opened in a web browser.

- __run_all__ __.bat__ and __.sh__ files exist for both the set of __data.py__ scripts and the __data_html.py__ scripts. These files will execute the respective python scripts for all data in the repository. 
    + If there are issues executing the script, in particular the __.sh__ script, a change mode may be needed. In a command prompt type:
```
chmod +x run_all_data.sh OR chmod +x run_all_data_html.sh
```
- When the scripts, either __data.py__ or __data_html.py__, are run, the current version of the repository (i.e., Github hash) is appended to the figure. 


To successfully execute the Python scripts in this repository, several additional packages (outside of base Python) will need to be installed. One way to do this is through _pip_ with following commands:

```
pip install pandas              #used for data wrangling/processing
pip install numpy               #used for math analysis
pip install scipy               #used for stats analysis
pip install matplotlib          #used for plot styling and pdf plots
pip install plotly              #used for interactive html plots
pip install GitPython           #used for add repo hash to plots
pip install pybaselines         #used for melting analysis in STA
```

## 03_Charts
The material sub directories get generated upon executing of the plotting scripts. The sub directories are broken down by material and further by test apparatus.
