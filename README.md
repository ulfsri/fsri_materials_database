# Database of Contemporary Material Properties for Fire Investigation Analysis
This project was supported by Award No. 2019-DU-BX-0018, awarded by the National Institute of Justice, Office of Justice Programs, U.S. Department of Justice. 

***
***

## Database Structure

## 01_Data/
Raw data generated from each apparatus are included here in plain-text format. The data is organized first by material, next by the short name of test apparatus used, and where applicable additional filtering by test settings.

### Test Apparatus and Data File Structure
- __Cone Calorimeter__ (Cone)
    + Data file structure includes the material, the short name for the apparatus, heat flux setting and scan/scalar, date of test, and replicate number. Scan is for the raw output data and scalar is for the initial conditions and high-level test parameters. 
        * Example: __MDF_Cone_HF25scalar_210323_R1__ stands for the first replicate of scalar data generated from a medium density fiber board test in cone calorimeter exposed to 25 kW/m<sup>2</sup> on March 23, 2021.
- __Fourier Transform Infrared Spectrometer__ (FTIR)
    + The FTIR directory contains subdirectories for the Attenuated Total Reflectance method (ATR) and the Integrating Sphere (IS)
    + Data file structure for the ATR experiment files includes the material, the abbreviated name for the apparatus, date of test, and replicate number. 
        * Example: __MDF_ATR_210323_R5__ stands for the fifth replicate of data collected with the ATR accessory conducted on June 8, 2021.
    + Data file structure for the IS experiment files includes the material, the abbreviated name for the apparatus, the measurand, whether the test is a sample measurement or standard reference, date of test, and replicate number.
        * Example: __OSB_IS_REFLECT_MEAS_210623_R7__ stands for the seventh replicate conducted on oriented strand board to measure spectral reflection in the integrating sphere on June 23, 2021.
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

## 02_Scripts/
Python processing scripts exist for analyzing the experimental data to generate derived quantities and to plot the experimental data. The scripts are apparatus specific and cycle through all materials upon execution. Each apparatus has a pair of scripts: __data.py__ and __data_html.py__. 

- The __data.py__ script produces any reduced data files, computes any derived quantities, and produces _.pdf_ graphs in __03_Charts/Material/Apparatus__. 
    + Derived quantities and/or reduced data files will get dropped into __02_Data/Material/Apparatus__. These files get updated each time the script gets executed.

- The __data_html.py__ script produces interactive _.html_ files that allow interactions such as hover, zoom, and pan. This script requires the __plotly__ package which can be installed using:
```
pip install plotly
```
Similar to the __data.py__ script, _.html_ graphs are output to __03_Charts/Material/Apparatus__ and can be opened in a web browser.

- __.bat__ and __.sh__ files exist for both the set of __data.py__ scripts and the __data_html.py__ scripts. These files will execute the respective python scripts for all data in the repository. 
    + If there are issues executing the script, in particular the __.sh__ script, a change mode may be needed. In a command prompt type:
```
chmod +x run_all_data.sh OR chmod +x run_all_data_html.sh
```
- When the scripts, either __data.py__ or __data_html.py__, are run, the current version of the repository (i.e., Github hash) is appended to the figure. To successfully execute the scripts, the GitPython package will need to be installed using:
```
pip install GitPython
```

## 03_Charts
This directory and resulting material sub directories get generated upon executing of the plotting scripts. The sub directories are broken down by material and further by test apparatus. The Charts directory is included in the .gitignore file for the repository since all the contents can be produced using the included python scripts.

## 04_Computed
Derived material properties require the use of models to interpret the raw data acquired from the test apparatus. This directory and resulting material sub directories is generated upon execution of the processing scripts included in the repository and is populated by derived material properties. Similar to the Charts directory, this directory is also included in the .gitignore as the contents are produced by the included python scripts.


