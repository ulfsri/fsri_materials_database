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