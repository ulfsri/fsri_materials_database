# JSON file writer
#   by: ULRI's Fire Safety Research Institute
#   Questions? Submit them here: https://github.com/ulfsri/fsri_materials_database/issues

# ***************************** Usage Notes *************************** #
# - Script outputs as a function of heat flux                           #
#   -  HTML Graphs dir: /03_Charts/{Material}/Cone                      #
#      Graphs: Extinction_Coefficient, Heat Release Rate Per Unit Area, #
#      Mass Loss Rate, Specific Extinction Area, Smoke Production Rate  #
#                                                                       #
#      HTML Tables dir: /01_Data/{Material}/Cone                        #
#      Tables: Heat Release Per Unit Area, CO Table, Soot Table         #
# ********************************************************************* #

# --------------- #
# Import Packages #
# --------------- #
import os
import glob
import numpy as np
import pandas as pd
import math
import git
import re

data_dir = '../../01_Data/'
charts_dir = '../../03_Charts/'
header_dir = 'material_headers/'

# # *** Extract test descriptions from existing material.json files ***

# element_df = pd.DataFrame(columns = ["specific heat", "thermal conductivity", "mass loss rate", "heat release rate per unit area", "carbon monoxide yield", "specific heat release rate", "soot yield", "effective heat of combustion", "heat of reaction", "heat of gasification", "ignition temperature", "melting temperature and enthalpy of melting", "emissivity"])
# search_term = '|'.join(element_df.columns)

# for d in sorted((f for f in os.listdir(data_dir) if not f.startswith(".")), key=str.lower):
#     if os.path.isdir(f'{data_dir}/{d}'):
#         material = d
#         print(material)

#         if os.path.isfile(f'{data_dir}{material}/material.json'):
#             with open(f'{data_dir}{material}/material.json', "r") as rf:
#                 for line in rf:
#                     match = re.search(search_term, line)
#                     if match:
#                         match_col = match.group()
#                         line = rf.readline()
#                         desc = line.split(': ')[1].split(',')[0]
#                         element_df.loc[material, match_col] = desc                      

# element_df.fillna('\"\"', inplace=True)
# element_df.to_csv('test_description.csv', index_label = 'materials')


# # *** Extract {material}_header.json files from existing material.json files ***

# for d in sorted((f for f in os.listdir(data_dir) if not f.startswith(".")), key=str.lower):
#     if os.path.isdir(f'{data_dir}/{d}'):
#         material = d
#         print(material)
#         if os.path.isfile(f'{data_dir}{material}/material.json'):
#             if not os.path.isdir('material_headers'):
#                 os.makedirs('material_headers')
#             with open(f'{data_dir}{material}/material.json', "r") as rf, open(f'material_headers/{material}_header.json', "w") as wf:
#                 for line in rf:
#                     if "measured property" in line:
#                         break
#                     wf.write(line)

# # *** Scan through repo to populate material.json files ***

df = pd.DataFrame(columns = ['Wet_cp', 'Dry_cp', 'Wet_k', 'Dry_k', 'STA_MLR', 'CONE_MLR_25', 'CONE_MLR_50', 'CONE_MLR_75', 'CONE_HRRPUA_25', 'CONE_HRRPUA_50', 'CONE_HRRPUA_75', 'CO_Yield', 'MCC_HRR', 'Soot_Yield', 'MCC_HoC', 'Cone_HoC', 'HoR', 'HoG', 'MCC_Ign_Temp', 'Melting_Temp', 'Emissivity', 'Full_JSON', "Picture"])
element_df = pd.read_csv('test_description.csv', index_col = 'materials')

for d in sorted((f for f in os.listdir(data_dir) if not f.startswith(".")), key=str.lower):
    if os.path.isdir(f'{data_dir}/{d}'):
        material = d
        print(material)

        # if material != 'Plastic_Laminate_Countertop':
        #     continue

        r = np.empty((23, ))
        r[:] = np.nan
        df.loc[material, :] = r

        measured = ['\t"measured property": [\n']
        derived = ['\t"derived property": [\n']

        # *** Specific Heat Capacity ***
        cp_list = []
        if os.path.isfile(f'{data_dir}{material}/HFM/{material}_HFM_Wet_specific_heat.html') or os.path.isfile(f'{data_dir}{material}/HFM/{material}_HFM_Dry_conductivity.html'):
            try:
                desc = element_df[material, 'specific heat']
            except:
                desc = '\"\"'

            cp_list.append('\t{\n')
            cp_list.append('\t\t"test name": "specific heat",\n')
            cp_list.append(f'\t\t"test description": {desc},\n')
            cp_list.append('\t\t"display name": "",\n')
            cp_list.append('\t\t"nested tests": [{\n')
            if os.path.isfile(f'{data_dir}{material}/HFM/{material}_HFM_Wet_specific_heat.html'):
                cp_list.append('\t\t\t\t"display name": "Unconditioned",\n')
                cp_list.append(f'\t\t\t\t"graph": "HFM/{material}_HFM_Wet_specific_heat.html",\n')
                cp_list.append(f'\t\t\t\t"table": "HFM/{material}_HFM_Wet_specific_heat.html"\n')
                cp_list.append('\t\t\t}')
                df.loc[material, 'Wet_cp'] = 'TRUE'
            if os.path.isfile(f'{data_dir}{material}/HFM/{material}_HFM_Wet_specific_heat.html') and os.path.isfile(f'{data_dir}{material}/HFM/{material}_HFM_Dry_specific_heat.html'):
                cp_list.append(',\n\t\t\t{\n')
            if os.path.isfile(f'{data_dir}{material}/HFM/{material}_HFM_Wet_specific_heat.html'):
                cp_list.append('\t\t\t\t"display name": "Dried",\n')
                cp_list.append(f'\t\t\t\t"graph": "HFM/{material}_HFM_Dry_specific_heat.html",\n')
                cp_list.append(f'\t\t\t\t"table": "HFM/{material}_HFM_Dry_specific_heat.html"\n')
                cp_list.append('\t\t\t}\n')
                df.loc[material, 'Dry_cp'] = 'TRUE'
            cp_list.append('\t\t]\n\t}')

        # *** Thermal Conductivity ***

        k_list = []
        if os.path.isfile(f'{data_dir}{material}/HFM/{material}_HFM_Wet_conductivity.html') or os.path.isfile(f'{data_dir}{material}/HFM/{material}_HFM_Dry_conductivity.html'):
            try:
                desc = element_df[material, 'thermal conductivity']
            except:
                desc = '\"\"'

            k_list.append('\t{\n')
            k_list.append('\t\t"test name": "thermal conductivity",\n')
            k_list.append(f'\t\t"test description": {desc},\n')
            k_list.append('\t\t"display name": "",\n')
            k_list.append('\t\t"nested tests": [{\n')
            if os.path.isfile(f'{data_dir}{material}/HFM/{material}_HFM_Wet_conductivity.html'):
                k_list.append('\t\t\t\t"display name": "Unconditioned",\n')
                k_list.append(f'\t\t\t\t"graph": "HFM/{material}_HFM_Wet_conductivity.html",\n')
                k_list.append(f'\t\t\t\t"table": "HFM/{material}_HFM_Wet_conductivity.html"\n')
                k_list.append('\t\t\t}')
                df.loc[material, 'Wet_k'] = 'TRUE'
            if os.path.isfile(f'{data_dir}{material}/HFM/{material}_HFM_Wet_conductivity.html') and os.path.isfile(f'{data_dir}{material}/HFM/{material}_HFM_Dry_conductivity.html'):
                k_list.append(',\n\t\t\t{\n')
            if os.path.isfile(f'{data_dir}{material}/HFM/{material}_HFM_Dry_conductivity.html'):
                k_list.append('\t\t\t\t"display name": "Dried",\n')
                k_list.append(f'\t\t\t\t"graph": "HFM/{material}_HFM_Dry_conductivity.html",\n')
                k_list.append(f'\t\t\t\t"table": "HFM/{material}_HFM_Dry_conductivity.html"\n')
                k_list.append('\t\t\t}\n')
                df.loc[material, 'Dry_k'] = 'TRUE'
            k_list.append('\t\t]\n\t}')

        # *** Mass Loss Rate ***

        mlr_list = []
        if os.path.isfile(f'{charts_dir}{material}/STA/N2/{material}_STA_MLR.html') or (os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_25.html') or os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_50.html') or os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_75.html')):
            try:
                desc = element_df[material, 'mass loss rate']
            except:
                desc = '\"\"'

            mlr_list.append('\t{\n')
            mlr_list.append('\t\t"test name": "mass loss rate",\n')
            mlr_list.append(f'\t\t"test description": {desc},\n')
            mlr_list.append('\t\t"display name": "",\n')
            mlr_list.append('\t\t"nested tests": [{\n')
            if os.path.isfile(f'{charts_dir}{material}/STA/N2/{material}_STA_MLR.html'):
                mlr_list.append('\t\t\t\t"display name": "Simultaneous Thermal Analyzer",\n')
                mlr_list.append(f'\t\t\t\t"graph": "STA/N2/{material}_STA_MLR.html"\n')
                mlr_list.append('\t\t\t}')
                df.loc[material, 'STA_MLR'] = 'TRUE'
            if os.path.isfile(f'{charts_dir}{material}/STA/N2/{material}_STA_MLR.html') and (os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_25.html') or os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_50.html') or os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_75.html')):
                mlr_list.append(',\n\t\t\t{\n')
            if os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_25.html'):
                mlr_list.append('\t\t\t\t"display name": "Cone Calorimeter: 25 kW/m<sup>2</sup>",\n')
                mlr_list.append(f'\t\t\t\t"graph": "Cone/{material}_Cone_MLR_25.html"\n')
                mlr_list.append('\t\t\t}')
                df.loc[material, 'CONE_MLR_25'] = 'TRUE'
            if (os.path.isfile(f'{charts_dir}{material}/STA/N2/{material}_STA_MLR.html') or os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_25.html')) and os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_50.html'):
                mlr_list.append(',\n\t\t\t{\n')
            if os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_50.html'):
                mlr_list.append('\t\t\t\t"display name": "Cone Calorimeter: 50 kW/m<sup>2</sup>",\n')
                mlr_list.append(f'\t\t\t\t"graph": "Cone/{material}_Cone_MLR_50.html"\n')
                mlr_list.append('\t\t\t}')
                df.loc[material, 'CONE_MLR_50'] = 'TRUE'
            if (os.path.isfile(f'{charts_dir}{material}/STA/N2/{material}_STA_MLR.html') or os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_25.html') or os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_50.html')) and os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_75.html'):
                mlr_list.append(',\n\t\t\t{\n')
            if os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_75.html'):
                mlr_list.append('\t\t\t\t"display name": "Cone Calorimeter: 75 kW/m<sup>2</sup>",\n')
                mlr_list.append(f'\t\t\t\t"graph": "Cone/{material}_Cone_MLR_75.html"\n')
                mlr_list.append('\t\t\t}\n')
                df.loc[material, 'CONE_MLR_75'] = 'TRUE'
            mlr_list.append('\t\t]\n\t}')

        # *** HRRPUA ***

        hrrpua_list = []
        if os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_HRRPUA_25.html') or os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_HRRPUA_50.html') or os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_HRRPUA_75.html'):
            try:
                desc = element_df[material, 'heat release rate per unit area']
            except:
                desc = '\"\"'                

            hrrpua_list.append('\t{\n')
            hrrpua_list.append('\t\t"test name": "heat release rate per unit area",\n')
            hrrpua_list.append(f'\t\t"test description": {desc},\n')
            hrrpua_list.append('\t\t"display name": "",\n')
            hrrpua_list.append('\t\t"nested tests": [{\n')
            if os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_HRRPUA_25.html'):
                hrrpua_list.append('\t\t\t\t"display name": "Cone Calorimeter: 25 kW/m<sup>2</sup>",\n')
                hrrpua_list.append(f'\t\t\t\t"graph": "Cone/{material}_Cone_HRRPUA_25.html",\n')
                hrrpua_list.append(f'\t\t\t\t"table": "Cone/{material}_Cone_Analysis_HRRPUA_Table_25.html"\n')
                hrrpua_list.append('\t\t\t}')
                df.loc[material, 'CONE_HRRPUA_25'] = 'TRUE'
            if os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_HRRPUA_25.html') and os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_HRRPUA_50.html'):
                hrrpua_list.append(',\n\t\t\t{\n')
            if os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_HRRPUA_50.html'):
                hrrpua_list.append('\t\t\t\t"display name": "Cone Calorimeter: 50 kW/m<sup>2</sup>",\n')
                hrrpua_list.append(f'\t\t\t\t"graph": "Cone/{material}_Cone_HRRPUA_50.html",\n')
                hrrpua_list.append(f'\t\t\t\t"table": "Cone/{material}_Cone_Analysis_HRRPUA_Table_50.html"\n')
                hrrpua_list.append('\t\t\t}')
                df.loc[material, 'CONE_HRRPUA_50'] = 'TRUE'
            if (os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_HRRPUA_25.html') or os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_HRRPUA_50.html')) and os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_HRRPUA_75.html'):
                hrrpua_list.append(',\n\t\t\t{\n')
            if os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_HRRPUA_75.html'):
                hrrpua_list.append('\t\t\t\t"display name": "Cone Calorimeter: 75 kW/m<sup>2</sup>",\n')
                hrrpua_list.append(f'\t\t\t\t"graph": "Cone/{material}_Cone_HRRPUA_75.html",\n')
                hrrpua_list.append(f'\t\t\t\t"table": "Cone/{material}_Cone_Analysis_HRRPUA_Table_75.html"\n')
                hrrpua_list.append('\t\t\t}\n')
                df.loc[material, 'CONE_HRRPUA_75'] = 'TRUE'
            hrrpua_list.append('\t\t]\n\t}')


        # *** CO Yield ***

        co_list = []
        if os.path.isfile(f'{data_dir}{material}/Cone/{material}_Cone_Analysis_CO_Table.html'):
            try:
                desc = element_df[material, 'carbon monoxide yield']
            except:
                desc = '\"\"'  
            co_list.append('\t{\n')
            co_list.append('\t\t"test name": "carbon monoxide yield",\n')
            co_list.append(f'\t\t"test description": {desc},\n')
            co_list.append('\t\t"display name": "",\n')
            co_list.append(f'\t\t"table": "Cone/{material}_Cone_Analysis_CO_Table.html"\n')
            co_list.append('\t}')
            df.loc[material, 'CO_Yield'] = 'TRUE'

        # *** Specific Heat Release Rate ***

        shrr_list = []
        if os.path.isfile(f'{charts_dir}{material}/MCC/{material}_MCC_HRR.html'):
            try:
                desc = element_df[material, 'specific heat release rate']
            except:
                desc = '\"\"'  
            shrr_list.append('\t{\n')
            shrr_list.append('\t\t"test name": "specific heat release rate",\n')
            shrr_list.append(f'\t\t"test description": {desc},\n')
            shrr_list.append('\t\t"display name": "",\n')
            shrr_list.append(f'\t\t"graph": "MCC/{material}_MCC_HRR.html"\n')
            shrr_list.append('\t}')
            df.loc[material, 'MCC_HRR'] = 'TRUE'

        # *** Derived Quantities ***
        # *** Soot Yield ***

        soot_list = []
        if os.path.isfile(f'{data_dir}{material}/Cone/{material}_Cone_Analysis_Soot_Table.html'):
            try:
                desc = element_df[material, 'soot yield']
            except:
                desc = '\"\"'  
            soot_list.append('\t{\n')
            soot_list.append('\t\t"test name": "soot yield",\n')
            soot_list.append(f'\t\t"test description": {desc},\n')
            soot_list.append('\t\t"display name": "",\n')
            soot_list.append(f'\t\t"table": "Cone/{material}_Cone_Analysis_Soot_Table.html"\n')
            soot_list.append('\t}')
            df.loc[material, 'Soot_Yield'] = 'TRUE'

        # *** Heat of Combustion ***

        ehc_list = []
        if os.path.isfile(f'{data_dir}{material}/MCC/{material}_MCC_Heats_of_Combustion.html') or os.path.isfile(f'{data_dir}{material}/Cone/{material}_Cone_Analysis_EHC_Table.html'):
            try:
                desc = element_df[material, 'effective heat of combustion']
            except:
                desc = '\"\"'  
            ehc_list.append('\t{\n')
            ehc_list.append('\t\t"test name": "effective heat of combustion",\n')
            ehc_list.append(f'\t\t"test description": {desc},\n')
            ehc_list.append('\t\t"display name": "",\n')
            ehc_list.append('\t\t"nested tests": [{\n')
            if os.path.isfile(f'{data_dir}{material}/MCC/{material}_MCC_Heats_of_Combustion.html'):
                ehc_list.append('\t\t\t\t"display name": "Micro-scale Combustion Calorimeter",\n')
                ehc_list.append(f'\t\t\t\t"table": "MCC/{material}_MCC_Heats_of_Combustion.html"\n')
                ehc_list.append('\t\t\t}')
                df.loc[material, 'MCC_HoC'] = 'TRUE'
            if os.path.isfile(f'{data_dir}{material}/MCC/{material}_MCC_Heats_of_Combustion.html') and os.path.isfile(f'{data_dir}{material}/Cone/{material}_Cone_Analysis_EHC_Table.html'):
                ehc_list.append(',\n\t\t\t{\n')
            if os.path.isfile(f'{data_dir}{material}/Cone/{material}_Cone_Analysis_EHC_Table.html'):        
                ehc_list.append('\t\t\t\t"display name": "Cone Calorimeter",\n')
                ehc_list.append(f'\t\t\t\t"table": "Cone/{material}_Cone_Analysis_EHC_Table.html"\n')
                ehc_list.append('\t\t\t}\n')
                df.loc[material, 'Cone_HoC'] = 'TRUE'
            ehc_list.append('\t\t]\n\t}')

        # *** Heat of Reaction ***

        hr_list = []
        if os.path.isfile(f'{data_dir}{material}/STA/{material}_STA_Heat_of_Reaction_Table.html'):
            try:
                desc = element_df[material, 'heat of reaction']
            except:
                desc = '\"\"'  
            hr_list.append('\t{\n')
            hr_list.append('\t\t"test name": "heat of reaction",\n')
            hr_list.append(f'\t\t"test description": {desc},\n')
            hr_list.append('\t\t"display name": "",\n')
            hr_list.append(f'\t\t"table": "STA/{material}_STA_Heat_of_Reaction_Table.html"\n') 
            hr_list.append('\t}')
            df.loc[material, 'HoR'] = 'TRUE'

        # *** Heat of Gasification ***

        hg_list = []
        if os.path.isfile(f'{data_dir}{material}/STA/{material}_STA_Heat_of_Gasification_Table.html'):
            try:
                desc = element_df[material, 'heat of gasification']
            except:
                desc = '\"\"'
            hg_list.append('\t{\n')
            hg_list.append('\t\t"test name": "heat of gasification",\n')
            hg_list.append(f'\t\t"test description": {desc},\n')
            hg_list.append('\t\t"display name": "",\n')
            hg_list.append(f'\t\t"table": "STA/{material}_STA_Heat_of_Gasification_Table.html"\n')
            hr_list.append('\t}')
            df.loc[material, 'HoG'] = 'TRUE'

        # *** Ignition Temperature ***

        it_list = []
        if os.path.isfile(f'{data_dir}{material}/MCC/{material}_MCC_Ignition_Temperature_Table.html'):
            try:
                desc = element_df[material, 'ignition temperature']
            except:
                desc = '\"\"'  
            it_list.append('\t{\n')
            it_list.append('\t\t"test name": "ignition temperature",\n')
            it_list.append(f'\t\t"test description": {desc},\n')
            it_list.append('\t\t"display name": "",\n')
            it_list.append(f'\t\t"table": "MCC/{material}_MCC_Ignition_Temperature_Table.html"\n')
            it_list.append('\t}')
            df.loc[material, 'MCC_Ign_Temp'] = 'TRUE'

        # *** Melting Temperature ***

        mt_list = []
        if os.path.isfile(f'{data_dir}{material}/STA/{material}_STA_Analysis_Melting_Temp_Table.html'):
            try:
                desc = element_df[material, 'melting temperature and enthalpy of melting']
            except:
                desc = '\"\"'  
            mt_list.append('\t{\n')
            mt_list.append('\t\t"test name": "melting temperature and enthalpy of melting",\n')
            mt_list.append(f'\t\t"test description": {desc},\n')
            mt_list.append('\t\t"display name": "",\n')
            mt_list.append(f'\t\t"table": "STA/{material}_STA_Analysis_Melting_Temp_Table.html"\n')
            mt_list.append('\t}')
            df.loc[material, 'Melting_Temp'] = 'TRUE'

        # *** Emissivity ***

        em_list = []
        if os.path.isfile(f'{data_dir}{material}/FTIR/IS/{material}_Emissivity.html'):
            try:
                desc = element_df[material, 'emissivity']
            except:
                desc = '\"\"'  
            em_list.append('\t{\n')
            em_list.append('\t\t"test name": "emissivity",\n')
            em_list.append(f'\t\t"test description": {desc},\n')
            em_list.append('\t\t"display name": "",\n')
            em_list.append(f'\t\t"table": "FTIR/IS/{material}_Emissivity.html",\n')
            em_list.append(f'\t\t"graph": "FTIR/IS/{material}_Emissivity.html"\n')
            em_list.append('\t}')
            df.loc[material, 'Emissivity'] = 'TRUE'

        measured_block_list = [cp_list, k_list, mlr_list, hrrpua_list, co_list, shrr_list]
        derived_block_list = [soot_list, ehc_list, hr_list, hg_list, it_list, mt_list, em_list]


        for m in measured_block_list:
            if m:
                measured.extend(m)
                measured.append(',\n')

        measured[-1] = '],\n'

        for d in derived_block_list:
            if d:
                derived.extend(d)
                derived.append(',\n')

        if len(derived)>1:
            derived[-1] = ']\n}'
        else:
            derived.append(']\n}')

        if os.path.isfile(f'{header_dir}/{material}_header.json'):
            df.loc[material, 'Full_JSON'] = 'TRUE'
            with open(f'{header_dir}/{material}_header.json', "r") as rf, open(f'{data_dir}{material}/material.json', "w") as wf:
                for line in rf:
                    wf.write(line)

            with open(f'{data_dir}{material}/material.json', 'a') as fid:
                for l in [measured, derived]:
                    for line in l:
                        fid.write(line)

        if os.path.isfile(f'{data_dir}{material}/{material}.jpg'):
            df.loc[material, 'Picture'] = 'TRUE'

df.fillna('FALSE', inplace=True)
df.to_csv('Material_Status.csv', index_label = 'material')
