# JSON file writer
#   by: ULRI's Fire Safety Research Institute
#   Questions? Submit them here: https://github.com/ulfsri/fsri_materials_database/issues

# ***************************** Usage Notes *************************** #
# - Script writes material.json file for materials in 01_Data           #
# - material.json file stores necessary information for front end       #
#   website to display data.                                            #
#   - this includes introductory information from *_header.json         #
#   - also includes file paths to any data to be displayed              #
#                                                                       #
# - 02_Scripts/run_all_data_html.* should be run prior to this script   #
# - json files only generated for materials with matching header files  #
#   in 02_Scripts/Utilities/material_headers/                           #
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
import json
import re

data_dir = '../../01_Data/'
charts_dir = '../../03_Charts/'
header_dir = 'material_headers/'


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def json_linter_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            # Try to load the JSON from the file
            json.load(file)
    except json.JSONDecodeError as e:
        # If a JSON decoding error occurs, return the error message
        return f"Invalid JSON: {e}"
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except Exception as e:
        # Catch any other errors
        return f"An error occurred: {e}"

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

# initialize material status dataframe
if os.path.isfile('material_status.csv'):
    mat_status_df = pd.read_csv('material_status.csv', index_col = 'material')
else:
    print('ERROR: material_status.csv does not exist. Run \"run_all_data_html\" to generate html output files and material_status.csv then rerun json_writer.py')
    exit()
# df = pd.DataFrame(columns = ['Wet_cp', 'Dry_cp', 'Wet_k', 'Dry_k', 'STA_MLR', 'CONE_MLR_25', 'CONE_MLR_50', 'CONE_MLR_75', 'CONE_HRRPUA_25', 'CONE_HRRPUA_50', 'CONE_HRRPUA_75', 'CO_Yield', 'MCC_HRR', 'Soot_Yield', 'MCC_HoC', 'Cone_HoC', 'HoR', 'HoG', 'MCC_Ign_Temp', 'Melting_Temp', 'Emissivity', 'Full_JSON', "Picture"])
mat_status_df['JSON_Header'] = np.nan
mat_status_df['test_description'] = np.nan
element_df = pd.read_csv('test_description.csv', index_col = 'materials')
test_notes = json.load(open('test_description.json'))
global_json = json.load(open('../../global.json'))

for d in sorted((f for f in os.listdir(data_dir) if not f.startswith(".")), key=str.lower):
    if os.path.isdir(f'{data_dir}/{d}'):
        material = d

        # if material != 'Basswood_Panel': continue
        print(material)

        # r = np.empty((23, ))
        # r[:] = np.nan
        # df.loc[material, :] = r

        measured = ['\t"measured property": [\n']
        derived = ['\t"derived property": [\n']

        # *** Specific Heat Capacity ***
        cp_list = []
        if os.path.isfile(f'{data_dir}{material}/HFM/{material}_HFM_Wet_specific_heat.html') or os.path.isfile(f'{data_dir}{material}/HFM/{material}_HFM_Dry_conductivity.html'):
            try:
                # desc = element_df[material, 'specific heat']
                desc = f"\"{test_notes[material, 'HFM_cp']}\""
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
                # df.loc[material, 'Wet_cp'] = 'TRUE'
            if os.path.isfile(f'{data_dir}{material}/HFM/{material}_HFM_Wet_specific_heat.html') and os.path.isfile(f'{data_dir}{material}/HFM/{material}_HFM_Dry_specific_heat.html'):
                cp_list.append(',\n\t\t\t{\n')
            if os.path.isfile(f'{data_dir}{material}/HFM/{material}_HFM_Dry_specific_heat.html'):
                cp_list.append('\t\t\t\t"display name": "Dried",\n')
                cp_list.append(f'\t\t\t\t"graph": "HFM/{material}_HFM_Dry_specific_heat.html",\n')
                cp_list.append(f'\t\t\t\t"table": "HFM/{material}_HFM_Dry_specific_heat.html"\n')
                cp_list.append('\t\t\t}\n')
                # df.loc[material, 'Dry_cp'] = 'TRUE'
            cp_list.append('\t\t]\n\t}')


        # *** Thermal Conductivity ***

        k_list = []
        if os.path.isfile(f'{data_dir}{material}/HFM/{material}_HFM_Wet_conductivity.html') or os.path.isfile(f'{data_dir}{material}/HFM/{material}_HFM_Dry_conductivity.html'):
            try:
                # desc = element_df[material, 'thermal conductivity']
                desc = f"\"{test_notes[material]['HFM_k']}\""
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
                # df.loc[material, 'Wet_k'] = 'TRUE'
            if os.path.isfile(f'{data_dir}{material}/HFM/{material}_HFM_Wet_conductivity.html') and os.path.isfile(f'{data_dir}{material}/HFM/{material}_HFM_Dry_conductivity.html'):
                k_list.append(',\n\t\t\t{\n')
            if os.path.isfile(f'{data_dir}{material}/HFM/{material}_HFM_Dry_conductivity.html'):
                k_list.append('\t\t\t\t"display name": "Dried",\n')
                k_list.append(f'\t\t\t\t"graph": "HFM/{material}_HFM_Dry_conductivity.html",\n')
                k_list.append(f'\t\t\t\t"table": "HFM/{material}_HFM_Dry_conductivity.html"\n')
                k_list.append('\t\t\t}\n')
                # df.loc[material, 'Dry_k'] = 'TRUE'
            k_list.append('\t\t]\n\t}')


        # *** Mass Loss Rate ***
        try: 
            cone_file_ls = [f'{charts_dir}{material}/Cone/{ff}' 
                            for ff in os.listdir(f'{charts_dir}{material}/Cone/') 
                            if ff.startswith(f'{material}_Cone_MLR_') and ff.endswith('.html')]
            cone_file_ls = natural_sort(cone_file_ls)
        except: cone_file_ls = []

        if os.path.isfile(f'{charts_dir}{material}/STA/N2/{material}_STA_MLR.html'):
            sta_file_ls = f'{charts_dir}{material}/STA/N2/{material}_STA_MLR.html'
        else: sta_file_ls = []

        mlr_list = []
        if cone_file_ls or sta_file_ls:
        # if os.path.isfile(f'{charts_dir}{material}/STA/N2/{material}_STA_MLR.html') or (os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_25.html') or os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_50.html' or os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_75.html')) or os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_35.html')):
            
            # if os.path.isfile(f'{charts_dir}{material}/STA/N2/{material}_STA_MLR.html'):
            #     try:
            #         desc = element_df[material, 'mass loss rate']
            #     except:
            #         desc = '\"\"'

            try:
                cone_notes = test_notes[material]['Cone']
                cone_notes_bool = True
            except: cone_notes_bool = False

            try:
                sta_notes = test_notes[material]['STA']
                sta_notes_bool = True
            except: sta_notes_bool = False

            # if STA exists
            if sta_file_ls:
            # if os.path.isfile(f'{charts_dir}{material}/STA/N2/{material}_STA_MLR.html'):
                # if Cone exists
                if cone_file_ls:
                # if (os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_25.html') or os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_50.html') or os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_75.html') or os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_35.html')):
                    if sta_notes_bool and cone_notes_bool:
                    # 'both exist and notes for both -- use notes and concat. store notes in same json to make flow obvious. throw error if <global_intro>'
                        if '<global_intro>' in sta_notes or '<global_intro>' in cone_notes:
                            print(f'ERROR: Cannot note use global test description for {material}. Test notes exist for both STA and cone, which conflict with global test description for MLR. Remove <global_intro> from test notes and write standalone notes in full.')
                            exit()
                        desc = f'\"{sta_notes}<br><br>{cone_notes}\"'
                    elif sta_notes_bool:
                        # 'both exist and only notes for STA -- use notes, must be written as standalone. throw error if <global_intro>. concat hard coded variation of global description for cone notes'
                        if '<global_intro>' in sta_notes:
                            print(f'ERROR: Cannot use global test description for {material}. Test notes exist for STA but not for cone, which conflicts with global test description for MLR. Remove <global_intro> from STA test notes and write standalone notes in full (no mention of specific output). Default notes for the cone will be added automatically.')
                            exit()
                        desc = f'\"{sta_notes}<br><br>Mass loss rate [g/s] was measured in the cone calorimeter experiments at three heat fluxes: 25 kW/m<sup>2</sup>, 50 kW/m<sup>2</sup>, and 75 kW/m<sup>2</sup>.\"'
                    elif cone_notes_bool:
                        # 'both exist but only notes for cone -- cone notes must begin with <global_intro>. throw error otherwise'
                        if '<global_intro>' not in cone_notes:
                                print(f'ERROR: Need to include global test description for {material}. Test notes exist for the cone but not for STA, which overrides default STA test description for MLR. Either begin notes with <global_intro> or write independent test notes for the STA.')
                                exit()

                        intro = [global_json['measured property'][i]['test description'] for i in range(0,len(global_json['measured property'])) if global_json['measured property'][i]['test name'] == 'mass loss rate'][0]

                        desc = f'\"{intro}<br><br>Cone Calorimeter Test Notes:<br>{cone_notes.split("<global_intro>")[1]}\"'
                    else:
                        # 'both exist and no notes -- notes are empty, use global'
                        desc = '\"\"'
                # STA exists but not cone
                else: 
                    if sta_notes_bool: 
                        # 'only STA exists and has notes -- use notes, must be written as standalone. throw error if <global_intro>'
                        if '<global_intro>' in sta_notes:
                            print(f'ERROR: Cannot use global test description. Cone tests have not been run for {material}, which conflicts with global test description for MLR. STA notes much be written as standalone.')
                            exit()
                        desc = f'\"{sta_notes}\"'
                    else: 
                        # 'only STA exists and no notes -- hard coded variation of global description'
                        desc = '\"Initial-mass-normalized mass loss rate [1/s] was measured in the simultaneous thermal analyzer experiments at three heating rates: 3 K/min, 10 K/min, and 30 K/min.\"'
            # if only cone tests exists
            elif cone_file_ls:
            # elif (os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_25.html') or os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_50.html') or os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_75.html') or os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_35.html')):
                if cone_notes_bool: 
                    # only cone exists and has notes:
                    if '<global_intro>' in cone_notes:
                        #  use notes combined with hardcoded variation of global description
                        desc = f'\"Mass loss rate [g/s] was measured in cone calorimeter experiments at three heat fluxes: 25 kW/m<sup>2</sup>, 50 kW/m<sup>2</sup>, and 75 kW/m<sup>2</sup>.<br><br>{cone_notes.split("<global_intro>")[1]}\"'
                    else:
                        print(f'WARNING: Cone notes for {material} are written without "global_intro". Be sure that notes begin with an introduction that fits for all measurements.')
                        print()
                        desc = f'\"{cone_notes}\"'
                else: 
                    # 'only cone exists and no notes -- hard coded variation of global description'
                    desc = '\"Mass loss rate [g/s] was measured in cone calorimeter experiments at three heat fluxes: 25 kW/m<sup>2</sup>, 50 kW/m<sup>2</sup>, and 75 kW/m<sup>2</sup>.\"'


            mlr_list.append('\t{\n')
            mlr_list.append('\t\t"test name": "mass loss rate",\n')
            mlr_list.append(f'\t\t"test description": {desc},\n')
            mlr_list.append('\t\t"display name": "",\n')
            mlr_list.append('\t\t"nested tests": [{\n')
            if sta_file_ls:
            # if os.path.isfile(f'{charts_dir}{material}/STA/N2/{material}_STA_MLR.html'):
                mlr_list.append('\t\t\t\t"display name": "Simultaneous Thermal Analyzer",\n')
                mlr_list.append(f'\t\t\t\t"graph": "STA/N2/{material}_STA_MLR.html"\n')
                mlr_list.append('\t\t\t}')
                # df.loc[material, 'STA_MLR'] = 'TRUE'
            for ff in cone_file_ls:
                if cone_file_ls.index(ff) != 0 or sta_file_ls:  
                    mlr_list.append(',\n\t\t\t{\n')
                hf = ff[:-5].split('_')[-1]
                mlr_list.append(f'\t\t\t\t"display name": "Cone Calorimeter: {hf} kW/m<sup>2</sup>",\n')
                mlr_list.append(f'\t\t\t\t"graph": "Cone/{material}_Cone_MLR_{hf}.html"\n')
                if cone_file_ls.index(ff) == len(cone_file_ls)-1:
                    mlr_list.append('\t\t\t}\n')
                else:
                    mlr_list.append('\t\t\t}')  
            mlr_list.append('\t\t]\n\t}')

            # if os.path.isfile(f'{charts_dir}{material}/STA/N2/{material}_STA_MLR.html') and (os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_25.html') or os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_50.html') or os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_75.html')):
            #     mlr_list.append(',\n\t\t\t{\n')
            # if os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_25.html'):
            #     mlr_list.append('\t\t\t\t"display name": "Cone Calorimeter: 25 kW/m<sup>2</sup>",\n')
            #     mlr_list.append(f'\t\t\t\t"graph": "Cone/{material}_Cone_MLR_25.html"\n')
            #     mlr_list.append('\t\t\t}')
            #     # df.loc[material, 'CONE_MLR_25'] = 'TRUE'
            # if (os.path.isfile(f'{charts_dir}{material}/STA/N2/{material}_STA_MLR.html') or os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_25.html')) and os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_50.html'):
            #     mlr_list.append(',\n\t\t\t{\n')
            # if os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_50.html'):
            #     mlr_list.append('\t\t\t\t"display name": "Cone Calorimeter: 50 kW/m<sup>2</sup>",\n')
            #     mlr_list.append(f'\t\t\t\t"graph": "Cone/{material}_Cone_MLR_50.html"\n')
            #     mlr_list.append('\t\t\t}')
            #     # df.loc[material, 'CONE_MLR_50'] = 'TRUE'
            # if (os.path.isfile(f'{charts_dir}{material}/STA/N2/{material}_STA_MLR.html') or os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_25.html') or os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_50.html')) and os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_75.html'):
            #     mlr_list.append(',\n\t\t\t{\n')
            # if os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_75.html'):
            #     mlr_list.append('\t\t\t\t"display name": "Cone Calorimeter: 75 kW/m<sup>2</sup>",\n')
            #     mlr_list.append(f'\t\t\t\t"graph": "Cone/{material}_Cone_MLR_75.html"\n')
            #     mlr_list.append('\t\t\t}\n')
            #     # df.loc[material, 'CONE_MLR_75'] = 'TRUE'
            # if (os.path.isfile(f'{charts_dir}{material}/STA/N2/{material}_STA_MLR.html') or os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_25.html') or os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_50.html') or os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_75.html')) and os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_35.html'):
            #     mlr_list.append(',\n\t\t\t{\n')
            # if os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_MLR_35.html'):
            #     mlr_list.append('\t\t\t\t"display name": "Cone Calorimeter: 35 kW/m<sup>2</sup>",\n')
            #     mlr_list.append(f'\t\t\t\t"graph": "Cone/{material}_Cone_MLR_35.html"\n')
            #     mlr_list.append('\t\t\t}\n')
            #     # df.loc[material, 'CONE_MLR_35'] = 'TRUE'
            # mlr_list.append('\t\t]\n\t}')

        # *** HRRPUA ***
        try: 
            cone_file_ls = [f'{charts_dir}{material}/Cone/{ff}' 
                            for ff in os.listdir(f'{charts_dir}{material}/Cone/') 
                            if ff.startswith(f'{material}_Cone_HRRPUA_') and ff.endswith('.html')]
            cone_file_ls = natural_sort(cone_file_ls)
        except: cone_file_ls = []
        
        hrrpua_list = []
        if cone_file_ls:        
        # if os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_HRRPUA_25.html') or os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_HRRPUA_50.html') or os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_HRRPUA_75.html'):
            try:
                # desc = element_df[material, 'heat release rate per unit area']
                desc = f"\"{test_notes[material]['Cone']}\""
                if '<global_intro>' in desc:
                    intro = [global_json['measured property'][i]['test description'] for i in range(0,len(global_json['measured property'])) if global_json['measured property'][i]['test name'] == 'heat release rate per unit area'][0]
                    desc = f'\"{intro}<br><br>{desc.split("<global_intro>")[1]}'
            except:
                desc = '\"\"'        
            hrrpua_list.append('\t{\n')
            hrrpua_list.append('\t\t"test name": "heat release rate per unit area",\n')
            hrrpua_list.append(f'\t\t"test description": {desc},\n')
            hrrpua_list.append('\t\t"display name": "",\n')
            hrrpua_list.append('\t\t"nested tests": [{\n')

            for ff in cone_file_ls:
                if cone_file_ls.index(ff) != 0:  
                    hrrpua_list.append(',\n\t\t\t{\n')
                hf = ff[:-5].split('_')[-1]
                hrrpua_list.append(f'\t\t\t\t"display name": "Cone Calorimeter: {hf} kW/m<sup>2</sup>",\n')
                hrrpua_list.append(f'\t\t\t\t"graph": "Cone/{material}_Cone_HRRPUA_{hf}.html",\n')
                hrrpua_list.append(f'\t\t\t\t"table": "Cone/{material}_Cone_Analysis_HRRPUA_Table_{hf}.html"\n') 
                if cone_file_ls.index(ff) == len(cone_file_ls)-1:
                    hrrpua_list.append('\t\t\t}\n')
                else:
                    hrrpua_list.append('\t\t\t}')  
            hrrpua_list.append('\t\t]\n\t}')

            # if os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_HRRPUA_25.html'):
            #     hrrpua_list.append('\t\t\t\t"display name": "Cone Calorimeter: 25 kW/m<sup>2</sup>",\n')
            #     hrrpua_list.append(f'\t\t\t\t"graph": "Cone/{material}_Cone_HRRPUA_25.html",\n')
            #     hrrpua_list.append(f'\t\t\t\t"table": "Cone/{material}_Cone_Analysis_HRRPUA_Table_25.html"\n')
            #     hrrpua_list.append('\t\t\t}')
            #     # df.loc[material, 'CONE_HRRPUA_25'] = 'TRUE'
            # if os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_HRRPUA_25.html') and os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_HRRPUA_50.html'):
            #     hrrpua_list.append(',\n\t\t\t{\n')
            # if os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_HRRPUA_50.html'):
            #     hrrpua_list.append('\t\t\t\t"display name": "Cone Calorimeter: 50 kW/m<sup>2</sup>",\n')
            #     hrrpua_list.append(f'\t\t\t\t"graph": "Cone/{material}_Cone_HRRPUA_50.html",\n')
            #     hrrpua_list.append(f'\t\t\t\t"table": "Cone/{material}_Cone_Analysis_HRRPUA_Table_50.html"\n')
            #     hrrpua_list.append('\t\t\t}')
            #     # df.loc[material, 'CONE_HRRPUA_50'] = 'TRUE'
            # if (os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_HRRPUA_25.html') or os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_HRRPUA_50.html')) and os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_HRRPUA_75.html'):
            #     hrrpua_list.append(',\n\t\t\t{\n')
            # if os.path.isfile(f'{charts_dir}{material}/Cone/{material}_Cone_HRRPUA_75.html'):
            #     hrrpua_list.append('\t\t\t\t"display name": "Cone Calorimeter: 75 kW/m<sup>2</sup>",\n')
            #     hrrpua_list.append(f'\t\t\t\t"graph": "Cone/{material}_Cone_HRRPUA_75.html",\n')
            #     hrrpua_list.append(f'\t\t\t\t"table": "Cone/{material}_Cone_Analysis_HRRPUA_Table_75.html"\n')
            #     hrrpua_list.append('\t\t\t}\n')
            #     # df.loc[material, 'CONE_HRRPUA_75'] = 'TRUE'
            # hrrpua_list.append('\t\t]\n\t}')


        # *** CO Yield ***

        co_list = []
        if os.path.isfile(f'{data_dir}{material}/Cone/{material}_Cone_Analysis_CO_Table.html'):
            try:
                # desc = element_df[material, 'CO Yield']
                desc = f"\"{test_notes[material]['Cone']}\""
                if '<global_intro>' in desc:
                    intro = [global_json['measured property'][i]['test description'] for i in range(0,len(global_json['measured property'])) if global_json['measured property'][i]['test name'] == 'carbon monoxide yield'][0]
                    desc = f'\"{intro}<br><br>{desc.split("<global_intro>")[1]}'
            except:
                desc = '\"\"'   

            co_list.append('\t{\n')
            co_list.append('\t\t"test name": "carbon monoxide yield",\n')
            co_list.append(f'\t\t"test description": {desc},\n')
            co_list.append('\t\t"display name": "",\n')
            co_list.append(f'\t\t"table": "Cone/{material}_Cone_Analysis_CO_Table.html"\n')
            co_list.append('\t}')
            # df.loc[material, 'CO_Yield'] = 'TRUE'

        # *** Specific Heat Release Rate ***

        shrr_list = []
        if os.path.isfile(f'{charts_dir}{material}/MCC/{material}_MCC_HRR.html'):  
            try:
                # desc = element_df[material, 'specific heat release rate']
                desc = f"\"{test_notes[material]['MCC']}\""
            except:
                desc = '\"\"'  
            shrr_list.append('\t{\n')
            shrr_list.append('\t\t"test name": "specific heat release rate",\n')
            shrr_list.append(f'\t\t"test description": {desc},\n')
            shrr_list.append('\t\t"display name": "",\n')
            shrr_list.append(f'\t\t"graph": "MCC/{material}_MCC_HRR.html"\n')
            shrr_list.append('\t}')
            # df.loc[material, 'MCC_HRR'] = 'TRUE'

        # *** Derived Quantities ***
        # *** Soot Yield ***

        soot_list = []
        if os.path.isfile(f'{data_dir}{material}/Cone/{material}_Cone_Analysis_Soot_Table.html'): 
            try:
                # desc = element_df[material, 'soot yield']
                desc = f"\"{test_notes[material]['Cone']}\""
                if '<global_intro>' in desc:
                    intro = [global_json['derived property'][i]['test description'] for i in range(0,len(global_json['derived property'])) if global_json['derived property'][i]['test name'] == 'soot yield'][0]
                    desc = f'\"{intro}<br><br>{desc.split("<global_intro>")[1]}'
            except:
                desc = '\"\"' 
            soot_list.append('\t{\n')
            soot_list.append('\t\t"test name": "soot yield",\n')
            soot_list.append(f'\t\t"test description": {desc},\n')
            soot_list.append('\t\t"display name": "",\n')
            soot_list.append(f'\t\t"table": "Cone/{material}_Cone_Analysis_Soot_Table.html"\n')
            soot_list.append('\t}')
            # df.loc[material, 'Soot_Yield'] = 'TRUE'

        # *** Heat of Combustion ***

        ehc_list = []
        if os.path.isfile(f'{data_dir}{material}/MCC/{material}_MCC_Heats_of_Combustion.html') or os.path.isfile(f'{data_dir}{material}/Cone/{material}_Cone_Analysis_EHC_Table.html'):
            try:
                desc = f"\"{element_df[material, 'effective heat of combustion']}\""
            except:
                desc = '\"\"'  

            try:
                cone_notes = test_notes[material]['Cone']
                cone_notes_bool = True
            except: cone_notes_bool = False

            try:
                mcc_notes = test_notes[material]['MCC']
                mcc_notes_bool = True
            except: mcc_notes_bool = False

            # if MCC exists
            if os.path.isfile(f'{data_dir}{material}/MCC/{material}_MCC_Heats_of_Combustion.html'):
                # if Cone exists
                if os.path.isfile(f'{data_dir}{material}/Cone/{material}_Cone_Analysis_EHC_Table.html'):             
                    if mcc_notes_bool and cone_notes_bool:
                    # 'both exist and notes for both -- use notes and concat. store notes in same json to make flow obvious. throw error if <global_intro>'
                        if '<global_intro>' in mcc_notes or '<global_intro>' in cone_notes:
                            print(f'ERROR: Cannot note use global test description for {material}. Test notes exist for both MCC and cone, which conflict with global test description for HoC. Remove <global_intro> from test notes and write standalone notes in full.')
                            exit()
                        desc = f'\"{mcc_notes}<br><br>{cone_notes}\"'
                    elif mcc_notes_bool:
                        # 'both exist and only notes for MCC -- use notes, must be written as standalone. throw error if <global_intro>. concat hard coded variation of global description for cone notes'
                        if '<global_intro>' in sta_notes:
                            print(f'ERROR: Cannot use global test description for {material}. Test notes exist for MCC but not for cone, which conflicts with global test description for HoC. Remove <global_intro> from MCC test notes and write standalone notes in full (no mention of specific output). Default notes for the cone will be added automatically.')
                            exit()
                        desc = f'\"{sta_notes}<br><br>Effective heat of combustion [MJ/kg] is calculated from data collected in cone calorimeter experiments at three heat fluxes: 25 kW/m<sup>2</sup>, 50 kW/m<sup>2</sup>, and 75 kW/m<sup>2</sup>.\"'
                    elif cone_notes_bool:
                        # 'both exist but only notes for cone -- cone notes must begin with <global_intro>. throw error otherwise'
                        if '<global_intro>' not in cone_notes:
                                print(f'ERROR: Need to include global test description for {material}. Test notes exist for the cone but not for MCC, which overrides default MCC test description for HoC. Either begin notes with <global_intro> or write independent test notes for the MCC.')
                                exit()
                        intro = [global_json['derived property'][i]['test description'] for i in range(0,len(global_json['derived property'])) if global_json['derived property'][i]['test name'] == 'effective heat of combustion'][0]
                        desc = f'\"{intro}<br><br>Cone Calorimeter Test Notes:<br>{cone_notes.split("<global_intro>")[1]}\"'
                    else:
                        # 'both exist and no notes -- notes are empty, use global'
                        desc = '\"\"'
                # MCC exists but not cone
                else: 
                    # 'only MCC exists and no notes -- hard coded variation of global description'
                    desc = '\"Effective heat of combustion [MJ/kg] is calculated from data collected in micro-scale combustion calorimeter experiments.\"'
            # if only cone tests exists
            elif os.path.isfile(f'{data_dir}{material}/Cone/{material}_Cone_Analysis_EHC_Table.html'):
                if cone_notes_bool: 
                    # 'only cone exists and has notes: -- use notes, must be written as standalone. throw error if  <global_intro>'
                    if '<global_intro>' in cone_notes:
                        #  use notes combined with hardcoded variation of global description
                        desc = f'\"Effective heat of combustion [MJ/kg] is calculated from data collected in cone calorimeter experiments.<br><br>{cone_notes.split("<global_intro>")[1]}\"'
                    else:
                        print(f'WARNING (MCC): Cone notes for {material} are written without "global_intro". Be sure that notes begin with an introduction that fits for all measurements.')
                        desc = f"\"{cone_notes}\""
                else: 
                    # 'only cone exists and no notes -- hard coded variation of global description'
                    desc = '\"Effective heat of combustion [MJ/kg] is calculated from data collected in cone calorimeter experiments.\"'

            
            ehc_list.append('\t{\n')
            ehc_list.append('\t\t"test name": "effective heat of combustion",\n')
            ehc_list.append(f'\t\t"test description": {desc},\n')
            ehc_list.append('\t\t"display name": "",\n')
            ehc_list.append('\t\t"nested tests": [{\n')
            if os.path.isfile(f'{data_dir}{material}/MCC/{material}_MCC_Heats_of_Combustion.html'):
                ehc_list.append('\t\t\t\t"display name": "Micro-scale Combustion Calorimeter",\n')
                ehc_list.append(f'\t\t\t\t"table": "MCC/{material}_MCC_Heats_of_Combustion.html"\n')
                ehc_list.append('\t\t\t}')
                # df.loc[material, 'MCC_HoC'] = 'TRUE'
            if os.path.isfile(f'{data_dir}{material}/MCC/{material}_MCC_Heats_of_Combustion.html') and os.path.isfile(f'{data_dir}{material}/Cone/{material}_Cone_Analysis_EHC_Table.html'):
                ehc_list.append(',\n\t\t\t{\n')
            if os.path.isfile(f'{data_dir}{material}/Cone/{material}_Cone_Analysis_EHC_Table.html'):        
                ehc_list.append('\t\t\t\t"display name": "Cone Calorimeter",\n')
                ehc_list.append(f'\t\t\t\t"table": "Cone/{material}_Cone_Analysis_EHC_Table.html"\n')
                ehc_list.append('\t\t\t}\n')
                # df.loc[material, 'Cone_HoC'] = 'TRUE'
            ehc_list.append('\t\t]\n\t}')

        # *** Heat of Reaction ***

        hr_list = []
        if os.path.isfile(f'{data_dir}{material}/STA/{material}_STA_Heat_of_Reaction_Table.html'):
            try:
                # desc = element_df[material, 'heat of reaction']
                desc = f"\"{test_notes[material]['STA']}\""
            except:
                desc = '\"\"'  
            hr_list.append('\t{\n')
            hr_list.append('\t\t"test name": "heat of reaction",\n')
            hr_list.append(f'\t\t"test description": {desc},\n')
            hr_list.append('\t\t"display name": "",\n')
            hr_list.append(f'\t\t"table": "STA/{material}_STA_Heat_of_Reaction_Table.html"\n') 
            hr_list.append('\t}')
            # df.loc[material, 'HoR'] = 'TRUE'

        # *** Heat of Gasification ***

        hg_list = []
        if os.path.isfile(f'{data_dir}{material}/STA/{material}_STA_Heat_of_Gasification_Table.html'):
            try:
                # desc = element_df[material, 'heat of gasification']
                desc = f"\"{test_notes[material]['STA']}\""
            except:
                desc = '\"\"'
            hg_list.append('\t{\n')
            hg_list.append('\t\t"test name": "heat of gasification",\n')
            hg_list.append(f'\t\t"test description": {desc},\n')
            hg_list.append('\t\t"display name": "",\n')
            hg_list.append(f'\t\t"table": "STA/{material}_STA_Heat_of_Gasification_Table.html"\n')
            hr_list.append('\t}')
            # df.loc[material, 'HoG'] = 'TRUE'

        # *** Ignition Temperature ***

        it_list = []
        if os.path.isfile(f'{data_dir}{material}/MCC/{material}_MCC_Ignition_Temperature_Table.html'):
            try:
                # desc = element_df[material, 'ignition temperature']
                desc = f"\"{test_notes[material]['MCC']}\""
            except:
                desc = '\"\"'  
            it_list.append('\t{\n')
            it_list.append('\t\t"test name": "ignition temperature",\n')
            it_list.append(f'\t\t"test description": {desc},\n')
            it_list.append('\t\t"display name": "",\n')
            it_list.append(f'\t\t"table": "MCC/{material}_MCC_Ignition_Temperature_Table.html"\n')
            it_list.append('\t}')
            # df.loc[material, 'MCC_Ign_Temp'] = 'TRUE'

        # *** Melting Temperature ***

        mt_list = []
        if os.path.isfile(f'{data_dir}{material}/STA/{material}_STA_Analysis_Melting_Temp_Table.html'):
            try:
                # desc = element_df[material, 'melting temperature and enthalpy of melting']
                desc = f"\"{test_notes[material]['STA']}\""
            except:
                desc = '\"\"'  
            mt_list.append('\t{\n')
            mt_list.append('\t\t"test name": "melting temperature and enthalpy of melting",\n')
            mt_list.append(f'\t\t"test description": {desc},\n')
            mt_list.append('\t\t"display name": "",\n')
            mt_list.append(f'\t\t"table": "STA/{material}_STA_Analysis_Melting_Temp_Table.html"\n')
            mt_list.append('\t}')
            # df.loc[material, 'Melting_Temp'] = 'TRUE'

        # *** Emissivity ***

        em_list = []
        if os.path.isfile(f'{data_dir}{material}/FTIR/IS/{material}_Emissivity.html'):
            try:
                # desc = element_df[material, 'emissivity']
                desc = f"\"{test_notes[material]['IS']}\""
            except:
                desc = '\"\"'  
            em_list.append('\t{\n')
            em_list.append('\t\t"test name": "emissivity",\n')
            em_list.append(f'\t\t"test description": {desc},\n')
            em_list.append('\t\t"display name": "",\n')
            em_list.append(f'\t\t"table": "FTIR/IS/{material}_Emissivity.html",\n')
            em_list.append(f'\t\t"graph": "FTIR/IS/{material}_Emissivity.html"\n')
            em_list.append('\t}')
            # df.loc[material, 'Emissivity'] = 'TRUE'

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

        # write to JSON file and update JSON status in material_status_df
        if os.path.isfile(f'{header_dir}/{material}_header.json'):
            mat_status_df.loc[material, 'JSON_Header'] = True


            mat_status_df.loc[material, 'Full_JSON'] = True

            with open(f'{header_dir}/{material}_header.json', "r") as rf, open(f'{data_dir}{material}/material.json', "w") as wf:
                for line in rf:
                    wf.write(line)

            with open(f'{data_dir}{material}/material.json', 'a') as fid:
                for l in [measured, derived]:
                    for line in l:
                        fid.write(line)
            
            error = json_linter_from_file(f'{data_dir}{material}/material.json')
            if error:
                print(error)
                exit()

        else:
            mat_status_df.loc[material, 'JSON_Header'] = False

            if os.path.isfile(f'{data_dir}{material}/material.json'):
                mat_status_df.loc[material, 'Full_JSON'] = 'Existing json without header file'
                # print('Existing json without header file')
                error = json_linter_from_file(f'{data_dir}{material}/material.json')
                if error:
                    print(error)
                    exit()
            else:
                mat_status_df.loc[material, 'Full_JSON'] = False
                # print('No json')
            

        if os.path.isfile(f'{data_dir}{material}/{material}.jpg'):
            mat_status_df.loc[material, 'Picture'] = True


        # check which materials need to be added to test_description
        # if cone data exists, it needs to be in test_description
        # even if no cone specific notes, add empty entry for material to record that no cone notes are necessary
        cone_test = False
        for col in mat_status_df:
            if 'CONE' in col:
                if mat_status_df.loc[material, col]: 
                    cone_test = True
                    # print(col)
                    break
        
        if cone_test: 
            try:
                if test_notes[material]:
                    mat_status_df.loc[material, 'test_description'] = True
                    # print('True')
            except:
                mat_status_df.loc[material, 'test_description'] = False
                # print('False')

# df.fillna('FALSE', inplace=True)

mat_status_df.to_csv('Material_Status.csv', index_label = 'material')

test_notes = {key: test_notes[key] for key in sorted(test_notes)}
with open('test_description.json',  'w') as file:
    json.dump(test_notes, file, indent=4)


full_json = (mat_status_df['Full_JSON'] == True).sum()
existing = (mat_status_df['Full_JSON'] == 'Existing json without header file').sum()
no_json = (mat_status_df['Full_JSON'] == False).sum()

print()
print('Full json: ', full_json)
print('No json: ', no_json)
print('Existing json without header file: ', existing)
print(mat_status_df[mat_status_df['Full_JSON'] == 'Existing json without header file'].index.values)

# 
print()
no_notes = (mat_status_df['test_description'] == False).sum()
print('Has cone data but no notes in test_description.json: ', no_notes)
print(mat_status_df[mat_status_df['test_description'] == False].index.values)
# print('In test_description: ', (mat_status_df['test_description'] == True).sum())
# print('Total materials: ', len(mat_status_df))

