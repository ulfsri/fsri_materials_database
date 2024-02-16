# JSON file validator
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
import json
from pathlib import Path

data_dir = '../../01_Data/'

no_file_list = []
valid_file_list = []
invalid_file_list = []

for d in sorted((f for f in os.listdir(data_dir) if not f.startswith(".")), key=str.lower):
    if os.path.isdir(f'{data_dir}/{d}'):
        material = d
        print(material)

        if not os.path.isfile(f'{data_dir}{d}/material.json'):
            print('NO FILE')
            no_file_list.append(material)
        else:
            try:
                data = Path.read_text(Path(f'{data_dir}{d}/material.json'))
                json_data = json.loads(data)
                print('VALID')
                valid_file_list.append(material)
            except:
                print('NOT VALID')
                invalid_file_list.append(material)

print(f'NO MATERIAL.JSON FILE: {no_file_list}')
print(f'VALID MATERIAL.JSON FILE: {valid_file_list}')
print(f'INVALID MATERIAL.JSON FILE: {invalid_file_list}')