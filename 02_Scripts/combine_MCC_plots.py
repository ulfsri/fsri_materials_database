# MCC plots combined into a single PDF
#   by: Conor McCoy
# ***************************** Run Notes ***************************** #
# - Requires that the plots have already been made                      #
# - Useful for quick evaluation of all results 							#
# - MATERIAL / FILENAMES ARE SEEN ON BOOKMARKS 	                        #
#                                                                       #
# TO DO:                                                                #
# - Add filenames on plots?												#
# - Clean up package list (some likely unused)			  	            #
#                                                                       #
# ********************************************************************* #

# --------------- #
# Import Packages #
# --------------- #
import os
import os.path # check for file existence
import glob
import numpy as np
import pandas as pd
import math
from tkinter import Tk
from tkinter.filedialog import askdirectory
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import integrate
import git
import pdfcombine # https://github.com/tdegeus/pdfcombine/blob/master/README.md#basic-usage


plot_dir = '../03_Charts/'

filenames_list=[]

for d in os.scandir(plot_dir):
	material = d.path.split('/')[-1]
	if material == '.DS_Store':
		continue
	print(f'{material} MCC')
	if d.is_dir():
		if os.path.isdir(f'{d.path}/MCC'):
			for f in glob.iglob(f'{d.path}/MCC/*.pdf'):
				g=f.replace("\\","/")
				# print(g)
				filenames_list.append(g)
# print(filenames_list)

pdfcombine.combine(filenames_list,output = "../03_Charts/combined_MCC_plots.pdf")

