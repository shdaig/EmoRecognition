import pyedflib
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
import mne
import openpyxl


def all_name(path):
    names = []
    object = Path(path)
    subfolder = object.iterdir()
    for elem in subfolder:
        if elem.is_dir():
            names.append(elem.name)
    return names


def all_file(path, exaple):
    exaple = exaple
    filelist = [] 
    file_fif = []
    for root, dirs, files in os.walk(path): 
        for file in files:
            filelist.append(os.path.join(root,file))
            # print(root,file)
            if(file.endswith(exaple)): 
                file_fif.append(os.path.join(root, file))
    return file_fif


def all_dates(path, names, file_fif):
    dates = []
    for i in range(len(names)):
        helper = []
        for elem in file_fif:
            if names[i] in elem:
                helper.append(elem[(len(path) + len(names[i])+1):(len(path) + len(names[i])+1) + 8])
                # print(file_fif[i][(len(path) + len(elem) + 1):(len(path) + len(elem) + 1) + 8])
            # else:
            #     helper.append('no date')
        dates.append(helper)
    for i in range(len(dates)):
        if len(dates[i]) == 0:
            dates[i] = ['']
    return dates

def dataframe_create(names, dates):
    all_len = []
    for elem in dates:
        all_len.append(len(elem))
    
    table = []
    new_table = []
    for i in range(len(names)):
        table = []
        if len(dates[i]) < max(all_len):
            table.append(names[i])
            for elem in dates[i]:
                table.append(elem)
            for j in range(max(all_len) - len(table) + 1):
                table.append('')
        else:
            table.append(names[i])
            for elem in dates[i]:
                table.append(elem)
        new_table.append(table)
    data = pd.DataFrame(new_table)
    return data


















