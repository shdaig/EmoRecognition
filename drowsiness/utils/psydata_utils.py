import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


def name_date(path):
    exaple = '.json'
    key_word = 'psydata'
    dates = []
    names = []
    all_psydata = []
    object = Path(path)
    subfolder = object.iterdir()
    for elem in subfolder:
        if '.' not in str(elem):
            # names.append(str(elem)[len(path):])
            obj = Path(str(elem))
            sub = obj.iterdir()
            for date in sub:
                # print(date)
                obj2 = Path(str(date))
                sub2 = obj2.iterdir()
                for file in sub2:
                    # print(file)
                    if key_word in str(file):
                        if exaple in str(file):
                            all_psydata.append(str(file))
                            idx = str(file).find('202')
                            names.append(str(file)[len(path):idx-1])
    names = list(np.unique(names))
    for name in names:
        helper = []
        for file in all_psydata:
            if name in file:
                idx = file.find('202')
                helper.append(file[idx:idx+8])
        dates.append(helper)
    return all_psydata


def psy_data(path_file):
    before = []
    after = []
    with open(path_file) as f:
        templates = json.load(f)
        if len(templates) == 4:
                before.append(templates[0])
                before.append(templates[1])
                after.append(templates[2])
                after.append(templates[3])
        else:
            for elem in templates:
                before.append(elem)
    return before, after


def s_a_n(before):
    s,a,n = [],[],[]
    # s_sum, a_sum, n_sum = [],[],[]
    stai_quest, san_quest = [], []
    for elem1 in before:
        dic = elem1['V']
        for key in dic:
            if key.find('stai')!=-1:
                stai_quest.append(dic[key])
            elif key.find('san')!=-1:
                san_quest.append(dic[key])
    # print(san_quest)
    san_quest = np.array(san_quest)
    if len(san_quest) % 6 == 0:
        san_quest = san_quest.reshape(-1,2)
        san_quest = san_quest.reshape(-1,6)
        for elem1 in san_quest:
            # print(elem1)
            s.append(round(elem1[0]+4))
            s.append(round(elem1[1]+4))
            a.append(round(elem1[2]+4))
            a.append(round(elem1[3]+4))
            n.append(round(elem1[4]+4))
            n.append(round(elem1[5]+4))
    #         print(s)
    # print('------')
    return sum(s), sum(a), sum(n), round(sum(stai_quest))
# ------------------------------------------------------------------------------------------------------------------------------------------------------|Для Таганрога
# --------------------------------------------------|Создание списка нужных столбцов (ничего лишнего)
def column_func(df):
    column_name = []
    all_column = []
    for elem in df:
        column_name.append(elem)
    for elem in column_name:
        for elem1 in df[elem].tolist():
            if 'до' == str(elem1).lower() or 'после' == str(elem1).lower():
                all_column.append(df[elem].tolist())
    return all_column

def lst_column_create(all_column, names, dates, state):
# --------------------------------------------------|Выделение только с Ф
    # print(all_column)
    col_state = []
    for i in range(len(all_column)):
        for elem in all_column[i]:
            if f'Аксай {state}' in str(elem):
                col_state.append(all_column[i])
    # print(col_state)
# --------------------------------------------------|Распределение по 
    main_col = []
    for i in range(len(names)):
        helper = []
        for col in col_state:
            if names[i].lower() in col[0].lower():
                for elem in dates[i]: 
                    dat = elem
                    if type(col[1]) is str:
                        col_dat = str(col[1])
                        idx = dat.find(' ')
                        if idx > 0:
                            col_date_new = datetime.strptime(col_dat, '%d.%m.%y.')
                            our_date_new = datetime.strptime(dat[2:], '%y_ %m_%d')
                        else:
                            col_date_new = datetime.strptime(col_dat, '%d.%m.%y.')
                            our_date_new = datetime.strptime(dat[2:], '%y_%m_%d')
                        if col_date_new == our_date_new:
                            helper.append(col)
                    else:
                        idx = dat.find(' ')
                        if idx > 0:
                            our_date_new = datetime.strptime(dat[2:], '%y_ %m_%d')
                        else:
                            our_date_new = datetime.strptime(dat[2:], '%y_%m_%d')
                        if col[1] == our_date_new:
                            helper.append(col)
        if len(helper) > 1:
            main_col.append(helper)
    return main_col

def main_func(state, xlsx_path, path_edf):
# --------------------------------------------------|Определение списков имен и дат
    path_xlsx = xlsx_path
    edf_files = []
    names = []
    dates = []
    exaple = '.EDF'
    obj = Path(path_edf)
    subfolder = obj.iterdir()
    for elem in subfolder:
        if exaple in str(elem):
            edf_files.append(str(elem))
    for elem in edf_files:
        idx = elem.find('202')
        if idx > 0:
            new_el = elem[len(path_edf)+1:idx-1]
            idx1 = new_el.find(' ')
            if idx1 > 0:
                names.append(new_el[:idx1])
            else:
                names.append(new_el)
    names = list(np.unique(names))
    for i in range(len(names)):
        helper = []
        for elem in edf_files:
            if names[i] in elem:
                idx = elem.find('202')
                if idx > 0:
                    helper.append(elem[idx:idx+10])
        dates.append(helper)
# --------------------------------------------------|Открытие файла эксель
    f = pd.read_excel(path_xlsx, sheet_name = 2)
    f_t = f[57:84]
    f_san = f[10:49]
# ------------------|san
    all_column_san = column_func(f_san)
    main_col_san = lst_column_create(all_column_san, names, dates, state)
# ------------------|t
    all_column_t = column_func(f_t)
    main_col_t = lst_column_create(all_column_t, names, dates, state)
    return main_col_san, main_col_t


def main_func_taganrog(state, xlsx_path, path_edf):
# --------------------------------------------------|Определение списков имен и дат
    path_xlsx = xlsx_path
    edf_files = []
    names = []
    dates = []
    exaple = '.EDF'
    obj = Path(path_edf)
    subfolder = obj.iterdir()
    for elem in subfolder:
        if exaple in str(elem):
            edf_files.append(str(elem))
    for elem in edf_files:
        idx = elem.find('202')
        if idx > 0:
            new_el = elem[len(path_edf)+1:idx-1]
            idx1 = new_el.find(' ')
            if idx1 > 0:
                names.append(new_el[:idx1])
            else:
                names.append(new_el)
    names = list(np.unique(names))
    for i in range(len(names)):
        helper = []
        for elem in edf_files:
            if names[i] in elem:
                idx = elem.find('202')
                if idx > 0:
                    helper.append(elem[idx:idx+10])
        dates.append(helper)
# --------------------------------------------------|Открытие файла эксель
    f = pd.read_excel(path_xlsx, sheet_name = 2)
    f_t = f[57:84]
    f_san = f[10:49]
# ------------------|san
    all_column_san = column_func(f_san)
    main_col_san = lst_column_create_tagan(all_column_san, names, dates, state)
# ------------------|t
    all_column_t = column_func(f_t)
    main_col_t = lst_column_create_tagan(all_column_t, names, dates, state)
    return main_col_san, main_col_t

def lst_column_create_tagan(all_column, names, dates, state):
# --------------------------------------------------|Выделение только с Ф
    # print(all_column)
    col_state = []
    for i in range(len(all_column)-1):
        for elem in all_column[i]:
            if f'Аксай {state}' in str(elem):
                col_state.append(all_column[i])
                col_state.append(all_column[i+1])
    return col_state