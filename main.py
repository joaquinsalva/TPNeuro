# -*- coding: utf-8 -*-
"""
Creamos los dataframes necesarios para el proyecto

"""
#%% Importar Librerías

# Importar Librerías
from pathlib import Path
import os

import numpy as np
import pandas as pd

from julearn import PipelineCreator, run_cross_validation
from julearn.utils import configure_logging
from julearn.inspect import preprocess
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectPercentile, chi2, SelectKBest, f_classif

from julearn.viz import plot_scores


#%% import dataSets (path noe)
# data_path = Path("/home/noefa/neuro/dataSets/TPneuro")

rois = pd.read_csv( "./rois/rois1.txt", sep="\t")

thicknessL = np.fromfile("./thickness/thickness_lh.bin", dtype=np.float32)
thicknessR = np.fromfile("./thickness/thickness_rh.bin", dtype=np.float32)
volumeL = np.fromfile("./volume/volume_lh.bin", dtype=np.float32)
volumeR = np.fromfile("./volume/volume_rh.bin", dtype=np.float32)

#%% import dataSets (path joaco)

# data_path = "ezyZip/"

# rois = pd.read_csv(data_path + "rois/rois1.txt", sep="\t")

# thicknessL = np.fromfile(data_path + "thickness/thickness_lh.bin", dtype=np.float32)
# thicknessR = np.fromfile(data_path + "thickness/thickness_rh.bin", dtype=np.float32)
# volumeL = np.fromfile(data_path + "volume/volume_lh.bin", dtype=np.float32)
# volumeR = np.fromfile(data_path + "volume/volume_rh.bin", dtype=np.float32)

#%% DataFrames1
# DataFrames

target = pd.DataFrame(rois["group"])

n_subjects = 383
#thicknessL1 = thicknessL.reshape([-1, n_subjects])
thicknessL2 = thicknessL.reshape([n_subjects, -1])


#thicknessR1 = thicknessR.reshape([-1, n_subjects])
thicknessR2 = thicknessR.reshape([n_subjects, -1])

thickness2 = pd.concat([target,pd.DataFrame(np.concatenate((thicknessR2, thicknessL2), axis=1))], axis=1)
#thickness1 = np.concatenate((thicknessR1, thicknessL1), axis=0)

n_subjects = 383
#volumeL1 = volumeL.reshape([-1, n_subjects])
volumeL2 = volumeL.reshape([n_subjects, -1])


#volumeR1 = volumeR.reshape([-1, n_subjects])
volumeR2 = volumeR.reshape([n_subjects, -1])

volume2 = pd.concat([target,pd.DataFrame(np.concatenate((volumeR2, volumeL2), axis=1))], axis=1)
#volume1 = np.concatenate((volumeR1, volumeL1), axis=0)

#%% DataFrames2
# DataFrames2
bip_sch_rois = rois[rois['group'].isin(['bip', 'sch'])].drop('id', axis=1).reset_index(drop=True)
bip_cnt_rois = rois[rois['group'].isin(['bip', 'cnt'])].drop('id', axis=1).reset_index(drop=True)
sch_cnt_rois = rois[rois['group'].isin(['cnt', 'sch'])].drop('id', axis=1).reset_index(drop=True)

rois_columns = bip_cnt_rois.columns.tolist()
rois_columns = rois_columns[1:]

rois = rois.drop('id', axis=1).reset_index(drop=True)


thickness2.columns = ['group'] + [str(col) for col in thickness2.columns[1:]]

bip_sch_thickness = thickness2[thickness2['group'].isin(['bip', 'sch'])].reset_index(drop=True)
bip_cnt_thickness = thickness2[thickness2['group'].isin(['bip', 'cnt'])].reset_index(drop=True)
sch_cnt_thickness = thickness2[thickness2['group'].isin(['cnt', 'sch'])].reset_index(drop=True)

thickness2 = thickness2.reset_index(drop=True)

thickness_columns = bip_cnt_thickness.columns.tolist()
thickness_columns = thickness_columns[1:]


volume2.columns = ['group'] + [str(col) for col in volume2.columns[1:]]

bip_sch_volume2 = volume2[volume2['group'].isin(['bip', 'sch'])].reset_index(drop=True)
bip_cnt_volume2 = volume2[volume2['group'].isin(['bip', 'cnt'])].reset_index(drop=True)
sch_cnt_volume2 = volume2[volume2['group'].isin(['cnt', 'sch'])].reset_index(drop=True)

volume2 = volume2.reset_index(drop=True)

volume_columns = bip_cnt_volume2.columns.tolist()
volume_columns = volume_columns[1:]


rois_all = [bip_sch_rois, bip_cnt_rois, sch_cnt_rois, rois]
thickness_all = [bip_sch_thickness, bip_cnt_thickness, sch_cnt_thickness, thickness2]
volume_all = [bip_sch_volume2, bip_cnt_volume2, sch_cnt_volume2, volume2]

rois_all_names = ['bip_sch_rois', 'bip_cnt_rois', 'sch_cnt_rois', 'rois']
thickness_all_names = ['bip_sch_thickness', 'bip_cnt_thickness', 'sch_cnt_thickness', 'thickness2']
volume_all_names = ['bip_sch_volume2', 'bip_cnt_volume2', 'sch_cnt_volume2', 'volume2']

