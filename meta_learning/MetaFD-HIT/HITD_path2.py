# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 20:11:22 2022

@author: Owner
"""

#####
###########数据进行降采样，效果不好
###########
import os
root_dir1 = r"D:\\元学习与强化学习 - Meta-Learning Representations for Continual Learning\\mrcl_guzhangzhenuan\\metaFD-HIT\\HITdataset2"


def get_file(root_path):
    file_list = os.listdir(path=root_path)
    file_list = [os.path.join(root_path, f) for f in file_list]
    # if len(file_list) != 1:  # 若文件中存在不止一个文件，则存在歧义
    #     print('There are {} files in [{}]'.format(len(file_list), root_path))
    #     exit()
    assert len(file_list) == 1, 'There are {} files in [{}]'.format(len(file_list), root_path)  #
    return file_list[0]


# NC
NC_50= get_file(os.path.join(root_dir1, r'NC\NC_PRE50'))
NC_70 = get_file(os.path.join(root_dir1, r'NC\NC_PRE70'))
NC_90 = get_file(os.path.join(root_dir1, r'NC\NC_PRE90'))
#NC_07_10 = get_file(os.path.join(root_dir1, r'Health\N15_M07_F10'))


# PM
PM_50 = get_file(os.path.join(root_dir1, r'PM\PRE50_PM'))
PM_70 = get_file(os.path.join(root_dir1, r'PM\PRE70_PM'))
PM_90= get_file(os.path.join(root_dir1, r'PM\PRE90_PM'))
#IR_07_10 = get_file(os.path.join(root_dir1, r'IR\N15_M07_F10'))

# BDZ03
BDZ03_50 = get_file(os.path.join(root_dir1, r'BDZ_03\BDZ03_PRE50'))
BDZ03_70 = get_file(os.path.join(root_dir1, r'BDZ_03\BDZ03_PRE70'))
BDZ03_90 = get_file(os.path.join(root_dir1, r'BDZ_03\BDZ03_PRE90'))
#OR_07_10 = get_file(os.path.join(root_dir1, r'OR\N15_M07_F10'))

# BDZ09
BDZ09_50 = get_file(os.path.join(root_dir1, r'BDZ_09\BDZ09_PRE50'))
BDZ09_70 = get_file(os.path.join(root_dir1, r'BDZ_09\BDZ09_PRE70'))
BDZ09_90 = get_file(os.path.join(root_dir1, r'BDZ_09\BDZ09_PRE90'))

#BDZ03_PM

BDZ03_PM_50 = get_file(os.path.join(root_dir1, r'PM_BDZ_03\PRE50_PM_BDZ_03'))
BDZ03_PM_70 = get_file(os.path.join(root_dir1, r'PM_BDZ_03\PRE70_PM_BDZ_03'))
BDZ03_PM_90 = get_file(os.path.join(root_dir1, r'PM_BDZ_03\PRE90_PM_BDZ_03'))
#OR_07_10 = get_file(os.path.join(root_dir1, r'OR\N15_M07_F10'))

#BDZ09_PM

BDZ09_PM_50 = get_file(os.path.join(root_dir1, r'PM_BDZ_09\PRE50_PM_BDZ09'))
BDZ09_PM_70 = get_file(os.path.join(root_dir1, r'PM_BDZ_09\PRE70_PM_BDZ09'))
BDZ09_PM_90 = get_file(os.path.join(root_dir1, r'PM_BDZ_09\PRE90_PM_BDZ09'))

# Tasks with 10-way
T0 = [NC_50, PM_50,BDZ03_50,BDZ09_50,BDZ03_PM_50,BDZ09_PM_50]
T1 = [NC_70, PM_70,BDZ03_70,BDZ09_70,BDZ03_PM_70,BDZ09_PM_70]
T2 = [NC_90, PM_90,BDZ03_90,BDZ09_90,BDZ03_PM_90,BDZ09_PM_90]
#T2 = [NC_07_10 , IR_07_10,OR_07_10]
# Tasks with 7-way
# T7w = [NC_0, IF_7[0], IF_14[0], OF_7[0], OF_14[0], RoF_7[0], RoF_14[0]]
# T4w = [NC_1, IF_21[0], OF_21[0], RoF_21[0]]

# Tasks with 6-way
#T6w = [IF_7[0], IF_14[0], IF_21[0], OF_7[0], OF_14[0], OF_21[0]]
#T4w = [NC_0, RoF_7[0], RoF_14[0], RoF_21[0]]  # brand new categories

# T_sq = [NC_3, IF_7[3], OF_7[3]]
# T_sa = [NC_3, OF_7[3], RoF_7[3]]




##################
#######没有进行降采样的数据
######




import os
root_dir1 = r"D:\\元学习与强化学习 - Meta-Learning Representations for Continual Learning\\mrcl_guzhangzhenuan\\metaFD-HIT\\HITdataset3"


def get_file(root_path):
    file_list = os.listdir(path=root_path)
    file_list = [os.path.join(root_path, f) for f in file_list]
    # if len(file_list) != 1:  # 若文件中存在不止一个文件，则存在歧义
    #     print('There are {} files in [{}]'.format(len(file_list), root_path))
    #     exit()
    assert len(file_list) == 1, 'There are {} files in [{}]'.format(len(file_list), root_path)  #
    return file_list[0]


# NC
NC_50 = get_file(os.path.join(root_dir1, r'NC\NC_PRE50'))
NC_70 = get_file(os.path.join(root_dir1, r'NC\NC_PRE70'))
NC_90 = get_file(os.path.join(root_dir1, r'NC\NC_PRE90'))
#NC_07_10 = get_file(os.path.join(root_dir1, r'Health\N15_M07_F10'))


# PM
PM_50 = get_file(os.path.join(root_dir1, r'PM\PRE50_PM'))
PM_70 = get_file(os.path.join(root_dir1, r'PM\PRE70_PM'))
PM_90= get_file(os.path.join(root_dir1, r'PM\PRE90_PM'))
#IR_07_10 = get_file(os.path.join(root_dir1, r'IR\N15_M07_F10'))

# BDZ03
BDZ03_50 = get_file(os.path.join(root_dir1, r'BDZ_03\BDZ03_PRE50'))
BDZ03_70 = get_file(os.path.join(root_dir1, r'BDZ_03\BDZ03_PRE70'))
BDZ03_90 = get_file(os.path.join(root_dir1, r'BDZ_03\BDZ03_PRE90'))
#OR_07_10 = get_file(os.path.join(root_dir1, r'OR\N15_M07_F10'))


#BDZ09_PM
BDZ09_PM_50 = get_file(os.path.join(root_dir1, r'PM_BDZ_09\PRE50_PM_BDZ09'))
BDZ09_PM_70 = get_file(os.path.join(root_dir1, r'PM_BDZ_09\PRE70_PM_BDZ09'))
BDZ09_PM_90 = get_file(os.path.join(root_dir1, r'PM_BDZ_09\PRE90_PM_BDZ09'))

# Tasks with 10-way
T0 = [NC_50, PM_50,BDZ03_50,BDZ09_PM_50]
T1 = [NC_70, PM_70,BDZ03_70,BDZ09_PM_70]
T2 = [NC_90, PM_90,BDZ03_90,BDZ09_PM_90]







