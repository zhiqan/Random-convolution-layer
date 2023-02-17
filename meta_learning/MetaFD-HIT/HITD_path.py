
import os
root_dir1 = r"D:\\元学习与强化学习 - Meta-Learning Representations for Continual Learning\\mrcl_guzhangzhenuan\\metaFD-HIT\\HITdataset"


def get_file(root_path):
    file_list = os.listdir(path=root_path)
    file_list = [os.path.join(root_path, f) for f in file_list]
    # if len(file_list) != 1:  # 若文件中存在不止一个文件，则存在歧义
    #     print('There are {} files in [{}]'.format(len(file_list), root_path))
    #     exit()
    assert len(file_list) == 1, 'There are {} files in [{}]'.format(len(file_list), root_path)  #
    return file_list[0]


# NC
NC_800 = get_file(os.path.join(root_dir1, r'NC\NC_800'))
NC_1000 = get_file(os.path.join(root_dir1, r'NC\NC_1000'))
NC_1200 = get_file(os.path.join(root_dir1, r'NC\NC_1200'))
#NC_07_10 = get_file(os.path.join(root_dir1, r'Health\N15_M07_F10'))


# PM
PM_800 = get_file(os.path.join(root_dir1, r'PM\800RPM_PM'))
PM_1000 = get_file(os.path.join(root_dir1, r'PM\1000RPM_PM'))
PM_1200= get_file(os.path.join(root_dir1, r'PM\1200RPM_PM'))
#IR_07_10 = get_file(os.path.join(root_dir1, r'IR\N15_M07_F10'))

# BDZ
BDZ_800 = get_file(os.path.join(root_dir1, r'BDZ\BDZ_800rpm'))
BDZ_1000 = get_file(os.path.join(root_dir1, r'BDZ\BDZ_1000rpm'))
BDZ_1200 = get_file(os.path.join(root_dir1, r'BDZ\BDZ_1200rpm'))
#OR_07_10 = get_file(os.path.join(root_dir1, r'OR\N15_M07_F10'))

#BDZ_PM
BDZ_PM_800 = get_file(os.path.join(root_dir1, r'PM_BDZ\800RPM_PM_BDZ'))
BDZ_PM_1000 = get_file(os.path.join(root_dir1, r'PM_BDZ\1000RPM_PM_BDZ'))
BDZ_PM_1200 = get_file(os.path.join(root_dir1, r'PM_BDZ\1200RPM_PM_BDZ'))
#OR_07_10 = get_file(os.path.join(root_dir1, r'OR\N15_M07_F10'))

#PRE50NC
PRE50NC_800 = get_file(os.path.join(root_dir1, r'Pre_50_NC\800RPM_PRE_50'))
PRE50NC_1000 = get_file(os.path.join(root_dir1, r'Pre_50_NC\1000RPM_PRE_50'))
PRE50NC_1200 = get_file(os.path.join(root_dir1, r'Pre_50_NC\1200RPM_PRE_50'))
#OR_07_10 = get_file(os.path.join(root_dir1, r'OR\N15_M07_F10'))
#PRE90NC
PRE90NC_800 = get_file(os.path.join(root_dir1, r'Pre_90_NC\800RPM_PRE_90'))
PRE90NC_1000 = get_file(os.path.join(root_dir1, r'Pre_90_NC\1000RPM_PRE_90'))
PRE90NC_1200 = get_file(os.path.join(root_dir1, r'Pre_90_NC\1200RPM_PRE_90'))


# Tasks with 10-way
T0 = [NC_1000, PM_1000,BDZ_1000,BDZ_PM_1000,PRE50NC_1000,PRE90NC_1200]
T1 = [NC_1200, PM_1200,BDZ_1200,BDZ_PM_1200,PRE50NC_1200,PRE90NC_1200]
T2 = [NC_800, PM_800,BDZ_800,BDZ_PM_800,PRE50NC_800,PRE90NC_800]
#T2 = [NC_07_10 , IR_07_10,OR_07_10]
# Tasks with 7-way
# T7w = [NC_0, IF_7[0], IF_14[0], OF_7[0], OF_14[0], RoF_7[0], RoF_14[0]]
# T4w = [NC_1, IF_21[0], OF_21[0], RoF_21[0]]

# Tasks with 6-way
#T6w = [IF_7[0], IF_14[0], IF_21[0], OF_7[0], OF_14[0], OF_21[0]]
#T4w = [NC_0, RoF_7[0], RoF_14[0], RoF_21[0]]  # brand new categories

# T_sq = [NC_3, IF_7[3], OF_7[3]]
# T_sa = [NC_3, OF_7[3], RoF_7[3]]













