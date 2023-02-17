"""
齿轮
"""

import os
root_dir1 = r"D:\论文+无监督下基于小样本数据的旋转机械故障诊断研究\故障诊断例子\MetaFD-SEU\SEU_dataset"


def get_file(root_path):
    file_list = os.listdir(path=root_path)
    file_list = [os.path.join(root_path, f) for f in file_list]
    # if len(file_list) != 1:  # 若文件中存在不止一个文件，则存在歧义
    #     print('There are {} files in [{}]'.format(len(file_list), root_path))
    #     exit()
    assert len(file_list) == 1, 'There are {} files in [{}]'.format(len(file_list), root_path)  #
    return file_list[0]


# NC
NC_0 = get_file(os.path.join(root_dir1, r'Health\20'))
NC_1 = get_file(os.path.join(root_dir1, r'Health\30'))


# Chipped

Chipped_20 = get_file(os.path.join(root_dir1, r'Chipped\20'))
Chipped_30 = get_file(os.path.join(root_dir1, r'Chipped\30'))

# Miss
Miss_20 = get_file(os.path.join(root_dir1, r'Miss\20'))
Miss_30 = get_file(os.path.join(root_dir1, r'Miss\30'))

# Surface
Surface_20 = get_file(os.path.join(root_dir1, r'Surface\20'))
Surface_30 = get_file(os.path.join(root_dir1, r'Surface\30'))


# Root
Root_20 = get_file(os.path.join(root_dir1, r'Root\20'))
Root_30 = get_file(os.path.join(root_dir1, r'Root\30'))

# Tasks with 10-way
T2 = [NC_0, Chipped_20, Miss_20, Surface_20, Root_20]
T3 = [NC_1, Chipped_30, Miss_30, Surface_30, Root_30]

# Tasks with 7-way
# T7w = [NC_0, IF_7[0], IF_14[0], OF_7[0], OF_14[0], RoF_7[0], RoF_14[0]]
# T4w = [NC_1, IF_21[0], OF_21[0], RoF_21[0]]

# Tasks with 6-way
#T6w = [IF_7[0], IF_14[0], IF_21[0], OF_7[0], OF_14[0], OF_21[0]]
#T4w = [NC_0, RoF_7[0], RoF_14[0], RoF_21[0]]  # brand new categories

# T_sq = [NC_3, IF_7[3], OF_7[3]]
# T_sa = [NC_3, OF_7[3], RoF_7[3]]





'''
轴承
'''



import os
root_dir = r"D:\论文+无监督下基于小样本数据的旋转机械故障诊断研究\故障诊断例子\MetaFD-SEU\SEU_bear"


def get_file(root_path):
    file_list = os.listdir(path=root_path)
    file_list = [os.path.join(root_path, f) for f in file_list]
    # if len(file_list) != 1:  # 若文件中存在不止一个文件，则存在歧义
    #     print('There are {} files in [{}]'.format(len(file_list), root_path))
    #     exit()
    assert len(file_list) == 1, 'There are {} files in [{}]'.format(len(file_list), root_path)  #
    return file_list[0]


# NC
Health_0 = get_file(os.path.join(root_dir, r'Health\20'))
Health_1 = get_file(os.path.join(root_dir, r'Health\30'))


# ball

ball_20 = get_file(os.path.join(root_dir, r'ball\20'))
ball_30 = get_file(os.path.join(root_dir, r'ball\30'))

# com
com_20 = get_file(os.path.join(root_dir, r'com\20'))
com_30 = get_file(os.path.join(root_dir, r'com\30'))

# IR
IR_20 = get_file(os.path.join(root_dir, r'IR\20'))
IR_30 = get_file(os.path.join(root_dir, r'IR\30'))


# OR
OR_20 = get_file(os.path.join(root_dir, r'OR\20'))
OR_30 = get_file(os.path.join(root_dir, r'OR\30'))

# Tasks with 10-way
T0 = [Health_0, com_20,ball_20,IR_20,OR_20]
T1 = [Health_1, com_30,ball_30,IR_30,OR_30]




#T0 = [NC_0, IF_7[0], IF_14[0], IF_21[0], OF_7[0], OF_14[0], OF_21[0], RoF_7[0], RoF_14[0], RoF_21[0]]
#T1 = [NC_1, IF_7[1], IF_14[1], IF_21[1], OF_7[1], OF_14[1], OF_21[1], RoF_7[1], RoF_14[1], RoF_21[1]]
#T2 = [NC_2, RoF_21[2],IF_21[2], OF_21[2]]
#T3 = [NC_3, RoF_21[3],IF_21[3], OF_21[3]]



T2 = [NC_0, Chipped_20, Miss_20, Surface_20, Root_20]
T3 = [NC_1, Chipped_30, Miss_30, Surface_30, Root_30]


# Tasks with 7-way
# T7w = [NC_0, IF_7[0], IF_14[0], OF_7[0], OF_14[0], RoF_7[0], RoF_14[0]]
# T4w = [NC_1, IF_21[0], OF_21[0], RoF_21[0]]

# Tasks with 6-way
#T6w = [IF_7[0], IF_14[0], IF_21[0], OF_7[0], OF_14[0], OF_21[0]]
#T4w = [NC_0, RoF_7[0], RoF_14[0], RoF_21[0]]  # brand new categories

# T_sq = [NC_3, IF_7[3], OF_7[3]]
# T_sa = [NC_3, OF_7[3], RoF_7[3]]


if __name__ == "__main__":
    print(T0)
    print(T1)
    pass



























