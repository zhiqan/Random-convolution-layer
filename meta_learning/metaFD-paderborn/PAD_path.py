"""
德国帕得恩轴承
"""

import os
root_dir1 = r"D:\\元学习与强化学习 - Meta-Learning Representations for Continual Learning\\mrcl_guzhangzhenuan\\metaFD-paderborn\\PAD_dataset"


def get_file(root_path):
    file_list = os.listdir(path=root_path)
    file_list = [os.path.join(root_path, f) for f in file_list]
    # if len(file_list) != 1:  # 若文件中存在不止一个文件，则存在歧义
    #     print('There are {} files in [{}]'.format(len(file_list), root_path))
    #     exit()
    assert len(file_list) == 1, 'There are {} files in [{}]'.format(len(file_list), root_path)  #
    return file_list[0]


# NC
NC_01_10 = get_file(os.path.join(root_dir1, r'Health\N15_M01_F10'))
NC_07_04 = get_file(os.path.join(root_dir1, r'Health\N15_M07_F04'))
NC_07_10 = get_file(os.path.join(root_dir1, r'Health\N15_M07_F10'))


# IR
IR_01_10 = get_file(os.path.join(root_dir1, r'IR\N15_M01_F10'))
IR_07_04 = get_file(os.path.join(root_dir1, r'IR\N15_M07_F04'))
IR_07_10 = get_file(os.path.join(root_dir1, r'IR\N15_M07_F10'))
# OR
OR_01_10 = get_file(os.path.join(root_dir1, r'OR\N15_M01_F10'))
OR_07_04 = get_file(os.path.join(root_dir1, r'OR\N15_M07_F04'))
OR_07_10 = get_file(os.path.join(root_dir1, r'OR\N15_M07_F10'))


# Tasks with 10-way
T0 = [NC_01_10, IR_01_10,OR_01_10]
T1 = [NC_07_04 , IR_07_04,OR_07_04]
T2 = [NC_07_10 , IR_07_10,OR_07_10]
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
