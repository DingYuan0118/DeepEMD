import os
import numpy as np
from os import listdir
from os.path import isfile, isdir, join
from pathlib import Path



cwd = os.getcwd() 
data_path = 'filelists/recognition36_crop/image_crops'
savedir = 'filelists/recognition36_crop/'
dataset_list = ["novel"]

file_list = listdir(data_path)

classfile_list_all = []
class_names = []
for file_name in file_list:
    class_name = "_".join(file_name.split("_")[1:4])
    class_names.append(class_name)
class_names_unique = np.unique(class_names).tolist()


for class_name in class_names_unique:
    image_name = []
    for file_name in file_list:
        if "_".join(file_name.split("_")[1:4]) == class_name:
            image_name.append(Path(join(data_path, file_name)).as_posix())
    classfile_list_all.append(image_name)
print(len(classfile_list_all))


# for mode in ["car", "plane", "all"]:
#     if mode == "all":
#         for dataset in dataset_list:
#             dataset += "_all"
#             file_list = []
#             label_list = []
#             for i, classfile_list in enumerate(classfile_list_all):
#                 file_list = file_list + classfile_list
#                 label_list = label_list + np.repeat(i, len(classfile_list)).tolist()

#             fo = open(savedir + dataset + ".json", "w")
#             fo.write('{"label_names": [')
#             fo.writelines(['"%s",' % item  for item in class_names_unique])
#             fo.seek(0, os.SEEK_END) 
#             fo.seek(fo.tell()-1, os.SEEK_SET)
#             fo.write('],')

#             fo.write('"image_names": [')
#             fo.writelines(['"%s",' % item  for item in file_list])
#             fo.seek(0, os.SEEK_END) 
#             fo.seek(fo.tell()-1, os.SEEK_SET)
#             fo.write('],')

#             fo.write('"image_labels": [')
#             fo.writelines(['%d,' % item  for item in label_list])
#             fo.seek(0, os.SEEK_END) 
#             fo.seek(fo.tell()-1, os.SEEK_SET)
#             fo.write(']}')

#             fo.close()
#             print("%s -OK" %dataset)

#     if mode == "car":
#         for dataset in dataset_list:
#             dataset += "_car"
#             file_list = []
#             label_list = []
#             for i, classfile_list in enumerate(classfile_list_all[:19]):
#                 file_list = file_list + classfile_list
#                 label_list = label_list + np.repeat(i, len(classfile_list)).tolist()

#             fo = open(savedir + dataset + ".json", "w")
#             fo.write('{"label_names": [')
#             fo.writelines(['"%s",' % item  for item in class_names_unique])
#             fo.seek(0, os.SEEK_END) 
#             fo.seek(fo.tell()-1, os.SEEK_SET)
#             fo.write('],')

#             fo.write('"image_names": [')
#             fo.writelines(['"%s",' % item  for item in file_list])
#             fo.seek(0, os.SEEK_END) 
#             fo.seek(fo.tell()-1, os.SEEK_SET)
#             fo.write('],')

#             fo.write('"image_labels": [')
#             fo.writelines(['%d,' % item  for item in label_list])
#             fo.seek(0, os.SEEK_END) 
#             fo.seek(fo.tell()-1, os.SEEK_SET)
#             fo.write(']}')

#             fo.close()
#             print("%s -OK" %dataset)

#     if mode == "plane":
#         for dataset in dataset_list:
#             dataset += "_plane"
#             file_list = []
#             label_list = []
#             for i, classfile_list in enumerate(classfile_list_all[19:]):
#                 file_list = file_list + classfile_list
#                 label_list = label_list + (np.repeat(i, len(classfile_list)) + 19).tolist()

#             fo = open(savedir + dataset + ".json", "w")
#             fo.write('{"label_names": [')
#             fo.writelines(['"%s",' % item  for item in class_names_unique])
#             fo.seek(0, os.SEEK_END) 
#             fo.seek(fo.tell()-1, os.SEEK_SET)
#             fo.write('],')

#             fo.write('"image_names": [')
#             fo.writelines(['"%s",' % item  for item in file_list])
#             fo.seek(0, os.SEEK_END) 
#             fo.seek(fo.tell()-1, os.SEEK_SET)
#             fo.write('],')

#             fo.write('"image_labels": [')
#             fo.writelines(['%d,' % item  for item in label_list])
#             fo.seek(0, os.SEEK_END) 
#             fo.seek(fo.tell()-1, os.SEEK_SET)
#             fo.write(']}')

#             fo.close()
#             print("%s -OK" %dataset)
