import os
import random
import shutil
def split_data(category, file_path):

    source = file_path + 'train\' + category
    filenames = os.listdir(file_path + 'train\' + category)
    
    print (len(filenames))
    print ('First Ten Files: ', filenames[0:10])

    filenames.sort()

    random.seed(256)
    random.shuffle(filenames)
    print ()
    split = 2000
    test_filenames = filenames[:split]
    print (len(test_filenames))

    destination = file_path + 'test\' + category

    for name in test_filenames:
        ini_path = os.path.join(source, name)
        fin_path = os.path.join(destination, name)

        shutil.move(ini_path, fin_path)
