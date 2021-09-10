import os
import random
import shutil
def split_data(category):

    source = 'C:\\Users\sumit\\Desktop\\Explicit Content Classifier\\Input\\train\\' + category
    filenames = os.listdir('C:\\Users\sumit\\Desktop\\Explicit Content Classifier\\Input\\train\\' + category)
    # print (type(filenames))
    print (len(filenames))
    print (filenames[0:10])

    filenames.sort()

    random.seed(256)
    random.shuffle(filenames)
    print ()
    split = 2000
    test_filenames = filenames[:split]
    print (len(test_filenames))

    destination = 'C:\\Users\\sumit\\Desktop\\Explicit Content Classifier\\Input\\test\\' + category

    for name in test_filenames:
        ini_path = os.path.join(source, name)
        fin_path = os.path.join(destination, name)

        shutil.move(ini_path, fin_path)

# split_data('Hentai')
# split_data('Neutral')
# split_data('Porn')
# split_data('Drawing')
# split_data('Sexy')
