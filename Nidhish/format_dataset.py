import os
import shutil
from tqdm import tqdm
labelsPath = "./datasets/archive/identity_CelebA.txt"
source='./datasets/archive/img_align_celeba/img_align_celeba'
destination='./datasets/celebAV2/'
with open(labelsPath,'r') as file:
    for line in tqdm(file.read().split('\n')[:-1]):
        img=line.split(' ')[0]
        label=line.split(' ')[1]
        # print(img)
        # print(label)

        if not os.path.exists(os.path.join(destination,label)):
            os.mkdir(os.path.join(destination,label))
        shutil.copy(os.path.join(source,img),os.path.join(destination,label,img))
        