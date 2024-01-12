##This code is used to resize the input data to 112_112

import cv2
from tqdm import tqdm
import os
outpath='./datasets/lfw-fish'+'_aligned_112_112'
train_data_path=outpath
if not os.path.exists(outpath):
    print("Generateing resized images")
    os.mkdir(outpath)
    print('outpath', outpath)
    for root, dir, files in tqdm(os.walk('./datasets/lfw-fish')):
        print('root ',root)
        outpath_im=os.path.join(outpath,root.split('/')[-1])
        print('outpath_im ',outpath_im)
        #print(outpath)
        for image in files:
            if not os.path.exists(outpath_im):
                os.mkdir(outpath_im)
            print(os.path.join(root,image))
            target_im = os.path.join(outpath_im,image.split('.')[0]+".png")
            if os.path.exists(target_im):
                print('image exists')
                continue
            im=cv2.imread(os.path.join(root,image))
            if im is None:
                print('im is None')
                continue
            print('Resizing image')
            target_im = os.path.join(outpath_im,image.split('.')[0]+".png")
            cv2.imwrite(target_im, cv2.resize(im, (112, 112), interpolation=cv2.INTER_CUBIC))
