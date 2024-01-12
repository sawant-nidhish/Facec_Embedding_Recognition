####This code is to add a rectilinear sample to the images that have random distortions
import os
import tqdm
import imageio
import sys
import shutil
i=0
for root, dir, files in tqdm(os.walk('./datasets/celebA_detect_aligned_112_112')):
    outpath=os.path.join('./datasets/celebA-fish-plus-rect_aligned_112_112',root.split('/')[-1])
    # print(outpath)
    if os.path.exists(outpath):
        for image in files:
            i=i+1
            shutil.copy2(os.path.join(root,image),outpath)
    else:
        print("This class doesnot exits in the destination",outpath)
print("Total",i,"images added to the destination")
