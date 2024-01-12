#This code is used to generate reference datasets containg n number og images per class
import os
from tqdm import tqdm
import shutil
import random
imagePerClass={}
src='./datasets/lfw/'
rfr1='./datasets/lfw-rect-refs/lfw1/'
infr1='./datasets/lfw-rect-infs/lfw1-infr/'
rfr2='./datasets/lfw-rect-refs/lfw2/'
infr2='./datasets/lfw-rect-infs/lfw2-infr/'
rfr3='./datasets/lfw-rect-refs/lfw3/'
infr3='./datasets/lfw-rect-infs/lfw3-infr/'
rfr4='./datasets/lfw-rect-refs/lfw4/'
infr4='./datasets/lfw-rect-infs/lfw4-infr/'
rfr5='./datasets/lfw-rect-refs/lfw5/'
infr5='./datasets/lfw-rect-infs/lfw5-infr/'
for root, dir, files in tqdm(os.walk(src)):
        outpath=os.path.join(src,root.split('/')[-1])
        # print(outpath)
        numImgs=0
        for image in files:
            numImgs=numImgs+1
        imagePerClass[root]=numImgs

for key, value in tqdm(imagePerClass.items()):
    if value >1:
        if not os.path.exists(os.path.join(rfr1,key.split('/')[-1])):
            os.makedirs(os.path.join(rfr1,key.split('/')[-1]),exist_ok=True)
        if not os.path.exists(os.path.join(infr1,key.split('/')[-1])):
            os.makedirs(os.path.join(infr1,key.split('/')[-1]),exist_ok=True)
        imgs=os.listdir(key)
        refImgs=random.sample(imgs,1)
        # print(refImgs)
        infrImgs=[img for img in imgs if img not in refImgs]
        for img in refImgs:
            shutil.copy2(os.path.join(src,key.split('/')[-1],img),os.path.join(rfr1,key.split('/')[-1]))
        for img in infrImgs:
            shutil.copy2(os.path.join(src,key.split('/')[-1],img),os.path.join(infr1,key.split('/')[-1]))
    if value >2:
        if not os.path.exists(os.path.join(rfr2,key.split('/')[-1])):
            os.makedirs(os.path.join(rfr2,key.split('/')[-1]),exist_ok=True)
        if not os.path.exists(os.path.join(infr2,key.split('/')[-1])):
            os.makedirs(os.path.join(infr2,key.split('/')[-1]),exist_ok=True)
        imgs=os.listdir(key)
        refImgs=random.sample(imgs,2)
        # print(refImgs)
        infrImgs=[img for img in imgs if img not in refImgs]
        for img in refImgs:
            shutil.copy2(os.path.join(src,key.split('/')[-1],img),os.path.join(rfr2,key.split('/')[-1]))
        for img in infrImgs:
            shutil.copy2(os.path.join(src,key.split('/')[-1],img),os.path.join(infr2,key.split('/')[-1]))
    if value >3:
        if not os.path.exists(os.path.join(rfr3,key.split('/')[-1])):
            os.makedirs(os.path.join(rfr3,key.split('/')[-1]),exist_ok=True)
        if not os.path.exists(os.path.join(infr3,key.split('/')[-1])):
            os.makedirs(os.path.join(infr3,key.split('/')[-1]),exist_ok=True)
        imgs=os.listdir(key)
        refImgs=random.sample(imgs,3)
        # print(refImgs)
        infrImgs=[img for img in imgs if img not in refImgs]
        for img in refImgs:
            shutil.copy2(os.path.join(src,key.split('/')[-1],img),os.path.join(rfr3,key.split('/')[-1]))
        for img in infrImgs:
            shutil.copy2(os.path.join(src,key.split('/')[-1],img),os.path.join(infr3,key.split('/')[-1]))
    if value >4:
        if not os.path.exists(os.path.join(rfr4,key.split('/')[-1])):
            os.makedirs(os.path.join(rfr4,key.split('/')[-1]),exist_ok=True)
        if not os.path.exists(os.path.join(infr4,key.split('/')[-1])):
            os.makedirs(os.path.join(infr4,key.split('/')[-1]),exist_ok=True)
        imgs=os.listdir(key)
        refImgs=random.sample(imgs,4)
        # print(refImgs)
        infrImgs=[img for img in imgs if img not in refImgs]
        for img in refImgs:
            shutil.copy2(os.path.join(src,key.split('/')[-1],img),os.path.join(rfr4,key.split('/')[-1]))
        for img in infrImgs:
            shutil.copy2(os.path.join(src,key.split('/')[-1],img),os.path.join(infr4,key.split('/')[-1]))
    if value >5:
        if not os.path.exists(os.path.join(rfr5,key.split('/')[-1])):
            os.makedirs(os.path.join(rfr5,key.split('/')[-1]),exist_ok=True)
        if not os.path.exists(os.path.join(infr5,key.split('/')[-1])):
            os.makedirs(os.path.join(infr5,key.split('/')[-1]),exist_ok=True)
        imgs=os.listdir(key)
        refImgs=random.sample(imgs,5)
        # print(refImgs)
        infrImgs=[img for img in imgs if img not in refImgs]
        for img in refImgs:
            shutil.copy2(os.path.join(src,key.split('/')[-1],img),os.path.join(rfr5,key.split('/')[-1]))
        for img in infrImgs:
            shutil.copy2(os.path.join(src,key.split('/')[-1],img),os.path.join(infr5,key.split('/')[-1]))
    
