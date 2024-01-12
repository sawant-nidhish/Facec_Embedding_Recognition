import os
from tqdm import tqdm
import shutil
import random
imagePerClass={}
src='./datasets/lfw-fish-single/'
rfr1='./datasets/lfw-fish-single-refs/lfw1/'
infr1='./datasets/lfw-fish-single-infs/lfw1-infr/'
rfr2='./datasets/lfw-fish-single-refs/lfw2/'
infr2='./datasets/lfw-fish-single-infs/lfw2-infr/'
rfr3='./datasets/lfw-fish-single-refs/lfw3/'
infr3='./datasets/lfw-fish-single-infs/lfw3-infr/'
rfr4='./datasets/lfw-fish-single-refs/lfw4/'
infr4='./datasets/lfw-fish-single-infs/lfw4-infr/'
rfr5='./datasets/lfw-fish-single-refs/lfw5/'
infr5='./datasets/lfw-fish-single-infs/lfw5-infr/'
for root, dir, files in tqdm(os.walk(src)):
        outpath=os.path.join(src,root.split('/')[-1])
        # print(outpath)
        numImgs=0
        for image in files:
            numImgs=numImgs+1
        imagePerClass[root]=numImgs

ref1=0
ref2=0
ref3=0
ref4=0
ref5=0
for key, value in imagePerClass.items():
    if value >1:
        # if not os.path.exists(os.path.join(rfr1,key.split('/')[-1])):
        #     os.makedirs(os.path.join(rfr1,key.split('/')[-1]),exist_ok=True)
        # if not os.path.exists(os.path.join(infr1,key.split('/')[-1])):
        #     os.makedirs(os.path.join(infr1,key.split('/')[-1]),exist_ok=True)
        imgs=os.listdir(key)
        refImgs=random.sample(imgs,1)
        # print(refImgs)
        infrImgs=[img for img in imgs if img not in refImgs]
        for img in refImgs:
            ref1+=1
            # shutil.copy2(os.path.join(src,key.split('/')[-1],img),os.path.join(rfr1,key.split('/')[-1]))
        #for img in infrImgs:
            
            # shutil.copy2(os.path.join(src,key.split('/')[-1],img),os.path.join(infr1,key.split('/')[-1]))
    if value >2:
        # if not os.path.exists(os.path.join(rfr2,key.split('/')[-1])):
        #     os.makedirs(os.path.join(rfr2,key.split('/')[-1]),exist_ok=True)
        # if not os.path.exists(os.path.join(infr2,key.split('/')[-1])):
        #     os.makedirs(os.path.join(infr2,key.split('/')[-1]),exist_ok=True)
        imgs=os.listdir(key)
        refImgs=random.sample(imgs,2)
        # print(refImgs)
        infrImgs=[img for img in imgs if img not in refImgs]
        for img in refImgs:
            ref2+=1
            # shutil.copy2(os.path.join(src,key.split('/')[-1],img),os.path.join(rfr2,key.split('/')[-1]))
        # for img in infrImgs:
        #     shutil.copy2(os.path.join(src,key.split('/')[-1],img),os.path.join(infr2,key.split('/')[-1]))
    if value >3:
        # if not os.path.exists(os.path.join(rfr3,key.split('/')[-1])):
        #     os.makedirs(os.path.join(rfr3,key.split('/')[-1]),exist_ok=True)
        # if not os.path.exists(os.path.join(infr3,key.split('/')[-1])):
        #     os.makedirs(os.path.join(infr3,key.split('/')[-1]),exist_ok=True)
        imgs=os.listdir(key)
        refImgs=random.sample(imgs,3)
        # print(refImgs)
        infrImgs=[img for img in imgs if img not in refImgs]
        for img in refImgs:
            ref3+=1
            # shutil.copy2(os.path.join(src,key.split('/')[-1],img),os.path.join(rfr3,key.split('/')[-1]))
        # for img in infrImgs:
        #     shutil.copy2(os.path.join(src,key.split('/')[-1],img),os.path.join(infr3,key.split('/')[-1]))
    if value >4:
        # if not os.path.exists(os.path.join(rfr4,key.split('/')[-1])):
        #     os.makedirs(os.path.join(rfr4,key.split('/')[-1]),exist_ok=True)
        # if not os.path.exists(os.path.join(infr4,key.split('/')[-1])):
        #     os.makedirs(os.path.join(infr4,key.split('/')[-1]),exist_ok=True)
        imgs=os.listdir(key)
        refImgs=random.sample(imgs,4)
        # print(refImgs)
        infrImgs=[img for img in imgs if img not in refImgs]
        for img in refImgs:
            ref4+=1
            # shutil.copy2(os.path.join(src,key.split('/')[-1],img),os.path.join(rfr4,key.split('/')[-1]))
        # for img in infrImgs:
        #     shutil.copy2(os.path.join(src,key.split('/')[-1],img),os.path.join(infr4,key.split('/')[-1]))
    if value >5:
        # if not os.path.exists(os.path.join(rfr5,key.split('/')[-1])):
        #     os.makedirs(os.path.join(rfr5,key.split('/')[-1]),exist_ok=True)
        # if not os.path.exists(os.path.join(infr5,key.split('/')[-1])):
        #     os.makedirs(os.path.join(infr5,key.split('/')[-1]),exist_ok=True)
        imgs=os.listdir(key)
        refImgs=random.sample(imgs,5)
        # print(refImgs)
        infrImgs=[img for img in imgs if img not in refImgs]
        for img in refImgs:
            ref5+=1
            # shutil.copy2(os.path.join(src,key.split('/')[-1],img),os.path.join(rfr5,key.split('/')[-1]))
        # for img in infrImgs:
        #     shutil.copy2(os.path.join(src,key.split('/')[-1],img),os.path.join(infr5,key.split('/')[-1]))

print("Num images perclass 1",ref1)
print("Num images perclass 2",ref2)
print("Num images perclass 3",ref3)
print("Num images perclass 4",ref4)
print("Num images perclass 5",ref5)

