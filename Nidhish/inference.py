#!/usr/bin/env python3

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from glob2 import glob
from skimage import transform
from skimage.io import imread, imsave
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import sys
np.set_printoptions(threshold=sys.maxsize)
sys.path.insert(0,'./')
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class Eval_folder:
    def __init__(self, model_interf, data_path, num_refrences, batch_size=128, save_embeddings=None):
        #if isinstance(model_interf, str) and model_interf.endswith("h5"):
        if isinstance(model_interf, str):
            model = tf.keras.models.load_model(model_interf, compile=False)
            self.model_interf = lambda imms: model((imms - 127.5) * 0.0078125).numpy()
        else:
            self.model_interf = model_interf
        self.dist_func = lambda aa, bb: np.dot(aa, bb)
        print('>>>>>',save_embeddings)
        self.embs, self.imm_classes, self.filenames, self.class_idx = self.prepare_images_and_embeddings(data_path, num_refrences, batch_size, save_embeddings)
        self.data_path = data_path
        self.newEmb, self.newImm_classes, self.newFilenames, self.infer_class_idx = self.getEmb(data_path, num_refrences, batch_size)
        self.num_refrences=num_refrences

    def prepare_images_and_embeddings(self, data_path, num_refrences, batch_size=128, save_embeddings=None):
        print(save_embeddings)
        if save_embeddings and os.path.exists(save_embeddings):
            print(">>>> Reloading from backup:", save_embeddings)
            aa = np.load(save_embeddings)
            print("Size of embeddings is",aa["embs"].shape)
            embs, imm_classes, filenames = aa["embs"], aa["imm_classes"], aa["filenames"]
            embs, imm_classes = embs.astype("float32"), imm_classes.astype("int")
        else:
            raise Exception("Embeddings not found")
        # print(imm_classes)
        # print(filenames)
        with open('./embeddings/class_idx'+str(num_refrences)+'.json','r') as file:
            class_idx=json.load(file)
        return embs, imm_classes, filenames, class_idx

    def do_evaluation(self):
        register_ids = np.unique(self.imm_classes)
        newRegister_ids = np.unique(self.newImm_classes)
        print(">>>> [base info] embs:", self.embs.shape, "imm_classes:", self.imm_classes.shape, "register_ids:", register_ids.shape)
        print(">>>> [base info] New embs:", self.newEmb.shape, "New imm_classes:", self.newImm_classes.shape, "New register_ids:", newRegister_ids.shape)

        print(">>>self.filename.shape",self.filenames.shape)
        print(">>>self.filename",self.filenames[0])
        # print(">>>>Cosine similarity",cosineSimMatrix(self.embs))
        # matrix=pd.DataFrame(data=cosineSimMatrix(self.embs),index=self.filenames,columns=self.filenames)
        # matrix.to_csv('./cosineSimMatrix.csv')
        register_base_embs = np.array([]).reshape(0, self.embs.shape[-1])
        register_base_dists = []
        for register_id in tqdm(register_ids, "Evaluating"):
            pos_pick_cond = self.imm_classes == register_id
            pos_embs = self.embs[pos_pick_cond]
            register_base_emb = normalize([np.sum(pos_embs, 0)])[0]
            # print("NIDHISH: register_id, embs shape, register_base_embs",register_id,self.embs.shape, register_base_emb.shape)
            register_base_dist = self.dist_func(self.newEmb, register_base_emb)
            # print("NIDHISH: register_base_dist",register_base_dist.shape)
            #print()
            register_base_dists.append(register_base_dist)
            register_base_embs = np.vstack([register_base_embs, register_base_emb])
        
        print("Embeddings shape is",register_base_embs.shape)
        register_base_dists = np.array(register_base_dists).T
        #print("NIDHISH: register_base_dist",register_base_dists.shape)
        #print(register_base_dists.argmax(1))
        accuracy=0
        un=0
        goodSim=0
        badSim=0
        for i in tqdm(range(len(register_base_dists.argmax(1))),'Getting Accuracy'):
            #print(i)
            #print("Predicted Class:",self.class_idx[str(register_base_dists.argmax(1)[i])],"True Class:",self.infer_class_idx[self.newImm_classes[i]],"File Name:",self.newFilenames[i],"Similarity",register_base_dists.max(axis=1)[i])
            #print(self.newFilenames[i].split("\\")[1])
            if self.newFilenames[i].split("/")[0]=="Unkown":
                un=un+1
            if self.class_idx[str(register_base_dists.argmax(1)[i])]==self.newFilenames[i].split("/")[0]:
                #print('Class Name',self.newFilenames[i].split("/")[0])
                goodSim=goodSim+register_base_dists.max(axis=1)[i]
                accuracy=accuracy+1
                #print(self.class_idx[str(register_base_dists.argmax(1)[i])])
            else:
                if self.newFilenames[i].split("/")[0]!="Unkown":
                    badSim=badSim+register_base_dists.max(axis=1)[i]
            
                #print(">>>>>>>>>Bad classification")
        print('Number of Referencec images',self.num_refrences)
        print("Total images",register_base_dists.shape[0])
        print("Total known images",register_base_dists.shape[0]-un)
        print("Good Calssification cluster mean",goodSim/accuracy)
        print("Bad Calssification cluster mean",badSim/(register_base_dists.shape[0]-un-accuracy))
        print("Good mean - Bad mean",goodSim/accuracy-badSim/(register_base_dists.shape[0]-un-accuracy))
        print("Good mean - Bad mean % difference",(goodSim/accuracy-badSim/(register_base_dists.shape[0]-un-accuracy))/(badSim/(register_base_dists.shape[0]-un-accuracy)))
        print("Accuracy",accuracy/(register_base_dists.shape[0]-un))
        # accuracy = (register_base_dists.argmax(1) == self.newImm_classes).sum() / register_base_dists.shape[0]
        # print("True Classes",self.imm_classes)
        # print("Pred Classes",register_base_dists.argmax(1))
        # print(np.expand_dims(self.imm_classes, 1).shape)
        # print(register_ids.shape)
        # reg_pos_cond = np.equal(register_ids, np.expand_dims(self.imm_classes, 1))
        # print(reg_pos_cond)
        # reg_pos_dists = register_base_dists[reg_pos_cond].ravel()
        # print(reg_pos_dists.shape)
        # reg_neg_dists = register_base_dists[np.logical_not(reg_pos_cond)].ravel()
        # print(reg_neg_dists.shape)
        # print(np.ones_like(reg_pos_dists))
        # label = np.concatenate([np.ones_like(reg_pos_dists), np.zeros_like(reg_neg_dists)])
        # score = np.concatenate([reg_pos_dists, reg_neg_dists])

        # self.register_base_embs, self.register_ids = register_base_embs, register_ids
        return accuracy, None, None

    def generate_eval_pair_bin(self, save_dest, pos_num=3000, neg_num=3000, min_pos=0, max_neg=1.0, nfold=10):
        import pickle

        p1_images, p2_images, pos_scores = [], [], []
        n1_images, n2_images, neg_scores = [], [], []

        for idx, register_id in tqdm(enumerate(self.register_ids), "Evaluating", total=self.register_ids.shape[0]):
            register_emb = self.register_base_embs[idx]

            """ Pick pos images """
            pos_pick_cond = self.imm_classes == register_id
            pos_embs = self.embs[pos_pick_cond]
            pos_dists = self.dist_func(pos_embs, pos_embs.T)

            curr_pos_num = pos_embs.shape[0]
            # xx, yy = np.meshgrid(np.arange(1, curr_pos_num), np.arange(curr_pos_num - 1))
            # triangle_pick = np.triu(np.ones_like(xx)).astype('bool')
            # p1_ids, p2_ids = yy[triangle_pick], xx[triangle_pick]
            p1_ids = []
            for ii in range(curr_pos_num - 1):
                p1_ids.extend([ii] * (curr_pos_num - 1 - ii))
            p2_ids = []
            for ii in range(1, curr_pos_num):
                p2_ids.extend(range(ii, curr_pos_num))
            # curr_pos_num = 5 --> p1_ids: [0, 0, 0, 0, 1, 1, 1, 2, 2, 3], p2_ids: [1, 2, 3, 4, 2, 3, 4, 3, 4, 4]
            pos_images = self.filenames[pos_pick_cond]
            p1_images.extend(pos_images[p1_ids])
            p2_images.extend(pos_images[p2_ids])
            pos_scores.extend([pos_dists[ii, jj] for ii, jj in zip(p1_ids, p2_ids)])

            """ Pick neg images for current register_id """
            if idx == 0:
                continue

            neg_argmax = self.dist_func(self.register_base_embs[:idx], register_emb).argmax()
            # print(idx, register_id, neg_argmax)
            neg_id = self.register_ids[neg_argmax]
            neg_pick_cond = self.imm_classes == neg_id
            neg_embs = self.embs[neg_pick_cond]
            neg_dists = self.dist_func(pos_embs, neg_embs.T)

            curr_neg_num = neg_embs.shape[0]
            xx, yy = np.meshgrid(np.arange(curr_pos_num), np.arange(curr_neg_num))
            n1_ids, n2_ids = xx.ravel().tolist(), yy.ravel().tolist()
            neg_images = self.filenames[neg_pick_cond]
            n1_images.extend(pos_images[n1_ids])
            n2_images.extend(neg_images[n2_ids])
            neg_scores.extend([neg_dists[ii, jj] for ii, jj in zip(n1_ids, n2_ids)])

        print(">>>> len(pos_scores):", len(pos_scores), "len(neg_scores):", len(neg_scores))
        pos_scores, neg_scores = np.array(pos_scores), np.array(neg_scores)
        pos_score_cond, neg_score_cond = pos_scores > min_pos, neg_scores < max_neg
        pos_scores, neg_scores = pos_scores[pos_score_cond], neg_scores[neg_score_cond]
        p1_images, p2_images = np.array(p1_images)[pos_score_cond], np.array(p2_images)[pos_score_cond]
        n1_images, n2_images = np.array(n1_images)[neg_score_cond], np.array(n2_images)[neg_score_cond]

        """ pick by sorted score values """
        pos_pick_cond = np.argsort(pos_scores)[:pos_num]
        neg_pick_cond = np.argsort(neg_scores)[-neg_num:]
        pos_scores, p1_images, p2_images = pos_scores[pos_pick_cond], p1_images[pos_pick_cond], p2_images[pos_pick_cond]
        neg_scores, n1_images, n2_images = neg_scores[neg_pick_cond], n1_images[neg_pick_cond], n2_images[neg_pick_cond]

        bins = []
        total = pos_num + neg_num
        for img_1, img_2 in tqdm(list(zip(p1_images, p2_images)) + list(zip(n1_images, n2_images)), "Creating bins", total=total):
            bins.append(tf.image.encode_png(imread(os.path.join(self.data_path, img_1))).numpy())
            bins.append(tf.image.encode_png(imread(os.path.join(self.data_path, img_2))).numpy())

        """ nfold """
        pos_fold, neg_fold = pos_num // nfold, neg_num // nfold
        issame_list = ([True] * pos_fold + [False] * neg_fold) * nfold
        pos_bin_fold = lambda ii: bins[ii * pos_fold * 2 : (ii + 1) * pos_fold * 2]
        neg_bin_fold = lambda ii: bins[pos_num * 2 :][ii * neg_fold * 2 : (ii + 1) * neg_fold * 2]
        bins = [pos_bin_fold(ii) + neg_bin_fold(ii) for ii in range(nfold)]
        bins = np.ravel(bins).tolist()

        print("Saving to %s" % save_dest)
        with open(save_dest, "wb") as ff:
            pickle.dump([bins, issame_list], ff)
        return p1_images, p2_images, pos_scores, n1_images, n2_images, neg_scores
    
    def getEmb(self,newIm,num_refrences,batch_size=128):
        img_shape = (112, 112)
        img_gen = ImageDataGenerator().flow_from_directory(newIm, class_mode="binary", target_size=img_shape, batch_size=batch_size, shuffle=False)
        steps = int(np.ceil(img_gen.classes.shape[0] / img_gen.batch_size))
        filenames = np.array(img_gen.filenames)

        class_indices=img_gen.class_indices
        embs, imm_classes = [], []
        for _ in tqdm(range(steps), "Embedding"):
            imm, imm_class = img_gen.next()
            emb = self.model_interf(imm)
            embs.extend(emb)
            imm_classes.extend(imm_class)
            
        embs, imm_classes = normalize(np.array(embs).astype("float32")), np.array(imm_classes).astype("int")
        #print("The im_classes are",imm_classes)
        class_idx={}
        for key,value in class_indices.items():
            class_idx[value]=key
        
        
        return embs, imm_classes, filenames, class_idx

def plot_tpr_far(score, label, new_figure=True, label_prefix=""):
    fpr, tpr, _ = roc_curve(label, score)
    roc_auc = auc(fpr, tpr)

    fpr_show = [10 ** (-ii) for ii in range(1, 7)[::-1]]
    fpr_reverse, tpr_reverse = fpr[::-1], tpr[::-1]
    tpr_show = [tpr_reverse[np.argmin(abs(fpr_reverse - ii))] for ii in fpr_show]
    print(pd.DataFrame({"FPR": fpr_show, "TPR": tpr_show}).set_index("FPR").T.to_markdown())

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure() if new_figure else None
        label = "AUC = %0.4f%%" % (roc_auc * 100)
        if label_prefix and len(label_prefix) > 0:
            label = label_prefix + " " + label
        plt.plot(fpr, tpr, lw=1, label=label)
        plt.xlim([10 ** -6, 0.1])
        plt.xscale("log")
        plt.xticks(fpr_show)
        plt.xlabel("False Positive Rate")
        plt.ylim([0, 1.0])
        plt.yticks(np.linspace(0, 1.0, 8, endpoint=True))
        plt.ylabel("True Positive Rate")

        plt.grid(linestyle="--", linewidth=1)
        plt.title("ROC")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()
    except:
        print("matplotlib plot failed")
        fig = None
    return fig

##Cosine similarity matrix
def cosineSimMatrix(embs):
    matrix=[]
    
    print("embs Shape",embs.shape)
    for i in range(embs.shape[0]):
        simRows=[]
        for j in range(embs.shape[0]):
            simRows.append(np.dot(embs[i],embs[j]))
        matrix.append(simRows)
    return np.array(matrix)


if __name__ == "__main__":
    import sys
    import argparse
    import tensorflow_addons as tfa
    
    def list_of_strings(arg):
        return arg.split(',')

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--data_path", type=list_of_strings, default=None, help="Data path, containing images in class folders")
    parser.add_argument("-m", "--model_file", type=str, default=None, help="Model file, keras h5")
    parser.add_argument("-n", "--num_refrences", type=list_of_ints, default=None, help="Number of refrence images")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("-D", "--detection", action="store_true", help="Run face detection before embedding")
    parser.add_argument("-S", "--save_embeddings", type=list_of_strings, default=None, help="Save / Reload embeddings data")
    parser.add_argument("-B", "--save_bins", type=str, default=None, help="Save evaluating pair bin")
    
    args = parser.parse_known_args(sys.argv[1:])[0]

    print('>>>>>>>>>>>>',args.save_embeddings)
    if args.model_file == None and args.data_path == None and args.save_embeddings == None:
        print(">>>> Please seee `--help` for usage")
        sys.exit(1)

    data_path = args.data_path
    for i, numRefs in enumerate(args.num_refrences):
        data_path=args.data_path[i]
        if args.detection:
            from face_detector import YoloV5FaceDetector

            data_path = YoloV5FaceDetector().detect_in_folder(args.data_path[i])
        #print()
    #print(args.num_refrences)
    #print(args.save_embeddings)
    #print(args.data_path)
    #for i, numRefs in enumerate(args.num_refrences):
        ee = Eval_folder(args.model_file, data_path, numRefs, args.batch_size, args.save_embeddings[i])
        accuracy, score, label = ee.do_evaluation()
        
#     from sklearn.manifold import TSNE
#     print(ee.embs.shape)
#     tsne=TSNE(n_components=2, perplexity=min(ee.embs.shape[0]-1,20)).fit_transform(ee.embs)
#     tx = tsne[:, 0]
#     ty = tsne[:, 1]
# # scale and move the coordinates so they fit [0; 1] range
#     def scale_to_01_range(x):
#         # compute the distribution range
#         value_range = (np.max(x) - np.min(x))
    
#         # move the distribution so that it starts from zero
#         # by extracting the minimal value from all its values
#         starts_from_zero = x - np.min(x)
    
#         # make the distribution fit [0; 1] by dividing by its range
#         return starts_from_zero / value_range
#     tx = scale_to_01_range(tx)
#     ty = scale_to_01_range(ty)
#     print(tsne.shape)
#     classes=np.unique(ee.imm_classes)
#     color_map={0:'red',1:'cyan',2:'yellow',3:'orange',4:'blue',5:'green'}
#     import matplotlib.pyplot as plt
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     print(ee.imm_classes)
#     # for every class, we'll add a scatter plot separately
#     for label in color_map:
#         # find the samples of the current class in the data
#         indices = [i for i, l in enumerate(ee.imm_classes) if l == label]
#         print(indices)
#         # extract the coordinates of the points of this class only
#         current_tx = np.take(tx, indices)
#         current_ty = np.take(ty, indices)
    
#         # convert the class color to matplotlib format
#         # color = np.array(color_map[label], dtype=np.float) / 255
    
#         # add a scatter plot with the corresponding color and label
#         ax.scatter(current_tx, current_ty, c=color_map[label], label=label)
    
#         # build a legend using the labels we set previously
#         ax.legend(loc='best')
        
#     # finally, show the plot
#     # print(ee.newEmb.shape)
#     # newtsne=TSNE(n_components=2, perplexity=1).fit_transform(ee.newEmb)
#     # newtx = scale_to_01_range(newtsne[:,0])
#     # newty = scale_to_01_range(newtsne[:,1])
#     # ax.scatter(newtx, newty, c='black', label='testSample')
#     plt.show()
#     # print(ee.classDict)
    
    # print(">>>> top1 accuracy:", accuracy)

    # if args.save_bins is not None:
    #     _ = ee.generate_eval_pair_bin(args.save_bins)

    # plot_tpr_far(score, label)
elif __name__ == "__test__":
    data_path = "temp_test/faces_emore_test/"
    model_file = "checkpoints/GhostFaceNet_W1.3_S1_ArcFace.h5"
    batch_size = 64
    save_embeddings = None
