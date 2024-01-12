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
sys.path.insert(0,'./')
# gpus = tf.config.experimental.list_physical_devices("GPU")
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


class Prepare_Embeddings:
    def __init__(self, model_interf, data_path, num_references, batch_size=128, save_embeddings=None):
        #if isinstance(model_interf, str) and model_interf.endswith("h5"):
        if isinstance(model_interf, str):
            model = tf.keras.models.load_model(model_interf, compile=False)
            self.model_interf = lambda imms: model((imms - 127.5) * 0.0078125).numpy()
        else:
            self.model_interf = model_interf
        self.embs, self.imm_classes, self.filenames = self.prepare_images_and_embeddings(data_path, num_references, batch_size, save_embeddings)
        self.data_path = data_path

    def prepare_images_and_embeddings(self, data_path, num_references, batch_size=128, save_embeddings=None):
        #if save_embeddings and os.path.exists(save_embeddings):
        #    print(">>>> Reloading from backup:", save_embeddings)
        #    aa = np.load(save_embeddings)
        #    embs, imm_classes, filenames = aa["embs"], aa["imm_classes"], aa["filenames"]
        #    embs, imm_classes = embs.astype("float32"), imm_classes.astype("int")
        #else:
        #   print(">>>> Stored embeddings not found")
        embs, imm_classes = np.array([]), np.array([])
        #This code maints a count of classes for correct labelling
        if len(imm_classes)==0:
            startClass=0
        else:
            startClass=np.unique(imm_classes)[-1] + 1

        print("Start class is",startClass)
        
        print(">>>> Creating new embeddings")
        img_shape = (112, 112)

        img_gen = ImageDataGenerator().flow_from_directory(data_path, class_mode="binary", target_size=img_shape, batch_size=batch_size, shuffle=False)
        print('Batch size',img_gen.batch_size,'type',type(img_gen.batch_size))
        steps = int(np.ceil(img_gen.classes.shape[0] / img_gen.batch_size))
        filenames = np.array(img_gen.filenames)
        print("The image indices are",img_gen.class_indices)
        class_indices = img_gen.class_indices
        for _ in tqdm(range(steps), "Embedding"):
            imm, imm_class = img_gen.next()
            imm_class=imm_class + startClass
            print("Type is",imm_class)
            emb = self.model_interf(imm)
            if len(embs)==0:
                embs=emb
            else:
                embs=np.concatenate((embs,emb), axis=0)
            
            if len(imm_classes)==0:
                imm_classes=imm_class
            else:
                imm_classes=np.concatenate((imm_classes,imm_class), axis=0)
        embs, imm_classes = normalize(np.array(embs).astype("float32")), np.array(imm_classes).astype("int")
        print("The im_classes are",imm_classes)
        if save_embeddings:
            print(">>>> Saving embeddings to:", save_embeddings)
            print(embs.shape)
            np.savez(save_embeddings.split('.npz')[0]+'-'+str(num_references)+'.npz', embs=embs, imm_classes=imm_classes, filenames=filenames)
            class_idx={}
            for key,value in class_indices.items():
                class_idx[value]=key
            print(type(class_indices))
            with open('./embeddings/class_idx'+str(num_references)+'.json', 'w') as file:
                json.dump(class_idx,file)
        
        return embs, imm_classes, filenames


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
    parser.add_argument("-n", "--num_refrences", type=list_of_ints, default=[1], help="Number of reference images")
    parser.add_argument("-m", "--model_file", type=str, default=None, help="Model file, keras h5")

    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("-D", "--detection", action="store_true", help="Run face detection before embedding")
    parser.add_argument("-S", "--save_embeddings", type=str, default=None, help="Save / Reload embeddings data")
    parser.add_argument("-B", "--save_bins", type=str, default=None, help="Save evaluating pair bin")
    args = parser.parse_known_args(sys.argv[1:])[0]

    if args.model_file == None and args.data_path == None and args.save_embeddings == None:
        print(">>>> Please seee `--help` for usage")
        sys.exit(1)

    data_path = args.data_path
    for i,numRefs in enumerate(args.num_refrences):
        data_path=args.data_path[i]
        if args.detection:
            from face_detector import YoloV5FaceDetector
            data_path = YoloV5FaceDetector().detect_in_folder(args.data_path[i],100)
    #for i,numRefs in enumerate(args.num_refrences):
        #print('The embeddings path',args.save_embeddings)
        #print('The embeddings path',args.data_path[i])
        ee = Prepare_Embeddings(args.model_file, data_path, numRefs, args.batch_size, args.save_embeddings)
        
        
    # accuracy, score, label = ee.do_evaluation()
    # print(">>>> top1 accuracy:", accuracy)

    # if args.save_bins is not None:
    #     _ = ee.generate_eval_pair_bin(args.save_bins)

    # plot_tpr_far(score, label)
elif __name__ == "__test__":
    data_path = "temp_test/faces_emore_test/"
    model_file = "checkpoints/GhostFaceNet_W1.3_S1_ArcFace.h5"
    batch_size = 64
    save_embeddings = None
