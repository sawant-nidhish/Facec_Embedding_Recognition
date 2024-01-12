import imageio.v2 as imageio
import numpy as np
from math import sqrt
import sys
sys.path.insert(0,'./')
import argparse
import os
from tqdm import tqdm
from face_detector import YoloV5FaceDetector

from tensorflow import keras
import losses, train, GhostFaceNets
import tensorflow as tf
import keras_cv_attention_models
import cv2
import json
#from numba import jit

#@jit(target_backend='cuda')
def get_fish_xn_yn(source_x, source_y, radius, distortion, cx, cy):
    """
    Get normalized x, y pixel coordinates from the original image and return normalized 
    x, y pixel coordinates in the destination fished image.
    :param distortion: Amount in which to move pixels from/to center.
    As distortion grows, pixels will be moved further from the center, and vice versa.
    """

    # if 1 - distortion*(radius**2) == 0:
    #     return source_x, source_y

    # return source_x / (1 - (distortion*(radius**2))), source_y / (1 - (distortion*(radius**2)))
    
    dx = source_x - cx
    dy = source_y - cy
    r2 = radius**2
    r4 = r2 * r2
    r6 = r4 * r2
    k1 = 2
    k2 = 2
    p1 = 0.01
    p2 = 0.01
    radial_distortion = 1.0 + k1 * r2 + k2 * r4
    tangential_x = 2.0 * p1 * dx * dy + p2 * (r2 + 2.0 * dx * dx)
    tangential_y = p1 * (r2 + 2.0 * dy * dy) + 2.0 * p2 * dx * dy

    distorted_x = cx + radial_distortion * dx 
    distorted_y = cy + radial_distortion * dy
    return distorted_x, distorted_y


# def fish(img, distortion_coefficient):
#     """
#     :type img: numpy.ndarray
#     :param distortion_coefficient: The amount of distortion to apply.
#     :return: numpy.ndarray - the image with applied effect.
#     """

#     # If input image is only BW or RGB convert it to RGBA
#     # So that output 'frame' can be transparent.
#     w, h = img.shape[0], img.shape[1]
#     if len(img.shape) == 2:
#         # Duplicate the one BW channel twice to create Black and White
#         # RGB image (For each pixel, the 3 channels have the same value)
#         bw_channel = np.copy(img)
#         img = np.dstack((img, bw_channel))
#         img = np.dstack((img, bw_channel))
#     if len(img.shape) == 3 and img.shape[2] == 3:
#         # print("RGB to RGBA")
#         img = np.dstack((img, np.full((w, h), 255)))

#     # prepare array for dst image
#     dstimg = np.zeros_like(img)

#     # floats for calcultions
#     w, h = float(w), float(h)

#     # easier calcultion if we traverse x, y in dst image
#     for x in range(len(dstimg)):
#         for y in range(len(dstimg[x])):

#             # normalize x and y to be in interval of [-1, 1]
#             xnd, ynd = float((2*x - w)/w), float((2*y - h)/h)

#             # get xn and yn distance from normalized center
#             rd = sqrt(xnd**2 + ynd**2)

#             # new normalized pixel coordinates
#             xdu, ydu = get_fish_xn_yn(xnd, ynd, rd, distortion_coefficient)

#             # convert the normalized distorted xdn and ydn back to image pixels
#             xu, yu = int(((xdu + 1)*w)/2), int(((ydu + 1)*h)/2)

#             # if new pixel is in bounds copy from source pixel to destination pixel
#             if 0 <= xu and xu < img.shape[0] and 0 <= yu and yu < img.shape[1]:
#                 dstimg[x][y] = img[xu][yu]

#     return dstimg.astype(np.uint8)

#@jit(target_backend='cuda')
def fish(img, distortion_coefficient, centerChoice):
    """
    :type img: numpy.ndarray
    :param distortion_coefficient: The amount of distortion to apply.
    :return: numpy.ndarray - the image with applied effect.
    """

    # If input image is only BW or RGB convert it to RGBA
    # So that output 'frame' can be transparent.
    w, h = img.shape[0], img.shape[1]
    if len(img.shape) == 2:
        # Duplicate the one BW channel twice to create Black and White
        # RGB image (For each pixel, the 3 channels have the same value)
        bw_channel = np.copy(img)
        img = np.dstack((img, bw_channel))
        img = np.dstack((img, bw_channel))
    if len(img.shape) == 3 and img.shape[2] == 3:
        # print("RGB to RGBA")
        img = np.dstack((img, np.full((w, h), 255)))

    # prepare array for dst image
    dstimg = np.zeros_like(img)

    # floats for calcultions
    w, h = float(w), float(h)
    if centerChoice=="random":
        cx, cy = np.random.uniform(-1,1), np.random.uniform(-1,1)
    elif centerChoice=="fixed-center":
        cx, cy= 0, 0
    # print("Center",cx,cy)
    # print("Center",cx,cy)
    # dist_x=min(np.abs(1-cx),np.abs(-1-cx))
    # dist_y=min(np.abs(1-cy),np.abs(-1-cy))
    # dist_r = min(dist_x,dist_y)
    # print("dist_r",dist_r)
    # easier calcultion if we traverse x, y in dst image
    max_x = -np.inf
    min_x = np.inf
    max_y = -np.inf
    min_y = np.inf
    for x in range(len(dstimg)):
        for y in range(len(dstimg[x])):

            # normalize x and y to be in interval of [-1, 1]
            xnd, ynd = float((x - (w/2))/(w/2)), float((y - (h/2))/(h/2))
            if xnd > max_x:
                max_x = xnd
            if xnd < min_x:
                min_x = xnd
            if ynd > max_y:
                max_y = ynd
            if ynd < min_y:
                min_y = ynd
            # xnd, ynd = float((2*x - w)/w), float((2*y - h)/h)

            # get xn and yn distance from normalized center
            rd = sqrt((xnd-cx)**2 + (ynd-cy)**2)
            # dtc = (2*distortion_coefficient-cx - cy)/2
            # if dtc<=0:
            #     dtc=0.5
            # if dtc>1:
            #     dtc=1
            # new normalized pixel coordinates
            # if distortion_coefficient>dist_r:
            #     distortion_coefficient=dist_r
            
            xdu, ydu = get_fish_xn_yn(xnd, ynd, rd, distortion_coefficient, cx, cy)

            # convert the normalized distorted xdn and ydn back to image pixels
            xu, yu = int((xdu + 1)*(w/2)), int((ydu + 1)*(h/2))
            # xu, yu = int(((xdu + 1)*w)/2), int(((ydu + 1)*h)/2)

            # if new pixel is in bounds copy from source pixel to destination pixel
            if 0 <= xu and xu < img.shape[0] and 0 <= yu and yu < img.shape[1]:
                dstimg[x][y] = img[xu][yu]
    # print("x",max_x,min_x)
    # print("y",max_y,min_y)
    # print("Unnormalized center",int((cx + 1)*(w/2)), int((cy + 1)*(h/2)))
    # print("OG image shape",img.shape)
    # print("Distorted image shape",dstimg.shape)
    return dstimg.astype(np.uint8)

def parse_args(args=sys.argv[1:]):
    """Parse arguments."""

    parser = argparse.ArgumentParser(
        description="Apply fish-eye effect to images.",
        prog='python3 fish.py')

    parser.add_argument("-i", "--image", help="path to image file."
                        " If no input is given, the supplied example 'grid.jpg' will be used.",
                        type=str, default="grid.jpg")

    parser.add_argument("-o", "--outpath", help="file path to write output to."
                        " format: <path>.<format(jpg,png,etc..)>",
                        type=str, default="fish.png")

    parser.add_argument("-d", "--distortion",
                        help="The distrotion method. If the distortion needs only to be at the center then input shuld be \"fixed-center\". Else, if the distortion needs to be random then input \"random\"",
                        type=str, default="fixed-center")

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args()
    
    ####Code for creating the fish images from a normal dataset
    '''if not os.path.exists(args.outpath):
        print("Generating fish eye images")
        os.mkdir(args.outpath)
        i=0
        for root, dir, files in tqdm(os.walk(args.image)):
            outpath=os.path.join(args.outpath,root.split('/')[-1])
            #print(outpath)
            
            for image in files:
                i=i+1
                if not os.path.exists(outpath):
                    os.mkdir(outpath)
                try:
                    imgobj = imageio.imread(os.path.join(root,image))
                except Exception as e:
                    print(e)
                    sys.exit(1)
                if args.distortion == 'random':
                    for i in range(3):
                        ##output_img = fish(imgobj, args.distortion,args.distortion)
                    
                        imageName=image.split('.')[0]+"-"+str(i)+".png"
                        if os.path.exists(os.path.join(outpath,imageName)):
                            continue
                        else:
                            output_img = fish(imgobj, args.distortion,args.distortion)
                            imageio.imwrite(os.path.join(outpath,imageName), output_img, format='png')
                #output_img = fish(imgobj, args.distortion,"fixed-center")
                # imageName=image.split('.')[0]+"-"+str(3)+".png"
                imageName=image.split('.')[0]+".png"
                if os.path.exists(os.path.join(outpath,imageName)):
                    continue
                else:
                    output_img = fish(imgobj, args.distortion,"fixed-center")
                    imageio.imwrite(os.path.join(outpath,imageName), output_img, format='png')
    else:
        print("Fishs eye images already exists")'''
    
    ###Code for performing detection on the trainig dataset
    data_path=args.outpath
    print(data_path)
    train_data_path=data_path+'_aligned_112_112'
    if not os.path.exists(data_path+'_aligned_112_112'):
        print("Generating detections")
        YoloV5FaceDetector().detect_in_folder(data_path,100)
    else:
        print("Detections already exists")

    '''outpath=args.outpath+'_aligned_112_112'
    train_data_path=outpath
    if not os.path.exists(outpath):
        print("Generating fish eye images")
        os.mkdir(outpath)
        print('outpath', outpath)
        for root, dir, files in tqdm(os.walk(args.outpath)):
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
'''
    ###Code to train model
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(tf.config.list_physical_devices('GPU'))

    # Remove the below for better accuracies and keep it for faster training
    keras.mixed_precision.set_global_policy("mixed_float16")

    # ms1m-retinaface-t1 (MS1MV3) dataset
    # data_basic_path = 'datasets/ms1m-retinaface-t1'
    # data_path = data_basic_path + '_112x112_folders'
    # eval_paths = [os.path.join(data_basic_path, ii) for ii in ['lfw.bin', 'cfp_fp.bin', 'agedb_30.bin']]

    # (MS1MV2) dataset
    # train_data_path = os.path.join(args.outpath,'_aligned_112_112')

    #GhostFaceNetV1
    # Strides of 2
    # basic_model = GhostFaceNets.buildin_models("ghostnetv1", dropout=0, emb_shape=512, output_layer='GDC', bn_momentum=0.9, bn_epsilon=1e-5)
    # basic_model = GhostFaceNets.add_l2_regularizer_2_model(basic_model, weight_decay=5e-4, apply_to_batch_normal=False)
    # basic_model = GhostFaceNets.replace_ReLU_with_PReLU(basic_model)

    # Strides of 1
    basic_model = GhostFaceNets.buildin_models("ghostnetv1", dropout=0, emb_shape=512, output_layer='GDC', bn_momentum=0.9, bn_epsilon=1e-5, scale=True, use_bias=True, strides=1)
    basic_model = GhostFaceNets.add_l2_regularizer_2_model(basic_model, weight_decay=5e-4, apply_to_batch_normal=False)
    basic_model = GhostFaceNets.replace_ReLU_with_PReLU(basic_model)

    # Strides of 2
    # tt = train.Train(data_path, eval_paths=eval_paths,
    #     save_path='ghostnetv1_w1.3_s2.h5',
    #     basic_model=basic_model, model=None, lr_base=0.1, lr_decay=0.5, lr_decay_steps=45, lr_min=1e-5,
    #     batch_size=128, random_status=0, eval_freq=1, output_weight_decay=1)

    # Strides of 1
    ####Code to train model fresh
    '''t1 = train.Train(train_data_path,
         save_path='ghost-S1-scratch.h5',
         basic_model=basic_model, model=None, lr_base=0.0001, lr_decay=0.005, lr_decay_steps=180, lr_min=1e-5,
         batch_size=1024, random_status=0, eval_freq=0, output_weight_decay=0.05)

    optimizer = keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9)
    sch = [
         {"loss": losses.ArcfaceLoss(scale=32), "epoch": 20, "optimizer": optimizer},
        
    ]
    # # {"loss": losses.ArcfaceLoss(scale=64), "epoch": 50},
    t1.train(sch, 0)'''

    with open('checkpoints/ghost-S1-fixed-centerpretrained_hist.json','r') as file:
        modelHist=json.load(file)

    #Code for continuation of training
    t1 = train.Train(train_data_path,
        save_path='ghost-S1-'+args.distortion+'_plus_rect_pretrained.h5',
        basic_model=None, model="checkpoints/GhostFaceNet_W1.3_S1_ArcFace.h5", lr_base=0.0001, lr_decay=0.005, lr_decay_steps=180, lr_min=1e-5,
        batch_size=1024, random_status=0, eval_freq=0, output_weight_decay=0.05)

    optimizer = keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9)
    sch = [
        {"loss": losses.ArcfaceLoss(scale=32), "epoch": 50, "optimizer": optimizer},
        
    ]
    # {"loss": losses.ArcfaceLoss(scale=64), "epoch": 50},
    t1.train(sch, 0)
    '''t1 = train.Train(train_data_path,
        save_path='ghost-S1-'+args.distortion+'scratch.h5',
        basic_model=basic_model, model=None, lr_base=0.01, lr_decay=0.5, lr_decay_steps=45, lr_min=1e-5,
        batch_size=128, random_status=0, eval_freq=0, output_weight_decay=1)

    optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    sch = [
        {"loss": losses.ArcfaceLoss(scale=32), "epoch": 20, "optimizer": optimizer},

    ]
    # {"loss": losses.ArcfaceLoss(scale=64), "epoch": 50},
    t1.train(sch, 0)'''




    

