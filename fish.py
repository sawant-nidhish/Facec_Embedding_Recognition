import imageio.v2 as imageio
import numpy as np
from math import sqrt
import sys
import argparse
import os
from tqdm import tqdm
from numba import jit,cuda
@jit(target_backend='cuda')
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

#     return dstimg.astype(np.uint8)i
@jit(target_backend='cuda')
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
                        help="The distoration coefficient. How much the move pixels from/to the center."
                        " Recommended values are between -1 and 1."
                        " The bigger the distortion, the further pixels will be moved outwars from the center (fisheye)."
                        " The Smaller the distortion, the closer pixels will be move inwards toward the center (rectilinear)."
                        " For example, to reverse the fisheye effect with --distoration 0.5,"
                        " You can run with --distortion -0.3."
                        " Note that due to double processing the result will be somewhat distorted.",
                        type=str, default='random')

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.outpath):
        os.mkdir(args.outpath)
    i=0
    for root, dir, files in tqdm(os.walk(args.image)):
        outpath=os.path.join(args.outpath,root.split('/')[-1])
        # print(outpath)
        
        for image in files:
            i=i+1
            if not os.path.exists(outpath):
                os.mkdir(outpath)
            try:
                imgobj = imageio.imread(os.path.join(root,image))
            except Exception as e:
                print(e)
                sys.exit(1)
            if args.distortion=='random':
                for i in range(3):
                    output_img = fish(imgobj, args.distortion,"random")
                    imageName=image.split('.')[0]+"-"+str(i)+".png"
                    imageio.imwrite(os.path.join(outpath,imageName), output_img, format='png')
            output_img = fish(imgobj, args.distortion,"fixed-center")
            # imageName=image.split('.')[0]+"-"+str(3)+".png"
            imageName=image.split('.')[0]+".png"
            imageio.imwrite(os.path.join(outpath,imageName), output_img, format='png')
        # if i>38218:
        #     break
            
            
    # try:
    #     imgobj = imageio.imread(args.image)
    # except Exception as e:
    #     print(e)
    #     sys.exit(1)
    # if os.path.exists(args.outpath):
    #     ans = input(
    #         args.outpath + " exists. File will be overridden. Continue? y/n: ")
    #     if ans.lower() != 'y':
    #         print("exiting")
    #         sys.exit(0)
    
    # output_img = fish(imgobj, args.distortion)
    # imageio.imwrite(args.outpath, output_img, format='png')
