#!/bin/bash
#SBATCH --job-name=gh-single
#SBATCH --partition=DGX
#SBATCH --mail-user=nidhish.sawant.18003@iitgoa.ac.in
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --output=log/gh-single%j.out

cd $SLURM_SUBMIT_DIR
#wget https://drive.google.com/u/0/uc?id=1JgmzL9OLTqDAZE86pBgETtSQL4USKTFy&export=download
#cp -r ./datasets/celebA-fish_detect_aligned_112_112 ./datasets/celebA-fish-plus-rect_aligned_112_112

#python ./Nidhish/addImgsToDest.py > log/copyImgs
python ./Nidhish/ogEvaluation.py > ogEvaluation
#python prepare_data.py -D datasets/faces_emore -T lfw.bin cfp_fp.bin agedb_30.bin > prepare_data
#zip -r datasets/celebA-fish-single-detect.zip datasets/celebA-fish-single_detect_aligned_112_112/ > zipDir
#python ./Nidhish/train.py -i datasets/celebA/ -o datasets/celebA-fish-plus-rect -d random > log/train_log_bs512_rect
#python fish.py -i datasets/lfw/ -o datasets/lfw-fish -d random > log/lfw-fish-random
#python ./Nidhish/resize.py > lfw-resize
#python ./Nidhish/generateRefs.py > gen-refs
#python ./Nidhish/register_face.py -d 'datasets/lfw-rect-refs/lfw1_aligned_112_112/','datasets/lfw-rect-refs/lfw2_aligned_112_112/','datasets/lfw-rect-refs/lfw3_aligned_112_112/','datasets/lfw-rect-refs/lfw4_aligned_112_112/','datasets/lfw-rect-refs/lfw5_aligned_112_112/' -n 1,2,3,4,5 -m ./checkpoints/GhostFaceNet_W1.3_S1_ArcFace.h5 -S ./embeddings/rect_og.npz > register-face_rect_og
#python ./Nidhish/inference.py -d 'datasets/lfw-rect-infs/lfw1-infr_aligned_112_112/','datasets/lfw-rect-infs/lfw2-infr_aligned_112_112/','datasets/lfw-rect-infs/lfw3-infr_aligned_112_112/','datasets/lfw-rect-infs/lfw4-infr_aligned_112_112/','datasets/lfw-rect-infs/lfw5-infr_aligned_112_112/' -n 1,2,3,4,5 -m ./checkpoints/GhostFaceNet_W1.3_S1_ArcFace.h5 -S './embeddings/rect_og-1.npz','./embeddings/rect_og-2.npz','./embeddings/rect_og-3.npz','./embeddings/rect_og-4.npz','./embeddings/rect_og-5.npz' > infer-face_rect_og
#python ./Nidhish/inference.py -d 'datasets/lfw-fish-single-infs/lfw4-infr/','datasets/lfw-fish-single-infs/lfw5-infr/' -n 4,5 -m ./checkpoints/ghost-S1-fixed-centerpretrained_basic_model_latest.h5 -S './embeddings/fixed-center-4.npz','./embeddings/fixed-center-5.npz' -D > infer-face

#python ./Nidhish/register_face.py -d 'datasets/lfw-fish-single-refs/lfw1_aligned_112_112/','datasets/lfw-fish-single-refs/lfw2_aligned_112_112/','datasets/lfw-fish-single-refs/lfw3_aligned_112_112/','datasets/lfw-fish-single-refs/lfw4_aligned_112_112/','datasets/lfw-fish-single-refs/lfw5_aligned_112_112/' -n 1,2,3,4,5 -m ./checkpoints/GhostFaceNet_W1.3_S1_ArcFace.h5 -S './embeddings/og-fx.npz' > og-register-face

#python ./Nidhish/inference.py -d 'datasets/lfw-fish-single-infs/lfw1-infr_aligned_112_112/','datasets/lfw-fish-single-infs/lfw2-infr_aligned_112_112/','datasets/lfw-fish-single-infs/lfw3-infr_aligned_112_112/','datasets/lfw-fish-single-infs/lfw4-infr_aligned_112_112/','datasets/lfw-fish-single-infs/lfw5-infr_aligned_112_112/' -n 1,2,3,4,5 -m ./checkpoints/GhostFaceNet_W1.3_S1_ArcFace.h5 -S './embeddings/og-fx-1.npz','./embeddings/og-fx-2.npz','./embeddings/og-fx-3.npz','./embeddings/og-fx-4.npz','./embeddings/og-fx-5.npz' > og-infer
