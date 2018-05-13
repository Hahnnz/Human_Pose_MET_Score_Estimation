import os

ROOT_DIR = os.path.expanduser(os.getcwd()+'/')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'out/')

MET_DATASET_ROOT = os.path.join(ROOT_DIR, 'dataset/')

filewriter_path = ROOT_DIR + "out/tensorboard/"
checkpoint_path = ROOT_DIR + "out/checkpoints/"

if not os.path.isdir(filewriter_path):
    os.mkdir(filewriter_path)

if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)
