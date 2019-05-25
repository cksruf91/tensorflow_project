import os

MY_SLACK_TOKEN = ''

PROJECT = "D:\\Projects\\gitrepo\\tensorflow_project"
IMAGE_LOC = os.path.join(PROJECT,"dataset\\cifar10\\image")
TRAIN_IMAGE = os.path.join(IMAGE_LOC,'train')
TEST_IMAGE = os.path.join(IMAGE_LOC,'test')

TRAIN_FILE = os.path.join(IMAGE_LOC,'train.tfrecod')
TEST_FILE = os.path.join(IMAGE_LOC,'test.tfrecod')

RESULT_PATH = os.path.join(PROJECT, "result")
# 
LABEL = {'deer': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'ship': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 'cat': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
 'dog': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
 'frog': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
 'airplane': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
 'truck': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
 'horse': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
 'bird': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
 'automobile': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}

# Image shape to resahpe the image
# h,w, chennel
IMAGE_SHAPE = (32,32,3)
COPY_IMAGE = 1

SEND_MESSAGE = False
SAVE_CHECKPOINT = True
EARLY_STOPPING = False


# training options
class TrainOption():
    def __init__(self):
        self.LEARNING_RATE = 1e-3
        self.LR_DEACY_STEPS = 2000
        self.LR_DECAY_RATE = 0.96
        self.MOMENTUM = 0.9
        self.WEIGHT_DECAY = 0.0001
        self.BATCH_SIZE = 32
        self.EPOCHS = 200
        self.STEP_PER_EPOCH = None #1562
        self.DROPOUT_RATE = 0.0
        
