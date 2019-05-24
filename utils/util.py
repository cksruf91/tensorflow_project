import numpy as np
import sys
sys.path.append("..")
import time
from matplotlib import pyplot as plt

from slacker import Slacker
from config import *


def history_graph(hist): 
    fig = plt.figure()
    plt.xlabel('epochs')
    plt.ylabel('acc(%)')
    ax_acc = fig.add_subplot(111)
    line1 = ax_acc.plot(hist['epoch'], hist['acc'], label='acc',color='#0613a3') 
    line2 = ax_acc.plot(hist['epoch'], hist['val_acc'], label='val_acc',color='#7311d6') 
    ax_loss = ax_acc.twinx()
    line3 = ax_loss.plot(hist['epoch'], hist['loss'], label='loss',color='#a52121')
    line4 = ax_loss.plot(hist['epoch'], hist['val_loss'], label='val_loss', color='#c48a03')
    plt.ylabel('loss')

    lines = line1+line2+line3+line4
    labels = [l.get_label() for l in lines]
    plt.legend(lines,labels, fancybox=True, bbox_to_anchor=(1.35, 1.05))
    plt.show()
    return None

def slack_message(chennel, message):
    slack = Slacker(MY_SLACK_TOKEN)
    slack.chat.post_message(chennel, message)

def normalize_img(img):
    shape = img.shape
    img = np.float64(img.reshape(-1))
    img -= img.mean()
    img /= img.std()
    img = img.reshape(shape)
#     img = img/ 255
    return img

def learning_rate_schedule(epoch_, lr):
    if epoch_ > 150:
        lr *= 0.5e-3
    elif epoch_ > 130:
        lr *= 1e-3
    elif epoch_ > 110:
        lr *= 1e-2
    elif epoch_ > 80:
        lr *= 1e-1
    return lr


def train_progressbar(iteration, total, epoch = '',epochs = '', loss = '', acc = '', decimals = 1, barLength = 100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '>' * filledLength + ' ' * (barLength - filledLength)
    sys.stdout.write("\r epoch: {}/{} [{}] {} % - loss : {:5.5f}, - acc : {:5.5f}".format(epoch,epochs,bar,percent,loss,acc)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


class progress():
    def __init__(self):
        self.count = 1
        
    def add_count(self):
        self.count += 1
    
    def print_progress(self,batch_size,total,i):
        dot_num = int(batch_size*self.count/total*100)
        dot = '>'*dot_num
        empty = '_'*(100-dot_num)
        sys.stdout.write(f'\r [{dot}{empty}] {i} Done')
        self.add_count()
        

if __name__ == "__main__":
#     slack_message('#resnet_project', 'hi hello')
    
    for i in range(0, 100):
        train_progressbar(iteration =i , total =100, epoch = 1 ,epochs = 100, loss = 1.1, acc = 0.8 , decimals = 1, barLength = 50)
        time.sleep(0.05)
        
        
        