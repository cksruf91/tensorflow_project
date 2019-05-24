import numpy as np
import os
import timeit
import time
# import argparse

from config import *
from model import *
from train_generator import batch_iterator
from utils.util import train_progressbar, slack_message, learning_rate_schedule

# def args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', type=int, required=True, choices=[0,1]
#                         , help='number of class')  # number of class
#     args = parser.parse_args()
#     return args
# arg = args()

opts = TrainOption()

import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


with tf.device('/device:GPU:0'):
    # placeholder for images
    shapes = list((None,)+IMAGE_SHAPE)
    images = tf.placeholder('float32', shape=shapes, name='images')  
    
    # placeholder for labels
    labels = tf.placeholder('float32', shape=[None, len(LABEL.keys())], name='labels')  
    
    # placeholder for training boolean (is training)
    training = tf.placeholder('bool', name='training') 
    
    global_step = tf.get_variable(name='global_step', shape=[], dtype='int64', trainable=False)  
    learning_rate = tf.placeholder('float32', name='images')
    # learning_rate = tf.train.exponential_decay(opts.LEARNING_RATE, global_step, opts.LR_DEACY_STEPS, opts.LR_DECAY_RATE)
    
    ## placeholder fot store Beat accuracy
    best_accuracy = tf.get_variable(name='best_accuracy', dtype='float32', trainable=False, initializer=0.0)
    
    # model build
    output = ResNetV2(training).build(images)
    # output = basic_model(images,opts.DROPOUT_RATE,training)
    # output = ResNet(images,training).build()
    

#loss and optimizer
with tf.variable_scope('losses'):
    loss = tf.keras.backend.categorical_crossentropy(labels, output)
    loss = tf.reduce_mean(loss, name='loss')
#     loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels
#                                                       , logits=output
#                                                       , name='cross_entropy')
#     loss = tf.losses.softmax_cross_entropy(labels, output, label_smoothing=0.1)
#     l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()], name='l2_loss')
#     loss = loss + l2_loss * opts.WEIGHT_DECAY

    
with tf.variable_scope('optimizers'):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = optimizer.minimize(loss, global_step=global_step) 
    train_op = tf.group([train_op, update_ops], name='train_op')
#     optimizer = tf.train.MomentumOptimizer(learning_rate=opts.LEARNING_RATE, momentum=opts.MOMENTUM, use_nesterov=True)
    
#accuracy
with tf.variable_scope('accuracy'):
    
    output = tf.nn.softmax(output, name='output')
    prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1), name='prediction')
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32), name='accuracy')
    
# method to save model
# 참조 : https://goodtogreate.tistory.com/entry/Saving-and-Restoring
saver = tf.train.Saver()


## train 데이터 사이즈를 따로 저장하여 불러옴
with open(TRAIN_FILE.split('.')[0] + '.length', 'r') as f:
    train_data_lenth = int(f.read())
    
with open(TEST_FILE.split('.')[0] + '.length', 'r') as f:
    test_data_lenth = int(f.read())


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    gstep = sess.run(global_step)
    
    ## load batch generator
    print(f"train data from : {TRAIN_FILE}")
    train_iterator = batch_iterator(TRAIN_FILE 
                                    , batch_size=opts.BATCH_SIZE
                                    , training=True, shuffle=True)
    train_images_batch, train_labels_batch = train_iterator.get_next()
    
    print(f"test data from : {TEST_FILE}")
    test_iterator = batch_iterator(TEST_FILE 
                                   , batch_size=opts.BATCH_SIZE
                                   , training=False, shuffle=False)
    test_images_batch, test_labels_batch = test_iterator.get_next()
    
    ## initialize batch generator
    sess.run(train_iterator.initializer)
    sess.run(test_iterator.initializer)
    
    ## try to restore last model checkpoint
    try: 
        saver.restore(sess, tf.train.latest_checkpoint(RESULT_PATH))
        check_point_name = tf.train.latest_checkpoint(RESULT_PATH)
        last_epoch = int(check_point_name.split('_')[-1].split('.')[0])
        patience = 0 
        print("checkpoint restored")
    except:
        last_epoch = 0
        print("failed to load checkpoint")
        
        # loss acc history
        with open(os.path.join(RESULT_PATH,"history.csv"),'w') as f:
            f.write("epoch,loss,acc,val_loss,val_acc\n")
    
    ## epoch
    EPOCHS = opts.EPOCHS
    
    """ run train """
    for epoch_ in range(EPOCHS - last_epoch):
        
        epoch_ += 1+last_epoch

        if opts.STEP_PER_EPOCH is None:
            step_per_epoch = train_data_lenth//opts.BATCH_SIZE
        else:
            step_per_epoch = opts.STEP_PER_EPOCH
            
        ## learning late schedule
        lr = learning_rate_schedule(epoch_, opts.LEARNING_RATE)
        
        ## epoch당 step 계산
        step = 0
        train_loss = []
        train_acc = []
        start = timeit.default_timer()
        
        """ Train """
        while step < step_per_epoch:
            train_images, train_labels = sess.run([train_images_batch, train_labels_batch])
            gstep, _, loss_, accuracy_ = sess.run(
                        [global_step, train_op, loss, accuracy],
                        feed_dict={images: train_images, labels: train_labels, learning_rate : lr
                                   , training: True}) #, tf.keras.backend.learning_phase():1
            train_loss.append(loss_)
            train_acc.append(accuracy_)
            ## EPOCH 진행상황 출력
            mean_train_loss = np.mean(train_loss)
            mean_train_acc = np.mean(train_acc)
            train_progressbar(step, step_per_epoch
                              , epoch_, EPOCHS
                              ,mean_train_loss ,mean_train_acc , 1, 50)
            step += 1
        
        epoch_time = time.strftime("%H:%M:%S", time.gmtime(timeit.default_timer()-start))
        
        
        val_loss = []
        val_acc = []
#         predictions= []
        """ Validation """
        while True:
            try:
                test_images, test_labels = sess.run([test_images_batch, test_labels_batch])
                loss_, accuracy_, prediction_= sess.run(
                            [loss, accuracy, output],
                            feed_dict={images: test_images, labels: test_labels
                                       , training: False}) #tf.keras.backend.learning_phase():0
#                 predictions.append(prediction_)
                val_loss.append(loss_)
                val_acc.append(accuracy_)
            except tf.errors.OutOfRangeError:
                sess.run(test_iterator.initializer)
                break
#             print(prediction_)
        
    
        mean_val_loss = np.mean(val_loss)
        mean_val_acc = np.mean(val_acc)
#         predictions = np.concatenate(predictions)
        print('\n time: ',epoch_time, '\tvalidation_acc - ',
              'best :', best_accuracy.eval(),
              '\tcurrent :', mean_val_acc)
        
        ## send result to slack
        
        if SEND_MESSAGE:
            message = "epoch : {} | time : {} \n loss : {:5.5f} \n acc : {:5.5f} \n val_loss : {:5.5f} \n val_acc : {:5.5f}".format(epoch_, epoch_time, mean_train_loss, mean_train_acc,mean_val_loss, mean_val_acc)
            slack_message('#resnet_project', message)
        
        # save checkpoint
        if SAVE_CHECKPOINT:
            save_path = saver.save(sess, os.path.join(RESULT_PATH,f'checkpoint_{epoch_}.ckpt'))
        
        if best_accuracy.eval() < mean_val_acc:
            best_accuracy = tf.assign(best_accuracy, mean_val_acc)
            patience = 0
        else:
            patience += 1
        
        # save history 
        with open(os.path.join(RESULT_PATH,"history.csv"),'a') as f:
            f.write(str(epoch_)+','+str(mean_train_loss)+','+str(mean_train_acc)+','+ str(mean_val_loss)+','+str(mean_val_acc)+'\n')
        
        # early stop training
        if EARLY_STOPPING:
            if patience > 7:
                break
            
