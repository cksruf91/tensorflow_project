from keras.layers import Input, regularizers, Conv1D , MaxPooling1D, Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from sklearn.utils import class_weight
from keras.regularizers import l2
import tensorflow as tf


""" ResNet """
class ResNetV2():
    def __init__(self,training):
#         self.X_input = X_input
        self.training = training
        self.name = 'ResNetV2'
        self.in_filter = 16
        self.out_filter = None
    
    def resnet_block(self,x, filter_size, kernels=3, stride=1 ,batch=True ,acti=True ):
        if batch:
            x = tf.layers.batch_normalization(inputs = x, training=self.training)
#             x = tf.keras.layers.BatchNormalization()(x)
        if acti:
            x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(filters=filter_size,
                                   kernel_size=kernels,
                                   padding="SAME", strides=stride,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
                                  )(x)
        return x

    def build(self,X_input) :
        
#         inputs = tf.keras.layers.Input((32,32,3))
        x = tf.keras.layers.Conv2D(filters=self.in_filter,
                                   kernel_size=3,
                                   padding="SAME", strides=1,
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4)
                                  )(X_input)
        x = tf.layers.batch_normalization(inputs = x, training=self.training)
#         x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        for stage in range(3):
            for blocks in range(3):
                stride_ = 1
                if stage == 0:
                    ## 0: 16 -> 64
                    self.out_filter = self.in_filter *4 
                else:
                    ## 1: 62 -> 128 | 2: 128 -> 256
                    self.out_filter = self.in_filter *2 
                    if blocks == 0:
                        stride_ = 2 ## down size
                
                if blocks == 0:    
                    block_layer = self.resnet_block(x,self.in_filter, kernels=1
                                          , stride=stride_, batch = False, acti =False)
                    x = self.resnet_block(x,self.out_filter,kernels=1
                                          , stride=stride_, batch = False, acti =False)
                else:
                    block_layer = self.resnet_block(x,self.in_filter, kernels=1, batch = True, acti =True)

                block_layer = self.resnet_block(block_layer, self.in_filter, kernels=3, batch = True, acti =True)
                block_layer = self.resnet_block(block_layer, self.out_filter, kernels=1, batch = True, acti =True)

                x = tf.keras.layers.Add()([block_layer, x])
            
            print(self.in_filter)
            self.in_filter = self.out_filter
        
        
        x = tf.layers.batch_normalization(inputs = x, training=self.training)
#         x = tf.keras.layers.BatchNormalization()(x) #
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.AveragePooling2D(pool_size = 8)(x)
        x = tf.keras.layers.Flatten()(x)
        out = tf.keras.layers.Dense(units=10, activation='softmax',kernel_initializer ='he_normal')(x)
        
        ## keras model summary
        # model = tf.keras.Model(inputs=inputs, outputs=out)
        # print(model.summary())
        
        return out



"""keras models"""
def cnn_model(input_dim):
        model = Sequential()
        # input_shape = []
        model.add(Conv1D(filters=128 ,
                        kernel_size=5,
                        input_shape=(input_dim[1], input_dim[2]),
                        strides=1,
                        padding='causal',
                        data_format='channels_last',
                        dilation_rate=1,
                        activation='tanh',
                        use_bias=True,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros',
                        kernel_regularizer=regularizers.l1(10e-5)
                        # bias_regularizer=None,
                        # activity_regularizer=None,
                        # kernel_constraint=None,
                        # bias_constraint=None
                        ))
        model.add(MaxPooling1D(3,strides=1,data_format='channels_last'))
        model.add(Conv1D(filters=16 ,
                        kernel_size=2,
                        strides=1,
                        padding='causal',
                        data_format='channels_last',
                        dilation_rate=1,
                        activation='tanh',
                        use_bias=True,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros',
                        kernel_regularizer=regularizers.l1(10e-5) 
                        # bias_regularizer=None,
                        # activity_regularizer=None,
                        # kernel_constraint=None,
                        # bias_constraint=None
                        ))
        # model.add(GlobalAveragePooling1D())
        model.add(MaxPooling1D(3,strides=1,data_format='channels_last'))
        model.add(Flatten())
        model.add(Dense(126, activation='tanh'))
        model.add(Dense(32, activation='tanh'))
        model.add(Dense(10, activation='sigmoid'))    
        return model
    
    
def model_2d(input_dim):
    model = Sequential()

    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=input_dim
                     , padding='same', kernel_regularizer=l2(0.001)))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (5, 5), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (5, 5), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (5, 5), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D())
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    return model


"""tensorflow models"""
def basic_model(inputs,dropout_rate,training):
    with tf.variable_scope('conv1'):
#         keras.layers.separable_conv2d
        inputs = tf.layers.separable_conv2d(inputs, filters=64
                                            , kernel_size=[5, 5]
                                            , strides=[1, 1]
                                            , padding='SAME')
#         keras.layers.batch_normalization
        inputs = tf.layers.batch_normalization(inputs, training=training)
#         keras.layers.dropout
        inputs = tf.layers.dropout(inputs, rate=dropout_rate, training=training)
        inputs = tf.nn.relu(inputs)
        
    with tf.variable_scope('conv2'):
        inputs = tf.layers.separable_conv2d(inputs, filters=32
                                            , kernel_size=[3, 3]
                                            , strides=[1, 1]
                                            , padding='SAME')
        inputs = tf.layers.batch_normalization(inputs, training=training)
        inputs = tf.nn.relu(inputs)
        
    with tf.variable_scope('pooling1'):
#         keras.layers.average_pooling2d
        inputs = tf.layers.average_pooling2d(inputs, pool_size=[2,2]
                                            , strides=[1, 1], name="layer3")
        inputs = tf.nn.relu(inputs)
    
    with tf.variable_scope('conv3'):
        inputs = tf.layers.separable_conv2d(inputs, filters=32
                                            , kernel_size=[3, 3]
                                            , strides=[1, 1]
                                            , padding='SAME')
        inputs = tf.layers.batch_normalization(inputs, training=training)
        inputs = tf.nn.relu(inputs)
    
    with tf.variable_scope('conv4'):
        inputs = tf.layers.separable_conv2d(inputs, filters=32
                                            , kernel_size=[3, 3]
                                            , strides=[1, 1]
                                            , padding='SAME')
        inputs = tf.layers.batch_normalization(inputs, training=training)
        inputs = tf.nn.relu(inputs)
        
    with tf.variable_scope('pooling1'):
        inputs = tf.layers.average_pooling2d(inputs, pool_size=[2,2]
                                            , strides=[1, 1], name="layer6")
        inputs = tf.nn.relu(inputs)
        
    with tf.variable_scope('classifier'):
        inputs = tf.layers.average_pooling2d(inputs, pool_size=inputs.shape[1:3], strides=[1, 1])
#         keras.layers.flatten
        inputs = tf.layers.flatten(inputs)
        inputs = tf.layers.dropout(inputs, rate=0.3, training=training)
#         keras.layers.dense
        output = tf.layers.dense(inputs, units=10)
    
    return output
        
""" ResNet """
class ResNet():
    def __init__(self, X_input,training):
        self.X_input = X_input
        self.training = training
        self.name = 'build'
        
    def residual_block(self, X_input, num_filter, chg_dim) :
        stride=1
        #stride=2일 경우
        if chg_dim :
            stride=2
#             pool1 = tf.layers.max_pooling2d(inputs= X_input, strides=2, pool_size=[1,1])
            pool1 = tf.keras.layers.MaxPool2D(strides=2, pool_size=[1,1])(X_input)
            pad1 = tf.pad(pool1, [[0,0], [0,0], [0,0], [int(num_filter/4),int(num_filter/4)]])
            shortcut = pad1
        else :
            shortcut = X_input

        bm1 = tf.layers.batch_normalization(inputs = X_input, training=self.training)
#         bm1 = tf.keras.layers.BatchNormalization(trainable=self.training)(X_input)
        relu1 = tf.nn.relu(bm1)
        conv1 = tf.keras.layers.SeparableConv2D(filters=num_filter
                                 , kernel_size=[3, 3]
                                 , padding="SAME", strides=stride
                                 , kernel_initializer=tf.contrib.layers.xavier_initializer())(relu1)
#         conv1 = tf.layers.conv2d(inputs = relu1, filters=num_filter
#                                  , kernel_size=[3, 3]
#                                  , padding="SAME", strides=stride
#                                  , kernel_initializer=tf.contrib.layers.xavier_initializer())

        bm2 = tf.layers.batch_normalization(inputs = conv1, training=self.training)
#         bm2 = tf.keras.layers.BatchNormalization(trainable=self.training)(conv1)
        relu2 = tf.nn.relu(bm2)
        conv2 = tf.keras.layers.SeparableConv2D(filters=num_filter
                                 , kernel_size=[3, 3]
                                 , padding="SAME", strides=1
                                 , kernel_initializer=tf.contrib.layers.xavier_initializer())(relu2)
#         conv2 = tf.layers.conv2d(inputs = relu2, filters=num_filter
#                                  , kernel_size=[3, 3]
#                                  , padding="SAME", strides=1
#                                  , kernel_initializer=tf.contrib.layers.xavier_initializer())

        bm3 = tf.layers.batch_normalization(inputs = conv2, training=self.training)
        relu3 = tf.nn.relu(bm3)
        conv3 = tf.keras.layers.SeparableConv2D(filters=num_filter
                                 , kernel_size=[3, 3]
                                 , padding="SAME", strides=1
                                 , kernel_initializer=tf.contrib.layers.xavier_initializer())(relu3)

        X_output = conv3 + shortcut

        return X_output

    def build(self) :
        with tf.variable_scope(self.name) :

            ###Input Layer
            #input : ? * 32 * 32 * 3
            #ouput : ? * 32 * 32 * 3
            #self.X = tf.placeholder(tf.float32, [None, 32, 32, 3])
            #self.Y = tf.placeholder(tf.float32, [None, 10])
            #self.training = tf.placeholder(tf.bool)

            ###Hidden Layer
            #input : ? * 32 * 32 * 3
            #ouput : ? * 32 * 32 * 16
#             conv = tf.layers.conv2d(inputs = self.X_input, filters = 16
#                                     , kernel_size=[3,3], padding="SAME"
#                                     , strides=1)
            conv = tf.keras.layers.SeparableConv2D(filters = 16
                                    , kernel_size=[3,3], padding="SAME"
                                    , strides=1)(self.X_input)
#             bm = tf.keras.layers.BatchNormalization(trainable=self.training)(conv1)
            bm = tf.layers.batch_normalization(inputs = conv, training=self.training)
            
    
            relu = tf.nn.relu(bm)
            
#             shape = relu.get_shape().as_list()
#             print(shape)
            
            #ouput : ? * 32 * 32 * 16
            res1 = self.residual_block(relu, 16, False)
            res2 = self.residual_block(res1, 16, False)
            res3 = self.residual_block(res2, 16, False)
            res4 = self.residual_block(res3, 16, False)
            res5 = self.residual_block(res4, 16, False)

            #input : ? * 32 * 32 * 16
            #ouput : ? * 16 * 16 * 32
            res6 = self.residual_block(res5, 32, True)
            res7 = self.residual_block(res6, 32, False)
            res8 = self.residual_block(res7, 32, False)
            res9 = self.residual_block(res8, 32, False)
            res10 = self.residual_block(res9, 32, False)

            #input : ? * 16 * 16 * 32
            #ouput : ? * 8 * 8 * 64
            res11 = self.residual_block(res10, 64, True)
            res12 = self.residual_block(res11, 64, False)
            res13 = self.residual_block(res12, 64, False)
            res14 = self.residual_block(res13, 64, False)
            res15 = self.residual_block(res14, 64, False)

            ###Global Average Pooling
            #input : ? * 8 * 8 * 64
            #ouput : ? * 1 * 1 * 64
            gap = tf.reduce_mean(res15, [1, 2], keep_dims=True)

            ###Output Layer
            #input : ? * 1 * 1 * 64
            #ouput : ? * 1 * 1 * 64
            shape = gap.get_shape().as_list()
            dimension = shape[1] * shape[2] * shape[3]
            flat = tf.reshape(gap, [-1, dimension])

#             fc = tf.layers.dense(inputs=flat, units=10, kernel_initializer=tf.contrib.layers.xavier_initializer())
            fc = tf.keras.layers.Dense(units=10, kernel_initializer=tf.contrib.layers.xavier_initializer())(flat)
#             logits = fc
            return fc

#             self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
#             update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name)
#             with tf.control_dependencies(update_ops):
#                 self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

#             correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1)) 
#             self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
