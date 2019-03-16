import numpy as np
import os
import time
import sys
import pickle
import matplotlib.pyplot as plt
from keras_robsanpam.image import ImageDataGeneratorMod
from keras_robsanpam.resnet152 import ResNet152
from keras.applications.resnet50 import ResNet50
from keras import backend as K
from keras.models import Model, Sequential, model_from_json
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D
from keras.engine.topology import get_source_inputs
import tensorflow as tf
from tqdm import tqdm
from keras.regularizers import l2
from keras.activations import get


def set_regularization(model,
                       apply_to,
                       kernel_regularizer=None,
                       bias_regularizer=None,
                       activity_regularizer=None,
                       beta_regularizer=None,
                       gamma_regularizer=None
                       ):
    
    if apply_to == 'all':
        layers = model.layers
    else:
        layers = [l for l in model.layers if str(l.name) in apply_to]
    
    for layer in layers:
        # set kernel_regularizer
        if kernel_regularizer is not None and hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = kernel_regularizer
        # set bias_regularizer
        if bias_regularizer is not None and hasattr(layer, 'bias_regularizer'):
            layer.bias_regularizer = bias_regularizer
        # set activity_regularizer
        if activity_regularizer is not None and hasattr(layer, 'activity_regularizer'):
            layer.activity_regularizer = activity_regularizer
        # set beta and gamma of BN layer
        if beta_regularizer is not None and hasattr(layer, 'beta_regularizer'):
            layer.beta_regularizer = beta_regularizer
        if gamma_regularizer is not None and hasattr(layer, 'gamma_regularizer'):
            layer.gamma_regularizer = gamma_regularizer
    out = model_from_json(model.to_json())
    out.set_weights(model.get_weights())
    return out


def create_generators(y_dict, batch_size=16):

    image_generator = ImageDataGeneratorMod(rescale=1./255)

    print("\nLoading train set...")
    train_generator = image_generator.flow_from_directory(
                traindir,
                target_size=(224,224),
                batch_size=batch_size,
                shuffle=True,
                class_mode='multilabel',
                multilabel_classes=y_dict,
                seed=1)

    print("\nLoading test set...")
    val_generator = image_generator.flow_from_directory(
                valdir,
                target_size=(224,224),
                batch_size=batch_size,
                shuffle=True,
                class_mode='multilabel',
                multilabel_classes=y_dict,
                seed=1)

    return train_generator, val_generator


def run_epoch_on_generator(sess,
                           num_classes,
                           generator,
                           batch_size=None,
                           epoch=None,
                           epochs=None,
                           training=True,
                           steps_per_epoch=None,
                           lr=None):
    
    loss_list = []
    acc_list = []
    
    tp = np.zeros((1, num_classes))
    fp = np.zeros((1, num_classes))
    fn = np.zeros((1, num_classes))

    for i in tqdm(range(steps_per_epoch), desc="Epoch {}/{} - {}: ".format(
        epoch, epochs, "Training" if training else " Testing")):
        
        batch_X, batch_y = next(generator)        

        if training:
            step, loss_value, acc, utp, ufp, ufn = sess.run([train_op, loss_op, acc_op, tp_op, fp_op, fn_op],
                                             feed_dict={X:batch_X, y:batch_y, 
                                                        K.learning_phase():1, lr_placeholder:lr})
        else:
            loss_value, acc, utp, ufp, ufn = sess.run([loss_op, acc_op, tp_op, fp_op, fn_op],
                                       feed_dict={X:batch_X, y:batch_y, 
                                                  K.learning_phase():0})
        loss_list.append(loss_value)
        acc_list.append(acc)
        
        tp += utp
        fp += ufp
        fn += ufn

    precision = tp / (tp + fp + 1e-07)
    recall = tp / (tp + fn + 1e-07)
    f1 = 2 * precision * recall / (precision + recall + 1e-07)

    return loss_list, acc_list, f1


# Optimizations
os.environ["KMP_BLOCKTIME"] = '30' # Sleep threads instantly after parallel exec.
os.environ["OMP_NUM_THREADS"]= '6' # Number of physical cores
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Filter out INFO and WARNING logs
config = tf.ConfigProto(inter_op_parallelism_threads=1)
session = tf.Session(config=config)
K.set_session(session)

logsdir = './logs/'
# sys.stdout = open(logsdir+'process.stdout', 'w')
print(time.strftime("%a, %d, %b %Y, %H:%M:%S"))
start_time = time.time()

#Main Settings
maindir = '/home/aurora-cc/robsanpam/FERA17DB-OFT-h224-w224-v6/'
traindir = maindir+"images/train/"
valdir = maindir+"images/valid/"
y_labels_path = maindir+"meta/all_y_labels.p"
K.set_image_data_format("channels_last")
input_size = (224, 224, 3)
num_classes = 5

batch_size = 32
epochs =  30

# Adam
beta1 = 0.5 # default is 0.9
beta2 = 0.999 # default is 0.999

# Weight decay
wdecay = True
wd_apply_to = 'all' #['fc1000', 'logits_last'] # 'all'
wdecay_val = 0.026

# Training
train_all = False
train_sections = 6
weights = 'imagenet' # 'imagenet' or None

# Early stopping
early_stopping = True
stopping_criteria = 4 # Epochs without improvement that stops the program

# Learning rate reduction
lr_reduction = True
lr = 0.001 # This is the starting learning rate
factor = 0.1 # Factor to multiply current learning rate
min_lr = 1e-6 # Minimum value of the learning rate
patience = 2 # Epochs without improvement that triggers a reduction

assert weights in ['imagenet', None]

#load labels
y_dict = pickle.load(open(y_labels_path, "rb"))

print("\nCreating generators...\n")

train_generator, val_generator = create_generators(y_dict, batch_size)
t_steps = train_generator.samples//batch_size
v_steps = val_generator.samples//batch_size

print("\nCreating model...")

X = Input(shape=(input_size[0], input_size[1], input_size[2],)) 
y = tf.placeholder(tf.float32, [None, num_classes], name='y')

with tf.variable_scope('resnet50'):
    resnet = ResNet50(include_top=False
                      ,input_tensor=X
                      ,weights=weights
                      ,pooling='avg' # original implementation uses avg
                      )

# Change activation of last fully connected layer...
#resnet.get_layer('fc1000').activation = get('relu')

newfc1000 = Dense(1000, activation='softmax', name='fc1000')(resnet.output)
output = Dense(num_classes, activation=None, name='logits_last')(newfc1000)
whole_model = Model(resnet.input, output, name='whole_Model')

# Show final model
whole_model.summary()

# Apply wd
if wdecay:
    l2_reg = l2(wdecay_val)
    whole_model = set_regularization(model=whole_model,
                            apply_to=wd_apply_to,
                            kernel_regularizer=l2_reg,
                            bias_regularizer=l2_reg,
                            activity_regularizer=None,
                            beta_regularizer=None,
                            gamma_regularizer=None)

model = model_from_json(whole_model.to_json())
model.set_weights(whole_model.get_weights())
whole_model=None

# Selecting variables to train
train_vars = []
train_layers = []
section_layers = np.array([])

if train_sections <= 6:
    section_layers = np.append(section_layers, ['fc1000'], axis = 0)

if train_sections <= 5:
    section_layers = np.append(section_layers, ['res5a_branch2a', 'bn5a_branch2a', 'activation_41', 
                           'res5a_branch2b', 'bn5a_branch2b', 'activation_42', 
                           'res5a_branch2c', 'res5a_branch1', 'bn5a_branch2c', 
                           'bn5a_branch1', 'add_14', 'activation_43',
                           'res5b_branch2a','bn5b_branch2a', 'activation_44',
                           'res5b_branch2b','bn5b_branch2b', 'activation_45',
                           'res5b_branch2c', 'bn5b_branch2c', 'add_15', 
                           'activation_46', 'res5c_branch2a', 'bn5c_branch2a',
                           'activation_47', 'res5c_branch2b', 'bn5c_branch2b',
                           'activation_48', 'res5c_branch2c', 'bn5c_branch2c',
                           'add_16', 'activation_49', 'global_max_pooling2d_1', 
                           'fc1000'], axis = 0)

if train_sections <= 4:
    section_layers = np.append(section_layers, ['res4a_branch2a', 'bn4a_branch2a', 'activation_23',
                           'res4a_branch2b', 'bn4a_branch2b', 'activation_24',
                           'res4a_branch2c', 'res4a_branch1', 'bn4a_branch2c',
                           'bn4a_branch1', 'add_8', 'activation_25',
                           'res4b_branch2a', 'bn4b_branch2a', 'activation_26',
                           'res4b_branch2b', 'bn4b_branch2b', 'activation_27',
                           'res4b_branch2c', 'bn4b_branch2c', 'add_9',
                           'activation_28', 'res4c_branch2a', 'bn4c_branch2a',
                           'activation_29', 'res4c_branch2b', 'bn4c_branch2b',
                           'activation_30', 'res4c_branch2c', 'bn4c_branch2c',
                           'add_10', 'activation_31', 'res4d_branch2a',
                           'bn4d_branch2a', 'activation_32', 'res4d_branch2b',
                           'bn4d_branch2b', 'activation_33', 'res4d_branch2c',
                           'bn4d_branch2c', 'add_11', 'activation_34',
                           'res4e_branch2a', 'bn4e_branch2a', 'activation_35',
                           'res4e_branch2b', 'bn4e_branch2b', 'activation_36',
                           'res4e_branch2c', 'bn4e_branch2c', 'add_12',
                           'activation_37', 'res4f_branch2a', 'bn4f_branch2a',
                           'activation_38', 'res4f_branch2b', 'bn4f_branch2b',
                           'activation_39', 'res4f_branch2c', 'bn4f_branch2c',
                           'add_13', 'activation_40'], axis =0)

for layer in model.layers:
    if train_all == True:
        train_vars.append(layer.trainable_weights)
        train_layers.append(layer.name)
    elif str(layer.name).split('_')[-1] == 'last' or str(layer.name) in section_layers:
        train_vars.append(layer.trainable_weights)
        train_layers.append(layer.name)

print('\nTraining layers:')
print(train_layers)

logits = model(X)

probabilities = tf.nn.sigmoid(logits)

xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)
loss_op = tf.cast(tf.reduce_mean(xentropy), tf.float32)

lr_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')
opt = tf.train.AdamOptimizer(learning_rate=lr_placeholder, 
                             beta1=beta1, 
                             beta2=beta2)

train_op = opt.minimize(loss_op, var_list=[train_vars])

correct = tf.equal(tf.round(probabilities), y)
acc_op = tf.reduce_mean(tf.cast(correct, tf.float32))

tp_op = tf.reduce_sum(
            tf.transpose(
                        tf.to_float(
                            tf.logical_and(
                                tf.equal(tf.transpose(y), True), 
                                tf.equal(tf.transpose(tf.round(probabilities)), True)
                            )
                        )
            ), 0)

fp_op = tf.reduce_sum(
            tf.transpose(
                        tf.to_float(
                            tf.logical_and(
                                tf.equal(tf.transpose(y), False), 
                                tf.equal(tf.transpose(tf.round(probabilities)), True)
                            )
                        )
            ), 0)

fn_op = tf.reduce_sum(
            tf.transpose(
                        tf.to_float(
                            tf.logical_and(
                                tf.equal(tf.transpose(y), True), 
                                tf.equal(tf.transpose(tf.round(probabilities)), True)
                            )
                        )
            ), 0)

best_f1 = -1.0
logs = np.empty((0,7))

with K.get_session() as sess:
    
    print("\nTraining ...\n")

    for epoch in range(epochs):

        train_loss_list, train_acc_list, train_f1 = run_epoch_on_generator(sess,
                                                                 num_classes,
                                                                 train_generator,
                                                                 batch_size=batch_size,
                                                                 epoch=epoch+1,
                                                                 epochs=epochs,
                                                                 training=True,
                                                                 steps_per_epoch=t_steps,
                                                                 lr=lr)

        valid_loss_list, valid_acc_list, valid_f1 = run_epoch_on_generator(sess,
                                                                 num_classes,
                                                                 val_generator,
                                                                 batch_size=batch_size,
                                                                 epoch=epoch+1,
                                                                 epochs=epochs,
                                                                 training=False,
                                                                 steps_per_epoch=v_steps)

        train_acc = np.array(train_acc_list).mean()
        valid_acc = np.array(valid_acc_list).mean()
        train_loss = np.hstack(train_loss_list).mean()
        valid_loss = np.hstack(valid_loss_list).mean()

        epoch_log = np.array((train_loss,
                              valid_loss,
                              train_acc,
                              valid_acc,
                              train_f1,
                              valid_f1,
                              lr), ndmin=2)
        
        logs = np.vstack((logs, epoch_log))
        
        info_str = "Training Loss: {} \nTraining Accuracy {} \nTraining F1: {} - Individual: {} \nValidation Loss: {} \nValidation accuracy: {} \nValidation F1: {} - Individual: {} \nLearning Rate: {}\n".format(
            train_loss,
            train_acc,
            np.mean(train_f1),
            train_f1,
            valid_loss,
            valid_acc,
            np.mean(valid_f1),
            valid_f1,
            lr)

        try:
            f = open(logsdir+'epoch_logs.pickle', 'wb')
            pickle.dump(logs, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except Exception as e:
            print('Unable to save data to '+logsdir+'epoch_logs.pickle:', e)
            raise

        print(info_str)

        if np.mean(valid_f1) > best_f1:
            best_f1 = np.mean(valid_f1)
            sc = stopping_criteria
            lrp = patience
            # Save the weights
            model.save_weights(logsdir+'model_weights.h5')
            # Save the model architecture
            with open(logsdir+'model_architecture.json', 'w') as f:
                f.write(model.to_json())
            print('New best model saved successfully...\n')

        else:
            if not epochs == epoch: 
                if early_stopping:
                    sc -= 1
                    if sc == 0:
                        print("Stopping early!")
                        break
                if lr_reduction:
                    lrp -= 1
                    if lrp == 0:
                        lrp = patience
                        lr *= factor
                        if lr<min_lr:
                            print('Minimum learning rate', str(lr),'reached')
                            lr = min_lr
                        else:
                            print('Reducing learning rate to', str(lr))
                        print()
            print()

print("\nTotal time: %.1d seconds." % (time.time()-start_time))

print('Done.\n')
