import tensorflow as tf
import cv2 #read in pixel data
import pong #our class
import numpy as np #math
import random #random 
from collections import deque #queue data structure. fast appends. and pops. replay memory



#hyper params
ACTIONS = 3 #up,down, stay
#define our learning rate
GAMMA = 0.99
#for updating our gradient or training over time
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
#how many frames to anneal epsilon
EXPLORE = 100
OBSERVE = 10
#store our experiences, the size of it
REPLAY_MEMORY = 100
#batch size to train on
BATCH = 30

#create tensorflow graph
def createGraph():

    #first convolutional layer. bias vector
    #creates an empty tensor with all elements set to zero with a shape
    W_conv1 = tf.Variable(tf.truncated_normal(shape=[8, 8, 4, 32], stddev=0.1))
    b_conv1 = tf.Variable(tf.constant(value=0.1, shape=[32]))

    W_conv2 = tf.Variable(tf.truncated_normal(shape=[4, 4, 32, 64], stddev=0.1))
    b_conv2 = tf.Variable(tf.constant(shape=[64], value=0.1))

    W_conv3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=0.1))
    b_conv3 = tf.Variable(tf.constant(shape=[64], value=0.1))

    W_fc4 = tf.Variable(tf.truncated_normal(shape=[3136, 784], stddev=0.1))
    b_fc4 = tf.Variable(tf.constant(shape=[784], value=0.1))

    W_fc5 = tf.Variable(tf.truncated_normal(shape=[784, ACTIONS], stddev=0.1))
    b_fc5 = tf.Variable(tf.constant(shape=[ACTIONS], value=0.1))

    #input for pixel data
    s = tf.placeholder("float", [None, 84, 84, 4])


    #Computes rectified linear unit activation fucntion on  a 2-D convolution given 4-D input and filter tensors. and 
    conv1 = tf.nn.relu(tf.nn.conv2d(s, W_conv1, strides = [1, 4, 4, 1], padding = "VALID") + b_conv1)

    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W_conv2, strides = [1, 2, 2, 1], padding = "VALID") + b_conv2)

    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W_conv3, strides = [1, 1, 1, 1], padding = "VALID") + b_conv3)

    conv3_flat = tf.reshape(conv3, [-1, 3136])

    fc4 = tf.nn.relu(tf.matmul(conv3_flat, W_fc4) + b_fc4)
    
    fc5 = tf.matmul(fc4, W_fc5) + b_fc5

    return s, fc5


#deep q network. feed in pixel data to graph session 
def trainGraph(inp, out, sess):

    #to calculate the argmax, we multiply the predicted output with a vector with one value 1 and rest as 0
    argmax = tf.placeholder("float", [None, ACTIONS]) 
    gt = tf.placeholder("float", [None]) #ground truth

    #action
    action = tf.reduce_sum(tf.multiply(out, argmax), reduction_indices = 1)
    #cost function we will reduce through backpropagation
    cost = tf.reduce_mean(tf.square(action - gt))
    #optimization fucntion to reduce our minimize our cost function 
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    #initialize our game
    game = pong.PongGame()
    
    #create a queue for experience replay to store policies
    Point = deque()
    D = deque()

    #intial frame
    frame = game.getPresentFrame()
    #convert rgb to gray scale for processing
    frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
    #binary colors, black or white
    ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    #stack frames, that is our input tensor
    inp_t = np.stack((frame, frame, frame, frame), axis = 2)

    #saver
    saver = tf.train.Saver()

    sess.run(tf.initialize_all_variables())

    t = 0
    epsilon = INITIAL_EPSILON
    
    #training time
    while(1):
        #output tensor
        out_t = out.eval(feed_dict = {inp : [inp_t]})[0]
        #argmax function
        argmax_t = np.zeros([ACTIONS])

        #
        if(random.random() <= epsilon):
            maxIndex = random.randrange(ACTIONS)
        else:
            maxIndex = np.argmax(out_t)
        argmax_t[maxIndex] = 1
        
        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        #reward tensor if score is positive
        reward_t, frame = game.getNextFrame(argmax_t)
        #get frame pixel data
        frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
        ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        frame = np.reshape(frame, (84, 84, 1))
        #new input tensor
        inp_t1 = np.append(frame, inp_t[:, :, 0:3], axis = 2)
        
        #add our input tensor, argmax tensor, reward and updated input tensor tos tack of experiences
        if (reward_t ==0):
            Point.append((inp_t, argmax_t, reward_t, inp_t1))
        else:
            framearray =np.empty(shape=(len(Point)+1), dtype=object)
            for i in range(len(Point) +1):
                if (i == 0):
                    framearray[i] = (inp_t, argmax_t, reward_t, inp_t1)
                else:
                    temp  = Point.pop()
                    framearray[i] = temp
            D.append(framearray)




        #if we run out of replay memory, make room
        if len(D) > REPLAY_MEMORY:
            D.popleft()
        
        #training iteration
        if len(D) > 0:
            if (BATCH > len(D)):
                currentBATCH = len(D)

            else:
                currentBATCH = BATCH

            #get values from our replay memory
            minibatch = random.sample(D, currentBATCH)
            inp_batch = []
            argmax_batch = []
            reward_batch = []
            inp_t1_batch = []
            for i in range(len(minibatch)):
                for j in range(len(minibatch[i])):
                    inp_batch.append(minibatch[i][j])
                    argmax_batch.append(minibatch[i][j])
                    reward_batch.append(minibatch[i][j])
                    inp_t1_batch.append(minibatch[i][j])


        
            gt_batch = []
            out_batch = out.eval(feed_dict={inp: inp_t1_batch})
            
            #add values to our batch
            for i in range(0, len(minibatch)):
                    gt_batch.append(reward_batch[i] + GAMMA * np.max(out_batch[i]))



            #train on that 
            train_step.run(feed_dict = {
                           gt : gt_batch,
                           argmax : argmax_batch,
                           inp : inp_batch
                           })
        
        #update our input tensor the the next frame
        inp_t = inp_t1
        t = t+1

        #print our where wer are after saving where we are
        if t % 10000 == 0:
            saver.save(sess, './' + 'pong' + '-dqn', global_step = t)

        print("TIMESTEP", t, "/ EPSILON", epsilon, "/ ACTION", maxIndex, "/ REWARD", reward_t, "/ Q_MAX %e" % np.max(out_t))


def main():
    #create session
    sess = tf.InteractiveSession()
    #input layer and output layer by creating graph
    inp, out = createGraph()
    #train our graph on input and output with session variables
    trainGraph(inp, out, sess)

if __name__ == "__main__":
    main()
