import tensorflow as tf
import cv2  # read in pixel data
import pong  # our class
import numpy as np  # math
import random  # random
from collections import deque  # queue data structure. fast appends. and pops. replay memory

# hyper params
ACTIONS = 3  # up,down, stay
# define our learning rate
GAMMA = 0.99
# for updating our gradient or training over time
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
# how many frames to anneal epsilon
EXPLORE = 500000
OBSERVE = 50000
# store our experiences, the size of it
REPLAY_MEMORY = 800
# batch size to train on

# create tensorflow graph
def createGraph():
    # first convolutional layer. bias vector
    # creates an empty tensor with all elements set to zero with a shape
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

    # input for pixel data
    s = tf.placeholder("float", [None, 84, 84, 4])

    # Computes rectified linear unit activation fucntion on  a 2-D convolution given 4-D input and filter tensors. and
    conv1 = tf.nn.relu(tf.nn.conv2d(s, W_conv1, strides=[1, 4, 4, 1], padding="VALID") + b_conv1)

    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W_conv2, strides=[1, 2, 2, 1], padding="VALID") + b_conv2)

    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W_conv3, strides=[1, 1, 1, 1], padding="VALID") + b_conv3)

    conv3_flat = tf.reshape(conv3, [-1, 3136])

    fc4 = tf.nn.relu(tf.matmul(conv3_flat, W_fc4) + b_fc4)

    fc5 = tf.matmul(fc4, W_fc5) + b_fc5

    return s, fc5


# deep q network. feed in pixel data to graph session
def trainGraph(inp, out, sess):
    game = pong.PongGame()

    # to calculate the argmax, we multiply the predicted output with a vector with one value 1 and rest as 0
    argmax = tf.placeholder("float", [None, ACTIONS])
    gt = tf.placeholder("float", [None])  # ground truth

    # action
    prob = tf.nn.softmax(out)
    action = tf.reduce_sum(tf.multiply(prob, argmax), reduction_indices=1)
    # cost function we will reduce through backpropagation
    cost = tf.reduce_mean(tf.multiply(action, gt)) #tf.square(action - gt))
    # optimization fucntion to reduce our minimize our cost function
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # initialize our game


    # create a queue for experience replay to store policies
    Point = deque()
    D = deque()

    # intial frame
    frame = game.getPresentFrame()
    # convert rgb to gray scale for processing
    frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
    # binary colors, black or white
    ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    # stack frames, that is our input tensor
    inp_t = np.stack((frame, frame, frame, frame), axis=2)

    # saver
    saver = tf.train.Saver()

    sess.run(tf.initialize_all_variables())

    t = 0
    epsilon = INITIAL_EPSILON

    # training time
    while (1):
        # output tensor
        prob_eval, out_eval = sess.run([prob, out], feed_dict={inp: [inp_t]})
        prob_t = prob_eval[0]
        out_t = out_eval[0]
        # argmax function
        argmax_t = np.zeros([ACTIONS])

        #
   #     maxIndex = np.argmax(out_t[0])
 #       if (random.random() > prob_t[maxIndex]):
#            maxIndex = random.randrange(ACTIONS)

    #    argmax_t[maxIndex] = 1

        if (random.random() <= epsilon):
            maxIndex = random.randrange(ACTIONS)
        else:
            maxIndex = np.argmax(out_t)
        argmax_t[maxIndex] = 1

        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # reward tensor if score is positive
        reward_t, frame = game.getNextFrame(argmax_t)
        # get frame pixel data
        frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
        ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        frame = np.reshape(frame, (84, 84, 1))
        # new input tensor
        inp_t1 = np.append(frame, inp_t[:, :, 0:3], axis=2)
        win_count = 0
        # add our input tensor, argmax tensor, reward and updated input tensor tos tack of experiences
        if (reward_t == 0):

            Point.append([inp_t, argmax_t, reward_t, inp_t1])
            if (len(Point) > 300):
                Point.popleft()
        else:
            reward_array = np.zeros(shape=[len(Point) +1])
            #filling values in the reward array
            for i in reversed(xrange(0, len(Point)+1)):
                if (i == len(Point)):
                    Point.append([inp_t, argmax_t, reward_t, inp_t1])

                    reward_array[i] = reward_t



                else:

                    reward_array[i] = reward_array[i+1] * GAMMA + reward_array[i]
            #adding reward array values back into frames and putting them into frame queue
            reward_array -= np.mean(reward_array)
            reward_array /= np.std(reward_array)

            print("how long did the point last?     ", len(Point))
            for i in range(len(reward_array)):
                QueueTransfer = Point.popleft()
                QueueTransfer[2] = reward_array[i]
                D.append(QueueTransfer)
            if reward_t == 1:
                win_count += 1


        # if we run out of replay memory, make room
        if (len(D) > REPLAY_MEMORY):
            for i in range(0, len(D) - REPLAY_MEMORY):
                D.popleft()

        # training iteration
        if (len(D) ==REPLAY_MEMORY):
            print("training")


            # get values from our replay memory
            np.random.shuffle(D)
            inp_batch = []
            argmax_batch = []
            reward_batch = []
            inp_t1_batch = []
            for i in range(len(D)):
                temp = D.pop()
                inp_batch.append(temp[0])
                argmax_batch.append(temp[1])
                reward_batch.append(temp[2])
                inp_t1_batch.append(temp[3])


            c1, a1, o1 = sess.run([cost, action, out],
                                  feed_dict={gt: [reward_batch[-1]],
                                             argmax: [argmax_batch[-1]],
                                             inp: [inp_batch[-1]]})

            # train on that
            train_step.run(feed_dict={
                gt: reward_batch,
                argmax: argmax_batch,
                inp: inp_batch
            })

        # update our input tensor the the next frame
        inp_t = inp_t1
        t = t + 1

        # print our where wer are after saving where we are
        if t % 10000 == 0:
            saver.save(sess, './' + 'pong' + '-dqn', global_step=t)

        print(
        "TIMESTEP", t, "/ ACTION", maxIndex, "Epsilon', ", epsilon, "/ REWARD", reward_t, "/ Wins ", win_count, "/ Q_MAX %e" % np.max(out_t))


def main():
    # create session
    sess = tf.InteractiveSession()
    # input layer and output layer by creating graph
    inp, out = createGraph()
    # train our graph on input and output with session variables
    trainGraph(inp, out, sess)


if __name__ == "__main__":
    main()
