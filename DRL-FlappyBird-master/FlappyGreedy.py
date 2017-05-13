# -------------------------
# Project: Deep Q-Learning on Flappy Bird
# Author: Flood Sung
# Date: 2016.3.21
# -------------------------

import cv2
import sys
import tensorflow as tf
sys.path.append("game/")
import wrapped_flappy_bird as game
import numpy as np
class BrainDQN:

  def __init__(self,actions):

    self.actions = actions
    # init Q network
    self.stateInput,self.QValue,self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2 = self.createQNetwork()

    # saving and loading networks
    self.saver = tf.train.Saver()
    self.session = tf.InteractiveSession()
    self.session.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        self.saver.restore(self.session, checkpoint.model_checkpoint_path)
        print "Successfully loaded:", checkpoint.model_checkpoint_path
    else:
        print "Could not find old network weights"
  def createQNetwork(self):
    # network weights
    W_conv1 = self.weight_variable([8,8,4,32])
    b_conv1 = self.bias_variable([32])

    W_conv2 = self.weight_variable([4,4,32,64])
    b_conv2 = self.bias_variable([64])

    W_conv3 = self.weight_variable([3,3,64,64])
    b_conv3 = self.bias_variable([64])

    W_fc1 = self.weight_variable([1600,512])
    b_fc1 = self.bias_variable([512])

    W_fc2 = self.weight_variable([512,self.actions])
    b_fc2 = self.bias_variable([self.actions])

    # input layer

    stateInput = tf.placeholder("float",[None,80,80,4])

    # hidden layers
    h_conv1 = tf.nn.relu(self.conv2d(stateInput,W_conv1,4) + b_conv1)
    h_pool1 = self.max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(self.conv2d(h_pool1,W_conv2,2) + b_conv2)

    h_conv3 = tf.nn.relu(self.conv2d(h_conv2,W_conv3,1) + b_conv3)

    h_conv3_flat = tf.reshape(h_conv3,[-1,1600])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1) + b_fc1)

    # Q Value layer
    QValue = tf.matmul(h_fc1,W_fc2) + b_fc2

    return stateInput,QValue,W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2
  def getAction(self):
    QValue = self.QValue.eval(feed_dict= {self.stateInput:[self.currentState]})[0]
    action = np.zeros(self.actions)
    action_index = 0
    action_index = np.argmax(QValue)
    action[action_index] = 1
    return action
  def setPerception(self,nextObservation,action,reward,terminal):
    #newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
    newState = np.append(self.currentState[:,:,1:],nextObservation,axis = 2)
    self.currentState = newState
    self.trainQNetwork()

  def trainQNetwork(self):
    currentStateT = np.reshape(self.currentState, (1,80,80,4))
    self.QValue.eval(feed_dict={self.stateInput:currentStateT})

  def setInitState(self,observation):
    self.currentState = np.stack((observation, observation, observation, observation), axis = 2)

  def weight_variable(self,shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

  def bias_variable(self,shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

  def conv2d(self,x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

  def max_pool_2x2(self,x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

# preprocess raw image to 80*80 gray image
def preprocess(observation):
  observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
  ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
  return np.reshape(observation,(80,80,1))

def playFlappyBird():
  # Step 1: init BrainDQN
  actions = 2
  brain = BrainDQN(actions)
  # Step 2: init Flappy Bird Game
  flappyBird = game.GameState()
  # Step 3: play game
  # Step 3.1: obtain init state
  action0 = np.array([1,0])  # do nothing
  observation0, reward0, terminal = flappyBird.frame_step(action0)
  observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
  ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
  brain.setInitState(observation0)

  # Step 3.2: run the game
  while 1!= 0:
    action = brain.getAction()
    nextObservation,reward,terminal = flappyBird.frame_step(action)
    nextObservation = preprocess(nextObservation)
    brain.setPerception(nextObservation,action,reward,terminal)

def main():
  playFlappyBird()

if __name__ == '__main__':
  main()