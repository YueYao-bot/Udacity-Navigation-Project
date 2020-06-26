# Project1: Navigation


## Project Details


For this project, an agent navigates (and collect bananas!) in a large, square world.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

- 0 - move forward.
- 1 - move backward.
- 2 - turn left.
- 3 - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

Tested with python 3.6.0 with
- pytorch 1.4.0
- unityagents 0.4.0

## Getting Started
The required environment *Banana.app* is uploaded in this project. You can simply start the project without any additional work.



## Instructions
*Navigation.ipynb* is the entry of the project and has two modes. One for training and one for testing. Be sure that you have q-network "checkpoint.pth". Otherwise you need train the DQN-Agent firstly. 

You also use *Navigation.py* if you do not want to use jupyter notebook. It has same code as in *Navigation.ipynb*.

*checkpoint.pth* saves the weights of a trained model, you can load it and start the test mode!