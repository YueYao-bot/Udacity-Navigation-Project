from unityagents import UnityEnvironment
import numpy as np
import torch
import matplotlib.pyplot as plt
from dqn_agent import Agent

# please do not modify the line below
env = UnityEnvironment(file_name="./Banana")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)





dqn_agent = Agent(state_size=len(state), action_size=brain.vector_action_space_size, seed=0)

scores = []                                          # initialize the score
eps, eps_decay, eps_min = 1.0, 0.99, 0.01
train_mode = False

if __name__ == "__main__":
    if train_mode:
        for i_eposide in range(2000):
            env_info = env.reset(train_mode=True)[brain_name]
            state = env_info.vector_observations[0]
            eposide_score = 0
            while True:
                action = dqn_agent.act(state, eps = eps )
                env_info = env.step(action)[brain_name]
                next_state, reward, done = env_info.vector_observations[0], env_info.rewards[0], env_info.local_done[0]
                dqn_agent.step(state, action, reward, next_state, done)
                state = next_state
                eposide_score += reward
                if done:
                    break

            scores.append(eposide_score)
            eps = max(eps_min, eps*eps_decay)

            if len(scores) > 100:
                print('\rEpisode {}\t Score: {:.2f} \t Avg Score: {:.2f}'.format(i_eposide, eposide_score, np.mean(scores[-100:])))

            if len(scores)>100 and np.mean(scores[-100:]) >= 13.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score over last 100 Episodes: {:.2f}'.format(i_eposide, np.mean(scores[-100:])))
                torch.save(dqn_agent.qnetwork_local.state_dict(), 'checkpoint.pth')
                plt.figure()
                plt.plot(range(i_eposide+1), scores)
                plt.xlabel("Episode")
                plt.ylabel("Score")
                plt.show()

                break
    else:
        dqn_agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
        env_info = env.reset(train_mode=train_mode)[brain_name]
        state = env_info.vector_observations[0]
        score = 0  # initialize the score
        while True:
            action = dqn_agent.act(state, eps=0.)
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            score += reward  # update the score
            state = next_state  # roll over the state to next time step
            if done:  # exit loop if episode finished
                print("Score: {:.2f}".format(score))
                break

