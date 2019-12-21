# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DQNAgent:
    def __init__(self, state_size, action_size, model=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=3000)
        self.gamma = 0.95    
        self.epsilon = 0.01 
        self.model = model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
    
    def memory_clear(self):
        self.memory.clear()

    def act_greedy(self, state):
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
Task_1_avg_scores = []
Task_2_avg_scores = []
#
def train_agent_in_env(agent, env, train_steps = 2000, distortion = 0, enable = True):
    agent.memory_clear()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    done = False
    batch_size = 32
    state = env.reset() + distortion
    state = np.reshape(state, [1, state_size]) # Dist
    score = 0
    for time in range(train_steps):
            if (enable):
            env2 = gym.make('CartPole-v1')
            Task_1_avg_scores.append(test_agent_in_env(agent, env2, test_episodes = 10, distortion = [-2.4, -1, -0.20943951, -1]))
            Task_2_avg_scores.append(test_agent_in_env(agent, env2, test_episodes = 10, distortion = [+2.4, +1, +0.20943951, +1]))
        #env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        score = score + reward
        next_state = next_state + distortion
        next_state = np.reshape(next_state , [1, state_size]) # Dist
        agent.remember(state, action, reward, next_state, done)
        state = next_state 
        if done:
            print(time)            
            score = 0
            state = env.reset() + distortion
            state = np.reshape(state, [1, state_size]) # Dist
                  
        if len(agent.memory) > batch_size:
            
            
            agent.replay(batch_size)

def test_agent_in_env(agent, env, test_episodes = 1000, distortion = 0):
    
    score = 0
    scores = []
    for e in range(test_episodes):
        state = env.reset() + distortion
        state = np.reshape(state, [1, state_size])
        while True:            
            #env.render()
            action = agent.act_greedy(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])	
            state = next_state + distortion
            score = score + reward
            if done:
                scores.append(score)            
                score = 0            
                break
    return np.average(scores)


###  ANN & ENV  ###   
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

model = Sequential()
model.add(Dense(48, input_dim=state_size, activation='relu'))
model.add(Dense(48, activation='relu'))
model.add(Dense(action_size)
model.compile(loss='mse', optimizer=Adam(lr=0.001))


### TEST ###
agent = DQNAgent(state_size, action_size, model)
train_agent_in_env(agent, env, train_steps = 2000, distortion = [-2.4, -1, -0.20943951, -1])
print("Avg Score in Task 1: " + str(test_agent_in_env(agent, env, distortion = [-2.4, -1, -0.20943951, -1])))

train_agent_in_env(agent, env, train_steps = 2000, distortion = [+2.4, +1, +0.20943951, +1])
print("Avg Score in Task 2: " + str(test_agent_in_env(agent, env, distortion = [+2.4, +2, +0.20943951, +2], test_episodes = 50)))
print("Avg Score in Task 1: " + str(test_agent_in_env(agent, env, distortion = [-2.4, -1, -0.20943951, -1], test_episodes = 50)))


Task_1_avg_100 = []
Task_2_avg_100 = []
for i in range(20):
        Task_1_avg_100.append(np.average(Task_1_avg_scores[2000:][i*100:i*100+100]))
	Task_2_avg_100.append(np.average(Task_2_avg_scores[2000:][i*100:i*100+100]))
	
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(range(20), Task_1_avg_100, 'g-')
ax2.plot(range(20), Task_2_avg_100, 'b-')
plt.show()


