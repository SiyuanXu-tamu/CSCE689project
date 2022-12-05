
import os, sys
sys.path.append('../game')
sys.path.append('../utils')

from collections import defaultdict
import json
import random
import numpy as np

import gym
from TemplateAgent import FlappyBirdAgent
from FlappyBirdGame import FlappyBirdNormal
import numpy as np

import warnings
warnings.filterwarnings('ignore')
        

class QLearningAgent(FlappyBirdAgent):
    ''' Q-Learning Agent. '''
    
    def __init__(self, actions, probFlap = 0.5, rounding = None):
        
        super().__init__(actions)
        self.soft_epsilon = probFlap
        self.qValues = defaultdict(float)
        self.env = FlappyBirdNormal(gym.make('FlappyBird-v0'), rounding = rounding)
        print('you choose Q-larning agent')

    def find_the_action(self, state):
        def random_choose_Act():
            if random.random() < self.soft_epsilon:
                return 0
            return 1
            
        if random.random() < self.epsilon:
            return random_choose_Act()
            
        qValue = [self.qValues.get((state, action), 0) for action in self.actions]
        ###action space 0 or 1    
        if qValue[0] < qValue[1]:
            return 1
        elif qValue[0] > qValue[1]:
            return 0
        else:
            return random_choose_Act()
            
    def save_QValues(self):
        toSave = {key[0] + ' action ' + str(key[1]) : self.qValues[key] for key in self.qValues}
        with open('qValues.json', 'w') as fp:
            json.dump(toSave, fp)
            
    def load_QValues(self):
        def parseKey(key):
            state = key[:-9]
            action = int(key[-1])
            return (state, action)

        with open('qValues.json') as fp:
            toLoad = json.load(fp)
            self.qValues = {parseKey(key) : toLoad[key] for key in toLoad}

    def train(self, order = 'forward', numIters = 20000, epsilon = 0.1, discount = 1,
              eta = 0.9, epsilonDecay = False, etaDecay = False, evalPerIters = 250,
              numItersEval = 1000):
        '''
        Trains the agent.
        '''
        self.epsilon = epsilon
        self.initialEpsilon = epsilon
        self.discount = discount
        self.eta = eta
        self.epsilonDecay = epsilonDecay
        self.etaDecay = etaDecay
        self.evalPerIters = evalPerIters
        self.numItersEval = numItersEval
        self.env.seed(random.randint(0, 100))

        done = False
        maxScore = 0
        maxReward = 0
        
        for i in range(numIters):
            if i % 50 == 0 or i == numIters - 1:###print per 50 epoch
                print("Iter: ", i)
            
            self.epsilon = self.initialEpsilon / (i + 1) if self.epsilonDecay \
                           else self.initialEpsilon
            score = 0
            totalReward = 0
            _ = self.env.reset()
            gameIter = []
            current_state = self.env.getGameState()
            
            while True:####create a training episode
                action = self.find_the_action(current_state)
                nextState, reward, done, _ = self.env.step(action)
                gameIter.append((current_state, action, reward, nextState))
                current_state = nextState
                #self.env.render()  # Uncomment it to display graphics.
                totalReward += reward
                if reward >= 1:
                    score += 1
                if done:
                    break
            
            if score > maxScore: 
                maxScore = score
            if totalReward > maxReward: 
                maxReward = totalReward
            
            if order == 'forward':####foward Q-learning 
                for (state, action, reward, nextState) in gameIter:
                    self.updateQ(state, action, reward, nextState)
            else:###order == 'backward' backward Q-learning 
                for (state, action, reward, nextState) in gameIter[::-1]:##reverse
                    self.updateQ(state, action, reward, nextState)
                
            if self.etaDecay:
                self.eta = (i + 1) / (i + 2) * self.eta
            
            if (i + 1) % self.evalPerIters == 0:
                output = self.test(numIters = self.numItersEval)
                self.save_result(output, i + 1)
                self.save_QValues()
        ####end training        
        self.env.close()
        print("Train Max Score: ", maxScore)
        print("Train Max Reward: ", maxReward)
        print('')
    
    def test(self, numIters = 20000):
        self.epsilon = 0
        self.env.seed(0)

        done = False
        maxScore = 0
        maxReward = 0
        output = defaultdict(int)
        
        for i in range(numIters):
            score = 0
            totalReward = 0
            _ = self.env.reset()
            state = self.env.getGameState()
            
            while True:
                action = self.find_the_action(state)
                state, reward, done, _ = self.env.step(action)
                totalReward += reward
                if reward >= 1: ####number if positive reward in a test
                    score += 1
                if done:
                    break
                    
            output[score] += 1
            if score > maxScore: maxScore = score
            if totalReward > maxReward: maxReward = totalReward
    
        self.env.close()
        print("Max Score Test: ", maxScore)
        print("Max Reward Test: ", maxReward)
        print()
        return output
            
    def updateQ(self, state, action, reward, nextState):#Updates the Q-values based on an observation.
        nextQValues = [self.qValues.get((nextState, nextAction), 0) for nextAction in self.actions]
        nextValue = max(nextQValues)
        self.qValues[(state, action)] = (1 - self.eta) * self.qValues.get((state, action), 0) \
                                        + self.eta * (reward + self.discount * nextValue)
        
    def save_result(self, output, iter):##Saves the scores.
        if not os.path.isdir('scores'):####path to your results file
            os.mkdir('scores')
        with open('./scores/scores_{}.json'.format(iter), 'w') as fp:
            json.dump(output, fp)
