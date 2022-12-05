'''
This file is to implement the SARSA algorithm and build an agent.
'''

import os, sys
sys.path.append('../game')
sys.path.append('../utils')


from agents.TemplateAgent import FlappyBirdAgent
from game.FlappyBirdGame import FlappyBirdNormal
import json
import random
import gym
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class SARSAAgent(FlappyBirdAgent):
    '''This class is to implement of the SARSA Agent and contains several functions. '''

    def __init__(self, actions, probFlap = 0.5, rounding = None):
        '''
        This function is to initialize the SARSA agent and set the environment.

        Args:
            actions : the action set which conatins 1 and 0
            qValus: The Q value of the algorithm
            probFlap : The probability to take the jump action
            rounding : The level to be discretized.
        '''
        super().__init__(actions)
        self.probFlap = probFlap
        self.qValues = defaultdict(float)
        self.env = FlappyBirdNormal(gym.make('FlappyBird-v0'), rounding = rounding)

    def act(self, state):
        '''
         This function randomly returns the next actions for the current state.

        Args:
            state : The state right now

        Returns:
            the action from actions: 0 or 1.
        '''
        def randomAct():
            if random.random() < self.probFlap:
                return 0
            return 1

        if random.random() < self.epsilon:
            return randomAct()

        qValues = [self.qValues.get((state, actions), 0) for actions in self.actions]

        if qValues[0] < qValues[1]:
            return 1
        elif qValues[0] > qValues[1]:
            return 0
        else:
            return randomAct()

    def saveQValues(self):
        '''
        This function is to save the Q-value from the training.

        Args:
            qSaved: to save the q value

        Returns:
            none

        '''
        qSaved = {key[0] + ' actions ' + str(key[1]) : self.qValues[key] for key in self.qValues}
        with open('qValues.json', 'w') as fp:
            json.dump(qSaved, fp)

    def loadQValues(self):
        '''
        This function is to load the Q-value from the local.

        Args:
            qLoaded: to load the q value from local

        Returns:
            none
        '''
        def parseKey(key):
            state = key[:-9]
            actions = int(key[-1])
            return (state, actions)

        with open('qValues.json') as fp:
            qLoaded = json.load(fp)
            self.qValues = {parseKey(key) : qLoaded[key] for key in qLoaded}

    def train(self, order = 'forward', numIters = 20000, epsilon = 0.1, discount = 1,
              eta = 0.9, epsilonDecay = False, etaDecay = False, evalPerIters = 250,
              numItersEval = 1000):
        '''
        This function is to train this SARSA agent.

        Args:
            order: The order of updates, 'forward' or 'backward'.
            numIters: The number of training iterations.
            epsilon: The epsilon value of the algorithm.
            discount: The discount factor of the alogorithm.
            eta: The eta value of the algorithm.
            epsilonDecay: Whether to use epsilon decay.
            etaDecay: Whether to use eta decay.
            evalPerIters: The number of iterations between two evaluation calls.
            numItersEval: The number of evaluation iterations.
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
            if i % 50 == 0 or i == numIters - 1:
                print("Iter: ", i)

            self.epsilon = self.initialEpsilon / (i + 1) if self.epsilonDecay \
                           else self.initialEpsilon
            score = 0
            totalReward = 0
            ob = self.env.reset()
            gameIter = []
            state = self.env.getGameState()
            actions = self.act(state)

            while True:
                nextState, reward, done, _ = self.env.step(actions)
                nextactions = self.act(nextState)
                gameIter.append((state, actions, reward, nextState, nextactions))
                state = nextState
                actions = nextactions
                self.env.render()
                totalReward += reward
                if reward >= 1:
                    score += 1
                if done:
                    break

            if score > maxScore: maxScore = score
            if totalReward > maxReward: maxReward = totalReward

            if order == 'forward':
                for (state, actions, reward, nextState, nextactions) in gameIter:
                    self.updateQ(state, actions, reward, nextState, nextactions)
            else:
                for (state, actions, reward, nextState, nextactions) in gameIter[::-1]:
                    self.updateQ(state, actions, reward, nextState, nextactions)

            if self.etaDecay:
                self.eta *= (i + 1) / (i + 2)

            if (i + 1) % self.evalPerIters == 0:
                output = self.test(numIters = self.numItersEval)
                self.saveOutput(output, i + 1)
                self.saveQValues()

        self.env.close()
        print("Max Score Train: ", maxScore)
        print("Max Reward Train: ", maxReward)
        print()

    def test(self, numIters = 20000):
        '''
        This function is to evaluate the agent.

        Args:
            numIters: The number of evaluation iterations.

        Returns:
            dict: A set of scores got.
        '''
        self.epsilon = 0
        self.env.seed(0)

        done = False
        maxScore = 0
        maxReward = 0
        output = defaultdict(int)

        for i in range(numIters):
            score = 0
            totalReward = 0
            ob = self.env.reset()
            state = self.env.getGameState()

            while True:
                actions = self.act(state)
                state, reward, done, _ = self.env.step(actions)
#                self.env.render()  # Uncomment it to display graphics.
                totalReward += reward
                if reward >= 1:
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

    def updateQ(self, state, actions, reward, nextState, nextactions):
        '''
        This function is to update the Q-values based on the observation.

        Args:
            state, nextState: Two states.
            actions, nextactions: 0 or 1.
            reward: The reward value.

        Returns:
            None.

        '''
        self.qValues[(state, actions)] = (1 - self.eta) * self.qValues.get((state, actions), 0) \
                                        + self.eta * (reward + self.discount * self.qValues.get((nextState, nextactions), 0))

    def saveOutput(self, output, iter):
        '''
        This function works to save all the scores of the game.

        Args:
            output (dict): A set of scores.
            iter (int): Current iteration.

        Returns:
            None.
        '''
        if not os.path.isdir('scores'):
            os.mkdir('scores')
        with open('./scores/scores_{}.json'.format(iter), 'w') as fp:
            json.dump(output, fp)
