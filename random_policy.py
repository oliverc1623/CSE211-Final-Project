import compiler_gym
from compiler_gym import wrappers
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

env = compiler_gym.make(
    "llvm-v0",
    benchmark="benchmark://cbench-v1/dijkstra",
    observation_space="Autophase",
    reward_space="IrInstructionCountOz",
)
env = wrappers.TimeLimit(env, 45)

observation = env.reset()

scores = []
for _ in range(1000): 
    observation, reward, done, info = env.step(env.action_space.sample()) # User selects action 
    scores.append(reward)
    if done:    
        env.reset()

plt.plot(scores)
plt.show()