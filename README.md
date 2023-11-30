# CSE211-Final-Project

Github repo for my final project for CSE211, Compiler Design. 


## DQN on Autophase LLVM

### Reward space

#### Runtime

If I want to use runtime use this wrapper
```{Python}
import compiler_gym
from compiler_gym import wrappers

env = compiler_gym.make(
    "llvm-v0",
    benchmark="benchmark://cbench-v1/dijkstra",
    observation_space="InstCount"
)
env = wrappers.TimeLimit(env, 45)
env = wrappers.RuntimePointEstimateReward(env)
```