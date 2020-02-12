# pomdp

Trying to reproduce the results from the paper *Point-based value iteration: An anytime algorithm for POMDPs* by Joelle Pineau, Geoff Gordon and Sebastian Thrun (2003) on the Tag environment. 

## Algorithm
The PBVI algorithm alternates between belief expansion in order to 

## Environment
See my implementation on pomdp environment on gym at:

The Tag environment has 870 states, 30 observations and 5 actions. 
It consists of an agent (a robot) which must track an opponent (a person) and tag this person. The partial observability arises from the fact that the robot does not have access to the person's location unless they are in the same cell.
Of the 5 actions, 4 are motion action and result in a -1 reward, the last is the tag action and results in a reward of -10 if the person is not in the cell, 10 otherwise. 
The motion actions of the agent are deterministic. 
At each timestep, the opponent moves away from the agent with probability 0.8 and stays in the same cell with probability 0.2.


## Expected results
The number of maintained beliefs should be around 1334 after 13 iterations of the algorithm.
The agent reaches the terminal state 59% of the time. 
The average reward is -9.180.
The agent reaches the target in 
More can be found in the original paper.

## Obtained results
Work in progress.
