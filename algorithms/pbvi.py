import numpy as np


class Pomdp:
  def __init__(self, states=None, observations=None, actions=None, transitions=None, observe=None, reward=None, discount=None):
    self.states = states
    self.observations = observations
    self.actions = actions
    self.transitions = transitions
    self.observe = observe
    self.reward = reward
    self.discount = discount



class PBVI:
  def __init__(self, pomdp):
    self.pomdp = pomdp
    # Set of maintained beliefs
    self.B = []
    self.values = [0]

  def 

  def backup(self):
    action_belief_terms = dict()
    for a in self.pomdp.actions:
      observation_terms = dict()
      for o in self.pomdp.observations:
        observation_terms[o] = [self.pomdp.discount * np.dot(self.pomdp.transitions[:,a,:] * self.pomdp.observe[a,:,o], v) for v in self.values]
      for i, b in enumerate(self.beliefs):
        action_belief_terms[(a, i)] = self.pomdp.reward[:,a]
        for o in self.pomdp.observations:
          action_belief_terms[(a, i)] += max([np.dot(w, b) for w in observation_terms[o]])
    new_values = []
    for i, b in enumerate(self.beliefs):
      f = lambda x : np.dot(x, b)
      new_values.append(max([action_belief_terms[(a, i)] for a in self.pomdp.actions], key=f))
    self.values = new_values

    def belief_expansion(self):
      for belief in self.B:
        # get next belief given aciton and observation
        ob = [self.pomdp.simulate_action(a, belief) for a in self.pomdp.action_space]
        obs = self.pomdp.observe[ob]
        next_belief = belief * self.pomdp.transition
        next_belief = next_belief * obs

        # normalize probability
        next_belief = next_belief / np.sum(next_belief, axis=0)

        # add farthest away belief
        l1_dists    = np.linalg.norm(b_s[:,None] - B_, ord=1, axis=2)
    

if __name__ == "__main__":
  # pomdp = Pomdp(states=np.array([0, 1]),
  #               observations=np.array([0, 1]),
  #               actions=np.array([0, 1]),
  #               transitions=np.array([[[.1, .2], [.3, .4]],
  #                                     [[.9, .8], [.7, .6]]]),
  #               observe=np.array([[[.7, .4], [.1, .2]],
  #                                 [[.3, .6], [.9, .8]]]),
  #               reward=np.array([[1, 0], [.2, .8]]),
  #               discount=0.9)
  pomdp = Tag(env_model=True)
  solver = Solver(pomdp)
  print solver.values
  solver.backup()
  print solver.values



