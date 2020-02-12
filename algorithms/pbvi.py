import numpy as np
import gym
import gym_pomdp
import logging

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
    assert self.pomdp.track_belief, 'PBVI requires computation of the belief'
    self.pomdp.reset() # Reset to get initial belief
    self.B = self.pomdp.belief.reshape(1,-1)
    self.values = np.zeros((1,self.pomdp.state_space.n))

  def value_backup(self):
    gamma_a_b = np.zeros((self.pomdp.action_space.n, self.B.shape[0], self.pomdp.state_space.n))
    for a in range(self.pomdp.action_space.n):
      gamma_a_o = np.array([self.pomdp.discount * np.matmul(self.pomdp.observe[:,:,a], self.pomdp.transition[:,a,:].T*v) for v in self.values])
      
      alpha_index = np.argmax(np.matmul(gamma_a_o, self.B.T),0)

      for b in range(self.B.shape[0]):
        gamma_a_b[a,b,:] = self.pomdp.reward[:,a] + np.sum(np.array([gamma_a_o[alpha_index[i,b],i,:] for i in range(self.pomdp.observation_space.n)]))
      
    index = np.argmax(np.sum(gamma_a_b*self.B,2),0)
    self.values = np.array([gamma_a_b[index[i],i,:] for i in range(self.B.shape[0])])

  def expand_belief(self):
    for belief in self.B:
      if np.where(belief>0)[0][0]<840:
        # get next belief given aciton and observation
        ob = [self.pomdp.simulate_action(a, belief) for a in range(self.pomdp.action_space.n)]
        obs = np.array([self.pomdp.observe[ob[i],:,i] for i in range(len(ob))]).T
        next_belief = np.sum(belief * np.transpose(self.pomdp.transition,(2,1,0)),2)
        next_belief = next_belief * obs

        # normalize probability
        if np.any(np.sum(next_belief, axis=0)==0):
          print(ob)
          print(np.sum(next_belief, axis=0))
        next_belief = next_belief / np.sum(next_belief, axis=0)

        # L1 distance for the new belief wrt B
        l1_dists = np.array([np.linalg.norm(b - self.B, ord=1, axis=1) for b in next_belief.T])

        min_dist = np.min(l1_dists, axis=1)
        belief_candidate =  np.argmax(min_dist)

        if min_dist[belief_candidate] > 0:
          self.B = np.append(self.B, next_belief[:,belief_candidate].reshape(1,-1), axis=0)



    

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
  logging.basicConfig(level=20, format='%(asctime)s: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
  pomdp = env = gym.make("Tag-v0", env_model=True, track_belief=True)
  solver = PBVI(pomdp)
  n_ite = 12
  for i in range(n_ite):
   solver.value_backup()
   solver.expand_belief()
   logging.info('Iteration %s done'%(i))
   logging.info('Belief maintained: %s'%(solver.B.shape[0]))



