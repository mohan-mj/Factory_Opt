import gym
env = gym.make("CartPole-v1")
observation = env.reset()
for _ in range(1000):
  env.render()
  # Input:
  #   Force to the cart with actions: 0=left, 1=right
  # Returns:
  #   obs = cart position, cart velocity, pole angle, rot rate
  #   reward = +1 for every timestep
  #   done = True when abs(angle)>15 or abs(cart pos)>2.4
  action = env.action_space.sample() # random action
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()
env.close()