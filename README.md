# D4PG

Added already:

- Prioritized Experience Replay
- N-Step Bootstrapping
- D2RL
- Distributional IQN Critic

TODO:
- add munchausen RL
- test runs for Pendulum and LunarLanderContinuous
- compare critic loss of base and iqn agent!

Notes:

- Performance depends a lot on good hyperparameter->> tau for Per bigger (pendulum 1e-2) for regular replay (1e-3)

- BatchNorm had good impact on the overall performance (!)
