# -*- coding: utf-8 -*-
import numpy as np
import os
from chainerrl.agents.a3c import A3C
from chainerrl.optimizers import RMSpropAsync
from chainer.optimizer import GradientClipping
from datetime import datetime as dt

from danmaku_env import DanmakuEnv
from model import Model

N_PROCESSES = 4
N_STEPS = 300000
SAVE_INTERVAL = 20000


def train_one_step(idx, env, agent, state, reward):
    action = agent.act_and_train(state, reward)
    state, reward, done, _ = env.step(action)

    if done:
        agent.stop_episode_and_train(state, reward, done)
        return idx, env.reset(), 0.
    else:
        return idx, state, reward


def main():
    out_dir = 'results_' + dt.now().strftime('%Y%m%d%H%M%S')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    assert os.path.isdir(out_dir)

    envs = [DanmakuEnv(random_seed=rs) for rs
            in np.random.randint(np.iinfo(np.uint32).max, size=N_PROCESSES)]
    obs_space = envs[0].observation_space
    action_space = envs[0].action_space

    model = Model(obs_space.shape[0], action_space.n)
    model(np.random.uniform(
        size=obs_space.shape).astype('float32')[np.newaxis, :, :, :])

    opt = RMSpropAsync(lr=7e-4, eps=1e-1, alpha=0.99)
    opt.setup(model)
    opt.add_hook(GradientClipping(40))

    agents = [A3C(model, opt, t_max=5, gamma=0.99,
                  beta=1e-2, process_idx=idx)
              for idx in range(N_PROCESSES)]

    states = [env.reset() for env in envs]
    rewards = [0.] * N_PROCESSES
    episode_count = 0
    step_count = 0
    print('episode: {0:06d}'.format(episode_count + 1))

    while step_count < N_STEPS:
        results = [train_one_step(idx, env, agent, state, reward)
                   for idx, env, agent, state, reward
                   in zip(range(N_PROCESSES),
                          envs, agents, states, rewards)]
        states = [result[1] for result in results]
        rewards = [result[2] for result in results]

        print(dt.now())
        print('passed steps: {0:06d}'.format(step_count + 1))
        print('statistics: {}'.format(agents[0].get_statistics()))

        if envs[0].t == 0:
            episode_count += 1
            print('episode: {0:06d}'.format(episode_count + 1))

        if (step_count + 1) % SAVE_INTERVAL == 0:
            save_agent_path = os.path.join(
                out_dir,
                'agent_step_{0:06d}_'.format(step_count + 1) +
                dt.now().strftime('%Y%m%d%H%M%S'))
            agents[0].save(save_agent_path)
            print('save ' + save_agent_path)
        step_count += 1


if __name__ == '__main__':
    main()
