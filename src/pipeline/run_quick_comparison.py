"""
Quick flat vs curriculum comparison on tiny scenarios only.
Uses consistent obs/action dims for fair comparison.
"""
import sys
import time
import numpy as np
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))

from nasim.scenarios import load_scenario
import nasim
from src.pipeline.curriculum_controller import (
    CurriculumController, CurriculumConfig, PhaseConfig, FlatController
)
from src.pipeline.simple_dqn_agent import SimpleDQNAgent

PROJECT_ROOT = Path(__file__).parent.parent.parent
COMPILED_DIR = str(PROJECT_ROOT / 'data' / 'scenarios' / 'generated' / 'compiled_tiny')
TOTAL_EPS = 500
MAX_STEPS = 100
SEED = 42


def make_env(path):
    sc = load_scenario(str(path))
    return nasim.NASimEnv(sc, fully_obs=True, flat_actions=True, flat_obs=True)


def train_ep(agent, env, max_steps):
    obs, info = env.reset()
    total_r = 0.0
    done = False
    trunc = False
    for step in range(max_steps):
        a = agent.select_action(obs)
        n_obs, r, done, trunc, info = env.step(a)
        agent.store_transition(obs, a, r, n_obs, float(done or trunc))
        agent.update()
        total_r += r
        obs = n_obs
        if done or trunc:
            break
    success = done and not trunc and total_r > 0
    return success, total_r, step + 1


def run_one(mode):
    print(f"\n{'='*50}")
    print(f"  {mode.upper()} TRAINING")
    print(f"{'='*50}")

    sample = make_env(sorted(Path(COMPILED_DIR).glob('*.yml'))[0])
    obs, _ = sample.reset()
    obs_dim = obs.shape[0]
    act_dim = sample.action_space.n
    sample.close()
    print(f"  obs_dim={obs_dim}, action_dim={act_dim}")

    agent = SimpleDQNAgent(
        obs_dim=obs_dim, action_dim=act_dim,
        lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay=1500, buffer_size=5000, batch_size=32,
        target_update=50, hidden_dim=64, seed=SEED
    )

    if mode == 'curriculum':
        config = CurriculumConfig(
            phases=[
                PhaseConfig(tier=1, sr_threshold=0.50, min_episodes=20,
                            max_episodes=150, sr_window=15, warmup_episodes=5),
                PhaseConfig(tier=2, sr_threshold=0.40, min_episodes=25,
                            max_episodes=150, sr_window=15, warmup_episodes=5),
                PhaseConfig(tier=3, sr_threshold=0.30, min_episodes=25,
                            max_episodes=150, sr_window=15, warmup_episodes=5),
                PhaseConfig(tier=4, sr_threshold=0.20, min_episodes=25,
                            max_episodes=150, sr_window=20, warmup_episodes=5),
            ],
            seed=SEED, scenarios_per_tier=5
        )
        ctrl = CurriculumController(config, COMPILED_DIR)
    else:
        ctrl = FlatController(COMPILED_DIR, max_episodes=TOTAL_EPS, seed=SEED)

    sr_win = deque(maxlen=50)
    reward_win = deque(maxlen=50)
    cur_env = None
    cur_sc = None
    t0 = time.time()
    ep = 0

    while not ctrl.is_complete() and ep < TOTAL_EPS:
        ep += 1
        sc_path = ctrl.get_next_scenario()
        if sc_path != cur_sc:
            if cur_env:
                cur_env.close()
            cur_env = make_env(sc_path)
            cur_sc = sc_path

        ok, r, steps = train_ep(agent, cur_env, MAX_STEPS)
        status = ctrl.record_episode(ok, r, steps)
        sr_win.append(1 if ok else 0)
        reward_win.append(r)

        if ep % 100 == 0:
            sr = np.mean(list(sr_win))
            avg_r = np.mean(list(reward_win))
            if mode == 'curriculum':
                tier_str = "T" + str(status.get('tier', '?'))
            else:
                tier_str = "ALL"
            print(f"  Ep {ep:4d} | {tier_str} | SR={sr:.2f} | "
                  f"R={avg_r:.1f} | eps={agent.epsilon:.3f}")

        transition = status.get('transition')
        if transition:
            print(f"  >>> TRANSITION: T{transition['from_tier']} -> "
                  f"T{transition.get('to_tier', 'end')} "
                  f"(SR={transition['final_sr']:.2f}, "
                  f"reason={transition['reason']})")

    if cur_env:
        cur_env.close()

    elapsed = time.time() - t0
    final_sr = np.mean(list(sr_win)) if sr_win else 0
    avg_reward = np.mean(list(reward_win)) if reward_win else 0
    print(f"\n  Result: {ep} eps | SR50={final_sr:.3f} | "
          f"R={avg_reward:.1f} | {elapsed:.1f}s")
    return {
        'mode': mode, 'eps': ep,
        'final_sr': final_sr, 'avg_reward': avg_reward,
        'time': elapsed
    }


if __name__ == '__main__':
    flat_r = run_one('flat')
    curr_r = run_one('curriculum')

    print(f"\n{'='*55}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Metric':<25s} {'FLAT':>10s} {'CURRICULUM':>12s}")
    print(f"  {'-'*49}")
    print(f"  {'Final SR (50-win)':<25s} {flat_r['final_sr']:>10.3f} "
          f"{curr_r['final_sr']:>12.3f}")
    print(f"  {'Avg Reward':<25s} {flat_r['avg_reward']:>10.1f} "
          f"{curr_r['avg_reward']:>12.1f}")
    print(f"  {'Episodes':<25s} {flat_r['eps']:>10d} "
          f"{curr_r['eps']:>12d}")
    print(f"  {'Time (s)':<25s} {flat_r['time']:>10.1f} "
          f"{curr_r['time']:>12.1f}")

    diff = curr_r['final_sr'] - flat_r['final_sr']
    r_diff = curr_r['avg_reward'] - flat_r['avg_reward']
    print(f"\n  SR diff: {diff:+.3f} ({'curriculum' if diff >= 0 else 'flat'} better)")
    print(f"  R  diff: {r_diff:+.1f} ({'curriculum' if r_diff >= 0 else 'flat'} better)")
    print(f"{'='*55}")
