import sys
import os
import json
from loguru import logger as logging
import time
from tqdm import tqdm, trange
import wandb
import random
import torch
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn, TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn
from torch.utils.tensorboard import SummaryWriter
from easydict import EasyDict

curr_path = os.path.dirname(__file__)
parent_path = os.path.dirname(curr_path)
sys.path.append(curr_path)
sys.path.append(parent_path)  # add current terminal path to sys.path
from src.agent.util import UTIL, color, Attack_Path_Transition, Matrix
from src.agent.policy.common import RewardScaling, Normalization
from src.agent.policy.config import *
from src.agent.actions import Action
from src.agent.host import StateEncoder, HOST
from src.agent.policy.PPO import PPO_agent


class BaseAgent():
    def __init__(self, logger: SummaryWriter, policy_name, cfg: config, seed,
                 use_wandb):
        self.policy_name = policy_name
        self.config = cfg
        self.seed = seed
        self.tf_logger = logger
        self.use_wandb = use_wandb
        self.get_policy(policy_name=policy_name, config=cfg)
        self.name = "separated"

    @classmethod
    def get_policy_config(cls, policy_name, args: dict=None):
        if policy_name == "PPO":
            config = PPO_Config(**args) if args else PPO_Config()
        else:
            return None
        return config

    def get_policy(self, policy_name, config: config):
        if policy_name == "PPO":

            self.config = PPO_Config() if not config else config
            self.Policy = PPO_agent(cfg=self.config,
                                    logger=self.tf_logger,
                                    use_wandb=self.use_wandb)

        
        else:
            self.Policy = None
            print("please imput agent name")
            exit()


class Agent(BaseAgent):
    def __init__(self,
                 logger: SummaryWriter = None,
                 use_wandb=False,
                 policy_name="PPO",
                 seed=0,
                 config: config = None):
        super().__init__(logger=logger,
                         use_wandb=use_wandb,
                         policy_name=policy_name,
                         seed=seed,
                         cfg=config)

        try:
            self.use_reward_scaling = self.config.use_reward_scaling
            self.use_state_norm = self.config.use_state_norm
            self.use_lr_decay = self.config.use_lr_decay
        except:
            self.use_reward_scaling = False
            self.use_state_norm = False
            self.use_lr_decay = False

        if self.use_reward_scaling:
            self.reward_scaling = RewardScaling(shape=1,
                                                gamma=self.config.gamma)
        if self.use_state_norm:
            self.state_norm = Normalization(shape=StateEncoder.state_space)
        self.num_episodes = 0
        self.eval_times = 0

        self.task_num_episodes = 0
        self.total_training_step = 0
        self.best_return = -float('inf')
        self.best_action_set = []
        self.best_episode = 0
        self.best_reward_episode = []
        self.eval_rewards = 0
        self.eval_success_rate = 0
        self.is_loaded_agent = False
        self.max_reward = 1000
        self.min_reward = -1000
        self.last_episode_reward = -float('inf')
        self.first_hit_step = -1
        self.first_hit_eps = -1
        self.convergence_eps = -1
        self.hit_convergence_gap_eps = -1
        self.convergence_judge_done_num = 10
        self.convergence_judge_done_list = [False
                                            ] * self.convergence_judge_done_num

    def decison_making(self, observation):
        if self.use_state_norm:
            observation = self.state_norm(observation, update=False)
        a = self.Policy.evaluate(observation)
        return a

    def train_with_rich(self, target_list, eval_freq=5, use_wandb=True):

        train_matrix = {}
        Train_Episode_Rewards = []
        Train_Episode_Steps = []
        Train_Success_Rate = []
        Train_Episode_Time = []
        Eval_Episode_Rewards = []
        Eval_Success_Rate = []
        '''
        explore stage: prepare transitions
        '''

        from colorama import deinit
        deinit()

        with Progress(SpinnerColumn(finished_text="[green]✔"),
                      "[progress.description]{task.description}",
                      MofNCompleteColumn(), BarColumn(),
                      "[progress.percentage]{task.percentage:>3.2f}%",
                      "[yellow]⏱", TimeElapsedColumn(), "[cyan]⏳",
                      TimeRemainingColumn(),
                      TextColumn("👋 [red]{task.fields}")) as progress:
            Exploring_description = "[red]Exploring"
            Training_description = "[red]Training"

            if self.config.explore_eps:
                Exploring = progress.add_task(Exploring_description,
                                              total=self.config.explore_eps)
                for p in range(self.config.explore_eps):
                    ep_results = self.run_train_episode(target_list,
                                                        explore=True)
                    ep_return, ep_steps, success_rate = ep_results
                    progress.update(Exploring, completed=p, r_t=ep_return)

            Training = progress.add_task(Training_description,
                                         total=self.config.train_eps)
            for p in range(self.config.train_eps + 1):

                # description = "[green]Training"
                start = time.time()
                ep_results = self.run_train_episode(target_list)
                end = time.time()
                run_time = float(end - start)
                ep_return, ep_steps, success_rate = ep_results

                Train_Episode_Rewards.append(ep_return)
                Train_Episode_Steps.append(ep_steps)
                Train_Success_Rate.append(success_rate)
                Train_Episode_Time.append(run_time)

                self.tf_logger.add_scalar("Train_Episode_Rewards", ep_return,
                                          self.num_episodes)
                self.tf_logger.add_scalar("Train_Episode_Steps", ep_steps,
                                          self.num_episodes)
                self.tf_logger.add_scalar("Train_Success_Rate", success_rate,
                                          self.num_episodes)
                self.tf_logger.add_scalar("Train_Episode_Time", run_time,
                                          self.num_episodes)

                if use_wandb:
                    wandb.log(
                        {
                            "Train_Episode_Rewards": ep_return,
                            "Train_Success_Rate": success_rate,
                            "Train_Episode_Steps": ep_steps,
                            "Train_Episode_Time": run_time,
                        }, self.num_episodes)

                if self.num_episodes % eval_freq == 0:
                    self.Evaluate(target_list=target_list,
                                  verbose=False,
                                  step_limit=self.config.eval_step_limit)
                    Eval_Episode_Rewards.append(self.eval_rewards)
                    Eval_Success_Rate.append(self.eval_success_rate)

                    self.tf_logger.add_scalar("Eval_Episode_Rewards",
                                              self.eval_rewards,
                                              self.num_episodes)
                    self.tf_logger.add_scalar("Eval_Success_Rate",
                                              self.eval_success_rate,
                                              self.num_episodes)
                    if use_wandb:
                        wandb.log(
                            {
                                "Eval_Episode_Rewards": self.eval_rewards,
                                "Eval_Success_Rate": self.eval_success_rate,
                            }, self.num_episodes)
                progress.update(
                    Training,
                    completed=p,
                    re_t=f"{ep_return}/{self.best_return}[green]",
                    step=f"{ep_steps}[blue]",
                    re_e=f"{self.eval_rewards}[cyan]",
                    rate_e=f"{self.eval_success_rate*100}%[yellow]",
                    rate_t=f"{success_rate*100}%",
                )

            train_matrix["Train_Episode_Rewards"] = Train_Episode_Rewards
            train_matrix["Train_Episode_Steps"] = Train_Episode_Steps
            train_matrix["Train_Success_Rate"] = Train_Success_Rate
            train_matrix["Train_Episode_Time"] = Train_Episode_Time
            train_matrix["Eval_Episode_Rewards"] = Eval_Episode_Rewards
            train_matrix["Eval_Success_Rate"] = Eval_Success_Rate

        return train_matrix

    def train_with_tqdm(self, task_list, eval_freq=5):
        Train_matrix = EasyDict({
            "signal": Matrix.Finished,
            "Train_Episode_Rewards": [],
            "Train_Episode_Steps": [],
            "Train_Success_Rate": [],
            "Train_Episode_Time": [],
            "Eval_Episode_Rewards": [],
            "Eval_Success_Rate": [],
            "last_task": -1
        })

        self.task_num_episodes = 0
        self.eval_rewards = 0
        '''
        explore stage: prepare transitions
        '''
        with tqdm(range(self.config.explore_eps),
                  desc=color.color_str('Exploring', c=color.RED)) as tbar:
            for _ in tbar:
                ep_results = self.run_train_episode(task_list, explore=True)
                ep_return, ep_steps, success_rate = ep_results
                tbar.set_postfix(ep_return=color.color_str(f"{ep_return}",
                                                           c=color.PURPLE),
                                 ep_steps=color.color_str(f"{ep_steps}",
                                                          c=color.GREEN))
        '''
        exploit stage: train policy
        
        '''
        with tqdm(range(self.config.train_eps),
                  desc=f"{color.color_str('Training',c=color.RED)}") as tbar:
            for _ in tbar:
                start = time.time()
                self.num_episodes += 1

                ep_results = self.run_train_episode(task_list)

                end = time.time()
                run_time = float(end - start)
                ep_return, ep_steps, success_rate = ep_results
                self.last_episode_reward = ep_return
                Train_matrix.Train_Episode_Rewards.append(ep_return)
                Train_matrix.Train_Episode_Steps.append(ep_steps)
                Train_matrix.Train_Success_Rate.append(success_rate)
                Train_matrix.Train_Episode_Time.append(run_time)
                if self.tf_logger:
                    self.tf_logger.add_scalar("Train/Episode_Rewards",
                                              ep_return, self.num_episodes)
                    self.tf_logger.add_scalar("Train/Episode_Steps", ep_steps,
                                              self.num_episodes)
                    self.tf_logger.add_scalar("Train/Success_Rate",
                                              success_rate, self.num_episodes)
                    self.tf_logger.add_scalar("Train/Episode_Time", run_time,
                                              self.num_episodes)
                    self.tf_logger.add_scalar("Train/Total_Steps",
                                              self.total_training_step,
                                              self.num_episodes)
                    # Log convergence milestones
                    if self.first_hit_eps > 0:
                        self.tf_logger.add_scalar("Convergence/First_Hit_Episode",
                                                  self.first_hit_eps,
                                                  self.num_episodes)
                    if self.convergence_eps > 0:
                        self.tf_logger.add_scalar("Convergence/Converged_Episode",
                                                  self.convergence_eps,
                                                  self.num_episodes)
                    if self.use_state_norm:
                        self.tf_logger.add_scalar(
                            "Auxillary/state_norm_mean",
                            self.state_norm.running_ms.mean.mean(),
                            self.num_episodes)
                if self.use_wandb:
                    wandb.log({
                        "Train/Episode_Rewards": ep_return,
                        "Train/Success_Rate": success_rate,
                        "Train/Episode_Steps": ep_steps,
                        "Train/Episode_Time": run_time,
                        "Total_Train_Steps": self.total_training_step,
                        "num_episodes": self.num_episodes
                    })

                if self.num_episodes % eval_freq == 0:
                    eval_attack_path, eval_rewards, eval_sr = self.Evaluate(
                        target_list=task_list,
                        verbose=False,
                        step_limit=self.config.eval_step_limit)
                    Train_matrix.Eval_Episode_Rewards.append(self.eval_rewards)
                    Train_matrix.Eval_Success_Rate.append(
                        self.eval_success_rate)
                    if self.tf_logger:
                        self.tf_logger.add_scalar("Eval/Episode_Rewards",
                                                  self.eval_rewards,
                                                  self.num_episodes)
                        self.tf_logger.add_scalar("Eval/Success_Rate",
                                                  self.eval_success_rate,
                                                  self.num_episodes)
                        # Per-target eval metrics
                        if eval_attack_path:
                            eval_steps_list = [len(t.get('path', [])) for t in eval_attack_path]
                            eval_success_count = sum(1 for t in eval_attack_path if t.get('success'))
                            avg_eval_steps = sum(eval_steps_list) / max(len(eval_steps_list), 1)
                            self.tf_logger.add_scalar("Eval/Avg_Steps_Per_Target",
                                                      avg_eval_steps,
                                                      self.num_episodes)
                            self.tf_logger.add_scalar("Eval/Targets_Succeeded",
                                                      eval_success_count,
                                                      self.num_episodes)
                    if self.use_wandb:
                        wandb.log({
                            "Eval/Episode_Rewards": self.eval_rewards,
                            "Eval/Success_Rate": self.eval_success_rate,
                        })
                '''
                display info
                '''
                tbar.set_postfix(
                    re_t=color.color_str(f"{ep_return}/{self.best_return}",
                                         c=color.PURPLE),
                    step=color.color_str(f"{ep_steps}", c=color.GREEN),
                    re_e=color.color_str(f"{self.eval_rewards}", c=color.BLUE),
                    rate_e=color.color_str(f"{self.eval_success_rate*100}%",
                                           c=color.CYAN),
                    rate_t=color.color_str(f"{success_rate*100}%",
                                           c=color.YELLOW),
                )

        return Train_matrix

    def run_train_episode(self, target_list, explore=False, update_norm=True):

        eps_steps = 0
        episode_return = 0
        self.action_set = []
        self.reward_set = []
        success_num = 0
        failed_num = 0
        target_id = 0
        self.task_num_episodes += 1

        if self.use_reward_scaling:
            self.reward_scaling.reset()
        # for target_id in range(len(self.target_list)):
        # random.shuffle(target_list)
        while target_id < len(target_list):
            done = 0
            target_step = 0
            target: HOST = target_list[target_id]
            '''
            Init observation
            '''
            o = target.reset()
            if self.use_state_norm:
                o = self.state_norm(o, update=update_norm)

            while not done:

                if target_step >= self.config.step_limit:
                    break
                '''
                Output an action
                '''
                action_info = self.Policy.select_action(
                    observation=o,
                    explore=explore,
                    is_loaded_agent=self.is_loaded_agent,
                    num_episode=self.task_num_episodes)
                a = action_info[0]  # action_info 中第一位为动作id
                self.action_set.append(a)
                '''
                Perform the action
                '''
                next_o, r, done, result = target.perform_action(a)
                self.total_training_step += 1
                eps_steps += 1
                target_step += 1
                episode_return += r
                self.reward_set.append(r)
                '''
                Store the transition
                '''
                if done:
                    success_num += 1
                    dw = True
                    if self.first_hit_step < 0:
                        self.first_hit_step = self.total_training_step
                    if self.first_hit_eps < 0:
                        self.first_hit_eps = self.task_num_episodes

                else:
                    dw = False

                self.convergence_judge_done_list[
                    (self.task_num_episodes - 1) %
                    self.convergence_judge_done_num] = dw
                if self.hit_convergence_gap_eps < 0:
                    if all(self.convergence_judge_done_list):
                        self.hit_convergence_gap_eps = (
                            self.task_num_episodes - self.first_hit_eps)
                        self.convergence_eps = self.task_num_episodes
                if self.use_state_norm:
                    next_o = self.state_norm(next_o, update=update_norm)
                if self.use_reward_scaling:
                    r = self.reward_scaling(r)[0]
                self.Policy.store_transtion(observation=o,
                                            action=action_info,
                                            reward=r,
                                            next_observation=next_o,
                                            done=dw)
                '''
                Update the policy
                '''
                if not explore:

                    self.Policy.update_policy(
                        num_episode=self.task_num_episodes,
                        train_steps=self.total_training_step)
                    if self.use_lr_decay:
                        #NOTE Only support PPO
                        rate = (
                            1 - self.task_num_episodes / self.config.train_eps
                        ) if self.task_num_episodes < self.config.train_eps else 1
                        if rate <= self.config.min_decay_lr:
                            rate = self.config.min_decay_lr
                        self.Policy.lr_decay(rate=rate)
                o = next_o
            if not done:
                failed_num += 1
                # break
            target_id += 1
        sucess_rate = float(format(success_num / len(target_list), '.3f'))
        if episode_return >= self.best_return:
            self.best_return = episode_return
            self.best_action_set = self.action_set
            self.best_reward_episode = self.reward_set
            self.best_episode = self.num_episodes
        return episode_return, eps_steps, sucess_rate

    def Evaluate(self,
                 target_list,
                 step_limit=10,
                 manual=False,
                 interactive=False,
                 verbose=True):
        sucess_rate = 0.0
        target_id = 0
        total_rewards = 0
        sucess_list = []
        faild_list = []
        attack_path = []
        inference_times = []
        attack_path_key = ["target", "step", "action", "result", "reward"]

        # random.shuffle(target_list)
        while target_id < len(target_list):
            host: HOST = target_list[target_id]
            host_attack_path = {}
            host_attack_path["ip"] = host.ip
            host_attack_path["path"] = []
            if interactive:
                UTIL.line_break(symbol='=', length=50)
                logging.info("testing: " + host.ip)
                UTIL.line_break(symbol='=', length=50)
            o = host.reset()
            if self.use_state_norm:
                o = self.state_norm(o, update=False)
            done = 0
            steps = 0
            task_return = 0
            if interactive:
                input("Press enter to continue...")
            while not done and steps < step_limit:
                # process = dict.fromkeys(attack_path_key, None)
                if not manual:
                    inf_start = time.time()
                    with torch.no_grad():
                        a = self.Policy.evaluate(o)
                    inf_end = time.time()
                    inference_times.append(inf_end - inf_start)
                else:
                    a = input(
                        "Please select action number, input '-1' to exit: ")
                    a = int(a)
                action_start = time.time()
                next_o, r, done, result = host.perform_action(a)
                action_end = time.time()
                if self.use_state_norm:
                    next_o = self.state_norm(next_o, update=False)
                o = next_o
                total_rewards += r
                task_return += r
                steps += 1
                if interactive:
                    UTIL.line_break(symbol='-', length=50)
                    logging.info(f"Step {steps}")
                    logging.info(f"Action Performed = {Action.get_action(a)}")
                    logging.info(f"Result = {result}")
                    logging.info(f"Reward = {r}")
                #logging.info(f"Done = {done}")
                #input("Press enter to continue..")

                process = Attack_Path_Transition(ip=host.ip,
                                                 step=steps,
                                                 action=Action.get_action(a),
                                                 result=result,
                                                 reward=r)

                host_attack_path["path"].append(process)

                if done:
                    if interactive:
                        logging.info("SUCCESS: " + host.ip)
                        logging.info(f"Total steps = {steps}")
                        logging.info(f"Total reward = {total_rewards}")
                    sucess_list.append(host.ip)
                    break
            if not done:

                faild_list.append(host.ip)
            host_attack_path['reward'] = task_return
            host_attack_path['success'] = done
            host_attack_path_ = EasyDict(host_attack_path)
            attack_path.append(host_attack_path)
            target_id += 1
        sucess_rate = float(format(len(sucess_list) / len(target_list), '.3f'))
        self.eval_rewards = total_rewards
        self.eval_success_rate = sucess_rate
        # Store inference timing stats
        self.last_eval_inference_times = inference_times
        self.last_eval_attack_path = attack_path
        if inference_times:
            avg_inf = sum(inference_times) / len(inference_times)
            if self.tf_logger:
                self.tf_logger.add_scalar("Eval/Avg_Inference_Time_ms",
                                          avg_inf * 1000,
                                          self.eval_times)
            if verbose:
                logging.info(f"Avg inference time: {avg_inf*1000:.2f} ms/action")
        self.eval_times += 1
        return attack_path, total_rewards, self.eval_success_rate

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        assert os.path.exists(path), f"{path} does not exist"
        if self.use_state_norm:
            mean = self.state_norm.running_ms.mean
            std = self.state_norm.running_ms.std
            mean_checkpoint = path / f"{self.policy_name}-norm_mean.pt"
            std_checkpoint = path / f"{self.policy_name}-norm_std.pt"
            torch.save(mean, mean_checkpoint)
            torch.save(std, std_checkpoint)
        self.Policy.save(path)

    def load(self, path):
        if self.use_state_norm:
            mean_checkpoint = path / f"{self.policy_name}-norm_mean.pt"
            std_checkpoint = path / f"{self.policy_name}-norm_std.pt"
            mean = torch.load(mean_checkpoint, weights_only=False)
            std = torch.load(std_checkpoint, weights_only=False)
            self.state_norm.running_ms.mean = mean
            self.state_norm.running_ms.std = std
        self.Policy.load(path)
        self.is_loaded_agent = True
