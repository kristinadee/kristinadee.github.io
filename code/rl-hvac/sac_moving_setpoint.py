# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 18:43:52 2025

@author: kresg
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
import collections
import argparse
import os
import matplotlib.pyplot as plt
from fmpy import read_model_description, extract, dump
from fmpy.fmi2 import FMU2Slave
from datetime import datetime

def warmup_fmu(fmu, warmup_seconds, step_size=60.0, max_csim_step=2.0):
    current_time = 0.0
    print(f"⏩ Warming up FMU to t={warmup_seconds:.0f} seconds...")
    while current_time < warmup_seconds:
        h = min(max_csim_step, warmup_seconds - current_time)
        try:
            fmu.doStep(
                currentCommunicationPoint=current_time,
                communicationStepSize=h,
                noSetFMUStatePriorToCurrentPoint=True
            )
        except Exception as e:
            raise RuntimeError(f"Warm-up failed at t={current_time}, h={h}") from e
        current_time += h
    return current_time

## FMU ENVIRONMENT ##
class FMUEnv(gym.Env):
    def __init__(
        self,
        fmu,
        vrs,
        outdoor_list,
        indoor_list,
        actions_list,
        setpoint=20.0,
        temp_factor=1.0,
        energy_factor=0.0,
        episode_length=144,
        max_power=11000.0,
        max_effective_power=10000.0,
        delta=60.0,
        max_csim_step=2.0,
    ):
        super().__init__()
        self.fmu                = fmu
        self.vrs                = vrs
        self.outdoor_list       = outdoor_list
        self.indoor_list        = indoor_list
        self.actions_list       = actions_list

        self.setpoint           = float(setpoint)
        self.temp_factor        = temp_factor
        self.energy_factor      = energy_factor
        self.episode_length     = int(episode_length)
        self.max_power          = float(max_power)
        self.max_effective_power = float(max_effective_power)
        self.delta              = float(delta)
        self.max_csim_step      = float(max_csim_step)
        self.current_step       = 0
        self.current_time       = 0.0

        self.action_space       = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        
        self.energy_history = []

    def compute_reward(self, temp_next, next_setpoint, power):
        upper_bound = next_setpoint + 1
        lower_bound = next_setpoint - 1
        temp_reward = -(temp_next - upper_bound)*(temp_next - upper_bound)
        
        if lower_bound <= temp_next <= upper_bound:
            temp_reward = 0
        elif temp_next < lower_bound:
            temp_reward = -(lower_bound - temp_next)*(lower_bound - temp_next)/1000
        else:
            temp_reward = -(temp_next - upper_bound)*(temp_next - upper_bound)/1000
        return float(self.temp_factor * temp_reward + self.energy_factor * power)

    def reset(self):
        self.setpoint = self.get_dynamic_setpoint()
        next_setpoint = self._predict_next_setpoint()
        tin  = round(self.fmu.getReal([self.vrs['IndoorAirTemp']])[0], 1)
        tout = round(self.fmu.getReal([self.vrs['OutdoorAirTemp']])[0], 1)
        self.indoor_list.append(tin)
        self.outdoor_list.append(tout)
        time_of_day = (self.current_time / 3600.0) % 24 / 24.0  # normalize to [0,1]
        return np.array([tin, tout, self.setpoint, next_setpoint, time_of_day], dtype=np.float32)

    def step(self, action):
        act_norm = float(np.clip(action[0], 0.0, 1.0))
        self.fmu.setReal([self.vrs['Action']], [act_norm])
        power = act_norm * self.max_effective_power
        
        # Run FMU step
        remaining = self.delta
        tloc = self.current_time
        while remaining > 1e-9:
            h = min(self.max_csim_step, remaining)
            try:
                self.fmu.doStep(
                    currentCommunicationPoint=tloc,
                    communicationStepSize=h,
                    noSetFMUStatePriorToCurrentPoint=True
                )
            except Exception as e:
                raise RuntimeError(f"fmi2DoStep failed at t={tloc}, h={h}") from e
            tloc += h
            remaining -= h
        self.current_time = tloc

        # Update setpoint dynamically
        self.setpoint = self.get_dynamic_setpoint()
        next_setpoint = self._predict_next_setpoint()
    
        # Read Temperatures
        tin  = round(self.fmu.getReal([self.vrs['IndoorAirTemp']])[0], 1)
        tout = round(self.fmu.getReal([self.vrs['OutdoorAirTemp']])[0], 1)
        self.indoor_list.append(tin)
        self.outdoor_list.append(tout)
        self.actions_list.append(power)

        # Read power and compute reward
        energy = self.fmu.getReal([self.vrs['PowerOut']])[0]
        self.energy_history.append(energy)

        reward = self.compute_reward(tin, self.setpoint, power)
        self.current_step += 1
        done = (self.current_step >= self.episode_length)
        time_of_day = (self.current_time / 3600.0) % 24 / 24.0
        obs = np.array([tin, tout, self.setpoint, next_setpoint, time_of_day], dtype=np.float32)
        info = {"heater_power": power,"setpoint": self.setpoint}

        return obs, reward, done, info
    def get_dynamic_setpoint(self):
        """
        Matches the DQL schedule:
        - Each day is split into 3 equal sections (low, high, low)
        - unoccupied_setpoint: 15 °C
        - occupied_setpoint: 20 °C
        """
        step_size_hours = self.delta / 3600.0
        steps_per_day = int(24 / step_size_hours)
    
        # number of steps in each section
        section_steps = steps_per_day // 3  
    
        # current step within the day
        current_step_in_day = int((self.current_time / self.delta) % steps_per_day)
    
        if current_step_in_day < section_steps:        # first unoccupied
            return 15.0
        elif current_step_in_day < 2 * section_steps:  # occupied
            return 20.0
        else:                                          # last unoccupied
            return 15.0
        
    def _predict_next_setpoint(self):
        # Predict setpoint at the next time step without changing current_time
        next_time = self.current_time + self.delta
        step_size_hours = self.delta / 3600.0
        steps_per_day = int(24 / step_size_hours)
        section_steps = steps_per_day // 3
        current_step_in_day = int((next_time / self.delta) % steps_per_day)
    
        if current_step_in_day < section_steps:
            return 15.0
        elif current_step_in_day < 2 * section_steps:
            return 20.0
        else:
            return 15.0

## ACTOR & CRITIC NETWORKS ##

class ActorNetwork(nn.Module):
    """
    MLP actor producing [mean, log_std].
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, act_dim * 2)]
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)

class CriticNetwork(nn.Module):
    """
    MLP critic Q(s,a).
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim + act_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)

## SOFT ACTOR-CRITIC ALGORITHM IMPLEMENTATION ##
class SAC:
    def __init__(self, env, initial_random_steps=200*15, n_step=5, update_interval=5):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # extras
        self.n_step        = 5
        self.update_interval = 5
        self.n_buffer      = collections.deque(maxlen=self.n_step)
        self.policy_delay  = 4
        self.initial_random_steps = initial_random_steps
        self.total_steps          = 0
        self._update_step = 0

        # hyperparams
        self._init_hyperparameters()
        
        # noise schedule for target-policy smoothing
        self.noise_start    = 0.3
        self.noise_end      = 0.3
        self.noise       = self.noise_start
        self.noise_sigma = self.noise_start #0.3
        self.noise_clip = 0.1 #0.6 

        # networks
        self.actor            = ActorNetwork(self.obs_dim, self.act_dim)
        self.actor_target = ActorNetwork(self.obs_dim, self.act_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1          = CriticNetwork(self.obs_dim, self.act_dim)
        self.critic2          = CriticNetwork(self.obs_dim, self.act_dim)
        self.critic1_target   = CriticNetwork(self.obs_dim, self.act_dim)
        self.critic2_target   = CriticNetwork(self.obs_dim, self.act_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # optimizers
        self.actor_optim   = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic1_optim = Adam(self.critic1.parameters(), lr=self.critic_lr)
        self.critic2_optim = Adam(self.critic2.parameters(), lr=self.critic_lr)
        init_alpha        = 0.3
        self.log_alpha    = torch.tensor(np.log(init_alpha), requires_grad=True)
        self.alpha_optim  = Adam([self.log_alpha], lr=self.alpha_lr)
        self.alpha        = init_alpha

        # replay buffer
        self.replay_buffer   = [None] * self.buffer_capacity
        self.buffer_counter  = 0
        self.buffer_size     = 0
        self.temp_history    = []
        self.outdoor_history = []
        self.power_history   = []
        self.setpoint_history = []

    def _init_hyperparameters(self):
        self.actor_lr         = 3e-4    
        self.critic_lr        = 3e-4 
        self.alpha_lr         = 5e-2
        self.gamma            = 0.995
        # faster target-net polyak average to slow drifting
        self.tau              = 0.01    # LAST CHANGED WAS O.05
        # more deterministic sooner
        self.target_entropy   = -2 * self.act_dim  # was -1*act_dim
        self.buffer_capacity  = int(1e6)
        self.batch_size       = 256     # was 256
        self.updates_per_step = 1

    def select_action(self, obs, evaluate=False):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        mean, log_std = torch.chunk(self.actor(obs_t), 2, dim=-1)
        std = log_std.clamp(-10, -1).exp()
        dist = torch.distributions.Normal(mean, std)
        z = mean if evaluate else dist.rsample()
        raw = torch.tanh(z)
        scaled = (raw + 1.0) / 2.0
        return scaled.detach().cpu().numpy().squeeze(0)

    def store_transition(self, transition):
        self.n_buffer.append(transition)
        if len(self.n_buffer) == self.n_step:
            R = sum(self.n_buffer[i][2] * (self.gamma ** i) for i in range(self.n_step))
            s0, a0 = self.n_buffer[0][0], self.n_buffer[0][1]
            s_n, d_n = self.n_buffer[-1][3], self.n_buffer[-1][4]
            self.replay_buffer[self.buffer_counter] = (s0, a0, R, s_n, d_n)
            self.buffer_counter = (self.buffer_counter + 1) % self.buffer_capacity
            self.buffer_size = min(self.buffer_size + 1, self.buffer_capacity)

    def sample_batch(self):
        idx = np.random.randint(self.buffer_size, size=self.batch_size)
        batch = [self.replay_buffer[i] for i in idx]
        s, a, r, s2, d = zip(*batch)
        return (
            torch.FloatTensor(s),
            torch.FloatTensor(a),
            torch.FloatTensor(r).unsqueeze(1),
            torch.FloatTensor(s2),
            torch.FloatTensor(d).unsqueeze(1),
        )

    def update(self):
        state, action, reward, next_state, done = self.sample_batch()

        with torch.no_grad():
            mean2, log_std2 = torch.chunk(self.actor_target(next_state), 2, dim=-1)
            std2 = log_std2.clamp(-20, 2).exp()
            dist2 = torch.distributions.Normal(mean2, std2)
            z2 = dist2.rsample()
            a2 = torch.tanh(z2)
            decay = (self.noise_start - self.noise_end) / float(self.total_steps_per_run)
            self.noise_sigma = max(self.noise_sigma - decay, self.noise_end)
            raw = torch.tanh(z2)
            scaled = (raw + 1.0) / 2.0
            decay = (self.noise_start - self.noise_end) / float(self.total_steps_per_run)
            self.noise_sigma = max(self.noise_sigma - decay, self.noise_end)
            noise = (torch.randn_like(scaled) * self.noise_sigma).clamp(-self.noise_clip, self.noise_clip)
            a2_noisy = (scaled + noise).clamp(0.0, 1.0)
            log_prob2 = (dist2.log_prob(z2) - torch.log(1 - a2.pow(2) + 1e-6)).sum(-1, keepdim=True)

            q1_t = self.critic1_target(next_state, a2_noisy)
            q2_t = self.critic2_target(next_state, a2_noisy)
            min_q = torch.min(q1_t, q2_t)
            target_v = min_q - self.alpha * log_prob2
            target_q = reward + (1 - done) * (self.gamma ** self.n_step) * target_v

        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        loss_q1 = nn.MSELoss()(q1, target_q)
        loss_q2 = nn.MSELoss()(q2, target_q)
        self.critic1_optim.zero_grad(); loss_q1.backward(); self.critic1_optim.step()
        self.critic2_optim.zero_grad(); loss_q2.backward(); self.critic2_optim.step()

        self._update_step += 1
        if self._update_step % self.policy_delay == 0:
            mean, log_std = torch.chunk(self.actor(state), 2, dim=-1)
            std = log_std.clamp(-20, 2).exp()
            dist = torch.distributions.Normal(mean, std)
            z = dist.rsample()
            tanh_z = torch.tanh(z)
            log_prob = (dist.log_prob(z) - torch.log(1 - tanh_z.pow(2) + 1e-6)).sum(-1, keepdim=True)

            raw = torch.tanh(z)
            scaled = (raw + 1.0) / 2.0
            q1_new = self.critic1(state, scaled)
            q2_new = self.critic2(state, scaled)
            actor_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new)).mean()
            self.actor_optim.zero_grad(); actor_loss.backward(); self.actor_optim.step()

            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad(); alpha_loss.backward(); self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()
        else:
            actor_loss = 0.0
            alpha_loss = 0.0

        for tp, p in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            tp.data.copy_(tp.data*(1-self.tau)+p.data*self.tau)
        for tp, p in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            tp.data.copy_(tp.data*(1-self.tau)+p.data*self.tau)
        for tp, p in zip(self.actor_target.parameters(), self.actor.parameters()):
            tp.data.copy_(tp.data*(1-self.tau) + p.data*self.tau)

        return loss_q1.item(), loss_q2.item(), actor_loss, alpha_loss

    def train_one_episode(self):
        obs = self.env.reset()
        ep_r = 0.0
        done = False
        # Reset data collectors per episode
        self.setpoint_history = []  
        self.temp_history = []
        self.outdoor_history = []
        self.power_history = []
        latest_q1, latest_q2, latest_actor, latest_alpha = 0.0, 0.0, 0.0, 0.0
        while not done:
            if self.total_steps < self.initial_random_steps:
                action = self.env.action_space.sample()
            else:
                action = self.select_action(obs)

            nxt, r, done, info = self.env.step(action)
            self.setpoint_history.append(info["setpoint"])
            self.store_transition((obs, action, r, nxt, float(done)))
            obs = nxt
            ep_r += r
            self.total_steps += 1

            if (self.total_steps >= self.initial_random_steps
                and self.total_steps % self.update_interval == 0
                and len(self.replay_buffer) > self.batch_size):
                for _ in range(self.updates_per_step):
                    latest_q1, latest_q2, latest_actor, latest_alpha = self.update()

            self.temp_history.append(obs[0])
            self.outdoor_history.append(obs[1])
            self.power_history.append(info["heater_power"])

        return ep_r, obs[0], latest_q1, latest_q2, latest_actor, latest_alpha

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def parse_args():
    parser = argparse.ArgumentParser("SAC on FMU Environment")
    parser.add_argument("--episodes",     type=int, default=200)
    parser.add_argument("--horizon_days", type=int, default=15,
                        help="How many days to simulate (from t=0)")
    return parser.parse_args()

def main():
    overall_start = datetime.now()
    print(f"🚀 Script started at: {overall_start.strftime('%Y-%m-%d %H:%M:%S')}")
    
    args = parse_args()

    # Load and dump FMU info
    fmu_filename = r"C:/Users/kresg/AppData/Local/Temp/OpenModelica/OMEdit/Building_model2_4element_ThermZone_Python_Control.fmu"
    dump(fmu_filename)

    # Read modelDescription and build vrs map
    md  = read_model_description(fmu_filename)
    vrs = {v.name: v.valueReference for v in md.modelVariables}

    # Extract & instantiate
    unzipdir = extract(fmu_filename)
    fmu = FMU2Slave(
        guid            = md.guid,
        unzipDirectory  = unzipdir,
        modelIdentifier = md.coSimulation.modelIdentifier,
        instanceName    = 'instance1'
    )

    # Simulation parameters
    step_size     = 60.0    
    substep_size  = 2.0
    steps_per_day = int(24*3600 / step_size)
    horizon_days  = args.horizon_days
    test_horizon_days = 30
    horizon_steps = horizon_days * steps_per_day
    test_horizon_steps = test_horizon_days * steps_per_day
    start_time = 0.0  # NEW start time in seconds (20 days)
    sim_end = start_time + horizon_steps * step_size
    
    script_name = os.path.basename(__file__)  # For metadata
    
    # Data Collectors
    all_episodes_rewards = []
    all_episodes_q1_loss = []
    all_episodes_q2_loss = []
    all_episodes_actor_loss = []
    all_episodes_alpha_loss = []
    all_episodes_indoor_temps = []
    all_episodes_outdoor_temps = []
    all_episodes_actions = []
    all_episodes_energy = []
    all_episodes_setpoints = []

    
    # Instantiate & initialize FMU at t=0 → sim_end
    fmu.instantiate()
    fmu.setupExperiment(startTime=start_time)
    fmu.enterInitializationMode()
    fmu.exitInitializationMode()

    # Create dummy env to infer spaces
    dummy = FMUEnv(
        fmu=fmu,
        vrs=vrs,
        outdoor_list=[],
        indoor_list=[],
        actions_list=[],
        setpoint=20.0,
        temp_factor=1.0,
        energy_factor=0.0,
        episode_length=horizon_steps,
        delta=step_size
    )
    sac = SAC(dummy)

#std2 = log_std2.clamp(-20, 2).exp()
    # ==== TRAINING ====  
    unzipdir = extract(fmu_filename)
    guid     = md.guid
    model_id = md.coSimulation.modelIdentifier

    sac.total_steps_per_run = (
        args.episodes
        * horizon_steps
        * sac.updates_per_step
    )
    train_rewards = []
    old_fmu = None
    for ep in range(1, args.episodes+1):
        if old_fmu is not None:
            old_fmu.terminate()
            old_fmu.freeInstance()
        fmu = FMU2Slave(
            guid            = guid,
            unzipDirectory  = unzipdir,
            modelIdentifier = model_id,
            instanceName    = f'instance_ep{ep}'
        )
        fmu.instantiate()
        fmu.setupExperiment(startTime=start_time)
        fmu.enterInitializationMode()
        fmu.exitInitializationMode()

        # 🔁 Warm up simulation time to 20 days
        warmup_seconds = 1728000.0  # 20 days
        train_current_time = warmup_fmu(fmu, warmup_seconds, step_size, substep_size)
        print(f"✅ Warm-up done. Current simulation time: {train_current_time:.1f}s")

        train_env = FMUEnv(
            fmu                = fmu,
            vrs                = vrs,
            outdoor_list       = [],
            indoor_list        = [],
            actions_list       = [],
            setpoint           = 20.0,
            temp_factor        = 1.0,
            energy_factor      = 0.0,
            episode_length     = horizon_steps,
            max_effective_power= 10000.0,
            delta              = step_size,
            max_csim_step      = substep_size
        )
        train_env.current_time = train_current_time
        sac.env = train_env
        old_fmu = fmu
        print(f"\n▶️ Starting episode {ep}/{args.episodes}")
        r, tin, q1, q2, actor_l, alpha_l = sac.train_one_episode()
        
        # append data
        all_episodes_rewards.append(r)
        all_episodes_q1_loss.append(q1)
        all_episodes_q2_loss.append(q2)
        all_episodes_actor_loss.append(actor_l)
        all_episodes_alpha_loss.append(alpha_l)
        all_episodes_setpoints.append(sac.setpoint_history.copy())
        
        # Episode-wise time-series
        all_episodes_indoor_temps.append(sac.temp_history.copy())
        all_episodes_outdoor_temps.append(sac.outdoor_history.copy())
        all_episodes_actions.append(sac.power_history.copy())
        all_episodes_energy.append(train_env.energy_history.copy())

        print(f"✅ Episode {ep}/{args.episodes} done | reward={r:.3f}, final Tin={tin:.1f}°C")

    os.makedirs("models", exist_ok=True)
    torch.save(sac.actor.state_dict(), "models_moving_noEF.pth")


    # ==== TESTING ====
    sac.actor.load_state_dict(torch.load("models_moving_noEF.pth", map_location=torch.device('cpu')))
    test_unzipdir = extract(fmu_filename)  # separate extract for test instance
    fmu_test = FMU2Slave(
        guid=guid,
        unzipDirectory=test_unzipdir,
        modelIdentifier=model_id,
        instanceName='test_instance'
    )
    fmu_test.instantiate()
    fmu_test.setupExperiment(startTime=0.0)
    fmu_test.enterInitializationMode()
    fmu_test.exitInitializationMode()
    
    test_env = FMUEnv(
        fmu=fmu_test,
        vrs=vrs,
        outdoor_list=[],
        indoor_list=[],
        actions_list=[],
        setpoint=20.0,
        temp_factor=1.0,
        energy_factor=0.0,
        episode_length=test_horizon_steps,
        max_effective_power=10000.0,
        delta=step_size,
        max_csim_step=substep_size
    )
    
    obs = test_env.reset()
    energy_kwh = 0.0
    done = False
    test_setpoint_hist = []
    while not done:
        action = sac.select_action(obs, evaluate=True)
        obs, _, done, info = test_env.step(action)
        energy_kwh += info["heater_power"] * (step_size/3600.0) / 1000.0
        test_setpoint_hist.append(info["setpoint"])
    print(f"Test energy over {horizon_days} days: {energy_kwh:.2f} kWh")
    print(len(test_env.indoor_list), len(test_env.outdoor_list), len(test_env.actions_list))
    
    # ==== PLOTTING ====

    plt.figure()
    plt.plot(all_episodes_rewards, marker='o')
    plt.title(f"Training Rewards ({args.horizon_days} days)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(alpha=0.3)
    plt.show()
    
    plt.figure()
    train_start_seconds = 1728000.0
    timesteps = np.arange(len(sac.temp_history)) * (step_size / 3600.0)
    timesteps += train_start_seconds / 3600.0  # offset by 20 days
    plt.plot(timesteps, sac.temp_history, label="Tin (train)")
    plt.plot(timesteps, sac.outdoor_history, label="Tout (train)")
    plt.plot(timesteps, sac.setpoint_history, linestyle="--", color="gray", label="Setpoint")
    plt.title("Training Temperature Trajectories (Day 20+)")
    plt.xlabel("Time (hours since t=0)")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    tin_hist  = test_env.indoor_list
    tout_hist = test_env.outdoor_list
    pwr_hist  = test_env.actions_list
    timesteps = np.arange(len(tin_hist)) * (step_size / 3600.0)
    
    # Adjust setpoint history to match indoor temp list
    if len(test_setpoint_hist) < len(tin_hist):
        test_setpoint_hist.insert(0, test_setpoint_hist[0])  # repeat first value
    
    plt.figure(figsize=(10,4))
    plt.plot(timesteps, tin_hist,  label="Indoor Temp")
    plt.plot(timesteps, tout_hist, label="Outdoor Temp")
    plt.plot(timesteps, test_setpoint_hist, linestyle="--", color="gray", label="Setpoint")
    plt.title(f"Test Temperatures over {test_horizon_steps} days")
    plt.xlabel("Time (hours)")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10,4))
    plt.plot(timesteps[:len(pwr_hist)], pwr_hist, label="Heater Power (W)")
    plt.title(f"Test Heater Power over {test_horizon_steps} days")
    plt.xlabel("Time (hours)")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("Writing Training Results to Excel File")
        # Metadata dictionary
    meta_data = {
        'script': script_name,
        'episodes': args.episodes,
        'start time': start_time * step_size,
        'number of steps': horizon_steps,
        'step size': step_size,
        'gamma': sac.gamma,
        'tau': sac.tau,
        'target entropy': sac.target_entropy,
        'batch size': sac.batch_size,
        'replay buffer size': sac.buffer_capacity,
        'network update frequency': sac.update_interval,
        'updates per step': sac.updates_per_step,
        'actor lr': sac.actor_lr,
        'critic lr': sac.critic_lr,
        'alpha lr': sac.alpha_lr,
        'alpha start': sac.alpha,
        'alpha end': sac.log_alpha.exp().item(),
        'noise start': sac.noise_start,
        'noise end': sac.noise_end,
        'noise clip': sac.noise_clip,
        'control type': 'Python',
        'fmu': fmu_filename,
        'temp factor': train_env.temp_factor,
        'energy factor': train_env.energy_factor,
        'loss function': 'MSELoss',
        'optimiser': 'Adam'
    }

    # DataFrames
    df_setpoints = pd.DataFrame(all_episodes_setpoints).T
    df_indoor_temp = pd.DataFrame(all_episodes_indoor_temps).T
    df_outdoor_temp = pd.DataFrame(all_episodes_outdoor_temps).T
    df_actions = pd.DataFrame(all_episodes_actions).T
    df_energy = pd.DataFrame(all_episodes_energy).T
    df_rewards = pd.DataFrame(all_episodes_rewards).T
    df_q1_loss = pd.DataFrame(all_episodes_q1_loss).T
    df_q2_loss = pd.DataFrame(all_episodes_q2_loss).T
    df_actor_loss = pd.DataFrame(all_episodes_actor_loss).T
    df_alpha_loss = pd.DataFrame(all_episodes_alpha_loss).T

    start_clock = datetime.now().strftime("%H:%M:%S")
    end_clock = datetime.now().strftime("%H:%M:%S")
    df_time = pd.DataFrame([start_clock, end_clock])
    df_meta = pd.DataFrame.from_dict(meta_data, orient='index')

    title = f"SAC_Results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    with pd.ExcelWriter(f'{title}.xlsx') as writer:
        df_meta.to_excel(writer, sheet_name='Metadata')
        df_setpoints.to_excel(writer, sheet_name='Setpoints')
        df_indoor_temp.to_excel(writer, sheet_name='Indoor Temps')
        df_outdoor_temp.to_excel(writer, sheet_name='Outdoor Temps')
        df_actions.to_excel(writer, sheet_name='Actions')
        df_energy.to_excel(writer, sheet_name='Energy')
        df_rewards.to_excel(writer, sheet_name='Rewards')
        df_q1_loss.to_excel(writer, sheet_name='Q1 Loss')
        df_q2_loss.to_excel(writer, sheet_name='Q2 Loss')
        df_actor_loss.to_excel(writer, sheet_name='Actor Loss')
        df_alpha_loss.to_excel(writer, sheet_name='Alpha Loss')
        df_time.to_excel(writer, sheet_name='Time')

    overall_end = datetime.now()
    runtime = overall_end - overall_start
    print(f"✅ Script completed at: {overall_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️ Total runtime: {runtime}")

if __name__ == "__main__":
    main()

