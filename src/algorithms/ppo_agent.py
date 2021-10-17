from src.algorithms.base_agent import BaseAgent
from src.algorithms.models import ActorCritic
from torch.optim import Adam
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class PpoAgent(BaseAgent):
    
    def __init__(self, env, config):
        super().__init__()
        self.config = config
        self.env = env
        self.buffer = RolloutBuffer()
        
        save_dir = self.config.model_dir
        if self.config.env == "LunarLander":
            save_dir += "/LunarLander"
        if self.config.env == "RocketLeague":
            save_dir += "/RocketLeague"
            
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.save_path = save_dir + "/policy.pth"
        
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.n    
        
        self.policy = ActorCritic(input_size, output_size)
        self.optimizer = Adam([
                        {'params': self.policy.actor.parameters(), 'lr': self.config.lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': self.config.lr_critic}
                    ])

        self.policy_old = ActorCritic(input_size, output_size)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
        
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action, log_probs = self.policy_old.act(state)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(log_probs)
        return action.item()
    
    def update_parameters(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.config.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()
        
        # Optimize policy for K epochs
        for _ in range(self.config.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.config.clip, 1+self.config.clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def train(self):
        # printing and logging variables
        print_running_reward = 0
        print_running_episodes = 0
        
        log_running_reward = 0
        log_running_episodes = 0
        
        time_step = 0
        i_episode = 0
        
        # track total training time
        start_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)
        print("============================================================================================")
        
        while time_step <= self.config.max_training_timesteps:
            state = self.env.reset()
            current_ep_reward = 0

            for t in range(1, self.config.max_ep_len+1):
                if self.config.render:
                    self.env.render()

                # select action with policy
                action = self.select_action(state)
                state, reward, done, _ = self.env.step(action)
    
                # saving reward and is_terminals
                self.buffer.rewards.append(reward)
                self.buffer.is_terminals.append(done)
    
                time_step +=1
                current_ep_reward += reward
    
                # update PPO agent
                if time_step % self.config.update_timestep == 0:
                    print(flush=True)
                    print("Updating policy",flush=True)
                    print(flush=True)
                    self.update_parameters()
                    
                # printing average reward
                if time_step % self.config.print_freq == 0:
    
                    # print average reward till last episode
                    print_avg_reward = print_running_reward / print_running_episodes
                    print_avg_reward = round(print_avg_reward, 2)
    
                    print(flush=True)
                    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward),flush=True)
                    print(flush=True)
                    
                    print_running_reward = 0
                    print_running_episodes = 0
    
                # save model weights
                if time_step % self.config.save_model_freq == 0:
                    print(flush=True)
                    print("--------------------------------------------------------------------------------------------",flush=True)
                    print("saving model",flush=True)
                    self.save()
                    print("model saved",flush=True)
                    print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time,flush=True)
                    print("--------------------------------------------------------------------------------------------",flush=True)
                    print(flush=True)
    
                # break; if the episode is over
                if done:
                    break
    
            print_running_reward += current_ep_reward
            print_running_episodes += 1
    
            log_running_reward += current_ep_reward
            log_running_episodes += 1
    
            i_episode += 1
        
        self.env.close()

        # print total training time
        print("============================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)
        print("Finished training at (GMT) : ", end_time)
        print("Total training time  : ", end_time - start_time)
        print("============================================================================================")
        
    def test(self):
        self.load()
        time_step = 0
        i_episode = 0
        print_running_reward = 0
        print_running_episodes = 0
        
        while True:
            current_ep_reward = 0
            state = self.env.reset() 
            
            for t in range(1, self.config.max_ep_len+1):
                self.env.render()
                action = self.select_action(state)
                state, reward, done, _ = self.env.step(action)
                time_step +=1
                current_ep_reward += reward
                
                # printing average reward
                if time_step % self.config.print_freq == 0:
    
                    # print average reward till last episode
                    print_avg_reward = print_running_reward / print_running_episodes
                    print_avg_reward = round(print_avg_reward, 2)
    
                    print(flush=True)
                    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward),flush=True)
                    print(flush=True)
                    
                    print_running_reward = 0
                    print_running_episodes = 0
                
                if done:
                    break
                
            print_running_reward += current_ep_reward
            print_running_episodes += 1
            i_episode += 1

    def save(self):
        torch.save(self.policy_old.state_dict(), self.save_path)
   
    def load(self):        
        self.policy_old.load_state_dict(torch.load(self.save_path))
        self.policy.load_state_dict(torch.load(self.save_path))

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            