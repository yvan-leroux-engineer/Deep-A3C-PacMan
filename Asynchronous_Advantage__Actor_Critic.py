import os
import time
import gym
import numpy as np
import matplotlib.pyplot as plt
from attrdict import AttrDict


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from torch.autograd import Variable


class A3C_Functions():
    '''
    Functions needed to control the network

    Functions :
        action : choose an action based on the state
        update : update the local parameters of the agent
        refresh_state : refresh the LSTM items
    '''
    def __init__(self, env, lr = 1e-3, gamma = 0.99):
        super().__init__()
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.input_shape = self.env.observation_space.shape
        self.model = A3C_Neural_Network(
            self.env.observation_space.shape[2],
            self.env.action_space.n,
        )
        # Same optimize as the paper
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=1e-3)
        self.hx = None
        self.cx = None

    def action(self, state, training=False):
        '''
        Choose an action in regard of the policy network.

        Args:
            state: An observation from the environment
            training: bool, whether is training or not

        Return:
            The action choosed according to softmax policy
            If training == True, it will also return value, log probability and entropy.
        '''
        state = torch.tensor(state)
        # Reshaping the state to 3x210x160, needed for the conv2D in A3C_Neural_Network
        state = state.view(1, self.input_shape[2], self.input_shape[0], self.input_shape[1])
        state = state.type(torch.FloatTensor)
        self.cx = Variable(self.cx.data)
        self.hx = Variable(self.hx.data)

        # Pass the state matrix threw the A3C_Neural_Network
        value, logit, (self.hx, self.cx) = self.model((state, (self.hx, self.cx)))

        # Determining the action to choose with the multinomial function
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        action = prob.multinomial(num_samples=1).data
        log_prob = log_prob.gather(1, Variable(action))
        action = action.numpy()

        # Return the variables
        if training:
            return action, value, log_prob, entropy
        else:
            return action, value.data.numpy()[0][0]

    def update(self,config, rewards, values, log_probs, entropies, R):
        '''
        Compute gradients and backpropagation

        Args:
            rewards = [r1, r2, ..., r_t-1]
            log_probs = [p1, p2, ..., p_t-1]
            entropies = [e1, e2, ..., e_t-1]
            R: 0 for terminal s_t otherwise V(s_t)
        '''
        value_loss = 0
        policy_loss = 0
        for t in reversed(range(len(rewards))):
            R = self.gamma * R + rewards[t]
            td_error = R - values[t]
            # maximize the return
            with torch.no_grad():
                advantage = td_error
            policy_loss = policy_loss - \
                (log_probs[t] * advantage + config.entropies_factor * entropies[t])
            # loss minimize
            value_loss = value_loss + td_error.pow(2)
        self.optimizer.zero_grad()
        loss = policy_loss + config.value_loss_factor * value_loss
        loss.backward()

    def refresh_state(self):
        '''
        Reinitialize the LSTM items cx, hx
        '''
        self.cx = Variable(torch.zeros(1, 256))
        self.hx = Variable(torch.zeros(1, 256))


class A3C_Neural_Network(torch.nn.Module):
    '''
    Initialize the network with the given paramaters

    Args :
        n_channels : number of channel for the state (ex 3 for RGB)
        n_actions : number of possible actions for the game (ex Ms PacMan = 9)
    '''
    def __init__(self, n_channels, n_actions):
        super(A3C_Neural_Network, self).__init__()
        self.n_channels = n_channels
        self.n_actions = n_actions
        # 3x 210 x 160
        self.conv1 = nn.Conv2d(n_channels, 16, 8, stride=4)
        self.bn1 = nn.BatchNorm2d(16)
        # 16 x 51 x 39
        self.conv2 = nn.Conv2d(16, 32, 6, stride=3, padding=(0, 2))
        self.bn2 = nn.BatchNorm2d(32)
        # 32 x 16 x 13
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        # 32 x 7 x 5 -> flatten before LSTM
        self.lstm = nn.LSTMCell(1120, 256)

        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, n_actions)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

    def forward(self, x):
        x, (hx, cx) = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(1, -1) # Similar to Flatten
        hx, cx = self.lstm(x, (hx, cx))

        x = hx
        # Return the state value, the policy, and LSTM items
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)


def main(config):
    '''
    Main function called when the script is initalized

    It initializes the global agent with saved paramaters if exists
    Then start the multiprocessing process with :
        - One core to evaluate periodically the global agent
        - All other cores for local agents to train

    '''
    # initialize and load global shared network
    mp.set_start_method('spawn', force=True)
    env = gym.make('MsPacman-v0')
    n_channels = env.observation_space.shape[2]
    n_actions = env.action_space.n
    shared_model = A3C_Neural_Network(n_channels, n_actions)

    # load parameters
    weight_path = os.path.join(config.path,'last_model.pt')
    if os.path.exists(weight_path):
        print(f"Load weights from {weight_path}")
        saved_state = torch.load(weight_path)
        shared_model.load_state_dict(saved_state())
    
    # Function to let local agent running with multiprocessing to access the global paramaters
    shared_model.share_memory()

    # Handling list for all processes
    processes = []
    # start 1 evaluating process
    p = Process(target=evaluate, args=(config, shared_model))
    p.start()
    processes.append(p)
    # start multiple training process
    for rank in range(0, config.max_workers):
        p = Process(target=train, args=(config, shared_model, rank))
        p.start()
        processes.append(p)
        time.sleep(0.1)
    for p in processes:
        p.join()


def train(config, shared_model, rank):
    ''' One training process
    Create local model and environment, update parameters according to A3C algorithm, synchronous shared model and local model.

    Args:
        config: paramaters of the model
        shared_model: the model of global agent
    '''
    print(f"Process {rank} start training")
    # Seed fixed for reprodution purpose
    torch.manual_seed(rank) 

    # Local enviroment creation
    env = gym.make('MsPacman-v0')
    controller = A3C_Functions(env, config)
    env.seed(rank)
    controller.model.train()
    optimizer = optim.RMSprop(shared_model.parameters(), lr=1e-3)

    t = 1
    #Load saved step if exists
    if os.path.exists(f"{config.path}saved_model/Process_{rank}.txt") :
        done_step = np.loadtxt(f"{config.path}saved_model/Process_{rank}.txt")
        done_step = int(done_step)
    else :
        done_step =0

    done = True
    while True:
        # Local lists reinitializing
        actions = [] 
        rewards = []
        log_probs = []
        values = []
        entropies = []
        # Reset the environement and LSTM states are done
        if done:
            observation = env.reset()
            controller.refresh_state()
            done = False
        # Load the global paramaters to the local agent
        controller.model.load_state_dict(shared_model.state_dict())
        # Start simulation
        t_start = t
        while not done and t - t_start < config.t_max: # Iterates for t_max set in config
            action, value, log_prob, entropy = controller.action(observation, training=True)
            observation, reward, done, _ = env.step(action)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            entropies.append(entropy)
            t += 1
        # Update local parameters
        R = torch.zeros(1, 1)
        if not done:
            _, value, _, _ = controller.action(observation, training=True)
            R = value.data
        optimizer.zero_grad()
        controller.update(config,rewards, values, log_probs, entropies, R)

        # Upload the gradient to global agent, then iterate his optimizer
        for param, shared_param in zip(controller.model.parameters(),
                                    shared_model.parameters()):
            shared_param._grad = param.grad
        optimizer.step()

        # Display the number of episodes done for each agent
        if done:
            done_step +=1
            print(f"Process {rank} done {done_step} episodes")
            np.savetxt(f"{config.path}saved_model/Process_{rank}.txt", [done_step])


def evaluate(config, shared_model):
    ''' The evaluation process
    Evaluate the shared model for a given times, return the mean score, then reupload global paramaters to revaluate

    Args:
        config: configurations of env and model
        shared_model: the PyTorch model of global actor
    '''
    print(f"Evaluation process starts")

    # Setting up local environement for testing
    env = gym.make('MsPacman-v0')
    controller = A3C_Functions(env, config)
    
    # Load saved rewards if exists
    if os.path.exists(os.path.join(config.path,'rewards_list')) :
        rewards = list(np.loadtxt(os.path.join(config.path,'rewards_list')))
        print(f"Load rewards list from {os.path.join(config.path,'rewards_list')}")
    else :
        rewards = []
    
    evaluate_episodes = config.evaluate_episodes

    while True:
        # Load paramaters of the global agent
        controller.model.load_state_dict(shared_model.state_dict())
        controller.model.eval()

        # Iterate for a given number of episodes and return the mean of the scores
        i, mean_reward, reward_l = 0, 0, 0
        while i < evaluate_episodes:
            reward_l = _model_testing(controller, env)
            mean_reward += reward_l
            i += 1
            print(f"Evaluation : Episode {i}/{evaluate_episodes} done, reward = {reward_l}")
        mean_reward /= evaluate_episodes
        rewards.append(mean_reward)

        # Ploting and saving the learning curve
        _plot_learning_curve(rewards, config)

        # Displaying some information
        print(f"Evaluation : Cycle n°{len(rewards)} , Mean reward = {mean_reward}")
        
        # Every 25 evaluation the model weights and reward list is saved with the iteration name 
        if len(rewards)%25 == 0 :
            torch.save(controller.model.state_dict, os.path.join(config.path,f'saved_model/model_iter_{len(rewards)}.pt')) 
            print(f"iter {len(rewards)}, model saved as model_iter_{len(rewards)}.pt")       

        # Saving model weights and rewards list
        np.savetxt(os.path.join(config.path,'rewards_list'),rewards)
        torch.save(controller.model.state_dict, os.path.join(config.path,'last_model.pt'))

        time.sleep(config.time_sleep)
        

def _plot_learning_curve(rewards, config):
    '''
    Plot the data from the evaluation of the global agent, and save it under learning curve
    '''
    def mean_for_five(x=5):
        return [np.mean(rewards[i:i+x]) for i in range(int(len(rewards)-x))]

    plt.clf()
    plt.cla()
    plt.close()
    plt.plot(rewards, 'g-')
    plt.plot([np.mean(rewards)]*len(rewards), 'r-')
    plt.plot(mean_for_five(), 'b-')
    plt.title('Learning Curve')
    plt.xlabel('Number of Evaluation')
    plt.ylabel(f'Mean Reward on {config.evaluate_episodes} episodes')
    plt.legend(['a3c','mean','means for 5 last values'], loc='best')
    plt.savefig(f"{config.path}/learning_curve.png")


def _model_testing(controller, env):
    '''
    Return the score for a given model
    '''
    controller.refresh_state()
    observation = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = controller.action(observation, training=False)
        _, reward, done, _ = env.step(action)
        total_reward += reward
        
    
    return total_reward

# Centralized dictionnary to let tune paramaters
config = AttrDict({
    "path": r"/home/ylk/pjt/",
    "max_workers": 14, # One is set free for the evaluation process
    "evaluate_episodes": 10, # Number of episodes to evaluate
    "t_max": 400, # Number of actions to take in the environement
    "value_loss_factor": 0.9, # Factor of loss value in the global loss
    "entropies_factor":0.01, # Factor of entropies in the objective function
    "time_sleep":60 # Pause between 2 evaluations cycles
    })


if __name__ == '__main__':
    print(time.strftime("%Hh %Mm %Ss"))
    main(config)