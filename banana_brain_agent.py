from banana_brain_model import BananaBrainQNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from collections import namedtuple, deque

# important if model can be trained on GPUs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

class BananaBrainAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size=1, action_size=1, seed=0, hidden_layers=[1,1], drop_p=0., model_checkpoint_file = None):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        
        if model_checkpoint_file is None:
            self.state_size = state_size
            self.action_size = action_size
            self.hidden_layers = hidden_layers
        else:
            checkpoint = torch.load(model_checkpoint_file)
            self.state_size = checkpoint['input_size']
            self.action_size = checkpoint['output_size']
            self.hidden_layers = checkpoint['hidden_layers']
            
        self.seed = random.seed(seed)

        # instantiate the two dual Q-Networks used for estimating and learning Q function (local and target)
        self.qnetwork_local = BananaBrainQNetwork(self.state_size, 
                                                  self.action_size, 
                                                  seed, 
                                                  self.hidden_layers, 
                                                  drop_p).to(device)
        if model_checkpoint_file is not None:
            self.qnetwork_local.load_state_dict(checkpoint['state_dict'])

        self.qnetwork_target = BananaBrainQNetwork(self.state_size, 
                                                   self.action_size, 
                                                   seed, 
                                                   self.hidden_layers, 
                                                   drop_p).to(device)
        
        # initilize the optimizer for the local BananaBrainQNetwork parameters - the one used for learning
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # initialize the criterion/loss fuction used by BananaBrainQNetwork traiing
        self.criterion = nn.MSELoss()
        
        # Initialize the Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # transform to torch tensor the input state from the environment
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # set the local BananaBrainQNetwork in evaluation mode
        self.qnetwork_local.eval()
        
        # Turn off gradients for evaluating Q function for the provided state (saves memory and computations)
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
            
        # set the local BananaBrainQNetwork in evaluation mode
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

        
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get the max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = self.criterion(Q_expected, Q_targets)
        
        # set the optimizer gradients to zero (should not cumulate)
        self.optimizer.zero_grad()
        
        # perform the back-propagation through the network and calculate the gradients
        loss.backward()
        
        # perform update of the BananaBrainQNetwork' weitghts
        self.optimizer.step()
        

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def save_agent_model(self, model_file_name):
        # save model when environment solved (more than 13 points in avrage over last 100 consecutive episodes)
        checkpoint = {
                'input_size': self.state_size,
                'output_size': self.action_size,
                'hidden_layers': [each.out_features for each in self.qnetwork_local.hidden_layers],
                'state_dict': self.qnetwork_local.state_dict()
        }
        torch.save(checkpoint, "./banana-model-checkpoint.pth")   

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)