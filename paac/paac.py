import gym
import gym.wrappers
import gym.vector
import torch
import argparse
from torch import nn
from torch.nn import functional as F


class StackFrames(nn.Module):
    def __init__(self, num_stack, axis=-1, batch_first=False):
        super().__init__()
        self.num_stack = num_stack
        self.axis = axis
        self.time_axis = 1 if batch_first else 0
        self.batch_first = batch_first

    def forward(self, *args):
        if len(args) == 1:
            seqs, lens = nn.utils.rnn.pad_packed_sequence(args[0], self.batch_first)
            is_packed = True
        else:
            seqs, lens = args
            is_packed = False
        first_frame = seqs.select(self.time_axis, 0).unsqueeze(self.time_axis)
        first_frame_shape = list(seqs.shape)
        output_shape = list(seqs.shape)
        first_frame_shape[self.time_axis] = self.num_stack - 1 
        output_shape[self.axis] *= self.num_stack
        first_frame = first_frame.expand(*first_frame_shape)
        padded_seqs = torch.cat((first_frame, seqs), self.time_axis) 
        output_seqs = torch.zeros(output_shape, dtype=seqs.dtype, device=seqs.device)
        for i in range(self.num_stack):
            values = padded_seqs.index_select(self.time_axis, torch.arange(i, i + seqs.shape[self.time_axis]))
            output_seqs.index_add_(self.axis, torch.arange(i*seqs.shape[self.axis],(i+1)*seqs.shape[self.axis]), values)
        mask = torch.arange(seqs.shape[self.time_axis]).expand(len(lens), seqs.shape[self.time_axis]) >= lens.unsqueeze(1)
        if not self.batch_first:
            mask = mask.t()
        mask = mask.view(*(mask.shape + tuple(1 for _ in seqs.shape[2:])))
        output_seqs.masked_fill_(mask, 0)
        if not is_packed:
            return output_seqs, lens
        else:
            return nn.utils.rnn.pack_padded_sequence(output_seqs, lens, self.batch_first)


class TorchWrapper(gym.Wrapper):
    def __init__(self, env, device=torch.device('cpu')):
        super().__init__(env)
        self.device = device

    def step(self, actions):
        obs, rew, done, stats = self.env.step(actions.detach().cpu().numpy())
        obs = torch.from_numpy(obs)
        rew = torch.from_numpy(rew)
        done = torch.from_numpy(done)
        obs = obs.transpose(1, 3).float() / 255.0
        return obs, rew, done, stats

    def reset(self):
        obs = torch.from_numpy(self.env.reset())
        obs = obs.transpose(1, 3).float() / 255.0
        return obs



class Network(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.stack_frames = StackFrames(4, -3)
        self.conv_stack = nn.Sequential(
            nn.Conv2d(args.num_stack, 32, 8, stride = 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride = 1),
            nn.ReLU(),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(32 * 7 * 7, 512),
            nn.ReLU(),
        )

        self.critic = nn.Linear(512, 1)
        self.policy_logits = nn.Linear(512, args.num_actions)
        nn.init.orthogonal_(self.policy_logits.weight.data, 0.01) 

    def forward(self, x, lens):
        with torch.no_grad():
            x, lens = self.stack_frames(x, lens)
        batch_shape = x.shape[:2]
        x = x.view(-1, *x.shape[2:])
        x = self.conv_stack(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_layer(x)
        values = self.critic(x).view(*(batch_shape + (-1,)))
        policy_logits = self.policy_logits(x).view(*(batch_shape + (-1,)))
        return values, policy_logits


def build_env(name):
    env = gym.make(name)#, width=84, height=84)
    env = gym.wrappers.ResizeObservation(env, (84, 84)) 
    env = gym.wrappers.GrayScaleObservation(env, True)
    return env

class Mean:
    def __init__(self):
        self.reset()

    def __call__(self, value=None, weight=1.0):
        if value is not None:
            # Update mode
            self._cumsum += value
            self._counts = weight
        else:
            if self._counts == 0: return 0
            return self._cumsum / self._counts

    def reset(self): 
        self._counts = 0
        self._cumsum = 0

class Model:
    def __init__(self, args, device=torch.device('cpu')):
        self.args = args 
        self.env = gym.vector.SyncVectorEnv([lambda: build_env(args.env) for _ in range(args.num_envs)])
        self.env = TorchWrapper(self.env, device=device)
        args.num_actions = self.env.action_space[0].n
        self.model = Network(args).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.device = device
        self.metrics = {
            'loss': Mean(),
            'return': Mean(),
        }


    def _train_on_batch(self, states, actions, rewards, continues, lens):
        values, policy_logits = self.model(states, lens + 1)
        policy_logits = policy_logits[:-1]
        next_values = values[-1]
        values = values[:-1]

        with torch.no_grad():
            returns = torch.cat((rewards, torch.zeros_like(rewards[:1])), 0)
            returns[-1] = next_values.squeeze(-1).detach() * continues
            for i in range(values.shape[0] - 1, -1, -1):
                returns[i] += returns[i + 1] * self.args.gamma 
            returns = returns[:-1]

        d = torch.distributions.Categorical(logits=policy_logits)
        ploss = -d.log_prob(actions) * (returns - values.squeeze(-1).detach())
        vloss = 0.5 * F.mse_loss(values.squeeze(-1), returns, reduction='none')
        entropy = self.args.entropy_weight * d.entropy()

        mask = torch.arange(states.shape[0] - 1).expand(len(lens), states.shape[0] - 1) >= lens.unsqueeze(1)
        mask = mask.t()
        weight = (~mask).sum()

        ploss.masked_fill_(mask, 0)
        vloss.masked_fill_(mask, 0) 
        entropy.masked_fill_(mask, 0)
        ploss = ploss.sum() / weight
        vloss = vloss.sum() / weight
        entropy = entropy.sum() / weight
        loss = ploss + vloss + -entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
        self.optimizer.step()

        self.metrics['loss'](loss.item(), len(lens))
        self.metrics['return'](rewards.sum().item(), len(lens)) 
        return loss.item()

    def _act_on_batch(self, states): 
        lens = torch.ones((states.shape[0],), dtype=torch.int64, device=states.device)
        _, policy_logits = self.model(states.unsqueeze(0), lens) 
        d = torch.distributions.Categorical(logits=policy_logits)
        return d.sample().squeeze(0)

    def _train_step(self):
        batch_states = []
        batch_lens = []
        batch_continues = []
        batch_actions = []
        batch_rewards = []

        running_states = [list() for _ in range(self.args.num_envs)]        
        running_actions = [list() for _ in range(self.args.num_envs)]        
        running_rewards = [list() for _ in range(self.args.num_envs)]        
        states = self.env.reset()
        for _ in range(args.env_steps):
            actions = self._act_on_batch(states)
            next_states, rewards, dones, _ = self.env.step(actions)
            for i in range(self.args.num_envs):
                running_states[i].append(states[i])
                running_actions[i].append(actions[i])
                running_rewards[i].append(rewards[i])
                if dones[i]:
                    running_states[i].append(next_states[i])
                    batch_lens.append(len(running_actions[i]))
                    batch_continues.append(False)
                    batch_states.append(running_states[i])
                    batch_actions.append(running_actions[i])
                    batch_rewards.append(running_rewards[i])
                    running_actions[i] = []
                    running_states[i] = []
                    running_rewards[i] = []
            states = next_states

        for i in range(self.args.num_envs):
            running_states[i].append(next_states[i])
            batch_lens.append(len(running_actions[i]))
            batch_continues.append(True)
            batch_states.append(running_states[i])
            batch_actions.append(running_actions[i])
            batch_rewards.append(running_rewards[i])

        batch_lens = torch.tensor(batch_lens, dtype=torch.int64, device=self.device)
        batch_continues = torch.tensor(batch_continues, dtype=torch.float32, device=self.device)
        batch_states = nn.utils.rnn.pad_sequence([torch.stack(x).float() for x in batch_states])
        batch_actions = nn.utils.rnn.pad_sequence([torch.stack(x) for x in batch_actions])
        batch_rewards = nn.utils.rnn.pad_sequence([torch.stack(x).float() for x in batch_rewards])
        return self._train_on_batch(batch_states, batch_actions, batch_rewards, batch_continues, batch_lens)

    def train(self): 
        for i in range(self.args.num_steps // (self.args.env_steps * self.args.num_envs)):
            self._train_step() 

            if i % 20 == 0:
                metrics = { key: item() for key, item in self.metrics.items() }
                metrics['step'] = i * self.args.env_steps * self.args.num_envs
                print('step: {step}, loss: {loss}, return: {return}'.format(**metrics))
                for _, v in self.metrics.items(): v.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='DeepmindLabSeekavoidArena01-v0')
    parser.add_argument('--num-envs', default=16, type=int)
    parser.add_argument('--num-stack', default=4, type=int)
    parser.add_argument('--num-steps', default=10**6, type=int)
    parser.add_argument('--env-steps', default=20, type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--entropy-weight', default=0.01, type=float)
    parser.add_argument('--grad-norm', default=0.5, type=float)
    args = parser.parse_args()
    model = Model(args)
    model.train()

