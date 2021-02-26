# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
""" GALA-A2C agent """

import torch.nn as nn
import torch.optim as optim
import time


class GALA_A2C():

    def __init__(self, actor_critic, value_loss_coef, entropy_coef, lr=None,
                 eps=None, alpha=None, max_grad_norm=None,
                 rank=0, gossip_buffer=None):
        """ GALA_A2C """

        self.rank = rank
        self.gossip_buffer = gossip_buffer
        self.actor_critic = actor_critic

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.RMSprop(
            actor_critic.parameters(), lr, eps=eps, alpha=alpha)
        self.mix_counter = 0

    def update(self, rollouts):
        start_time = time.time()
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef).backward()

        nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                 self.max_grad_norm)

        self.optimizer.step()
        end_time = time.time()
        # Local-Gossip
        train_time = end_time - start_time
        
        start_time = time.time()

        if self.gossip_buffer is not None:
            write_start = time.time()
            #print(type(self.actor_critic))
            #model_write = self.actor_critic.cpu()

            self.gossip_buffer.write_message(self.rank, self.actor_critic)
            #self.gossip_buffer.write_message(self.rank, model_write)
            write_end = time.time()
            aggregate_start = time.time()

            self.gossip_buffer.aggregate_message(self.rank, self.actor_critic)

            aggregate_end = time.time()

            self.mix_counter += 1

        end_time = time.time()
        buffer_time = end_time - start_time
        write_time = write_end - write_start
        aggregate_time = aggregate_end - aggregate_start
        aggregate_lock_time = self.gossip_buffer.aggregate_time_end - self.gossip_buffer.aggregate_time_start
        aggregate_circle_time = self.gossip_buffer.aggregate_circle_time_end - self.gossip_buffer.aggregate_circle_time_start
        aggregate_time_middle = self.gossip_buffer.aggregate_time_start - self.gossip_buffer.aggregate_circle_time_end
        return value_loss.item(), action_loss.item(), dist_entropy.item(), self.mix_counter, buffer_time, train_time, write_time, aggregate_time, aggregate_lock_time,aggregate_circle_time, aggregate_time_middle
