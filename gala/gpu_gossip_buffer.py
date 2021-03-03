# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Gossip Buffer

:author: Mido Assran
:description: Class defines a shared-memory Gossip-Buffer, which allows
    multi-processed asynchronous agents on the same machine to communicate
    tensors to on one-another
"""

import copy
import torch
import time
import os


class GossipBuffer():

    def __init__(self, topology, model, read_events,
                 write_events, sync_list, proc_manager ,sync_freq=0, num_nodes = 1, msg_buffer = 0):
        """ GossipBuffer """

        self.topology = topology
        self.num_learners = len(topology) # all learners in the cluster
        self.sync_list = sync_list
        self.sync_freq = sync_freq

        self.aggregate_time_start = 0
        self.aggregate_time_end = 0
        self.aggregate_circle_time_start = 0
        self.aggregate_circle_time_end = 0
        self.num_nodes = num_nodes
        self.num_learner_per_node = int(self.num_learners/self.num_nodes)
        # Initialize message buffer (4-item object):
        # [0] -> Msg-Tensor
        # [1] -> Events recording peers that have read the message
        # [2] -> Events recording peer that has written the message
        # [3] -> Lock for safe access of Msg-Tensor
        #self.msg_buffer = []
        self.read_events = read_events
        self.write_events = write_events
        self.msg_buffer = proc_manager.get_msg_buffer()
        print('num_learners', self.num_learners)
        #all_learner = self.num_learners * self.num_nodes
        '''for rank in range(self.num_learners):
            msg = copy.deepcopy(model)
            msg.share_memory()
            #r_events = read_events[rank]
            #w_events = write_events[rank]

            r_events = copy.copy(read_events[rank*self.num_learners : (rank+1)*self.num_learners]) #shallow copy problem it is not shallow copy
            w_events = copy.copy(write_events[rank*self.num_learners : (rank+1)*self.num_learners])
            #r_events = read_events
            #w_events = write_events
            counter = 0
            #print(r_events)
            print(rank * self.num_learners, ':', (rank + 1) * self.num_learners)
            #lock = buffer_locks[rank]
            #self.msg_buffer.append(proc_manager.list([msg, r_events, w_events, lock]))
            self.msg_buffer.append(list([msg, r_events, w_events]))

        # Initialize each Read-Buffer as 'read'
        for i in range(self.num_learners):
            read_event_list = self.msg_buffer[i][1] #todo use shallow copy or the change won't work
            #for event in read_event_list:
                #event.set()
            print(len(self.read_events))
            for j in range(len(read_event_list)): #todo use shallow copy or itself
                read_event_list[j] = True
                self.read_events[i * self.num_learners + j] = True
            print(r_events)
        print('self.msg_buffer',len(self.msg_buffer))
        print('read_events',len(read_events))
        print('read_event_list',len(read_event_list))
        #print('self.msg_buffer', list(self.msg_buffer))'''

    def write_message(self, rank, model, rotate=False):
        """
        Write agent 'rank's 'model' to a local 'boradcast buffer' that will be
        read by the out-neighbours defined in 'self.topology'.

        :param rank: Agent's rank in multi-agent graph topology
        :param model: Agent's torch neural network model
        :param rotate: Whether to alternate peers in graph topology

        Agents should only write to their own broadcast buffer:
            i.e., ensure 'model' belongs to agent 'rank'
        WARNING: setting rotate=True with sync_freq > 1 not currently supported
        """
        with torch.no_grad():
            # Get local broadcast-buffer
            #MAYBE WE NEED to LOAD event from server directly
            '''msg_buffer = self.msg_buffer[rank]
            broadcast_buffer = msg_buffer[0]
            read_event_list = msg_buffer[1]
            write_event_list = msg_buffer[2]'''
            broadcast_buffer = self.msg_buffer[rank]
            read_event_list = self.read_events[rank*self.num_learners : (rank+1)*self.num_learners]
            write_event_list = self.write_events[rank*self.num_learners : (rank+1)*self.num_learners]
            #lock = msg_buffer[3]
            # Check if out-peers finished reading our last message
            out_peers, _ = self.topology[rank].get_peers()
            #print(os.getpid(),'outpeers', out_peers)
            read_complete = True
            for peer in out_peers:
                #if not read_event_list[peer].is_set():
                if not read_event_list[peer] == True: #todo this only define it's true or false so we don't need to
                    read_complete = False
                    break
            # If peers done reading our last message, wait and clear events
            if read_complete:
                for peer in out_peers:
                    #todo use itself to simulate event
                    #read_event = read_event_list[peer] #noNeedtodo each assignment need to give new value of list
                    #read_event.wait() #等待其他进程都set了
                    while True:
                        if read_event_list[peer] == True:
                            break
                        else:
                            pass
                    #read_event.clear()
                    read_event_list[peer] = False
                    self.read_events[rank*self.num_learners+peer] = False #broadcast the change to network
            # If not done reading, cannot write another message right now
            else:
                return

            # Update broadcast-buffer with new message
            # -- flatten params and multiply by mixing-weight
            num_peers = self.topology[rank].peers_per_itr
            #with lock:
            for bp, p in zip(broadcast_buffer.parameters(),
                             model.parameters()):
                bp.data.copy_(p)
                bp.data.div_(num_peers + 1)
            # -- mark message as 'written'
            out_peers, _ = self.topology[rank].get_peers(rotate)
            torch.cuda.current_stream().synchronize()
            for peer in out_peers:
                #write_event_list[peer].set()
                #todo use itself simulate event
                write_event_list[peer] = True
                self.write_events[rank*self.num_learners + peer] = True


    def aggregate_message(self, rank, model):
        """
        Average messages with local model:
        Average all in-neighbours' (defined in 'self.topology') parameters with
        agent 'rank's 'model' and copy the result into 'model'.

        Agents should only aggregate messages into their own model:
            i.e., ensure 'model belongs to agent 'rank'
        """
        with torch.no_grad():
            # Check if in-peers finished writing messages to broadcast buffers
            _, in_peers = self.topology[rank].get_peers()

            write_complete = True
            self.aggregate_circle_time_start = time.time()
            for peer in in_peers:
                #peer_buffer = self.msg_buffer[peer]
                #write_event = peer_buffer[2][rank]
                write_event = self.write_events[peer*self.num_learners + rank]
                #if not write_event.is_set():
                #todo use itself to simulate event
                if not write_event == True:
                    write_complete = False #如果节点还没有set，则没写完
                    break
            # Check if any messages are excessively stale
            stale_assert = self.sync_list[rank] >= self.sync_freq
            
            self.aggregate_circle_time_end = time.time()
            # If peers done writing or message too stale, wait and clear events
            if write_complete or stale_assert:
                for peer in in_peers:
                    #peer_buffer = self.msg_buffer[peer]
                    #write_event = peer_buffer[2][rank]
                    #write_event.wait()
                    while True:
                        #if peer_buffer[2][rank] == True: #todo use peerbuffer itself to simulate event
                        if self.write_events[peer*self.num_learners + rank] == True:
                            break
                        else:
                            pass
                    #write_event.clear() #加锁
                    #peer_buffer[2][rank] = False

                    self.write_events[peer*self.num_learners + rank] = False
                    self.sync_list[rank] = 0
            # Not done writing, but staleness is still tolerable
            else:
                self.sync_list[rank] += 1
                print('%s: staleness %s' % (rank, self.sync_list[rank]))
                return
            
            self.aggregate_time_start = time.time()
            # Lazy-mixing of local params
            num_peers = self.topology[rank].peers_per_itr
            for p in model.parameters():
                p.data.div_(num_peers + 1)

            # Aggregate received messages
            for peer in in_peers:
                #peer_buffer = self.msg_buffer[peer]
                #lock = peer_buffer[3]
                #with lock:
                # Read message and update 'params'
                #peer_msg = peer_buffer[0]
                peer_msg = self.msg_buffer[peer]
                peer_msg.to('cuda:0')
                model.to('cuda:0')
                for p, bp in zip(model.parameters(),
                                 peer_msg.parameters()):
                    p.data.add_(bp.to(p.device, non_blocking=True))
                torch.cuda.current_stream().synchronize()
                # Mark message as 'read'
                #peer_buffer[1][rank].set() #聚合完成，设置为可写入
                #todo use itself to simulate
                #peer_buffer[1][rank] = True
                self.read_events[peer*self.num_learners + rank] = True
            #model.cuda()
            self.aggregate_time_end = time.time()

