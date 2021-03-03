# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed num_learnersunder the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import glob
import pprint as pp
import time
from collections import deque

import numpy as np
import torch
import torch.multiprocessing as mp

from gala.arguments import get_args
from gala.storage import RolloutStorage
from gala.model import Policy
from gala.gpu_gossip_buffer import GossipBuffer
from gala.gala_a2c import GALA_A2C
from gala.graph_manager import FullyConnectedGraph as Graph
from tensorboardX import SummaryWriter
from multiprocessing.managers import ListProxy, BarrierProxy, AcquirerProxy, EventProxy
mp.current_process().authkey = b'abc'

def actor_learner(args, rank, device, gossip_buffer):
    """ Single Actor-Learner Process """

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    #torch.cuda.set_device(device)
    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # (Hack) Import here to ensure OpenAI-gym envs only run on the CPUs
    # corresponding to the processes' affinity
    from gala import utils
    from gala.envs import make_vec_envs
    # Make envs
    envs = make_vec_envs(args.env_name, args.seed, args.num_procs_per_learner,
                         args.gamma, args.log_dir, device, False,
                         rank=rank)

    # Initialize actor_critic
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    # Initialize agent
    agent = GALA_A2C(
        actor_critic,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        alpha=args.alpha,
        max_grad_norm=args.max_grad_norm,
        rank=rank,
        gossip_buffer=gossip_buffer
    )

    rollouts = RolloutStorage(args.num_steps_per_update,
                              args.num_procs_per_learner,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    # Synchronize agents before starting training
    #barrier.wait() #TODO service
    #print('%s: barrier passed' % rank)

    # Start training
    start = time.time()
    num_updates = int(args.num_env_steps) // (
        args.num_steps_per_update
        * args.num_procs_per_learner
        * args.num_learners * 2)
    save_interval = int(args.save_interval) // (
        args.num_steps_per_update
        * args.num_procs_per_learner
        * args.num_learners * 2)
    writer = SummaryWriter('./runs/' + args.env_name+'/'+time.strftime("%Y-%m-%d", time.localtime())+'/ '+ str(time.time()))

    shake_dict = {}
    shake_dict['max'] = 0
    shake_dict['mean'] = 0
    shake_dict['median'] = 0
    shake_dict['min'] = 0

    for j in range(num_updates):
        loop_start = time.time()
        # Decrease learning rate linearly
        if args.use_linear_lr_decay:
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates, args.lr)
        # --/

        # Step through environment
        # --
        act_time = 0
        for step in range(args.num_steps_per_update):
            # Sample actions
            with torch.no_grad():

                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
            # Obser reward and next obs
            act_time_start = time.time()
            obs, reward, done, infos = envs.step(action)
            act_time_end = time.time()
            act_time += act_time_end - act_time_start
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)
        # --/

        # Update parameters
        # --
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()
        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)
        update_time_start = time.time()
        value_loss, action_loss, dist_entropy, mix_counter, buffer_time, train_time, write_time, aggregate_time, aggregate_lock_time, aggregate_circle_time, aggregate_time_middle = agent.update(rollouts)
        update_time_end = time.time()
        rollouts.after_update()
        # --/

        # Save every "save_interval" local environment steps (or last update)
        if (j % save_interval == 0
                or j == num_updates - 1) and args.save_dir != '':
            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(args.save_dir,
                            '%s.%.3d.%s.pt' % (rank, j // save_interval, np.mean(episode_rewards))))
        # --/

        # Log every "log_interval" local environment steps
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            num_steps = (j + 1) * args.num_procs_per_learner \
                * args.num_steps_per_update

            sub_max = np.max(episode_rewards) - shake_dict['max']
            sub_mean = np.mean(episode_rewards) - shake_dict['mean']
            sub_median = np.median(episode_rewards) - shake_dict['median']
            sub_min = np.min(episode_rewards) - shake_dict['min']

            # 绘图
            writer.add_scalar('ray/tune/episode_reward_mean_step', np.mean(episode_rewards), num_steps)
            writer.add_scalar('ray/tune/episode_reward_median_step', np.median(episode_rewards), num_steps)
            writer.add_scalar('ray/tune/episode_reward_min_step', np.min(episode_rewards), num_steps)
            writer.add_scalar('ray/tune/episode_reward_max_step', np.max(episode_rewards), num_steps)

            writer.add_scalar('ray/tune/episode_reward_mean', np.mean(episode_rewards), j)
            writer.add_scalar('ray/tune/episode_reward_median', np.median(episode_rewards), j)
            writer.add_scalar('ray/tune/episode_reward_min', np.min(episode_rewards), j)
            writer.add_scalar('ray/tune/episode_reward_max', np.max(episode_rewards), j)

            writer.add_scalar('ray/tune/episode_reward_shake_mean', sub_mean, num_steps)
            writer.add_scalar('ray/tune/episode_reward_shake_median', sub_median, num_steps)
            writer.add_scalar('ray/tune/episode_reward_shake_min', sub_min, num_steps)
            writer.add_scalar('ray/tune/episode_reward_shake_max', sub_max, num_steps)
            # 把这一次的数据复制过来
            shake_dict['max'] = np.max(episode_rewards)
            shake_dict['mean'] = np.mean(episode_rewards)
            shake_dict['median'] = np.median(episode_rewards)
            shake_dict['min'] = np.min(episode_rewards)

            end = time.time()
            writer.add_scalar('ray/tune/episode_reward_time_mean', np.mean(episode_rewards), (end - start))
            writer.add_scalar('ray/tune/episode_reward_time_median', np.median(episode_rewards), (end - start))
            writer.add_scalar('ray/tune/episode_reward_time_min', np.min(episode_rewards), (end - start))
            writer.add_scalar('ray/tune/episode_reward_time_max', np.max(episode_rewards), (end - start))
            
            writer.add_scalar('ray/tune/buffer_time_percent', buffer_time/(update_time_end - update_time_start), num_steps)
            writer.add_scalar('ray/tune/update_time_percent', (update_time_end - update_time_start)/(end-loop_start), num_steps)
            writer.add_scalar('ray/tune/buffer_time', buffer_time, num_steps)
            writer.add_scalar('ray/tune/loop_time', (end - loop_start),num_steps)
            writer.add_scalar('ray/tune/write_time', write_time, num_steps)
            writer.add_scalar('ray/tune/aggregate_time', aggregate_time, num_steps)
            writer.add_scalar('ray/tune/loop_time_except_update', (end - loop_start) - (update_time_end - update_time_start), num_steps)
            
            writer.add_scalar('ray/tune/aggregate_circle_time', aggregate_circle_time, num_steps)
            writer.add_scalar('ray/tune/aggregate_time_middle', aggregate_time_middle, num_steps)
            writer.add_scalar('ray/tune/aggregate_lock_time', aggregate_lock_time, num_steps)
            writer.add_scalar('ray/tune/act_time', act_time, num_steps)
            #print('counter:', mix_counter)
            #writer.add_scalar('ray/tune/episode_reward_gather_mean', np.mean(episode_rewards), mix_counter)
            #for i, (name, param) in enumerate(actor_critic.named_parameters()):
            #    writer.add_histogram(tag=name, values=param.cpu().clone().data.numpy(), global_step=j)
            print(('{}: Updates {}, num timesteps {}, FPS {} ' +
                   '\n {}: Last {} training episodes: ' +
                   'mean/median reward {:.1f}/{:.1f}, ' +
                   'min/max reward {:.1f}/{:.1f}\n').
                  format(rank, j, num_steps,
                         int(num_steps / (end - start)), rank,
                         len(episode_rewards),
                         np.mean(episode_rewards),
                         np.median(episode_rewards),
                         np.min(episode_rewards),
                         np.max(episode_rewards),
                         dist_entropy, value_loss, action_loss
                         ))
            print('j:', j)
        # --/


def make_gossip_buffer(args, mng, device):

    # Make local-gossip-buffer
    if args.num_learners * args.num_nodes> 1:
        # Make Topology
        topology = [] #every learner keep a local version of topology
        for rank in range(args.num_learners * args.num_nodes): #todo change the index for distribute peers
            graph = Graph(rank = rank, world_size= args.num_learners * args.num_nodes,
                          peers_per_itr=args.num_peers)
            topology.append(graph)
        
        # Initialize "actor_critic-shaped" parameter-buffer
        actor_critic = Policy(
            (4, 84, 84),
            base_kwargs={'recurrent': args.recurrent_policy},
            env_name=args.env_name)
        actor_critic.to('cpu') #todo model going through server

        # Keep track of local iterations since learner's last sync
        #sync_list = mng.list([0 for _ in range(args.num_learners)])
        sync_list = mng.get_sync_list()
        # Used to ensure proc-safe access to agents' message-buffers
        #buffer_locks = mng.list([mng.Lock() for _ in range(args.num_learners)])
        #buffer_locks = mng.get_buffer_locks()
        # Used to signal between processes that message was read
        #read_events = mng.list([mng.list([mng.Event() for _ in range(args.num_learners)])
        #    for _ in range(args.num_learners)])
        read_events = mng.get_read_events()
        # Used to signal between processes that message was written
        #write_events = mng.list([
        #    mng.list([mng.Event() for _ in range(args.num_learners)])
        #    for _ in range(args.num_learners)])
        write_events = mng.get_write_events()
        msg_buffer = mng.get_msg_buffer()

        # Need to maintain a reference to all objects in main processes
        _references = [topology, actor_critic,
                       read_events, write_events, sync_list]
        gossip_buffer = GossipBuffer(topology, actor_critic,
                                     read_events, write_events, sync_list,
                                     mng,sync_freq=args.sync_freq, num_nodes=args.num_nodes, msg_buffer = msg_buffer)
    else:
        _references = None
        gossip_buffer = None

    return gossip_buffer, _references


def train(args):
    pp.pprint(args)

    proc_manager = mp.Manager()
    #proc_manager.register('get_barrier')
    proc_manager.register('get_sync_list')
    #proc_manager.register('get_buffer_locks')
    proc_manager.register('get_read_events')
    proc_manager.register('get_write_events')
    proc_manager.register('get_msg_buffer')

    proc_manager.__init__(address=('127.0.0.1', 5000), authkey=b'abc')
    proc_manager.connect()
    #proc_manager.Barrier()
    #barrier = proc_manager.Barrier(args.num_learners) #todo barrier service
    #barrier = proc_manager.get_barrier()
    #barrier = barrier(args.num_learners)
    # Shared-gossip-buffer on GPU-0
    device = torch.device('cuda:%s' % 0 if args.cuda else 'cpu')
    #device = torch.device('cpu')
    shared_gossip_buffer, _references = make_gossip_buffer(
        args, proc_manager, device)

    # Make actor-learner processes
    proc_list = []
    for rank in range(0, args.num_learners): #todo change the rank for distribute peers

        # Uncomment these lines to use 2 GPUs
        # gpu_id = int(rank % 2)  # Even-rank agents on gpu-0, odd-rank on gpu-1
        # device = torch.device('cuda:%s' % gpu_id if args.cuda else 'cpu')
        proc = mp.Process(
            target=actor_learner,
            args=(args, rank + args.dist_rank * args.num_learners, device, shared_gossip_buffer),
            daemon=False
        )
        proc.start()
        proc_list.append(proc)

        # # Bind agents to specific hardware-threads (generally not necessary)
        # avail = list(os.sched_getaffinity(proc.pid))  # available-hwthrds
        # cpal = math.ceil(len(avail) / args.num_learners)  # cores-per-proc
        # mask = [avail[(rank * cpal + i) % len(avail)] for i in range(cpal)]
        # print('process-mask:', mask)
        # os.sched_setaffinity(proc.pid, mask)

    for proc in proc_list:
        proc.join()


if __name__ == "__main__":
    mp.set_start_method('spawn')
    args = get_args()
    torch.set_num_threads(1)

    # Make/clean save & log directories
    # --
    args.save_dir = args.save_dir + args.env_name
    def remove_files(files):
        for f in files:
            try:
                os.remove(f)
            except Exception:
                pass
    try:
        os.makedirs(args.log_dir)
    except OSError as e:
        print(e)
        files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
        remove_files(files)
    try:
        os.makedirs(args.save_dir)
    except OSError as e:
        print(e)
        files = glob.glob(os.path.join(args.save_dir, '*.pt'))
        remove_files(files)
    # --/

    train(args)
