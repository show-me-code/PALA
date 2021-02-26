# -*-coding:utf-8-*-
import torch.multiprocessing as mp
from multiprocessing.managers import ListProxy, BarrierProxy, AcquirerProxy, EventProxy
from gala.arguments import get_args
mp.current_process().authkey = b'abc'



def server(manager,host, port, key, args):
    barrier = manager.Barrier(4)
    '''sync_list = manager.list()
    buffer_locks = manager.list()
    read_events = manager.list()
    write_events = manager.list()'''
    num_learners = args.num_learners * args.num_nodes

    sync_list = manager.list([0 for _ in range(num_learners)])

    #buffer_locks = manager.list([manager.Lock() for _ in range(num_learners)])


    #read_events = manager.list([manager.list([manager.Event() for _ in range(num_learners)])
    #                        for _ in range(num_learners)]) #2 dim array is supported
    read_events = manager.list([manager.list([False for _ in range(num_learners)])
                            for _ in range(num_learners)]) #2 dim array is supported
    #write_events = manager.list([
    #    manager.list([manager.Event() for _ in range(num_learners)])
    #    for _ in range(num_learners)])
    write_events = manager.list([
            manager.list([False for _ in range(num_learners)])
            for _ in range(num_learners)])

    msg_buffer = manager.list()
    manager.register('get_barrier', callable=lambda: barrier, proxytype=BarrierProxy)
    manager.register('get_sync_list', callable=lambda :sync_list, proxytype=ListProxy)
    #manager.register('get_buffer_locks', callable=lambda : buffer_locks, proxytype=ListProxy)
    manager.register('get_read_events', callable=lambda : read_events, proxytype=ListProxy)
    manager.register('get_write_events', callable= lambda : write_events, proxytype=ListProxy)
    manager.register('get_msg_buffer', callable=lambda :msg_buffer, proxytype=ListProxy)
    manager.__init__(address=(host, port), authkey=key)
    print('start service at', host)
    s = manager.get_server()
    s.serve_forever()

if __name__ == '__main__':
    mp.set_start_method('spawn') #need to set start method into spawn to transmate cuda tensor
    args = get_args()
    manager = mp.Manager()
    server(manager,'127.0.0.1', 5000, b'abc', args)