import time,sys,os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"meshcnn")))

from meshcnn.options.train_options import TrainOptions
from data import DistributedDataLoader
from models import create_model
from meshcnn.util.writer import Writer
from test import run_validation

import torch.distributed as dist
import torch.multiprocessing as mp

def setup():
    # initialize the process group
    dist.init_process_group("nccl", init_method='env://')

def train(local_rank, numGPUs, opt):
    setup()

    dataset = DistributedDataLoader(opt)
    dataset_size = len(dataset)
    print('#training meshes = %d' % dataset_size)

    model = create_model(opt, local_rank)
    writer = Writer(opt)
    total_steps = 0
    global_rank = dist.get_rank()

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        dataset.set_epoch(epoch)

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.print_freq == 0:
                loss = model.loss
                t = (time.time() - iter_start_time) / opt.batch_size
                writer.print_current_losses(epoch, epoch_iter, loss, t, t_data)
                writer.plot_loss(loss, epoch, epoch_iter, dataset_size)

            if global_rank == 0 and i % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_network('latest')

            iter_data_time = time.time()
        if global_rank == 0 and epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_network('latest')
            model.save_network(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
        if opt.verbose_plot:
            writer.plot_model_wts(model, epoch)

        if epoch % opt.run_test_freq == 0:
            if global_rank == 0:
                acc = run_validation(model, epoch)
                writer.plot_acc(acc, epoch)
            dist.barrier()

    writer.close()
    dist.destroy_process_group()

if __name__ == '__main__':
    opt = TrainOptions().parse()
    numGPUs = len(opt.gpu_ids)
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # Non dist mode, set dummy env variables.
        local_rank = 0
        os.environ['LOCAL_RANK'] = '0'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12500'
    train(local_rank,numGPUs, opt)
    #mp.spawn(train, args=(numGPUs,opt), nprocs=numGPUs, join=True)