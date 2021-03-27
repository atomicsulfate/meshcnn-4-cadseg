import sys,os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"meshcnn")))

from meshcnn.options.test_options import TestOptions
from data import DataLoader, DistributedDataLoader
from models import create_model
from meshcnn.util.writer import Writer
import torch.distributed as dist
import torch.multiprocessing as mp

def setup():
    # initialize the process group
    dist.init_process_group("nccl", init_method='env://')

def create_test_options():
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    return opt

def test_model(model, opt, dataset, epoch=-1):
    writer = Writer(opt)
    # test
    writer.reset_counter()
    for i, data in enumerate(dataset):
        model.set_input(data)
        ncorrect, nexamples, ncorrectPerClass, nexamplesPerClass = model.test()
        writer.update_counter(ncorrect, nexamples,  ncorrectPerClass, nexamplesPerClass )
    writer.print_acc(epoch, writer.acc, writer.classAcc)
    return writer.acc

def run_test(rank, numGPUs, opt):
    print('Running Test on rank {}'.format(rank))
    setup()

    dataset = DistributedDataLoader(opt)
    model = create_model(opt, rank)
    return test_model(model, opt, dataset)

def run_validation(model, epoch):
    print('Running Validation')
    opt = create_test_options()
    dataset = DataLoader(opt)
    model.set_train(False)
    acc = test_model(model, opt, dataset, epoch)
    model.set_train(True)
    return acc

if __name__ == '__main__':
    opt = create_test_options()
    numGPUs = len(opt.gpu_ids)
    if (opt.export_folder):
        # When segmented meshes are exported, sparse tensors holding mesh pooling collapse
        # form part of the loaded data. Pytorch (up to v1.7) does not support multithreaded
        # loading of sparse tensors: https://github.com/pytorch/pytorch/issues/20248
        opt.num_threads = 0

    if 'LOCAL_RANK' in os.environ:
        local_rank = os.environ['LOCAL_RANK']
    else:
        # Non dist mode, set dummy env variables.
        local_rank = 0
        os.environ['LOCAL_RANK'] = '0'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12500'
    run_test(local_rank, numGPUs, opt)
