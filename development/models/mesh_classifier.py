import torch
from meshcnn.models import networks
from meshcnn.models.mesh_classifier import ClassifierModel
from os.path import join
from util.util import print_network
from torch.nn.parallel import DistributedDataParallel

class DistributedClassifierModel(ClassifierModel):
    def __init__(self, opt, rank):
        self.rank = rank
        self.opt = opt
        self.gpu_id = opt.gpu_ids[rank]

        print('Using distributed classifier model, GPU {}'.format(self.gpu_id))

        self.device = torch.device('cuda:{}'.format(self.gpu_id))
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.optimizer = None
        self.edge_features = None
        self.labels = None
        self.mesh = None
        self.soft_label = None
        self.loss = None

        #
        self.nclasses = opt.nclasses

        # load/define networks
        self.net = networks.define_classifier(opt.input_nc, opt.ncf, opt.ninput_edges, opt.nclasses, opt,
                                              [self.gpu_id], opt.arch, opt.init_type, opt.init_gain)
        self.net = DistributedDataParallel(self.net, device_ids=[self.gpu_id])
        self.set_train(opt.is_train)
        self.criterion = networks.define_loss(opt).to(self.device)

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler = networks.get_scheduler(self.optimizer, opt)
            print_network(self.net)

        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch)

    def set_train(self, train):
        self.is_train = train
        self.net.train(self.is_train)

    def forward(self):
        if self.is_train:
            return self.net(self.edge_features, self.mesh)
        else:
            # Do not use distributed model for test/validation.
            return self.net.module(self.edge_features, self.mesh)

    def load_network(self, which_epoch):
        """load model from disk"""
        save_filename = '%s_net.pth' % which_epoch
        load_path = join(self.save_dir, save_filename)
        net = self.net
        #if isinstance(net, torch.nn.DataParallel):
        #    net = net.module
        print('loading the model from %s' % load_path)
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.gpu_id}
        state_dict = torch.load(load_path, map_location=map_location)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)


    def save_network(self, which_epoch):
        # Only save model from process 0
        if (self.rank != 0):
            return

        save_filename = '%s_net.pth' % (which_epoch)
        save_path = join(self.save_dir, save_filename)
        torch.save(self.net.state_dict(), save_path)
