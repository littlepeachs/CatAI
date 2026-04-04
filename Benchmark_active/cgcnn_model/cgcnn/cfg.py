class CFG():
    def __init__(self):
        self.label_dir = r"E:\Research\Catalyst\Data\R1\be"
        self.structure_dir = r"E:\Research\Catalyst\Data\Raw_Crystal_Data_All"
        self.properties = []
        self.augmentation = False
        self.cuda = True
        self.workers = 0
        self.resume = False
        self.model_path = None
        self.checkpoint_path = None

        self.epochs = 30  # total epochs
        self.start_epoch = 0
        self.batch_size = 64
        self.lr = 0.01  # initial learning rate
        self.lr_milestones = [100]  # step change epoch
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.optim = "Adam"  # optimizer
        self.dropout = 0.2

        self.atom_fea_len = 64  # initial feature length
        self.h_fea_len = 64  # hidden feature length
        self.n_conv = 3  # num of conv layers
        self.n_h = 1  # number of hidden layers after pooling
        self.outsize = 1

        self.train_ratio = None
        self.train_size = None
        self.val_ratio = 0.0
        self.val_size = None
        self.test_ratio = 0.2
        self.test_size = None

        self.print_freq = 1  # print per num of epoch
        self.test_freq = 10

        self.save_name = 'model.pth'
        self.save_dir = 'default'
