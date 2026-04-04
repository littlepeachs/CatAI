from cgcnn.cfg import CFG
from cgcnn.main import Crystal_Trainer
import warnings


if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    cfg = CFG()

    cfg.label_dir = r"Cat_Data/R1"
    cfg.structure_dir = r"Cat_Data/Raw_Crystal_Data_All"

    cfg.properties = ["Binding_energy", "Energy_barrier"]

    cfg.test_ratio = 0.0

    cfg.augmentation = True
    cfg.epochs = 250
    cfg.lr_milestones = [150, ]
    cfg.lr = 0.001
    cfg.batch_size = 16
    cfg.weight_decay = 0
    cfg.dropout = 0.1

    cfg.n_conv = 3
    cfg.n_h = 2
    cfg.h_fea_len = 64
    cfg.atom_fea_len = 128
    cfg.out_size = 2

    cfg.print_freq = 100
    cfg.test_freq = 100

    cfg.save_dir = 'test'

    crystal_model = Crystal_Trainer(cfg)
    crystal_model.train()

