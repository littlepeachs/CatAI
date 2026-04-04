from cgcnn.cfg import CFG
from cgcnn.main import Crystal_Trainer
import warnings


if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    cfg = CFG()

    cfg.label_dir = r"Cat_Data/Slab_All"
    cfg.structure_dir = r"Cat_Data/Slab_All"

    cfg.model_path = "Results/Round_1/model.pth"
    cfg.properties = ["Binding_energy", "Energy_barrier"]
    cfg.test_ratio = 0
    cfg.dropout = 0.1

    cfg.n_conv = 3
    cfg.n_h = 2
    cfg.h_fea_len = 64
    cfg.atom_fea_len = 128
    cfg.out_size = 2

    cfg.save_dir = 'test'

    crystal_model = Crystal_Trainer(cfg)
    crystal_model.predict()
