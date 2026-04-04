import pickle
import numpy as np
import torch
from scipy.stats import spearmanr
import time
import pandas as pd
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='POE_lmdb')
parser.add_argument("--scaling", type=int, default=0)
args = parser.parse_args()

SCREENING_ROOT = Path(__file__).resolve().parents[2]
MODULE_SELECTION_OUTPUTS_ROOT = SCREENING_ROOT / 'module_selection_outputs'

def performance(hat_y, y):
    s, _ = spearmanr(hat_y, y)
    mae = np.mean(np.abs(hat_y - y))
    return s, mae

def cos_yh_hat_y(features, labels, K):
    # normalization for cosine similarity
    norms = torch.norm(features, dim=1, keepdim=True)
    normalized_features = features / norms

    # cosine similarity matrix  bsz * bsz
    cos_matrix = normalized_features @ normalized_features.transpose(0, 1)
    # print("cos matrix", cos_matrix)

    # Gaussion and leave one out
    adjacent_matrix = torch.exp(cos_matrix / 0.2)
    adjacent_matrix = adjacent_matrix - torch.diag(torch.diag(adjacent_matrix))

    # select top_k neighbors for prediction
    topk_value, topk_idx = torch.topk(adjacent_matrix, k=K, dim=1)
    # print("topk idx", topk_idx)
    mask = torch.zeros_like(adjacent_matrix)
    mask = mask.scatter(1, topk_idx, 1)
    # mask = ((mask + torch.t(mask))>0).type(torch.float32)
    adjacent_matrix = adjacent_matrix * mask

    # row normalization
    D = torch.sum(adjacent_matrix, 1, True)
    # print("D", D)
    adjacent_matrix = adjacent_matrix / D
    # print("normalized adj matrix", adjacent_matrix)

    # label propagation
    prediction = adjacent_matrix @ labels
    # print("predictions of label prop", prediction)

    prediction = prediction.numpy()

    # prediction.shape = (bsz, 1)
    return prediction

# Local POE dataset uses a single fixed split label for output bookkeeping.
splits = [0]
# regular datasets
# datasets = ['piezoelectric_tensor','expt_eform','mp_poly_total','mp_poisson_ratio','jarvis_2d_dielectric_opt','jarvis_2d_eform','jarvis_2d_exfoliation','jarvis_2d_gap_opt','mp_elastic_anisotropy','mp_eps_electronic','mp_eps_total','mp_phonons','mp_poly_electronic','expt_bandgap','jarvis_3d_eps_tbmbj','jarvis_3d_gap_tbmbj','mp_dielectric']
datasets = [args.dataset]
# regular modules
# module_names = ['mp_eform','mp_bandgap','n_Seebeck','n_avg_eff_mass','n_th_cond','p_Seebeck','p_avg_eff_mass','p_e_cond','n_e_cond','p_th_cond','mp_gvrh','castelli_eform','jarvis_gvrh','jarvis_bandgap','jarvis_eform','jarvis_kvrh','mp_kvrh','jarvis_dielectric_opt']

module_dict = {
    5:
    [
        "p_e_cond",
        "HL_Gap",
        "qmof_bg",
        "n_e_cond",
        "cepdb_pce"
    ],
    10:
    [
        "p_e_cond",
        "HL_Gap",
        "qmof_bg",
        "n_e_cond",
        "cepdb_pce",
        "qmof_e",
        "dos_fermi_level",
        "oqmd_delta_e",
        "oqmd_stability",
        "oqmd_band_gap",
    ],
    15:
    [
        "p_e_cond",
        "HL_Gap",
        "qmof_bg",
        "n_e_cond",
        "cepdb_pce",
        "qmof_e",
        "dos_fermi_level",
        "oqmd_delta_e",
        "oqmd_stability",
        "oqmd_band_gap",
        "surface_energy",
        "Ea_log_wo_cell",
        "gibbs_free_energy",
        "Electronic_E",
        "Metal_q",
    ],
    20:
    [
        "p_e_cond",
        "HL_Gap",
        "qmof_bg",
        "n_e_cond",
        "cepdb_pce",
        "qmof_e",
        "dos_fermi_level",
        "oqmd_delta_e",
        "oqmd_stability",
        "oqmd_band_gap",
        "surface_energy",
        "Ea_log_wo_cell",
        "gibbs_free_energy",
        "Electronic_E",
        "Metal_q",
        "castelli_eform",
        "jarvis_bandgap",
        "jarvis_eform",
        "jarvis_total_energy",
        "adsorption_energy",
    ],
    25:
    [
        "p_e_cond",
        "HL_Gap",
        "qmof_bg",
        "n_e_cond",
        "cepdb_pce",
        "qmof_e",
        "dos_fermi_level",
        "oqmd_delta_e",
        "oqmd_stability",
        "oqmd_band_gap",
        "surface_energy",
        "Ea_log_wo_cell",
        "gibbs_free_energy",
        "Electronic_E",
        "Metal_q",
        "castelli_eform",
        "jarvis_bandgap",
        "jarvis_eform",
        "jarvis_total_energy",
        "adsorption_energy",
        "mp_efermi",
        "n_th_cond",
        "mp_energy_above_hull",
        "mp_bandgap",
        "p_th_cond",
    ],
}
if args.scaling == 0:
    module_names = [
        "mp_eform",
        "mp_bandgap",
        "mp_energy_above_hull",
        "mp_efermi",
        'n_th_cond',
        'p_e_cond',
        'n_e_cond',
        'p_th_cond',
        'mp_gvrh',
        'mp_kvrh',
        "castelli_eform",
        'jarvis_gvrh',
        'jarvis_kvrh',
        "jarvis_bandgap",
        "jarvis_eform",
        "jarvis_total_energy",
        "adsorption_energy",
        "surface_energy",
        "Ea_log_wo_cell",
        "gibbs_free_energy",
        "Electronic_E",
        "Metal_q",
        "HL_Gap",
        "dos_fermi_level",
        "oqmd_delta_e",
        "oqmd_stability",
        "oqmd_band_gap",
        "qmof_bg",
        "qmof_e",
        "cepdb_pce",
    ]
else:
    module_names = module_dict[args.scaling]

# dict_module_names = {
#     0: "mp_eform",
#     1: "mp_bandgap",
#     2: "mp_energy_above_hull",
#     3: "mp_efermi",
#     4: "n_th_cond",
#     5: "p_e_cond",
#     6: "n_e_cond",
#     7: "p_th_cond",
#     8: "mp_gvrh",
#     9: "mp_kvrh",
#     10: "castelli_eform",
#     11: "jarvis_gvrh",
#     12: "jarvis_kvrh",
#     13: "jarvis_bandgap",
#     14: "jarvis_eform",
#     15: "jarvis_total_energy",
#     16: "adsorption_energy",
#     17: "surface_energy",
#     18: "Ea_log_wo_cell",
#     19: "gibbs_free_energy",
#     20: "Electronic_E",
#     21: "Metal_q",
#     22: "HL_Gap",
#     23: "dos_fermi_level",
#     24: "oqmd_delta_e",
#     25: "oqmd_stability",
#     26: "oqmd_band_gap",
#     27: "qmof_bg",
#     28: "qmof_e",
#     29: "cepdb_pce",
# }

print(f"len(module_names): {len(module_names)}")

# define base paths
embeddings_base_path = MODULE_SELECTION_OUTPUTS_ROOT / 'full_embeddings'
labels_base_path = MODULE_SELECTION_OUTPUTS_ROOT / 'full_labels'
predictions_base_path = MODULE_SELECTION_OUTPUTS_ROOT / 'full_predictions'
if args.scaling == 0:
    embeddings_base_path = MODULE_SELECTION_OUTPUTS_ROOT / 'catai_embeddings'
    labels_base_path = MODULE_SELECTION_OUTPUTS_ROOT / 'catai_labels'
    predictions_base_path = MODULE_SELECTION_OUTPUTS_ROOT / 'catai_predictions'
else:
    embeddings_base_path = MODULE_SELECTION_OUTPUTS_ROOT / 'catai_embeddings'
    labels_base_path = MODULE_SELECTION_OUTPUTS_ROOT / 'catai_labels'
    predictions_base_path = MODULE_SELECTION_OUTPUTS_ROOT / f'scaling_{args.scaling}_predictions'

predictions_base_path.mkdir(parents=True, exist_ok=True)
labels_base_path.mkdir(parents=True, exist_ok=True)

results = []

for dataset in datasets:
    for split in splits:
        all_hat_y = []
        # Start timing for the entire dataset split
        start_dataset_split = time.time()
        for module_name in module_names:
            module_start_time = time.time()
            with open(embeddings_base_path / f'{dataset}_split{split}_module_{module_name}.pkl', 'rb') as f:
                try:
                    data = pickle.load(f)
                except:
                    print(f'Error in {embeddings_base_path / f"{dataset}_split{split}_module_{module_name}.pkl"}')
                    exit(0)

            with open(labels_base_path / f'{dataset}_split{split}_module_{module_name}.pkl', 'rb') as f:
                labels = pickle.load(f)
            labels = labels.numpy()

            x = data[0]
            # print(x)
            hat_y = cos_yh_hat_y(x, labels, 5)
            s, mae = performance(hat_y, labels)

            module_end_time = time.time()
            module_elapsed = module_end_time - module_start_time
            print(f'Module {module_name} spearman for dataset {dataset} split {split}: {s}, mae: {mae}, time taken: {module_elapsed:.2f} sec')            
            # print(f'Module {module_name} spearman for dataset {dataset} split {split}: ', s, 'mae: ', mae)
            all_hat_y.append(torch.tensor(hat_y))
        print("="*20)
        dataset_split_elapsed = time.time() - start_dataset_split
        print("=" * 20)
        print(f'Time taken for dataset {dataset} split {split}: {dataset_split_elapsed:.2f} sec')

        results.append({
            "Dataset": dataset,
            "Split": split,
            "Total Time (sec)": dataset_split_elapsed
        })        
        
        torch.save(all_hat_y, predictions_base_path / f'{dataset}_split{split}.pt')