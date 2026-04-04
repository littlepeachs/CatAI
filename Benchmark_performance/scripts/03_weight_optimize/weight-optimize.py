import os
import torch
import numpy as np
import cvxpy as cp
import json
import pickle
import pandas as pd
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='Alloy-GMAE')
parser.add_argument("--scaling", type=int, default=0)
args = parser.parse_args()

BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
MODULE_SELECTION_OUTPUTS_ROOT = BENCHMARK_ROOT / 'module_selection_outputs'
JSON_ROOT = BENCHMARK_ROOT / 'json'

epsilon = 1e-6
lambda_reg = 0

T = 1
QUOTA = 800

# TEST: benchmark default modules
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

datasets = [args.dataset]

splits = ['0']
# K = 18  # Set to an integer if you want to limit the number of modules
N = len(module_names)
K = len(module_names)

optimized_obj_csv_path = "optimized_obj_csv_path"
optimized_obj_results = []

# define base paths
labels_base_path = 'labels_base_path'
predictions_base_path = 'predictions_base_path'
# data_name = 'data_name'

labels_base_path = MODULE_SELECTION_OUTPUTS_ROOT / 'catai_labels'
predictions_base_path = MODULE_SELECTION_OUTPUTS_ROOT / 'catai_predictions'

if args.scaling == 0:
    labels_base_path = MODULE_SELECTION_OUTPUTS_ROOT / 'catai_labels'
    predictions_base_path = MODULE_SELECTION_OUTPUTS_ROOT / 'catai_predictions'
else:
    labels_base_path = MODULE_SELECTION_OUTPUTS_ROOT / 'catai_labels'
    predictions_base_path = MODULE_SELECTION_OUTPUTS_ROOT / f'scaling_{args.scaling}_predictions'
# data_name = 'data_name' # TEST

for split in splits:
    module_dict = {}
    weight_dict = {}
    num_modules = {}
    for module_name in module_names:
        for data_name in datasets:
            print(data_name, split)

            predictions = torch.load(predictions_base_path / f"{data_name}_split{split}.pt")
            # print(predictions[0])
            # print(predictions[0].shape)
            
            for i in range(len(predictions)):
                result = sum(predictions[i]).numpy()
            
                if np.isnan(result).any():
                    print(f"Group {i} contains NaN, replacing with a large value.")
                    predictions[i] = torch.ones_like(predictions[i]) * 1e5

            # predictions[3] = torch.ones_like(predictions[3]) * 1e5
            # predictions[6] = torch.ones_like(predictions[6]) * 1e5
            # predictions[3] = torch.zeros_like(predictions[3]) * 1e5
            # predictions[6] = torch.zeros_like(predictions[6]) * 1e5 
            
            with open(labels_base_path / f'{data_name}_split{split}_module_{module_name}.pkl', 'rb') as f:
                labels = pickle.load(f)
            y = labels.numpy()
            
            hat_y = torch.cat(predictions, dim=1).numpy()
            m = y.shape[0]
            y = y.reshape(m)

            m = min(m, QUOTA)
            y = y[:m]
            hat_y = hat_y[range(m),:]
            print("hat_y.shape:", hat_y.shape)

            # Define variables
            w = cp.Variable(N * T)
            z = cp.Variable(N * T, boolean=True)

            # Compute E_D
            residuals = hat_y @ w - y
            E_D = (1 / m) * cp.sum_squares(residuals)
            # E_D = (1 / m) * cp.sum(residuals)

            # Objective function
            objective = cp.Minimize(E_D + lambda_reg * cp.sum(z))

            # Constraints
            constraints = [
                w >= 0,
                w <= z,            # Ensures w_t = 0 when z_t = 0
                w >= epsilon * z,
                cp.sum(w) == 1,
                cp.sum(z) <= K,
            ]

            # Define and solve the problem
            prob = cp.Problem(objective, constraints)

            prob.solve(solver=cp.CPLEX, verbose=True)  # Use 'GUROBI' or 'CPLEX' for better performance

            # Optimal weights and selected modules
            w_opt = w.value
            z_opt = z.value
            optimized_obj = prob.value
            print("Optimized Objective Value: ", optimized_obj)

            non_zero_indices = np.nonzero(z_opt)
            non_zero_indices = non_zero_indices[0].tolist()
            
            non_zero_indices = [i for i in non_zero_indices] #  if i not in [21,22]

            non_zero_weights = w_opt[non_zero_indices]
            print("non_zero_weights: ", non_zero_weights)

            module_dict[data_name] = ','.join([str(num) for num in non_zero_indices])
            num_modules[data_name] = len(non_zero_indices)
            weight_dict[data_name] = non_zero_weights.tolist()
            if sum(weight_dict[data_name]) != 1:
                weight_dict[data_name][-1] = 1 - sum(weight_dict[data_name][:-1])
            assert sum(weight_dict[data_name]) == 1

            optimized_obj_results.append({
                'data_name': data_name,
                'split': split,
                'optimized_objective': optimized_obj,
                'module_dict': json.dumps(module_dict[data_name]),
                'weight_dict': json.dumps(weight_dict[data_name]),
                'num_modules': num_modules[data_name]
            })
    if args.scaling == 0:
        with open(JSON_ROOT / f'catai_module_ids/split_{split}_{data_name}.json', 'w') as json_file:
            json.dump(module_dict, json_file, indent=4)
        
        with open(JSON_ROOT / f'catai_module_num/split_{split}_{data_name}.json', 'w') as json_file:
            json.dump(num_modules, json_file, indent=4)

        with open(JSON_ROOT / f'catai_module_weights/split_{split}_{data_name}.json', 'w') as json_file:
            json.dump(weight_dict, json_file, indent=4)
    else:
        (JSON_ROOT / f'scaling_{args.scaling}_module_ids').mkdir(parents=True, exist_ok=True)
        (JSON_ROOT / f'scaling_{args.scaling}_module_num').mkdir(parents=True, exist_ok=True)
        (JSON_ROOT / f'scaling_{args.scaling}_module_weights').mkdir(parents=True, exist_ok=True)
            
        with open(JSON_ROOT / f'scaling_{args.scaling}_module_ids/split_{split}_{data_name}.json', 'w') as json_file:
            json.dump(module_dict, json_file, indent=4)
        
        with open(JSON_ROOT / f'scaling_{args.scaling}_module_num/split_{split}_{data_name}.json', 'w') as json_file:
            json.dump(num_modules, json_file, indent=4)
            
        with open(JSON_ROOT / f'scaling_{args.scaling}_module_weights/split_{split}_{data_name}.json', 'w') as json_file:
            json.dump(weight_dict, json_file, indent=4)
    
    print(module_dict)
    print(num_modules)
    print(weight_dict)
    
    # selected_modules = list(map(int, module_dict['data_catalyst'].split(',')))
    # print(f"Selected modules: {[module_names[i] for i in selected_modules]}")


# df = pd.DataFrame(optimized_obj_results)
# df.to_csv(optimized_obj_csv_path, index=False)