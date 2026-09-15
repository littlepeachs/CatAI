# Asymmetric catalysis

CatAI extends task-composed JMP backbones with a role-aware attention readout for reaction-level asymmetric selectivity prediction on AHO and five CPA benchmarks. Three conformers are encoded for each reaction component and combined according to its chemical role.

After downloading the CatAI module weights and asymmetric-catalysis assets into `catai_data/`, run the complete workflow with:

```bash
cd Asymmetric_catalysis
bash run.sh
```

Install the repository environment plus `pip install -r Asymmetric_catalysis/requirements.txt` first. All generated checkpoints, predictions and logs are ignored by Git; paths can be overridden through the `CATAI_ASY_*` variables in `run_common.sh`.
