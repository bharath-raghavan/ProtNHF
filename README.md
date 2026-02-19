# ProtNHF: Neural Hamiltonian Flows for Controllable Protein Sequence Generation

**Authors:** Bharath Raghavan¹, David M. Rogers¹  

**Affiliations:**  
¹ National Center for Computational Sciences, Oak Ridge National Laboratory

## Introduction

ProtNHF is a generative model for protein sequences that enables continuous, controllable design without retraining. It leverages neural Hamiltonian flows with a Transformer-based energy function to map a latent Gaussian to protein embeddings. Sampling-time bias functions allow steering properties like amino acid composition or net charge smoothly and predictably. Generated sequences achieve high quality as measured by ESM-2 pseudo-perplexity and AlphaFold2 pLDDT scores. ProtNHF provides a flexible, physically interpretable framework for programmable protein sequence generation.

## Installation

Clone the repository:

```
git clone https://github.com/bharath-raghavan/ProtNHF.git
cd ProtNHF
```

ProtNHF requires both ProtNHF and `torch-scatter`, making sure the versions match. Instructions for this can be found on the respective websites.
After that, install ProtNHF:

```
pip install -e . # use `-e` for it to be editable locally. 
```

## Training

After installing, training can be initiated with:

```
protnhf train config.yaml
```

The training requires data parallelism and should be run on an HPC system. An example of running this with a SLURM batch script:

```
#!/bin/bash

#SBATCH -o %x-%j.out
#SBATCH --nodes=64
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 01:00:00

# load modules

srun python3 protnhf train config.yaml
```
Example `YAML` files are given in the `examples/` directory. Required sections are the `model` and `training` sections. The `model` section includes:

* `checkpoint` — path to save or load model weights
* `dt` — integrator timestep
* `niter` — number of leapfrog steps
* `std` - standard deviation of the latent Gaussian distribution
* `n_types` - Number of classes for embeddings, 20 for proteins
* `hidden_dims` — hidden size of NHF dynamics
* `energy.*` — transformer energy model architecture

The `training` section includes:
* `dataset.file` — path to HDF5 dataset
* `dataset.batch_size` — training batch size
* `num_epochs` — number of training epochs
* `optim.*` — optimizer warmup and cosine scheduler hyperparameters

## Sampling

Sampling is done on a single process with:

```
protnhf sample config.yaml out.fasta
```

Again, example `YAML` files are given in the `examples/` directory. Required sections are the `model` and `sample` sections. Furthermore, the path given in `model.checkpoint` given should correpond to a trained model with parameters matching those give in the rest of the `model` section. The `sample` section includes:

* `length` — sequence length
* `num` — number of sequences to generate of given length
* `bias` — optional inference-time bias terms

Without the bias section, ProtNHF will generated uncondtional samples. Supported bias types include:

* `coulomb`
* `gaussian`
* `restraint`
* `netchargerestraint`

Each bias term defines:
`k` — strength of the bias

Additional parameters depending on type (e.g., `target`, `residue`, `sigma`, `i`)

## Citing ProtNHF

If you use ProtNHF in your research, please cite:

XX

## License
