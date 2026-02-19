# ProtNHF: Neural Hamiltonian Flows for Controllable Protein Sequence Generation

by Bharath Raghavan and David M. Rogers
National Center for Computational Sciences, Oak Ridge National Laboratory, Oak
Ridge, United States of America

Controllable protein sequence generation remains a central challenge in computational protein design, as most existing approaches rely on retraining, classifier guidance, or architectural modification to impose conditioning. Here we introduce ProtNHF, a generative model that enables continuous, quantitative control over sequence-level properties through analytical bias functions applied exclusively at inference time. ProtNHF builds on neural Hamiltonian flows, where a lightweight Transformer-based potential energy function, inspired by ESM-2, is combined with an explicit kinetic term to define Hamiltonian dynamics in a continuous relaxation of protein sequence space. The model learns a symplectic transport map from a latent Gaussian distribution to protein sequence embeddings via deterministic leapfrog integration, enabling efficient and expressive sampling. In the unconditional setting, generated sequences achieve competitive quality as measured by ESM-2 pseudo-perplexity and AlphaFold2 pLDDT confidence scores.

A key advantage of the Hamiltonian formulation is its additive energy structure, which permits external bias potentials to be incorporated directly into the Hamiltonian at inference time without modifying or retraining the learned model. This casts controllable generation in a classical molecular modeling paradigm, where desired properties are enforced by explicit energy shaping. We demonstrate smooth, predictable, and approximately monotonic control over amino acid composition and global properties such as net charge by introducing simple analytical bias terms, including residue-specific chemical potentials and harmonic constraints. The bias strength modulates the values of these properties in generated sequences continuously while preserving structural plausibility and diversity. ProtNHF thus provides a flexible base distribution that can be steered toward different compositional regimes using transparent, physically interpretable energy terms, establishing a general framework for inference-time programmable protein sequence generation.

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

## Running

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
