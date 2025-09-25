# Data-Driven SDA Design and High-Throughput Experimentation for the Synthesis of High-Silica Small-Pore Zeolites Containing lta-cavities 

This is the official repository of

[**Data-Driven SDA Design and High-Throughput Experimentation for the Synthesis of High-Silica Small-Pore Zeolites Containing lta-cavities**](https://doi.org/10.26434/chemrxiv-2025-d7qzq)

Authors: Hwajun Lee<sup>1&dagger;&ddagger;</sup>, Soonhyoung Kwon<sup>1&dagger;&sect;</sup>, Alexander Hoffman<sup>2</sup>, Jie Zhu<sup>1</sup>, Mingrou Xie<sup>1</sup>, Elton Pan<sup>2</sup>, Vivek Vattipalli<sup>3</sup>, Ahmad Moini<sup>3</sup>, Anthony Debellis<sup>4</sup>, Elsa Olivetti<sup>2</sup>, Rafael Gómez-Bombarelli<sup>2</sup>&ast;, Yuriy Román-Leshkov<sup>1</sup>&ast;

Affiliations:	

<sup>1</sup>Department of Chemical Engineering, Massachusetts Institute of Technology, Cambridge MA 02139, USA.

<sup>2</sup>Department of Materials Science and Engineering, Massachusetts Institute of Technology, Cambridge MA 02139, USA.

<sup>3</sup>BASF Environmental Catalyst and Metal Solutions, Iselin, New Jersey 08830, United States

<sup>4</sup>BASF Quantum Chemistry and Hybrid Modeling Research, Tarrytown, New York 10591, United States

*Corresponding authors: rafagb@mit.edu, yroman@mit.edu

†These authors contributed equally to this work.

‡Present address: Extreme Materials Research Center, Korea Institute of Science and Technology, KIST, Seoul 02792, Republic of Korea. 
Division of Energy and Environment Technology, KIST School, Korea University of Science and Technology, Seoul 02792, Republic of Korea.

§Present address: Department of Chemical Engineering, Purdue University, West Lafayette, IN 47907, USA.

## 1) Dataset

The dataset is an updated version from ZeoSyn:

- Article: (https://pubs.acs.org/doi/10.1021/acscentsci.3c01615)

- Github: (https://github.com/eltonpan/zeosyn_dataset)


## 2) Setup and Installation

The code in this repo has been tested on a Jupyterlab and Jupyter Notebook running Python v.3.11.11. Most of the analysis code depends only on a few modules and should run after using `pip` to install them with the command below.

```bash
pip install ase imageio ipython jupyter matplotlib numpy pandas pymatgen rdkit scikit-learn scipy seaborn shap tqdm
```

Alternatively, the environment can be set up easily using `uv`:

```bash
pip install uv
uv sync
```

If you would like to reproduce the SHAP analysis, please see the installation instructions in the original `ZeoSyn` repository [here](https://github.com/eltonpan/zeosyn_dataset/tree/master) to install its dependencies.

## 3) Preprint Information

The preprint of this paper is available at (https://doi.org/10.26434/chemrxiv-2025-d7qzq).

