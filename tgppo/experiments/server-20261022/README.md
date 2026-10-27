# TGPPO: Learning Branching Policies for MILPs with Proximal Policy Optimization

**Accepted to the 40th AAAI Conference on Artificial Intelligence (AAAI 2026)**

[![AAAI](https://img.shields.io/badge/AAAI-2026-blue)](https://aaai.org/)
[![arXiv](https://img.shields.io/badge/arXiv-<2511.12986>-<green>.svg)](https://arxiv.org/abs/2511.12986)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

> **Authors:** Abdelouahed Ben Mhamed¹, Assia Kamal-Idrissi¹, Amal El Fallah Seghrouchni¹,²
>
> **Affiliations:**
> 1. Ai movement - International Artificial Intelligence Center of Morocco, University Mohammed VI Polytechnic
> 2. Lip6, Sorbonne University, France

---

## 📄 Paper & Full Version

The official version of this paper is published in the **Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 40, No. 29)**.

* **Official Proceedings:** [Read on AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/39619)
* **Full Version (with Appendix):** [Download from arXiv](https://arxiv.org/pdf/2511.12986)

> **Note:** Due to page limits, the version in the AAAI proceedings does not include the appendix. For detailed reward formulations, and additional experimental results, please refer to the **Extended Version** linked above.

---

## 📌 Abstract
Branch-and-Bound (B&B) is the dominant exact solution method for Mixed Integer Linear Programs (MILP), yet its exponential time complexity poses significant challenges for large-scale instances. The growing capabilities of machine learning have spurred efforts to improve B&B by learning data-driven branching policies. However, most existing approaches rely on Imitation Learning (IL), which tends to overfit to expert demonstrations and struggles to generalize to structurally diverse or unseen instances.

In this work, we propose **Tree-Gate Proximal Policy Optimization (TGPPO)**, a novel framework that employs Proximal Policy Optimization (PPO), a Reinforcement Learning (RL) algorithm, to train a branching policy aimed at improving generalization across heterogeneous MILP instances. Our approach builds on a parameterized state space representation that dynamically captures the evolving context of the search tree. Empirical evaluations show that TGPPO often outperforms existing learning-based policies in terms of reducing the number of nodes explored and improving p-Primal-Dual Integrals (PDI), particularly in out-of-distribution instances.

## 🚀 Key Contributions
1.  **On-Policy Learning:** We use PPO with a clipped-surrogate objective to learn directly from solver interactions, avoiding the bias inherent in imitation learning.
2.  **Tree-Gate Transformer:** A permutation-equivariant architecture that conditions attention via multiplicative gates driven by local tree statistics.
3.  **Difficulty-Adaptive Rewards:** A novel reward signal (H3) that adapts weights for node efficiency, gap closure, and PDI based on instance hardness.

## 💾 Datasets

The datasets used for training and evaluation (Set Cover, Combinatorial Auction, Capacitated Facility Location, and MIPLIB) are available for download.

👉 **[Download Datasets from here](https://data.mendeley.com/datasets/8msnxmvdgp/1)**

## 🛠️ Installation

### Prerequisites
* Linux or macOS (Recommended)
* Python 3.8+
* **SCIP Solver 6.0.1**: This project requires the SCIP solver. [Download SCIP here](https://www.scipopt.org/).

### Setup
```bash
# 1. Clone the repository
git clone https://github.com/benmhamed-a/tgppo-aaai-26.git
cd tgppo-aaai-26/code

# 2. Create a virtual environment
conda create -n tgppo python=3.8
conda activate tgppo

# 3. Install dependencies
# Ensure SCIP_HOME is set if PyScipOpt fails to find the library
pip install -r requirements.txt
```

---

## 🖊️ Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{mhamed2026learning,
  title     = {Learning Branching Policies for {MILPs} with Proximal Policy Optimization},
  author    = {Mhamed, A. B. and Kamal Idrissi, A. and Seghrouchni, A.},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {40},
  number    = {29},
  pages     = {24379--24387},
  year      = {2026},
  doi       = {10.1609/aaai.v40i29.39619},
  url       = {[https://doi.org/10.1609/aaai.v40i29.39619](https://doi.org/10.1609/aaai.v40i29.39619)}
}