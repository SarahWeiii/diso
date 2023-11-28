# Differentiable Iso-Surface Extraction Package (DISO)
This repo contains various differentiable iso-surface extraction algorithms implemented in `cuda`.
Currently two algorithms are implemented:
* Differentiable Marching Cubes (DiffMC)
* Differentiable Dual Marching Cubes (DualMC)

Both `DiffMC` and `DualMC` supports adding (learnable) deformation to grid vertices, which leads to better shape fitting ability in optimization problems.

# Installation
Requirements: torch, trimesh
```
pip install git+https://github.com/SarahWeiii/diso/
```

# Quick Start
You can simply try the following command, which turns a sphere SDF into triangle mesh using different algorithms.
```
python test.py
```

# Reference
If you find this repo useful, please cite the following paper:
```
@article{wei2023neumanifold,
  title={NeuManifold: Neural Watertight Manifold Reconstruction with Efficient and High-Quality Rendering Support},
  author={Wei, Xinyue and Xiang, Fanbo and Bi, Sai and Chen, Anpei and Sunkavalli, Kalyan and Xu, Zexiang and Su, Hao},
  journal={arXiv preprint arXiv:2305.17134},
  year={2023}
}
```