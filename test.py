import os
import torch
import torch.nn as nn
import trimesh
from diso import DiffMC
from diso import DualMC


class SphereSDF:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        self.aabb = torch.stack([center - radius, center + radius], dim=-1)

    def __call__(self, points):
        return torch.norm(points - self.center, dim=-1) - self.radius


os.makedirs("out", exist_ok=True)

diffmc = DiffMC(dtype=torch.float32).cuda()
dualmc = DualMC(dtype=torch.float32).cuda()
dimX, dimY, dimZ = 16, 16, 16
sphere = SphereSDF(torch.tensor([0.5, 0.5, 0.5]), 0.5)
grids = torch.stack(
    torch.meshgrid(
        torch.linspace(0, 1, dimX),
        torch.linspace(0, 1, dimY),
        torch.linspace(0, 1, dimZ),
        indexing="ij",
    ),
    dim=-1,
)

grids[..., 0] = (
    grids[..., 0] * (sphere.aabb[0, 1] - sphere.aabb[0, 0]) + sphere.aabb[0, 0]
)
grids[..., 1] = (
    grids[..., 1] * (sphere.aabb[1, 1] - sphere.aabb[1, 0]) + sphere.aabb[1, 0]
)
grids[..., 2] = (
    grids[..., 2] * (sphere.aabb[2, 1] - sphere.aabb[2, 0]) + sphere.aabb[2, 0]
)

sdf = sphere(grids)
sdf = sdf.requires_grad_(True).cuda()
sdf = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
deform = torch.nn.Parameter(
    torch.rand(
        (sdf.shape[0], sdf.shape[1], sdf.shape[2], 3),
        dtype=torch.float32,
        device="cuda",
    ),
    requires_grad=True,
)

# DiffMC with random grid deformation
verts, faces = diffmc(sdf, 0.5 * torch.tanh(deform))
mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(), faces=faces.cpu().numpy())
mesh.export("out/diffmc_sphere_w_deform.obj")

# DiffMC without grid deformation
verts, faces = diffmc(sdf, None)
mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(), faces=faces.cpu().numpy())
mesh.export("out/diffmc_sphere_wo_deform.obj")

# DualMC with random grid deformation
verts, faces = dualmc(sdf, 0.5 * torch.tanh(deform))
mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(), faces=faces.cpu().numpy())
mesh.export("out/dualmc_sphere_w_deform.obj")

# DualMC without grid deformation
verts, faces = dualmc(sdf, None)
mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(), faces=faces.cpu().numpy())
mesh.export("out/dualmc_sphere_wo_deform.obj")
