import os
import torch
import torch.nn as nn
import trimesh
from diso import DiffMC
from diso import DiffDMC

# define a sphere SDF
class SphereSDF:
    def __init__(self, center, radius, margin):
        self.center = center
        self.radius = radius
        self.aabb = torch.stack([center - radius - margin, center + radius + margin], dim=-1)

    def __call__(self, points):
        return torch.norm(points - self.center, dim=-1) - self.radius

device = "cuda:0"
os.makedirs("out", exist_ok=True)

# define a sphere
s_x = 0.
s_y = 0.
s_z = 0.
radius = 0.5
sphere = SphereSDF(torch.tensor([s_x, s_y, s_z]), radius, 1/64)

# create the iso-surface extractor
diffmc = DiffMC(dtype=torch.float32)
diffdmc = DiffDMC(dtype=torch.float32)

# create a grid
dimX, dimY, dimZ = 64, 64, 64
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

# query the SDF input
sdf = sphere(grids)
sdf = sdf.requires_grad_(True).to(device)
sdf = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)

# randomly deform the grid
deform = torch.nn.Parameter(
    torch.rand(
        (sdf.shape[0], sdf.shape[1], sdf.shape[2], 3),
        dtype=torch.float32,
        device=device,
    ),
    requires_grad=True,
)

# DiffMC with random grid deformation
## Test forward
verts, faces = diffmc(sdf, 0.5 * torch.tanh(deform), isovalue=0)
verts = verts.cpu() * (sphere.aabb[:, 1] - sphere.aabb[:, 0]) + sphere.aabb[:, 0]
mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(), faces=faces.cpu().numpy(), process=False)
mesh.export("out/diffmc_sphere_w_deform.obj")
## Test backward
L = torch.norm(verts)
L.backward()
grad_grid = sdf.grad
grad_deform = deform.grad
print("============ DiffMC w/ grid deformation ============")
print("grad_grid:", grad_grid.shape, grad_grid.min(), grad_grid.max())
print("grad_deform:", grad_deform.shape, grad_deform.min(), grad_deform.max())

# DiffMC without grid deformation
verts, faces = diffmc(sdf, None, isovalue=0)
verts = verts.cpu() * (sphere.aabb[:, 1] - sphere.aabb[:, 0]) + sphere.aabb[:, 0]
mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(), faces=faces.cpu().numpy(), process=False)
mesh.export("out/diffmc_sphere_wo_deform.obj")
## Test backward
L = torch.norm(verts)
L.backward()
grad_grid = sdf.grad
print("============ DiffMC w/o grid deformation ============")
print("grad_grid:", grad_grid.shape, grad_grid.min(), grad_grid.max())

# DiffDMC with random grid deformation
verts, faces = diffdmc(sdf, 0.5 * torch.tanh(deform), isovalue=0)
verts = verts.cpu() * (sphere.aabb[:, 1] - sphere.aabb[:, 0]) + sphere.aabb[:, 0]
mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(), faces=faces.cpu().numpy(), process=False)
mesh.export("out/diffdmc_sphere_w_deform.obj")
## Test backward
L = torch.norm(verts)
L.backward()
grad_grid = sdf.grad
grad_deform = deform.grad
print("============ DiffDMC w/ grid deformation ============")
print("grad_grid:", grad_grid.shape, grad_grid.min(), grad_grid.max())
print("grad_deform:", grad_deform.shape, grad_deform.min(), grad_deform.max())

# DiffDMC without grid deformation
verts, faces = diffdmc(sdf, None, isovalue=0)
verts = verts.cpu() * (sphere.aabb[:, 1] - sphere.aabb[:, 0]) + sphere.aabb[:, 0]
mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(), faces=faces.cpu().numpy(), process=False)
mesh.export("out/diffdmc_sphere_wo_deform.obj")
## Test backward
L = torch.norm(verts)
L.backward()
grad_grid = sdf.grad
print("============ DiffDMC w/o grid deformation ============")
print("grad_grid:", grad_grid.shape, grad_grid.min(), grad_grid.max())

print("forward results saved to out/")
