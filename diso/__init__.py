import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function

from . import _C


class DiffMC(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        if dtype == torch.float32:
            mc = _C.CUMCFloat()
        elif dtype == torch.float64:
            mc = _C.CUMCDouble()

        class DMCFunction(Function):
            @staticmethod
            def forward(ctx, grid, deform, isovalue):
                if deform is None:
                    verts, tris = mc.forward(grid, isovalue)
                else:
                    verts, tris = mc.forward(grid, deform, isovalue)
                ctx.grid = grid
                ctx.deform = deform
                ctx.isovalue = isovalue
                return verts, tris

            @staticmethod
            def backward(ctx, adj_verts, adj_faces):
                adj_grid = torch.zeros_like(ctx.grid)
                if ctx.deform is None:
                    mc.backward(
                        ctx.grid, ctx.isovalue, adj_verts, adj_grid
                    )
                    return adj_grid, None, None, None, None
                else:
                    adj_deform = torch.zeros_like(ctx.deform)
                    mc.backward(
                        ctx.grid, ctx.deform, ctx.isovalue, adj_verts, adj_grid, adj_deform
                    )
                    return adj_grid, adj_deform, None, None, None

        self.func = DMCFunction

    def forward(self, grid, deform=None, isovalue=0.0):
        if grid.min() > 0:
            return torch.zeros((0, 3), dtype=self.dtype, device=grid.device), torch.zeros((0, 3), dtype=torch.int32, device=grid.device)
        dimX, dimY, dimZ = grid.shape
        grid = F.pad(grid, (1, 1, 1, 1, 1, 1), "constant", 1)
        if deform is not None:
            deform = F.pad(deform, (0, 0, 1, 1, 1, 1, 1, 1), "constant", 0)
        verts, tris = self.func.apply(grid, deform, isovalue)
        verts = verts - 1
        verts = verts / (
            torch.tensor([dimX, dimY, dimZ], dtype=verts.dtype, device=verts.device) - 1
        )
        return verts, tris.long()

class DiffDMC(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        if dtype == torch.float32:
            dmc = _C.CUDMCFloat()
        elif dtype == torch.float64:
            dmc = _C.CUDMCDouble()

        class DDMCFunction(Function):
            @staticmethod
            def forward(ctx, grid, deform, isovalue):
                if deform is None:
                    verts, quads = dmc.forward(grid, isovalue)
                else:
                    verts, quads = dmc.forward(grid, deform, isovalue)
                ctx.grid = grid
                ctx.deform = deform
                ctx.isovalue = isovalue
                return verts, quads

            @staticmethod
            def backward(ctx, adj_verts, adj_faces):
                adj_grid = torch.zeros_like(ctx.grid)
                if ctx.deform is None:
                    dmc.backward(
                        ctx.grid, ctx.isovalue, adj_verts, adj_grid
                    )
                    return adj_grid, None, None, None, None
                else:
                    adj_deform = torch.zeros_like(ctx.deform)
                    dmc.backward(
                        ctx.grid, ctx.deform, ctx.isovalue, adj_verts, adj_grid, adj_deform
                    )
                    return adj_grid, adj_deform, None, None, None

        self.func = DDMCFunction

    def forward(self, grid, deform=None, isovalue=0.0, return_quads=False):
        if grid.min() > 0:
            return torch.zeros((0, 3), dtype=self.dtype, device=grid.device), torch.zeros((0, 4), dtype=torch.int32, device=grid.device)
        dimX, dimY, dimZ = grid.shape
        grid = F.pad(grid, (1, 1, 1, 1, 1, 1), "constant", 1)
        if deform is not None:
            deform = F.pad(deform, (0, 0, 1, 1, 1, 1, 1, 1), "constant", 0)
        verts, quads = self.func.apply(grid, deform, isovalue)
        verts = verts - 1
        verts = verts / (
            torch.tensor([dimX, dimY, dimZ], dtype=verts.dtype, device=verts.device) - 1
        )
        if return_quads:
            return verts, quads.long()
        else:
            # divide the quad into two triangles maximize the smallest angle within each triangle
            quads = quads.long()
            face_config1 = torch.tensor([[0, 1, 3], [1, 2, 3]])
            face_config2 = torch.tensor([[0, 1, 2], [0, 2, 3]])

            angles1, angles2 = [], []
            for i in range(len(face_config1)):
                v0, v1, v2 = torch.unbind(verts[quads[:, face_config1[i]]], dim=-2)
                cos1 = (F.normalize(v1-v0, dim=-1) * F.normalize(v2-v0, dim=-1)).sum(-1)
                cos2 = (F.normalize(v2-v1, dim=-1) * F.normalize(v0-v1, dim=-1)).sum(-1)
                cos3 = (F.normalize(v0-v2, dim=-1) * F.normalize(v1-v2, dim=-1)).sum(-1)
                angles1.append(torch.max(torch.stack([cos1, cos2, cos3], dim=-1), dim=-1)[0])
            for i in range(len(face_config2)):
                v0, v1, v2 = torch.unbind(verts[quads[:, face_config2[i]]], dim=-2)
                cos1 = (F.normalize(v1-v0, dim=-1) * F.normalize(v2-v0, dim=-1)).sum(-1)
                cos2 = (F.normalize(v2-v1, dim=-1) * F.normalize(v0-v1, dim=-1)).sum(-1)
                cos3 = (F.normalize(v0-v2, dim=-1) * F.normalize(v1-v2, dim=-1)).sum(-1)
                angles2.append(torch.max(torch.stack([cos1, cos2, cos3], dim=-1), dim=-1)[0])

            angles1 = torch.stack(angles1, dim=-1)
            angles2 = torch.stack(angles2, dim=-1)

            angles1 = torch.max(angles1, dim=1)[0]
            angles2 = torch.max(angles2, dim=1)[0]

            faces_1 = quads[angles1 < angles2]
            faces_2 = quads[angles1 >= angles2]
            faces = torch.cat([faces_1[:, [0, 1, 3, 1, 2, 3]].view(-1, 3), faces_2[:, [0, 1, 2, 0, 2, 3]].view(-1, 3)], dim=0)
            
            return verts, faces.long()
