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

    def forward(self, grid, deform=None, isovalue=0.0):
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
        return verts, quads.long()
