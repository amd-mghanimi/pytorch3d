# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
from torch.autograd import Function


_FACE_AREAS_NORMALS_EPS = 1e-6


def _face_areas_normals_forward(verts: torch.Tensor, faces: torch.Tensor):
    """
    Pure PyTorch implementation of face areas and normals.
    verts: (V, 3), faces: (F, 3) -> areas: (F,), normals: (F, 3)
    """
    v0 = verts[faces[:, 0]]  # (F, 3)
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    a = v1 - v0
    b = v2 - v0
    cross = torch.cross(a, b, dim=-1)
    norm = cross.norm(dim=-1, keepdim=True).clamp(min=_FACE_AREAS_NORMALS_EPS)
    areas = (norm.squeeze(-1) / 2.0).contiguous()
    normals = (cross / norm).contiguous()
    return areas, normals


class _MeshFaceAreasNormals(Function):
    """
    Torch autograd Function wrapper for face areas & normals.
    Uses pure PyTorch implementation (no C++/CUDA).
    """

    @staticmethod
    def forward(ctx, verts, faces):
        """
        Args:
            ctx: Context object used to calculate gradients.
            verts: FloatTensor of shape (V, 3), representing the packed
                batch verts tensor.
            faces: LongTensor of shape (F, 3), representing the packed
                batch faces tensor
        Returns:
            areas: FloatTensor of shape (F,) with the areas of each face
            normals: FloatTensor of shape (F,3) with the normals of each face
        """
        if not (verts.dim() == 2):
            raise ValueError("verts need to be of shape Vx3.")
        if not (verts.shape[1] == 3):
            raise ValueError("verts need to be of shape Vx3.")
        if not (faces.dim() == 2):
            raise ValueError("faces need to be of shape Fx3.")
        if not (faces.shape[1] == 3):
            raise ValueError("faces need to be of shape Fx3.")
        if not (faces.dtype == torch.int64):
            raise ValueError("faces need to be of type torch.int64.")
        # TODO(gkioxari) Change cast to floats once we add support for doubles.
        if not (verts.dtype == torch.float32):
            verts = verts.float()

        ctx.save_for_backward(verts, faces)
        areas, normals = _face_areas_normals_forward(verts, faces)
        return areas, normals

    @staticmethod
    def backward(ctx, grad_areas, grad_normals):
        verts, faces = ctx.saved_tensors
        if grad_areas is None and grad_normals is None:
            return None, None

        # Grad is disabled inside Function.backward; use enable_grad to build graph
        with torch.enable_grad():
            verts_grad = verts.detach().requires_grad_(True)
            areas, normals = _face_areas_normals_forward(verts_grad, faces)

            grad_outputs = []
            if grad_areas is not None:
                grad_outputs.append(grad_areas.contiguous().float())
            else:
                grad_outputs.append(torch.zeros_like(areas))
            if grad_normals is not None:
                grad_outputs.append(grad_normals.contiguous().float())
            else:
                grad_outputs.append(torch.zeros_like(normals))

            grad_verts, = torch.autograd.grad(
                outputs=[areas, normals],
                inputs=verts_grad,
                grad_outputs=grad_outputs,
                retain_graph=False,
            )
        return grad_verts, None


mesh_face_areas_normals = _MeshFaceAreasNormals.apply
