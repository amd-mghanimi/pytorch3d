# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable


def _interp_face_attrs_forward(
    pix_to_face: torch.Tensor,
    barycentric_coords: torch.Tensor,
    face_attrs: torch.Tensor,
) -> torch.Tensor:
    """
    Pure PyTorch implementation of face attribute interpolation forward.
    pix_attrs[p, d] = sum_i barycentric_coords[p, i] * face_attrs[f, i, d]
    where f = pix_to_face[p]. Pixels with pix_to_face < 0 output 0.
    """
    P = pix_to_face.shape[0]
    _, _, D = face_attrs.shape
    mask = pix_to_face >= 0
    pix_to_face_clamped = pix_to_face.clamp(min=0)
    # Gather face attributes: (P, 3, D)
    idx = pix_to_face_clamped.view(P, 1, 1).expand(P, 3, D)
    face_attrs_at_pixels = face_attrs.gather(0, idx)
    # Interpolate: sum over vertex dim
    pix_attrs = (barycentric_coords.unsqueeze(-1) * face_attrs_at_pixels).sum(
        dim=1
    )
    pix_attrs = torch.where(mask.unsqueeze(-1), pix_attrs, torch.zeros_like(pix_attrs))
    return pix_attrs


def _interp_face_attrs_backward(
    pix_to_face: torch.Tensor,
    barycentric_coords: torch.Tensor,
    face_attrs: torch.Tensor,
    grad_pix_attrs: torch.Tensor,
) -> tuple:
    """
    Pure PyTorch implementation of face attribute interpolation backward.
    """
    P = pix_to_face.shape[0]
    _, _, D = face_attrs.shape
    mask = pix_to_face >= 0
    pix_to_face_clamped = pix_to_face.clamp(min=0)
    idx = pix_to_face_clamped.view(P, 1, 1).expand(P, 3, D)
    face_attrs_at_pixels = face_attrs.gather(0, idx)

    # grad_barycentric_coords[p, i] = sum_d face_attrs[f,i,d] * grad_pix_attrs[p,d]
    grad_barycentric_coords = (
        face_attrs_at_pixels * grad_pix_attrs.unsqueeze(1)
    ).sum(dim=-1)
    grad_barycentric_coords = torch.where(
        mask.unsqueeze(-1), grad_barycentric_coords, torch.zeros_like(grad_barycentric_coords)
    )

    # grad_face_attrs[f, i, d] += bary[p,i] * grad_pix[p,d] for each p with pix_to_face[p]=f
    grad_face_attrs = torch.zeros_like(face_attrs)
    src = (
        barycentric_coords.unsqueeze(-1) * grad_pix_attrs.unsqueeze(1)
    )  # (P, 3, D)
    idx_expanded = pix_to_face_clamped.view(P, 1, 1).expand(P, 3, D)
    src = torch.where(
        mask.unsqueeze(-1).unsqueeze(-1), src, torch.zeros_like(src)
    )
    idx_expanded = torch.where(
        mask.unsqueeze(-1).unsqueeze(-1), idx_expanded, torch.zeros_like(idx_expanded)
    )
    grad_face_attrs.scatter_add_(0, idx_expanded, src)
    return grad_barycentric_coords, grad_face_attrs


def interpolate_face_attributes(
    pix_to_face: torch.Tensor,
    barycentric_coords: torch.Tensor,
    face_attributes: torch.Tensor,
) -> torch.Tensor:
    """
    Interpolate arbitrary face attributes using the barycentric coordinates
    for each pixel in the rasterized output.

    Args:
        pix_to_face: LongTensor of shape (...) specifying the indices
            of the faces (in the packed representation) which overlap each
            pixel in the image. A value < 0 indicates that the pixel does not
            overlap any face and should be skipped.
        barycentric_coords: FloatTensor of shape (N, H, W, K, 3) specifying
            the barycentric coordinates of each pixel
            relative to the faces (in the packed
            representation) which overlap the pixel.
        face_attributes: packed attributes of shape (total_faces, 3, D),
            specifying the value of the attribute for each
            vertex in the face.

    Returns:
        pixel_vals: tensor of shape (N, H, W, K, D) giving the interpolated
        value of the face attribute for each pixel.
    """
    # Check shapes
    F, FV, D = face_attributes.shape
    if FV != 3:
        raise ValueError("Faces can only have three vertices; got %r" % FV)
    N, H, W, K, _ = barycentric_coords.shape
    if pix_to_face.shape != (N, H, W, K):
        msg = "pix_to_face must have shape (batch_size, H, W, K); got %r"
        raise ValueError(msg % (pix_to_face.shape,))

    # Flatten and call the custom autograd function (pure PyTorch implementation)
    N, H, W, K = pix_to_face.shape
    pix_to_face = pix_to_face.view(-1)
    barycentric_coords = barycentric_coords.view(N * H * W * K, 3)
    args = (pix_to_face, barycentric_coords, face_attributes)
    out = _InterpFaceAttrs.apply(*args)
    out = out.view(N, H, W, K, -1)
    return out


class _InterpFaceAttrs(Function):
    @staticmethod
    def forward(ctx, pix_to_face, barycentric_coords, face_attrs):
        ctx.save_for_backward(pix_to_face, barycentric_coords, face_attrs)
        return _interp_face_attrs_forward(
            pix_to_face, barycentric_coords, face_attrs
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_pix_attrs):
        pix_to_face, barycentric_coords, face_attrs = ctx.saved_tensors
        grad_barycentric_coords, grad_face_attrs = _interp_face_attrs_backward(
            pix_to_face, barycentric_coords, face_attrs, grad_pix_attrs
        )
        return None, grad_barycentric_coords, grad_face_attrs


def interpolate_face_attributes_python(
    pix_to_face: torch.Tensor,
    barycentric_coords: torch.Tensor,
    face_attributes: torch.Tensor,
) -> torch.Tensor:
    F, FV, D = face_attributes.shape
    N, H, W, K, _ = barycentric_coords.shape

    # Replace empty pixels in pix_to_face with 0 in order to interpolate.
    mask = pix_to_face < 0
    pix_to_face = pix_to_face.clone()
    pix_to_face[mask] = 0
    idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
    pixel_face_vals = face_attributes.gather(0, idx).view(N, H, W, K, 3, D)
    pixel_vals = (barycentric_coords[..., None] * pixel_face_vals).sum(dim=-2)
    pixel_vals[mask] = 0  # Replace masked values in output.
    return pixel_vals
