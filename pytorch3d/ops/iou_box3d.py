# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from torch.autograd import Function


# -------------------------------------------------- #
#                  CONSTANTS                         #
# -------------------------------------------------- #
DOT_EPS = 1e-3
AREA_EPS = 1e-4

"""
_box_planes and _box_triangles define the 4- and 3-connectivity
of the 8 box corners.
_box_planes gives the quad faces of the 3D box
_box_triangles gives the triangle faces of the 3D box
"""
_box_planes = [
    [0, 1, 2, 3],
    [3, 2, 6, 7],
    [0, 1, 5, 4],
    [0, 3, 7, 4],
    [1, 2, 6, 5],
    [4, 5, 6, 7],
]
_box_triangles = [
    [0, 1, 2],
    [0, 3, 2],
    [4, 5, 6],
    [4, 6, 7],
    [1, 5, 6],
    [1, 6, 2],
    [0, 4, 7],
    [0, 7, 3],
    [3, 2, 6],
    [3, 6, 7],
    [0, 1, 5],
    [0, 4, 5],
]


def _check_coplanar(boxes: torch.Tensor, eps: float = 1e-4) -> None:
    faces = torch.tensor(_box_planes, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    P, V = faces.shape
    # (B, P, 4, 3) -> (B, P, 3)
    v0, v1, v2, v3 = verts.reshape(B, P, V, 3).unbind(2)

    # Compute the normal
    e0 = F.normalize(v1 - v0, dim=-1)
    e1 = F.normalize(v2 - v0, dim=-1)
    normal = F.normalize(torch.cross(e0, e1, dim=-1), dim=-1)

    # Check the fourth vertex is also on the same plane
    mat1 = (v3 - v0).view(B, 1, -1)  # (B, 1, P*3)
    mat2 = normal.view(B, -1, 1)  # (B, P*3, 1)
    if not (mat1.bmm(mat2).abs() < eps).all().item():
        msg = "Plane vertices are not coplanar"
        raise ValueError(msg)

    return


def _check_nonzero(boxes: torch.Tensor, eps: float = 1e-4) -> None:
    """
    Checks that the sides of the box have a non zero area
    """
    faces = torch.tensor(_box_triangles, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    T, V = faces.shape
    # (B, T, 3, 3) -> (B, T, 3)
    v0, v1, v2 = verts.reshape(B, T, V, 3).unbind(2)

    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (B, T, 3)
    face_areas = normals.norm(dim=-1) / 2

    if (face_areas < eps).any().item():
        msg = "Planes have zero areas"
        raise ValueError(msg)

    return


# -------------------------------------------------- #
#       PURE PYTORCH BOX3D OVERLAP HELPERS            #
# -------------------------------------------------- #


def _get_tri_verts(box: torch.Tensor) -> torch.Tensor:
    """Return vertex coords for box triangles. box: (8, 3) -> (12, 3, 3)."""
    faces = torch.tensor(_box_triangles, device=box.device, dtype=torch.int64)
    return box[faces]


def _get_plane_verts(box: torch.Tensor) -> torch.Tensor:
    """Return vertex coords for box planes. box: (8, 3) -> (6, 4, 3)."""
    faces = torch.tensor(_box_planes, device=box.device, dtype=torch.int64)
    return box[faces]


def _get_tri_center_normal(tris: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """tris: (T, 3, 3) -> center (T, 3), normal (T, 3)."""
    ctr = tris.mean(1)
    v0, v1, v2 = tris.unbind(1)
    ns = torch.stack(
        [
            torch.cross(v0 - ctr, v1 - ctr, dim=-1),
            torch.cross(v0 - ctr, v2 - ctr, dim=-1),
            torch.cross(v1 - ctr, v2 - ctr, dim=-1),
        ],
        dim=0,
    )
    i = torch.norm(ns, dim=-1).argmax(dim=0)
    normals = ns[i, torch.arange(tris.shape[0], device=tris.device)]
    return ctr, F.normalize(normals, dim=-1)


def _get_plane_center_normal(planes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """planes: (P, 4, 3) -> center (P, 3), normal (P, 3)."""
    ctr = planes.mean(1)
    v0, v1, v2, v3 = planes.unbind(1)
    ns = torch.stack(
        [
            torch.cross(v0 - ctr, v1 - ctr, dim=-1),
            torch.cross(v0 - ctr, v2 - ctr, dim=-1),
            torch.cross(v0 - ctr, v3 - ctr, dim=-1),
            torch.cross(v1 - ctr, v2 - ctr, dim=-1),
            torch.cross(v1 - ctr, v3 - ctr, dim=-1),
            torch.cross(v2 - ctr, v3 - ctr, dim=-1),
        ],
        dim=0,
    )
    i = torch.norm(ns, dim=-1).argmax(dim=0)
    normals = ns[i, torch.arange(planes.shape[0], device=planes.device)]
    return ctr, F.normalize(normals, dim=-1)


def _box_planar_dir(box: torch.Tensor) -> torch.Tensor:
    """Unit normals pointing inward for each of 6 faces. box: (8, 3) -> (6, 3)."""
    box_ctr = box.mean(0).view(1, 3)
    plane_verts = _get_plane_verts(box)
    plane_ctr, n = _get_plane_center_normal(plane_verts)
    direc = F.normalize(box_ctr - plane_ctr, dim=-1)
    c = (direc * n).sum(-1)
    n = n.clone()
    n[c < 0] *= -1.0
    return n


def _box_volume(box: torch.Tensor) -> torch.Tensor:
    """Volume of box. box: (8, 3) -> scalar."""
    ctr = box.mean(0).view(1, 1, 3)
    tri_verts = _get_tri_verts(box) - ctr
    vols = (
        tri_verts[:, 0]
        * torch.cross(tri_verts[:, 1], tri_verts[:, 2], dim=-1)
    ).sum(-1)
    return (vols.abs() / 6.0).sum()


def _tri_verts_area(tri_verts: torch.Tensor) -> torch.Tensor:
    """tri_verts: (3, 3) or (T, 3, 3) -> scalar or (T,)."""
    v0, v1, v2 = tri_verts.unbind(-2)
    return torch.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1) / 2.0


def _coplanar_tri_plane(
    tri: torch.Tensor, plane: torch.Tensor, n: torch.Tensor, eps: float = DOT_EPS
) -> bool:
    tri_ctr, tri_n = _get_tri_center_normal(tri.unsqueeze(0))
    tri_ctr, tri_n = tri_ctr[0], tri_n[0]
    check1 = tri_n.dot(n).abs() > 1 - eps
    dist12 = torch.norm(tri.unsqueeze(1) - plane.unsqueeze(0), dim=-1)
    i1, i2 = divmod(dist12.argmax().item(), 4)
    check2 = F.normalize(tri[i1] - plane[i2], dim=0).dot(n).abs() < eps
    return bool(check1.item() and check2.item())


def _is_inside(
    plane: torch.Tensor, n: torch.Tensor, points: torch.Tensor
) -> torch.Tensor:
    """points: (P, 3) -> bool (P,)."""
    plane_ctr = plane.mean(0)
    direc = (points - plane_ctr.unsqueeze(0)) * n.unsqueeze(0)
    return direc.sum(-1) >= 0.0


def _plane_edge_intersection(
    plane: torch.Tensor, n: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor
) -> Tuple[torch.Tensor, float]:
    direc = F.normalize(p1 - p0, dim=0)
    if direc.dot(n).abs() < DOT_EPS:
        return (p1 + p0) / 2.0, 0.5
    ctr = plane.mean(0)
    denom = (p1 - p0).dot(n)
    a = -(p0 - ctr).dot(n) / denom
    return p0 + a * (p1 - p0), a.item()


def _clip_tri_by_plane_oneout(
    plane: torch.Tensor,
    n: torch.Tensor,
    vout: torch.Tensor,
    vin1: torch.Tensor,
    vin2: torch.Tensor,
) -> torch.Tensor:
    pint1, _ = _plane_edge_intersection(plane, n, vin1, vout)
    pint2, _ = _plane_edge_intersection(plane, n, vin2, vout)
    verts = torch.stack((vin1, pint1, pint2, vin2), dim=0)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64, device=plane.device)
    return verts[faces]


def _clip_tri_by_plane_twoout(
    plane: torch.Tensor,
    n: torch.Tensor,
    vout1: torch.Tensor,
    vout2: torch.Tensor,
    vin: torch.Tensor,
) -> torch.Tensor:
    pint1, _ = _plane_edge_intersection(plane, n, vin, vout1)
    pint2, _ = _plane_edge_intersection(plane, n, vin, vout2)
    verts = torch.stack((vin, pint1, pint2), dim=0)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=plane.device)
    return verts[faces]


def _clip_tri_by_plane(
    plane: torch.Tensor, n: torch.Tensor, tri_verts: torch.Tensor
) -> Union[List[torch.Tensor], torch.Tensor]:
    if _coplanar_tri_plane(tri_verts, plane, n):
        return tri_verts.unsqueeze(0)
    v0, v1, v2 = tri_verts.unbind(0)
    isin0 = _is_inside(plane, n, v0.unsqueeze(0))[0]
    isin1 = _is_inside(plane, n, v1.unsqueeze(0))[0]
    isin2 = _is_inside(plane, n, v2.unsqueeze(0))[0]
    if isin0 and isin1 and isin2:
        return tri_verts.unsqueeze(0)
    if not isin0 and not isin1 and not isin2:
        return []
    if isin0:
        if isin1:
            return _clip_tri_by_plane_oneout(plane, n, v2, v0, v1)
        elif isin2:
            return _clip_tri_by_plane_oneout(plane, n, v1, v0, v2)
        else:
            return _clip_tri_by_plane_twoout(plane, n, v1, v2, v0)
    else:
        if isin1 and isin2:
            return _clip_tri_by_plane_oneout(plane, n, v0, v1, v2)
        elif isin1:
            return _clip_tri_by_plane_twoout(plane, n, v0, v2, v1)
        elif isin2:
            return _clip_tri_by_plane_twoout(plane, n, v0, v1, v2)
    return []


def _coplanar_tri_faces(tri1: torch.Tensor, tri2: torch.Tensor) -> bool:
    tri1_ctr, tri1_n = _get_tri_center_normal(tri1.unsqueeze(0))
    tri2_ctr, tri2_n = _get_tri_center_normal(tri2.unsqueeze(0))
    tri1_ctr, tri1_n = tri1_ctr[0], tri1_n[0]
    tri2_ctr, tri2_n = tri2_ctr[0], tri2_n[0]
    check1 = tri1_n.dot(tri2_n).abs() > 1 - DOT_EPS
    dist12 = torch.norm(tri1.unsqueeze(1) - tri2.unsqueeze(0), dim=-1)
    i1, i2 = divmod(dist12.argmax().item(), 3)
    check2 = (
        F.normalize(tri1[i1] - tri2[i2], dim=0).dot(tri1_n).abs() < DOT_EPS
        or F.normalize(tri1[i1] - tri2[i2], dim=0).dot(tri2_n).abs() < DOT_EPS
    )
    return bool(check1.item() and check2)


def _box3d_overlap_single(box1: torch.Tensor, box2: torch.Tensor) -> Tuple[float, float]:
    """Single-pair overlap. Returns (vol, iou)."""
    device = box1.device
    n1 = _box_planar_dir(box1)
    n2 = _box_planar_dir(box2)
    vol1 = _box_volume(box1)
    vol2 = _box_volume(box2)
    tri_verts1 = _get_tri_verts(box1)
    plane_verts1 = _get_plane_verts(box1)
    tri_verts2 = _get_tri_verts(box2)
    plane_verts2 = _get_plane_verts(box2)
    num_planes = 6

    for pidx in range(num_planes):
        plane = plane_verts2[pidx]
        nplane = n2[pidx]
        tri_updated: List[torch.Tensor] = []
        for i in range(tri_verts1.shape[0]):
            clipped = _clip_tri_by_plane(plane, nplane, tri_verts1[i])
            if isinstance(clipped, list) and len(clipped) == 0:
                continue
            tris = clipped if isinstance(clipped, torch.Tensor) else clipped
            for k in range(tris.shape[0]):
                tri_updated.append(tris[k])
        tri_verts1 = torch.stack(tri_updated, dim=0) if tri_updated else tri_verts1[:0]

    for pidx in range(num_planes):
        plane = plane_verts1[pidx]
        nplane = n1[pidx]
        tri_updated = []
        for i in range(tri_verts2.shape[0]):
            clipped = _clip_tri_by_plane(plane, nplane, tri_verts2[i])
            if isinstance(clipped, list) and len(clipped) == 0:
                continue
            tris = clipped if isinstance(clipped, torch.Tensor) else clipped
            for k in range(tris.shape[0]):
                tri_updated.append(tris[k])
        tri_verts2 = torch.stack(tri_updated, dim=0) if tri_updated else tri_verts2[:0]

    keep2 = torch.ones(tri_verts2.shape[0], device=device, dtype=torch.bool)
    for i1 in range(tri_verts1.shape[0]):
        for i2 in range(tri_verts2.shape[0]):
            if (
                _coplanar_tri_faces(tri_verts1[i1], tri_verts2[i2])
                and _tri_verts_area(tri_verts1[i1]).item() > AREA_EPS
            ):
                keep2[i2] = False
    keep2 = keep2.nonzero(as_tuple=True)[0]
    tri_verts2 = tri_verts2[keep2]

    overlap_tri_verts = torch.cat((tri_verts1, tri_verts2), dim=0)
    if overlap_tri_verts.shape[0] == 0:
        return 0.0, 0.0

    ctr = overlap_tri_verts.mean(0).mean(0)
    tetras = overlap_tri_verts - ctr.view(1, 1, 3)
    vol = (
        tetras[:, 0] * torch.cross(tetras[:, 1], tetras[:, 2], dim=-1)
    ).sum(-1).abs().sum() / 6.0

    iou_val = vol / (vol1 + vol2 - vol)
    return vol.item(), iou_val.item()


def _iou_box3d_python(boxes1: torch.Tensor, boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched pure PyTorch box3d overlap. boxes1: (N, 8, 3), boxes2: (M, 8, 3)."""
    N, M = boxes1.shape[0], boxes2.shape[0]
    vols = torch.zeros((N, M), dtype=boxes1.dtype, device=boxes1.device)
    ious = torch.zeros((N, M), dtype=boxes1.dtype, device=boxes1.device)
    for n in range(N):
        for m in range(M):
            vol, iou = _box3d_overlap_single(boxes1[n], boxes2[m])
            vols[n, m] = vol
            ious[n, m] = iou
    return vols, ious


class _box3d_overlap(Function):
    """
    Torch autograd Function wrapper for box3d_overlap C++/CUDA implementations.
    Backward is not supported.
    """

    @staticmethod
    def forward(ctx, boxes1, boxes2):
        """
        Arguments definitions the same as in the box3d_overlap function
        """
        vol, iou = _iou_box3d_python(boxes1, boxes2)
        return vol, iou

    @staticmethod
    def backward(ctx, grad_vol, grad_iou):
        raise ValueError("box3d_overlap backward is not supported")


def box3d_overlap(
    boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the intersection of 3D boxes1 and boxes2.

    Inputs boxes1, boxes2 are tensors of shape (B, 8, 3)
    (where B doesn't have to be the same for boxes1 and boxes2),
    containing the 8 corners of the boxes, as follows:

        (4) +---------+. (5)
            | ` .     |  ` .
            | (0) +---+-----+ (1)
            |     |   |     |
        (7) +-----+---+. (6)|
            ` .   |     ` . |
            (3) ` +---------+ (2)


    NOTE: Throughout this implementation, we assume that boxes
    are defined by their 8 corners exactly in the order specified in the
    diagram above for the function to give correct results. In addition
    the vertices on each plane must be coplanar.
    As an alternative to the diagram, this is a unit bounding
    box which has the correct vertex ordering:

    box_corner_vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]

    Args:
        boxes1: tensor of shape (N, 8, 3) of the coordinates of the 1st boxes
        boxes2: tensor of shape (M, 8, 3) of the coordinates of the 2nd boxes
    Returns:
        vol: (N, M) tensor of the volume of the intersecting convex shapes
        iou: (N, M) tensor of the intersection over union which is
            defined as: `iou = vol / (vol1 + vol2 - vol)`
    """
    if not all((8, 3) == box.shape[1:] for box in [boxes1, boxes2]):
        raise ValueError("Each box in the batch must be of shape (8, 3)")

    _check_coplanar(boxes1, eps)
    _check_coplanar(boxes2, eps)
    _check_nonzero(boxes1, eps)
    _check_nonzero(boxes2, eps)

    vol, iou = _box3d_overlap.apply(boxes1, boxes2)

    return vol, iou
