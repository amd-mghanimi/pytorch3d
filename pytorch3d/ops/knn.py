# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import heapq
from collections import namedtuple
from typing import Union

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable


_KNN = namedtuple("KNN", "dists idx knn")


def _knn_points_idx_forward(
    p1: torch.Tensor,
    p2: torch.Tensor,
    lengths1: torch.Tensor,
    lengths2: torch.Tensor,
    norm: int,
    K: int,
) -> tuple:
    """
    Pure PyTorch implementation of KNN forward.
    Finds K nearest neighbors in p2 for each point in p1.
    """
    N, P1, D = p1.shape
    P2 = p2.shape[1]

    idx = torch.zeros((N, P1, K), dtype=torch.int64, device=p1.device)
    dists = torch.zeros((N, P1, K), dtype=p1.dtype, device=p1.device)

    for n in range(N):
        len1 = lengths1[n].item()
        len2 = lengths2[n].item()
        for i in range(len1):
            # Compute distances to all points in p2[n]
            diff = p1[n, i : i + 1] - p2[n, :len2]  # (len2, D)
            if norm == 1:
                dist = diff.abs().sum(dim=1)  # (len2,)
            else:  # norm == 2
                dist = (diff * diff).sum(dim=1)  # (len2,)
            # Get K smallest using heap
            if len2 <= K:
                k_actual = len2
                sort_idx = torch.argsort(dist)
                idx[n, i, :k_actual] = sort_idx
                dists[n, i, :k_actual] = dist[sort_idx]
            else:
                # Use heap to get K smallest
                heap = [(dist[j].item(), j) for j in range(len2)]
                heapq.heapify(heap)
                k_smallest = heapq.nsmallest(K, heap)
                for k, (d, j) in enumerate(k_smallest):
                    idx[n, i, k] = j
                    dists[n, i, k] = d
    return idx, dists


def _knn_points_backward_python(
    p1: torch.Tensor,
    p2: torch.Tensor,
    idx: torch.Tensor,
    grad_dists: torch.Tensor,
    norm: int,
    lengths1: torch.Tensor,
    lengths2: torch.Tensor,
) -> tuple:
    """
    Pure PyTorch implementation of KNN backward.
    Supports norm=1 (L1) and norm=2 (L2 squared).
    """
    N, P1, K = idx.shape
    D = p1.shape[2]
    P2 = p2.shape[1]

    # Valid mask: (n,i,k) valid if k < min(lengths2[n], K) and idx[n,i,k] >= 0
    # Matches C++ which iterates k in range(min(length2, K)); idx>=0 handles ball_query padding
    k_valid = torch.arange(K, device=idx.device).unsqueeze(0)
    len2_clamped = torch.minimum(lengths2, torch.tensor(K, device=idx.device))
    valid_k = k_valid < len2_clamped.unsqueeze(1)
    valid_k = valid_k.unsqueeze(1).expand(-1, P1, -1)
    valid_i = (
        torch.arange(P1, device=idx.device).unsqueeze(0).unsqueeze(2)
        < lengths1.unsqueeze(1).unsqueeze(2)
    )
    mask = (
        (valid_k & valid_i & (idx >= 0))
        .unsqueeze(-1)
        .expand(-1, -1, -1, D)
    )

    idx_clamped = idx.clamp(min=0)
    p2_expanded = p2[:, :, None, :].expand(-1, -1, K, -1)
    idx_expanded = idx_clamped.unsqueeze(-1).expand(-1, -1, -1, D)
    p2_nn = p2_expanded.gather(1, idx_expanded)
    p2_nn = torch.where(mask, p2_nn, torch.zeros_like(p2_nn))

    diff = p1.unsqueeze(2) - p2_nn  # (N, P1, K, D)
    if norm == 1:
        # L1: d(|x|)/dx = sign(x), use 1 for x>0, -1 for x<=0
        sign = torch.where(diff > 0, torch.ones_like(diff), -torch.ones_like(diff))
        grad_diff = grad_dists.unsqueeze(-1) * sign
    else:  # norm == 2
        grad_diff = grad_dists.unsqueeze(-1) * 2.0 * diff
    grad_diff = torch.where(mask, grad_diff, torch.zeros_like(grad_diff))

    grad_p1 = grad_diff.sum(dim=2)

    grad_p2 = torch.zeros_like(p2)
    grad_diff_neg = -grad_diff
    grad_diff_neg = torch.where(mask, grad_diff_neg, torch.zeros_like(grad_diff_neg))
    # scatter_add: for each (n,i,k), add grad_diff_neg[n,i,k] to grad_p2[n, idx[n,i,k]]
    # Flatten (P1, K) for scatter: index (N, P1*K, D), src (N, P1*K, D)
    idx_flat = idx.reshape(N, -1, 1).expand(-1, -1, D)
    grad_flat = grad_diff_neg.reshape(N, -1, D)
    idx_flat = torch.where(
        (idx >= 0).reshape(N, -1, 1).expand(-1, -1, D),
        idx_flat,
        torch.zeros_like(idx_flat),
    )
    grad_flat = torch.where(
        (idx >= 0).reshape(N, -1, 1).expand(-1, -1, D),
        grad_flat,
        torch.zeros_like(grad_flat),
    )
    grad_p2.scatter_add_(1, idx_flat, grad_flat)
    return grad_p1, grad_p2


class _knn_points(Function):
    """
    Torch autograd Function wrapper for KNN C++/CUDA implementations.
    """

    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(
        ctx,
        p1,
        p2,
        lengths1,
        lengths2,
        K,
        version,
        norm: int = 2,
        return_sorted: bool = True,
    ):
        """
        K-Nearest neighbors on point clouds.

        Args:
            p1: Tensor of shape (N, P1, D) giving a batch of N point clouds, each
                containing up to P1 points of dimension D.
            p2: Tensor of shape (N, P2, D) giving a batch of N point clouds, each
                containing up to P2 points of dimension D.
            lengths1: LongTensor of shape (N,) of values in the range [0, P1], giving the
                length of each pointcloud in p1. Or None to indicate that every cloud has
                length P1.
            lengths2: LongTensor of shape (N,) of values in the range [0, P2], giving the
                length of each pointcloud in p2. Or None to indicate that every cloud has
                length P2.
            K: Integer giving the number of nearest neighbors to return.
            version: Which KNN implementation to use in the backend. If version=-1,
                the correct implementation is selected based on the shapes of the inputs.
            norm: (int) indicating the norm. Only supports 1 (for L1) and 2 (for L2).
            return_sorted: (bool) whether to return the nearest neighbors sorted in
                ascending order of distance.

        Returns:
            p1_dists: Tensor of shape (N, P1, K) giving the squared distances to
                the nearest neighbors. This is padded with zeros both where a cloud in p2
                has fewer than K points and where a cloud in p1 has fewer than P1 points.

            p1_idx: LongTensor of shape (N, P1, K) giving the indices of the
                K nearest neighbors from points in p1 to points in p2.
                Concretely, if `p1_idx[n, i, k] = j` then `p2[n, j]` is the k-th nearest
                neighbors to `p1[n, i]` in `p2[n]`. This is padded with zeros both where a cloud
                in p2 has fewer than K points and where a cloud in p1 has fewer than P1 points.
        """
        if not ((norm == 1) or (norm == 2)):
            raise ValueError("Support for 1 or 2 norm.")

        idx, dists = _knn_points_idx_forward(
            p1, p2, lengths1, lengths2, norm, K
        )

        # sort KNN in ascending order if K > 1
        if K > 1 and return_sorted:
            if lengths2.min() < K:
                P1 = p1.shape[1]
                mask = lengths2[:, None] <= torch.arange(K, device=dists.device)[None]
                # mask has shape [N, K], true where dists irrelevant
                mask = mask[:, None].expand(-1, P1, -1)
                # mask has shape [N, P1, K], true where dists irrelevant
                dists[mask] = float("inf")
                dists, sort_idx = dists.sort(dim=2)
                dists[mask] = 0
            else:
                dists, sort_idx = dists.sort(dim=2)
            idx = idx.gather(2, sort_idx)

        ctx.save_for_backward(p1, p2, lengths1, lengths2, idx)
        ctx.mark_non_differentiable(idx)
        ctx.norm = norm
        return dists, idx

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_dists, grad_idx):
        p1, p2, lengths1, lengths2, idx = ctx.saved_tensors
        norm = ctx.norm
        # TODO(gkioxari) Change cast to floats once we add support for doubles.
        if not (grad_dists.dtype == torch.float32):
            grad_dists = grad_dists.float()
        if not (p1.dtype == torch.float32):
            p1 = p1.float()
        if not (p2.dtype == torch.float32):
            p2 = p2.float()
        grad_p1, grad_p2 = _knn_points_backward_python(
            p1, p2, idx, grad_dists, norm, lengths1, lengths2
        )
        return grad_p1, grad_p2, None, None, None, None, None, None


def knn_points(
    p1: torch.Tensor,
    p2: torch.Tensor,
    lengths1: Union[torch.Tensor, None] = None,
    lengths2: Union[torch.Tensor, None] = None,
    norm: int = 2,
    K: int = 1,
    version: int = -1,
    return_nn: bool = False,
    return_sorted: bool = True,
) -> _KNN:
    """
    K-Nearest neighbors on point clouds.

    Args:
        p1: Tensor of shape (N, P1, D) giving a batch of N point clouds, each
            containing up to P1 points of dimension D.
        p2: Tensor of shape (N, P2, D) giving a batch of N point clouds, each
            containing up to P2 points of dimension D.
        lengths1: LongTensor of shape (N,) of values in the range [0, P1], giving the
            length of each pointcloud in p1. Or None to indicate that every cloud has
            length P1.
        lengths2: LongTensor of shape (N,) of values in the range [0, P2], giving the
            length of each pointcloud in p2. Or None to indicate that every cloud has
            length P2.
        norm: Integer indicating the norm of the distance. Supports only 1 for L1, 2 for L2.
        K: Integer giving the number of nearest neighbors to return.
        version: Which KNN implementation to use in the backend. If version=-1,
            the correct implementation is selected based on the shapes of the inputs.
        return_nn: If set to True returns the K nearest neighbors in p2 for each point in p1.
        return_sorted: (bool) whether to return the nearest neighbors sorted in
            ascending order of distance.

    Returns:
        dists: Tensor of shape (N, P1, K) giving the squared distances to
            the nearest neighbors. This is padded with zeros both where a cloud in p2
            has fewer than K points and where a cloud in p1 has fewer than P1 points.

        idx: LongTensor of shape (N, P1, K) giving the indices of the
            K nearest neighbors from points in p1 to points in p2.
            Concretely, if `p1_idx[n, i, k] = j` then `p2[n, j]` is the k-th nearest
            neighbors to `p1[n, i]` in `p2[n]`. This is padded with zeros both where a cloud
            in p2 has fewer than K points and where a cloud in p1 has fewer than P1
            points.

        nn: Tensor of shape (N, P1, K, D) giving the K nearest neighbors in p2 for
            each point in p1. Concretely, `p2_nn[n, i, k]` gives the k-th nearest neighbor
            for `p1[n, i]`. Returned if `return_nn` is True.
            The nearest neighbors are collected using `knn_gather`

            .. code-block::

                p2_nn = knn_gather(p2, p1_idx, lengths2)

            which is a helper function that allows indexing any tensor of shape (N, P2, U) with
            the indices `p1_idx` returned by `knn_points`. The output is a tensor
            of shape (N, P1, K, U).

    """
    if p1.shape[0] != p2.shape[0]:
        raise ValueError("pts1 and pts2 must have the same batch dimension.")
    if p1.shape[2] != p2.shape[2]:
        raise ValueError("pts1 and pts2 must have the same point dimension.")

    p1 = p1.contiguous()
    p2 = p2.contiguous()

    P1 = p1.shape[1]
    P2 = p2.shape[1]

    if lengths1 is None:
        lengths1 = torch.full((p1.shape[0],), P1, dtype=torch.int64, device=p1.device)
    if lengths2 is None:
        lengths2 = torch.full((p1.shape[0],), P2, dtype=torch.int64, device=p1.device)

    p1_dists, p1_idx = _knn_points.apply(
        p1, p2, lengths1, lengths2, K, version, norm, return_sorted
    )

    p2_nn = None
    if return_nn:
        p2_nn = knn_gather(p2, p1_idx, lengths2)

    return _KNN(dists=p1_dists, idx=p1_idx, knn=p2_nn if return_nn else None)


def knn_gather(
    x: torch.Tensor, idx: torch.Tensor, lengths: Union[torch.Tensor, None] = None
):
    """
    A helper function for knn that allows indexing a tensor x with the indices `idx`
    returned by `knn_points`.

    For example, if `dists, idx = knn_points(p, x, lengths_p, lengths, K)`
    where p is a tensor of shape (N, L, D) and x a tensor of shape (N, M, D),
    then one can compute the K nearest neighbors of p with `p_nn = knn_gather(x, idx, lengths)`.
    It can also be applied for any tensor x of shape (N, M, U) where U != D.

    Args:
        x: Tensor of shape (N, M, U) containing U-dimensional features to
            be gathered.
        idx: LongTensor of shape (N, L, K) giving the indices returned by `knn_points`.
        lengths: LongTensor of shape (N,) of values in the range [0, M], giving the
            length of each example in the batch in x. Or None to indicate that every
            example has length M.
    Returns:
        x_out: Tensor of shape (N, L, K, U) resulting from gathering the elements of x
            with idx, s.t. `x_out[n, l, k] = x[n, idx[n, l, k]]`.
            If `k > lengths[n]` then `x_out[n, l, k]` is filled with 0.0.
    """
    N, M, U = x.shape
    _N, L, K = idx.shape

    if N != _N:
        raise ValueError("x and idx must have same batch dimension.")

    if lengths is None:
        lengths = torch.full((x.shape[0],), M, dtype=torch.int64, device=x.device)

    idx_expanded = idx[:, :, :, None].expand(-1, -1, -1, U)
    # idx_expanded has shape [N, L, K, U]

    x_out = x[:, :, None].expand(-1, -1, K, -1).gather(1, idx_expanded)
    # p2_nn has shape [N, L, K, U]

    needs_mask = lengths.min() < K
    if needs_mask:
        # mask has shape [N, K], true where idx is irrelevant because
        # there is less number of points in p2 than K
        mask = lengths[:, None] <= torch.arange(K, device=x.device)[None]

        # expand mask to shape [N, L, K, U]
        mask = mask[:, None].expand(-1, L, -1)
        mask = mask[:, :, :, None].expand(-1, -1, -1, U)
        x_out[mask] = 0.0

    return x_out
