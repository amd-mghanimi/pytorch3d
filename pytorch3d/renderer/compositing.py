# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch


# Epsilon for numerical stability
_EPS = 1e-9
_EPS_NORM = 1e-4


def _accum_alphacomposite_python(features, alphas, points_idx):
    """Pure PyTorch alpha composite forward."""
    B, K, H, W = points_idx.shape
    C = features.shape[0]
    device = features.device
    dtype = features.dtype

    result = torch.zeros(B, C, H, W, dtype=dtype, device=device)
    for b in range(B):
        for c in range(C):
            for j in range(H):
                for i in range(W):
                    cum_alpha = 1.0
                    for k in range(K):
                        n_idx = points_idx[b, k, j, i].item()
                        if n_idx < 0:
                            continue
                        alpha = alphas[b, k, j, i]
                        result[b, c, j, i] = result[b, c, j, i] + (
                            cum_alpha * alpha * features[c, n_idx]
                        )
                        cum_alpha = cum_alpha * (1.0 - alpha)
    return result


def _accum_alphacomposite_backward_python(grad_output, features, alphas, points_idx):
    """Pure PyTorch alpha composite backward."""
    B, K, H, W = points_idx.shape
    C = features.shape[0]
    device = features.device
    dtype = features.dtype

    grad_features = torch.zeros_like(features)
    grad_alphas = torch.zeros_like(alphas)

    for b in range(B):
        for c in range(C):
            for j in range(H):
                for i in range(W):
                    cum_alpha = 1.0
                    for k in range(K):
                        n_idx = points_idx[b, k, j, i].item()
                        if n_idx < 0:
                            continue
                        alpha = alphas[b, k, j, i]
                        grad_alphas[b, k, j, i] = grad_alphas[b, k, j, i] + (
                            grad_output[b, c, j, i] * features[c, n_idx] * cum_alpha
                        )
                        grad_features[c, n_idx] = grad_features[c, n_idx] + (
                            grad_output[b, c, j, i] * cum_alpha * alpha
                        )
                        for t in range(k):
                            t_idx = points_idx[b, t, j, i].item()
                            if t_idx < 0:
                                continue
                            alpha_t = alphas[b, t, j, i]
                            grad_alphas[b, t, j, i] = grad_alphas[b, t, j, i] - (
                                grad_output[b, c, j, i]
                                * features[c, n_idx]
                                * cum_alpha
                                * alpha
                                / (1.0 - alpha_t + _EPS)
                            )
                        cum_alpha = cum_alpha * (1.0 - alpha)

    return grad_features, grad_alphas


def _accum_weightedsum_python(features, alphas, points_idx):
    """Pure PyTorch weighted sum forward."""
    B, K, H, W = points_idx.shape
    C = features.shape[0]
    device = features.device
    dtype = features.dtype

    result = torch.zeros(B, C, H, W, dtype=dtype, device=device)
    for b in range(B):
        for c in range(C):
            for j in range(H):
                for i in range(W):
                    for k in range(K):
                        n_idx = points_idx[b, k, j, i].item()
                        if n_idx < 0:
                            continue
                        alpha = alphas[b, k, j, i]
                        result[b, c, j, i] = result[b, c, j, i] + (
                            alpha * features[c, n_idx]
                        )
    return result


def _accum_weightedsum_backward_python(grad_output, features, alphas, points_idx):
    """Pure PyTorch weighted sum backward."""
    B, K, H, W = points_idx.shape
    C = features.shape[0]
    device = features.device

    grad_features = torch.zeros_like(features)
    grad_alphas = torch.zeros_like(alphas)

    for b in range(B):
        for c in range(C):
            for j in range(H):
                for i in range(W):
                    for k in range(K):
                        n_idx = points_idx[b, k, j, i].item()
                        if n_idx < 0:
                            continue
                        alpha = alphas[b, k, j, i]
                        grad_alphas[b, k, j, i] = grad_alphas[b, k, j, i] + (
                            grad_output[b, c, j, i] * features[c, n_idx]
                        )
                        grad_features[c, n_idx] = grad_features[c, n_idx] + (
                            grad_output[b, c, j, i] * alpha
                        )

    return grad_features, grad_alphas


def _accum_weightedsumnorm_python(features, alphas, points_idx):
    """Pure PyTorch normalized weighted sum forward."""
    B, K, H, W = points_idx.shape
    C = features.shape[0]
    device = features.device
    dtype = features.dtype

    result = torch.zeros(B, C, H, W, dtype=dtype, device=device)
    for b in range(B):
        for c in range(C):
            for j in range(H):
                for i in range(W):
                    t_alpha = torch.tensor(0.0, dtype=dtype, device=device)
                    for k in range(K):
                        n_idx = points_idx[b, k, j, i].item()
                        if n_idx < 0:
                            continue
                        t_alpha = t_alpha + alphas[b, k, j, i]

                    t_alpha_val = max(t_alpha.item(), _EPS_NORM)

                    for k in range(K):
                        n_idx = points_idx[b, k, j, i].item()
                        if n_idx < 0:
                            continue
                        alpha = alphas[b, k, j, i]
                        result[b, c, j, i] = result[b, c, j, i] + (
                            alpha * features[c, n_idx] / t_alpha_val
                        )
    return result


def _accum_weightedsumnorm_backward_python(grad_output, features, alphas, points_idx):
    """Pure PyTorch normalized weighted sum backward."""
    B, K, H, W = points_idx.shape
    C = features.shape[0]
    device = features.device
    dtype = features.dtype

    grad_features = torch.zeros_like(features)
    grad_alphas = torch.zeros_like(alphas)

    for b in range(B):
        for c in range(C):
            for j in range(H):
                for i in range(W):
                    t_alpha = torch.tensor(0.0, dtype=dtype, device=device)
                    t_alphafs = torch.tensor(0.0, dtype=dtype, device=device)
                    for k in range(K):
                        n_idx = points_idx[b, k, j, i].item()
                        if n_idx < 0:
                            continue
                        t_alpha = t_alpha + alphas[b, k, j, i]
                        t_alphafs = t_alphafs + alphas[b, k, j, i] * features[c, n_idx]

                    t_alpha_val = max(t_alpha.item(), _EPS_NORM)
                    t_alphafs_val = t_alphafs.item()

                    for k in range(K):
                        n_idx = points_idx[b, k, j, i].item()
                        if n_idx < 0:
                            continue
                        alpha = alphas[b, k, j, i]
                        grad_alphas[b, k, j, i] = grad_alphas[b, k, j, i] + (
                            grad_output[b, c, j, i]
                            * (features[c, n_idx] * t_alpha_val - t_alphafs_val)
                            / (t_alpha_val * t_alpha_val)
                        )
                        grad_features[c, n_idx] = grad_features[c, n_idx] + (
                            grad_output[b, c, j, i] * alpha / t_alpha_val
                        )

    return grad_features, grad_alphas


# Example functions for blending the top K features per pixel using the outputs
# from rasterization.
# NOTE: All blending function should return a (N, H, W, C) tensor per batch element.
# This can be an image (C=3) or a set of features.


class _CompositeAlphaPoints(torch.autograd.Function):
    """
    Composite features within a z-buffer using alpha compositing. Given a z-buffer
    with corresponding features and weights, these values are accumulated according
    to their weights such that features nearer in depth contribute more to the final
    feature than ones further away.

    Concretely this means:
        weighted_fs[b,c,i,j] = sum_k cum_alpha_k * features[c,pointsidx[b,k,i,j]]
        cum_alpha_k = alphas[b,k,i,j] * prod_l=0..k-1 (1 - alphas[b,l,i,j])

    Args:
        features: Packed Tensor of shape (C, P) giving the features of each point.
        alphas: float32 Tensor of shape (N, points_per_pixel, image_size,
            image_size) giving the weight of each point in the z-buffer.
            Values should be in the interval [0, 1].
        pointsidx: int32 Tensor of shape (N, points_per_pixel, image_size, image_size)
            giving the indices of the nearest points at each pixel, sorted in z-order.
            Concretely pointsidx[n, k, y, x] = p means that features[:, p] is the
            feature of the kth closest point (along the z-direction) to pixel (y, x) in
            batch element n. This is weighted by alphas[n, k, y, x].

    Returns:
        weighted_fs: Tensor of shape (N, C, image_size, image_size)
            giving the accumulated features at each point.
    """

    @staticmethod
    def forward(ctx, features, alphas, points_idx):
        pt_cld = _accum_alphacomposite_python(features, alphas, points_idx)
        ctx.save_for_backward(features.clone(), alphas.clone(), points_idx.clone())
        return pt_cld

    @staticmethod
    def backward(ctx, grad_output):
        grad_features = None
        grad_alphas = None
        grad_points_idx = None
        features, alphas, points_idx = ctx.saved_tensors

        grad_features, grad_alphas = _accum_alphacomposite_backward_python(
            grad_output, features, alphas, points_idx
        )

        return grad_features, grad_alphas, grad_points_idx, None


def alpha_composite(pointsidx, alphas, pt_clds) -> torch.Tensor:
    """
    Composite features within a z-buffer using alpha compositing. Given a z-buffer
    with corresponding features and weights, these values are accumulated according
    to their weights such that features nearer in depth contribute more to the final
    feature than ones further away.

    Concretely this means:
        weighted_fs[b,c,i,j] = sum_k cum_alpha_k * features[c,pointsidx[b,k,i,j]]
        cum_alpha_k = alphas[b,k,i,j] * prod_l=0..k-1 (1 - alphas[b,l,i,j])


    Args:
        pt_clds: Tensor of shape (N, C, P) giving the features of each point (can use
            RGB for example).
        alphas: float32 Tensor of shape (N, points_per_pixel, image_size,
            image_size) giving the weight of each point in the z-buffer.
            Values should be in the interval [0, 1].
        pointsidx: int32 Tensor of shape (N, points_per_pixel, image_size, image_size)
            giving the indices of the nearest points at each pixel, sorted in z-order.
            Concretely pointsidx[n, k, y, x] = p means that features[n, :, p] is the
            feature of the kth closest point (along the z-direction) to pixel (y, x) in
            batch element n. This is weighted by alphas[n, k, y, x].

    Returns:
        Combined features: Tensor of shape (N, C, image_size, image_size)
            giving the accumulated features at each point.
    """
    return _CompositeAlphaPoints.apply(pt_clds, alphas, pointsidx)


class _CompositeNormWeightedSumPoints(torch.autograd.Function):
    """
    Composite features within a z-buffer using normalized weighted sum. Given a z-buffer
    with corresponding features and weights, these values are accumulated
    according to their weights such that depth is ignored; the weights are used to
    perform a weighted sum.

    Concretely this means:
        weighted_fs[b,c,i,j] =
         sum_k alphas[b,k,i,j] * features[c,pointsidx[b,k,i,j]] / sum_k alphas[b,k,i,j]

    Args:
        features: Packed Tensor of shape (C, P) giving the features of each point.
        alphas: float32 Tensor of shape (N, points_per_pixel, image_size,
            image_size) giving the weight of each point in the z-buffer.
            Values should be in the interval [0, 1].
        pointsidx: int32 Tensor of shape (N, points_per_pixel, image_size, image_size)
            giving the indices of the nearest points at each pixel, sorted in z-order.
            Concretely pointsidx[n, k, y, x] = p means that features[:, p] is the
            feature of the kth closest point (along the z-direction) to pixel (y, x) in
            batch element n. This is weighted by alphas[n, k, y, x].

    Returns:
        weighted_fs: Tensor of shape (N, C, image_size, image_size)
            giving the accumulated features at each point.
    """

    @staticmethod
    def forward(ctx, features, alphas, points_idx):
        pt_cld = _accum_weightedsumnorm_python(features, alphas, points_idx)
        ctx.save_for_backward(features.clone(), alphas.clone(), points_idx.clone())
        return pt_cld

    @staticmethod
    def backward(ctx, grad_output):
        grad_features = None
        grad_alphas = None
        grad_points_idx = None
        features, alphas, points_idx = ctx.saved_tensors

        grad_features, grad_alphas = _accum_weightedsumnorm_backward_python(
            grad_output, features, alphas, points_idx
        )

        return grad_features, grad_alphas, grad_points_idx, None


def norm_weighted_sum(pointsidx, alphas, pt_clds) -> torch.Tensor:
    """
    Composite features within a z-buffer using normalized weighted sum. Given a z-buffer
    with corresponding features and weights, these values are accumulated
    according to their weights such that depth is ignored; the weights are used to
    perform a weighted sum.

    Concretely this means:
        weighted_fs[b,c,i,j] =
         sum_k alphas[b,k,i,j] * features[c,pointsidx[b,k,i,j]] / sum_k alphas[b,k,i,j]

    Args:
        pt_clds: Packed feature tensor of shape (C, P) giving the features of each point
            (can use RGB for example).
        alphas: float32 Tensor of shape (N, points_per_pixel, image_size,
            image_size) giving the weight of each point in the z-buffer.
            Values should be in the interval [0, 1].
        pointsidx: int32 Tensor of shape (N, points_per_pixel, image_size, image_size)
            giving the indices of the nearest points at each pixel, sorted in z-order.
            Concretely pointsidx[n, k, y, x] = p means that features[:, p] is the
            feature of the kth closest point (along the z-direction) to pixel (y, x) in
            batch element n. This is weighted by alphas[n, k, y, x].

    Returns:
        Combined features: Tensor of shape (N, C, image_size, image_size)
            giving the accumulated features at each point.
    """
    return _CompositeNormWeightedSumPoints.apply(pt_clds, alphas, pointsidx)


class _CompositeWeightedSumPoints(torch.autograd.Function):
    """
    Composite features within a z-buffer using normalized weighted sum. Given a z-buffer
    with corresponding features and weights, these values are accumulated
    according to their weights such that depth is ignored; the weights are used to
    perform a weighted sum. As opposed to norm weighted sum, the weights are not
    normalized to sum to 1.

    Concretely this means:
        weighted_fs[b,c,i,j] = sum_k alphas[b,k,i,j] * features[c,pointsidx[b,k,i,j]]

    Args:
        features: Packed Tensor of shape (C, P) giving the features of each point.
        alphas: float32 Tensor of shape (N, points_per_pixel, image_size,
            image_size) giving the weight of each point in the z-buffer.
            Values should be in the interval [0, 1].
        pointsidx: int32 Tensor of shape (N, points_per_pixel, image_size, image_size)
            giving the indices of the nearest points at each pixel, sorted in z-order.
            Concretely pointsidx[n, k, y, x] = p means that features[:, p] is the
            feature of the kth closest point (along the z-direction) to pixel (y, x) in
            batch element n. This is weighted by alphas[n, k, y, x].

    Returns:
        weighted_fs: Tensor of shape (N, C, image_size, image_size)
            giving the accumulated features at each point.
    """

    @staticmethod
    def forward(ctx, features, alphas, points_idx):
        pt_cld = _accum_weightedsum_python(features, alphas, points_idx)
        ctx.save_for_backward(features.clone(), alphas.clone(), points_idx.clone())
        return pt_cld

    @staticmethod
    def backward(ctx, grad_output):
        grad_features = None
        grad_alphas = None
        grad_points_idx = None
        features, alphas, points_idx = ctx.saved_tensors

        grad_features, grad_alphas = _accum_weightedsum_backward_python(
            grad_output, features, alphas, points_idx
        )

        return grad_features, grad_alphas, grad_points_idx, None


def weighted_sum(pointsidx, alphas, pt_clds) -> torch.Tensor:
    """
    Composite features within a z-buffer using normalized weighted sum.

    Args:
        pt_clds: Packed Tensor of shape (C, P) giving the features of each point
            (can use RGB for example).
        alphas: float32 Tensor of shape (N, points_per_pixel, image_size,
            image_size) giving the weight of each point in the z-buffer.
            Values should be in the interval [0, 1].
        pointsidx: int32 Tensor of shape (N, points_per_pixel, image_size, image_size)
            giving the indices of the nearest points at each pixel, sorted in z-order.
            Concretely pointsidx[n, k, y, x] = p means that features[:, p] is the
            feature of the kth closest point (along the z-direction) to pixel (y, x) in
            batch element n. This is weighted by alphas[n, k, y, x].

    Returns:
        Combined features: Tensor of shape (N, C, image_size, image_size)
            giving the accumulated features at each point.
    """
    return _CompositeWeightedSumPoints.apply(pt_clds, alphas, pointsidx)
