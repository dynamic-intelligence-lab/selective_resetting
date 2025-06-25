# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_parallel_scan as tps


class _UpdateOnRightWithSelectiveResetOnLeft(nn.Module):
    """
    Sample implementation of the selective-resetting method for parallel prefix
    scans proposed in "Generalized Orders of Magnitude for Scalable, Parallel,
    High-Dynamic-Range Computation" (Heinsen and Kozachkov, 2025).

    Given:
    
        matrices A1 and B1, on the left, and
        matrices A2 and B2, on the right,

    this module applies a specified select function to A1, to determine if its
    value should be reset by a specified reset function, and saved on B1, but
    only if B1 is still all-zeros, and then computes updated states as:
    
        matrices [A1 @ A2] and [B1 @ A2 + B2], on the right,

    broadcasting over preceding dimensions, if any.

    Please see Appendix C of the paper for an informal explanation of the
    intuition behind selective resetting, with step-by-step examples.

    Args:
        d: size of square matrices, each d x d.
        select_func: function for selecting matrix states that will be reset.
            The function must accept a float tensor of shape [..., d, d] as
            input, and return a bool tensor of shape [..., 1, 1] as output.
        reset_func: function that resets matrix states. The function must
            accept a float tensor of shape [..., d, d] as input, and return
            a float tensor of shape [..., d, d], with reset states, as output.

    Inputs:
        A1_atop_B1: float tensor of shape [..., 1, d + d, d], with each A1
            (d x d) stacked atop its corresponding B1 (d x d), on the left.
        A2_atop_B2: float tensor of shape [..., n, d + d, d], with each A2
            (d x d) stacked atop its corresponding B2 (d x d), on the right.

    Output:
        updated_A2_atop_B2: float tensor of shape [..., n, d + d, d], with each
        A1 @ A2 (d x d) stacked atop its corresponding B1 @ A2 + B2 (d x d).
    """
    def __init__(self, d, select_func, reset_func):
        super().__init__()
        self.d = d
        self.select_func = select_func
        self.reset_func = reset_func
        self.register_buffer('zeros_atop_I', torch.cat([
            torch.zeros(d, d),
            torch.eye(d)
        ], dim=0))

    def forward(self, A1_atop_B1, A2_atop_B2):
        d = self.d                                                        # for convenience

        # Get states and biases on the left:
        A1 = A1_atop_B1[..., :, :d, :]                                    # [..., 1, d, d]
        B1 = A1_atop_B1[..., :, d:, :]                                    # [..., 1, d, d]

        # Select which states will be reset, if any:
        is_selected = self.select_func(A1)                                # [..., 1, 1, 1], bool idx
        bias_is_unmodified = (B1 == 0).all(dim=(-2, -1), keepdim=True)    # [..., 1, 1, 1], bool idx
        should_reset_on_left = is_selected & bias_is_unmodified           # [..., 1, 1, 1], bool idx

        if torch.any(should_reset_on_left):
            # Replace resettable states on the left with new values:
            idx = should_reset_on_left.expand_as(A1)                      # [...., 1, d, d]
            subset_szs = [*A1.shape[:-4], -1, 1, d, d]                    # sizes of resettable subset
            subset_of_A1 = A1[idx].view(*subset_szs)                      # [..., <n in subset>, 1, d, d]
            new_values = self.reset_func(subset_of_A1)                    # [..., <n in subset>, 1, d, d]
            zeros_atop_new = F.pad(new_values, (0,0, d,0), value=0)       # [..., <n in subset>, 1, d + d, d]
            idx = should_reset_on_left.expand_as(A1_atop_B1)              # [...., 1, d + d, d]
            A1_atop_B1[idx] = zeros_atop_new.view(A1_atop_B1[idx].shape)  # [..., <n in subset>, 1, d + d, d]

        # Compute A1 @ A2, stacked atop B1 @ A2 + B2:
        preceding_szs = A1_atop_B1.shape[:-2]                             # sizes for broadcasting
        zeros_atop_I = self.zeros_atop_I.expand(*preceding_szs, -1, -1)   # [..., 1, d + d, n]

        A1_atop_B1_with_zeros_atop_I_on_right = torch.cat([
            A1_atop_B1,                                                   # [..., 1, d + d, d]
            zeros_atop_I,                                                 # [..., 1, d + d, d]
        ], dim=-1)                                                        # [..., 1, d + d, d + d]

        updated_A2_atop_B2 = torch.matmul(
            A1_atop_B1_with_zeros_atop_I_on_right,                        # [..., 1, d + d, d + d]
            A2_atop_B2,                                                   # [..., n, d + d, d]
        )                                                                 # [..., n, d + d, d]
        return updated_A2_atop_B2


class ParallelizedLeftToRightRecurrenceWithSelectiveResetting(nn.Module):
    """
    Computes a left-to-right non-diagonal linear recurrence with selective
    resets, in parallel, via a prefix scan, applying the selective-resetting
    method proposed in "Generalized Orders of Magnitude for Scalable, Parallel,
    High-Dynamic-Range Computation" (Heinsen and Kozachkov, 2025).

    Please see Appendix C of the paper for an informal explanation of the
    intuition behind selective resetting, with step-by-step examples.

    Args:
        d: size of square matrices, each d x d.
        select_func: function for selecting matrix states that will be reset.
            The function must accept a float tensor of shape [..., d, d] as
            input, and return a bool tensor of shape [..., 1, 1] as output.
        reset_func: function that resets matrix states. The function must
            accept a float tensor of shape [..., d, d] as input, and return
            a float tensor of shape [..., d, d], with reset states, as output.

    Inputs:
        A: float tensor of shape [..., n, d, d] with n left-to-right transition
            matrices, optionally with initial state in the first position.

    Output:
        X: float tensor of shape [..., n, d, d] with n compound state matrices,
            some of which may have been selectively reset.
    """
    def __init__(self, d, select_func, reset_func):
        super().__init__()
        self.d = d
        self.sr_transform = _UpdateOnRightWithSelectiveResetOnLeft(
            d=d, select_func=select_func, reset_func=reset_func)

    def forward(self, A):
        # Add a bias initialized with zeros below each transition matrix:
        A_atop_B = F.pad(A, (0, 0,  0, self.d), value=0)  # shape is [..., n, d + d, d]

        # Apply parallel prefix scan with selective-resetting transform:
        cumul_A_atop_B = tps.prefix_scan(A_atop_B, self.sr_transform, dim=-3)

        # Add cumulative transition matrices and biases (possibly reset):
        X = cumul_A_atop_B[..., :d, :] + cumul_A_atop_B[..., d:, :]

        return X
