from __future__ import division
import math
import torch
import torch.optim as optim
from collections import defaultdict
from math import sqrt


class SharedRMSprop(optim.Optimizer):
    """Implements RMSprop algorithm with shared states."""

    def __init__(
        self,
        params,
        lr=7e-4,
        alpha=0.99,
        eps=0.1,
        weight_decay=0,
        momentum=0,
        centered=False,
    ):
        defaults = defaultdict(
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
        )
        super(SharedRMSprop, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.zeros(1)
                state["grad_avg"] = p.data.new().resize_as_(p.data).zero_()
                state["square_avg"] = p.data.new().resize_as_(p.data).zero_()
                state["momentum_buffer"] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["square_avg"].share_memory_()
                state["step"].share_memory_()
                state["grad_avg"].share_memory_()
                state["momentum_buffer"].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("RMSprop does not support sparse gradients")
                state = self.state[p]

                square_avg = state["square_avg"]
                alpha = group["alpha"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                if group["centered"]:
                    grad_avg = state["grad_avg"]
                    grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                    avg = (
                        square_avg.addcmul(grad_avg, grad_avg, value=-1)
                        .sqrt_()
                        .add_(group["eps"])
                    )
                else:
                    avg = square_avg.sqrt().add_(group["eps"])

                if group["momentum"] > 0:
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).addcdiv_(grad, avg)
                    # Need to avoid version tracking for parameter.
                    p.data.add_(buf, alpha=-group["lr"])
                else:
                    # Need to avoid version tracking for parameter.
                    p.data.addcdiv_(grad, avg, value=-group["lr"])

        return loss


class SharedAdam(optim.Optimizer):
    """Implements Adam algorithm with shared states."""

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-3,
        weight_decay=0,
        amsgrad=False,
    ):
        defaults = defaultdict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        super(SharedAdam, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.zeros(1)
                state["exp_avg"] = p.data.new().resize_as_(p.data).zero_()
                state["exp_avg_sq"] = p.data.new().resize_as_(p.data).zero_()
                state["max_exp_avg_sq"] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"].share_memory_()
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()
                state["max_exp_avg_sq"].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group["amsgrad"]

                state = self.state[p]

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(group["weight_decay"], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                step_t = state["step"].item()
                bias_correction1 = 1 - beta1**step_t
                bias_correction2 = 1 - beta2**step_t

                step_size = group["lr"] / bias_correction1

                bias_correction2_sqrt = sqrt(bias_correction2)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(
                        group["eps"]
                    )
                else:
                    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(
                        group["eps"]
                    )

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
