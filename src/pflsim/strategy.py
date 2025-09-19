from __future__ import annotations
from typing import Dict, Iterable, Tuple, Optional
import torch
from torch import nn, Tensor

StateDict = Dict[str, Tensor]

class Strategy:
    def __init__(self) -> None:
        self._global: Optional[StateDict] = None

    def begin_round(self, global_state: Optional[StateDict]) -> None:
        self._global = None if global_state is None else {k: v.clone() for k, v in global_state.items()}

    @property
    def owns_local_update(self) -> bool:
        return False

    #def local_update(self, model: nn.Module, loader, epochs: int, loss_fn, **kwargs) -> None:
    #    raise NotImplementedError("This strategy does not implement local_update.")
        
    def augment_loss(self, base_loss: Tensor, model: nn.Module) -> Tensor:
        return base_loss

    # NOTE: items is (client_no, state_dict, optional_weight)
    def aggregate(self, items: Iterable[Tuple[int, StateDict, Optional[float]]]) -> StateDict:
        raise NotImplementedError

        
##################################################################################
################################################################################## 
class FedAvg(Strategy):
    """Equal-weight or sample-weighted FedAvg if weights are provided."""
    def aggregate(self, items: Iterable[Tuple[int, StateDict, Optional[float]]]) -> StateDict:
        items = list(items)
        if not items:
            raise RuntimeError("FedAvg.aggregate() called with no client states")

        # Prepare sums
        _, base, _ = items[0]
        agg = {k: v.detach().clone().mul_(0.0) for k, v in base.items()}

        total_w = 0.0
        any_weight = any(w is not None for _, _, w in items)

        for _, state, w in items:
            weight = float(w) if (any_weight and w is not None) else 1.0
            total_w += weight
            for k, v in state.items():
                agg[k].add_(v, alpha=weight)

        if total_w == 0.0:
            raise RuntimeError("Total weight is zero in FedAvg.aggregate")

        for k in agg:
            agg[k].div_(total_w)
        return agg

        
##################################################################################
################################################################################## 
class FedProx(Strategy):
    """FedProx: add (mu/2) * ||w - w_global||^2 to the task loss."""
    def __init__(self, mu: float = 0.01):
        super().__init__()
        self.mu = float(mu)

    def augment_loss(self, base_loss: Tensor, model: nn.Module) -> Tensor:
        if self.mu == 0.0 or self._global is None:
            return base_loss
        reg = 0.0
        g = self._global
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name in g:
                # move global tensor to the same device/dtype as p without allocating grads
                gp = g[name].to(p.device, dtype=p.dtype, non_blocking=True)
                reg = reg + (p - gp).pow(2).sum()
        return base_loss + 0.5 * self.mu * reg
        
    def aggregate(self, items: Iterable[Tuple[int, StateDict, Optional[float]]]) -> StateDict:
        items = list(items)
        if not items:
            raise RuntimeError("FedAvg.aggregate() called with no client states")

        # Prepare sums
        _, base, _ = items[0]
        agg = {k: v.detach().clone().mul_(0.0) for k, v in base.items()}

        total_w = 0.0
        any_weight = any(w is not None for _, _, w in items)

        for _, state, w in items:
            weight = float(w) if (any_weight and w is not None) else 1.0
            total_w += weight
            for k, v in state.items():
                agg[k].add_(v, alpha=weight)

        if total_w == 0.0:
            raise RuntimeError("Total weight is zero in FedAvg.aggregate")

        for k in agg:
            agg[k].div_(total_w)
        return agg
        
##################################################################################
##################################################################################        
class FedNova(Strategy):
    """
    FedNova-style server aggregation.

    Expects each client's 'state' to be a StateDict of cumulative gradients
    (already normalized on the client per FedNova), *not* model parameters.
    The aggregator maintains its own copy of the current global parameters.

    Usage:
        nova = FedNova(lr=1e-2, gmf=0.9)
        nova.set_global(initial_model_state_dict)   # call once before first round
        new_global = nova.aggregate(items)          # items: (cid, cum_grad_state, weight?)
    """
    def __init__(self, lr: float, gmf: float = 0.0):
        super().__init__()
        self.lr: float = float(lr)
        self.gmf: float = float(gmf)
        self._global: Optional[StateDict] = None
        self._momentum_buf: Dict[str, Tensor] = {}
        self.firstrun = True

    def set_global(self, state: StateDict) -> None:
        """Initialize/refresh the server's current global parameters."""
        # Deep copy to avoid aliasing
        self._global = {k: v.detach().clone() for k, v in state.items()}
        # Reset momentum whenever we (re)set the global params
        self._momentum_buf.clear()

    def aggregate(self, items: Iterable[Tuple[int, StateDict, Optional[float]]]) -> StateDict:
        items = list(items)
        if not items:
            raise RuntimeError("FedNova.aggregate() called with no client states")
        if self._global is None:
            raise RuntimeError("FedNova: call set_global(...) before aggregate()")

        # Prepare weighted average of cumulative gradients from clients
        _, first_state, _ = items[0]
        agg_grad: StateDict = {k: v.detach().clone().mul_(0.0) for k, v in first_state.items()}

        total_w = 0.0
        any_weight = any(w is not None for _, _, w in items)

        for _, cum_grad_state, w in items:
            weight = float(w) if (any_weight and w is not None) else 1.0
            total_w += weight
            for k, v in cum_grad_state.items():
                agg_grad[k].add_(v, alpha=weight)

        if total_w == 0.0:
            raise RuntimeError("Total weight is zero in FedNova.aggregate")

        for k in agg_grad:
            agg_grad[k].div_(total_w)

        # Apply FedNova server update with optional global momentum
        if self.gmf != 0.0:
            for k, g in agg_grad.items():
                if k not in self._momentum_buf:
                    # Initialize momentum buffer with g / lr (matches your snippet)
                    self._momentum_buf[k] = g.detach().clone().div(self.lr)
                else:
                    self._momentum_buf[k].mul_(self.gmf).add_(g, alpha=1.0 / self.lr)
                # x <- x - lr * m
                self._global[k].add_(self._momentum_buf[k], alpha=-self.lr)
        else:
            # Plain gradient step: x <- x - g   (your comment: cum_grad already carries LR multiples)
            for k, g in agg_grad.items():
                self._global[k].add_(g, alpha=-1.0)

        # Return the updated global parameters (like FedAvg.aggregate)
        return {k: v.detach().clone() for k, v in self._global.items()}
  


##################################################################################
##################################################################################   
class pFedME(Strategy):
    def __init__(self, lr: float = 0.001, eta: float = 0.001, lambd: float = 0.001):
        self.eta = float(eta)
        self.lambd = float(lambd)
        self.lr = float(lr)
    
    def owns_local_update(self) -> bool:
        return True
        
    @torch.no_grad()
    def _copy_params_(self, dst: nn.Module, src: nn.Module):
        for p_dst, p_src in zip(dst.parameters(), src.parameters()):
            p_dst.data.copy_(p_src.data)

    def pfedme_local_round(self,
        w_model: nn.Module,                 # client's global model (will be returned updated; to be sent for aggregation)
        make_theta: callable,               # factory -> nn.Module with same architecture as w_model
        dataloader: Iterable,               # local data D_i
        loss_fn: nn.Module,                 # task loss, e.g., nn.CrossEntropyLoss()
        inner_lr: float,                    # step size for optimizing theta_i
        lam: float,                         # λ (proximal strength)
        eta: float,                         # η (meta step size on w)
        inner_epochs: int = 1,              # how many passes over D_i for theta_i
        device: str = "cpu",
    ) -> Tuple[nn.Module, nn.Module]:
        """
        Runs one client-side pFedMe update.
        Returns (updated_w_model, personalized_theta_model).
        theta_i stays local; only w_model goes back to the server.
        """
        w_model.to(device)
        w_model.train()

        # ---- initialize θ_i from current w_i ----
        theta_model = make_theta().to(device)
        self._copy_params_(theta_model, w_model)

        optimizer = torch.optim.SGD(theta_model.parameters(), lr=inner_lr)

        # ---- optimize θ_i:  minimize f_i(θ_i) + (λ/2)||θ_i - w_i||^2 ----
        for _ in range(inner_epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = theta_model(x)
                task_loss = loss_fn(logits, y)

                # proximal term wrt fixed w_i
                prox = 0.0
                for p_theta, p_w in zip(theta_model.parameters(), w_model.parameters()):
                    prox = prox + torch.sum((p_theta - p_w.detach())**2)
                loss = task_loss + 0.5 * lam * prox

                loss.backward()
                optimizer.step()

        # ---- meta-update w_i:  w_i ← w_i - η λ (w_i - θ̂_i) ----
        with torch.no_grad():
            for p_w, p_theta in zip(w_model.parameters(), theta_model.parameters()):
                p_w.add_(-eta * lam * (p_w - p_theta))

        return w_model, theta_model
    
    
    def local_update(self, model: nn.Module, modgen: callable, dataloader: Iterable, epochs: int, loss_fn, **kwargs) -> None:   
        return self.pfedme_local_round(model, modgen, dataloader, loss_fn, self.lr, self.lambd, self.eta, epochs)
        
        
    def aggregate(self, items: Iterable[Tuple[int, StateDict, Optional[float]]]) -> StateDict:
        items = list(items)
        if not items:
            raise RuntimeError("pFedMe.aggregate() called with no client states")

        # Prepare sums
        _, base, _ = items[0]
        agg = {k: v.detach().clone().mul_(0.0) for k, v in base.items()}

        total_w = 0.0
        any_weight = any(w is not None for _, _, w in items)

        for _, state, w in items:
            weight = float(w) if (any_weight and w is not None) else 1.0
            total_w += weight
            for k, v in state.items():
                agg[k].add_(v, alpha=weight)

        if total_w == 0.0:
            raise RuntimeError("Total weight is zero in pFedMe.aggregate")

        for k in agg:
            agg[k].div_(total_w)
        return agg