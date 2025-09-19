from __future__ import annotations
from typing import Dict, Optional, Iterable, Tuple
import torch
from torch.nn import Module

from .strategy import Strategy, FedAvg, StateDict


class PFLSim:
    def __init__(self, num_clients: int, strategy: Strategy | None = None):
        """Initialize a PFL Simulation instance."""
        if num_clients <= 0:
            raise ValueError("num_clients must be positive")
        self._num_clients: int = num_clients
        self._strategy: Strategy = strategy if strategy is not None else FedAvg()
        self._client_models: Dict[int, StateDict] = {}
        self._client_weights: Dict[int, float] = {}  # e.g., n_samples for weighted FedAvg
        self._global: Optional[StateDict] = None

    # --- lifecycle ---
    def set_strategy(self, strategy: Strategy) -> None:
        """Swap strategy at runtime (preserves current global)."""
        self._strategy = strategy

    def begin_round(self) -> None:
        """Call at the start of each FL round so strategies see the current global weights."""
        self._strategy.begin_round(self._global)
        # Clear last round’s uploads; last-send-wins per client within a round.
        self._client_models.clear()
        self._client_weights.clear()

    # --- client-side API ---
    @torch.no_grad()
    def send(self, client_no: int, model: Module, *, n_samples: Optional[int] = None) -> None:
        """Upload latest weights for client_no. Optionally include sample count for weighting."""
        if not (0 <= client_no < self._num_clients):
            raise ValueError(f"client_no must be in [0, {self._num_clients-1}]")
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self._client_models[client_no] = state
        if n_samples is not None:
            if n_samples < 0:
                raise ValueError("n_samples must be non-negative")
            self._client_weights[client_no] = float(n_samples)

    def client_update(self, client_no: int, model: Module, loader, epochs: int, loss_fn, **kwargs) -> None:
        """Delegate local training to strategy (for non personalized algos)."""
        # Some strategies ignore client_no; it’s here for those that need per-client context.
        if isinstance(self._strategy, FedAvg) or isinstance(self._strategy, FedProx) or isinstance(self._strategy, FedNova):
            self._strategy.local_update(model, loader, epochs, loss_fn, **kwargs)
            
    def client_update_per(self, model: nn.Module, modgen: callable, dataloader: Iterable, epochs: int, loss_fn, **kwargs):
        """Local update done by strategy, for personalized algos"""
        w_model, theta_model = self._strategy.local_update(model, modgen, dataloader, epochs, loss_fn, **kwargs)
        return w_model, theta_model

    # --- server-side API ---
    def aggregate(self) -> None:
        """Aggregate uploaded client models; result goes into _global."""
        if not self._client_models:
            raise RuntimeError("No client models have been sent; call send(client_no, ...) first.")
        # Build iterable of (client_no, state_dict[, weight])
        items: Iterable[Tuple[int, StateDict, Optional[float]]] = (
            (cid, self._client_models[cid], self._client_weights.get(cid))
            for cid in self._client_models.keys()
        )
        self._global = self._strategy.aggregate(items)

    def get_global(self) -> Optional[StateDict]:
        return self._global

    @torch.no_grad()
    def load_global(self, model: Module) -> None:
        if self._global is None:
            raise RuntimeError("No global model available. Call aggregate() first.")
        device = next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else "cpu"
        model.load_state_dict({k: v.to(device) for k, v in self._global.items()}, strict=True)

    def augment_loss(self, base_loss: Tensor, model: Module):
        return self._strategy.augment_loss(base_loss, model)