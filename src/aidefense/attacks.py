"""Untargeted white-box attacks on images in the [0, 1] range."""

import torch
import torch.nn.functional as F


def fgsm(model, x: torch.Tensor, y: torch.Tensor, epsilon: float) -> torch.Tensor:
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    with torch.enable_grad():
        candidate = x.detach().requires_grad_(True)
        loss = F.cross_entropy(model(candidate), y)
        grad = torch.autograd.grad(loss, candidate)[0]
    return (x.detach() + epsilon * grad.sign()).clamp(0, 1).detach()


def pgd(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    steps: int = 10,
    step_size: float | None = None,
    random_start: bool = True,
) -> torch.Tensor:
    if epsilon < 0 or steps < 1:
        raise ValueError("epsilon must be nonnegative and steps must be positive")
    if step_size is None:
        step_size = max(epsilon / 4, 1 / 255)
    original = x.detach()
    adv = original.clone()
    if random_start and epsilon:
        adv = (adv + torch.empty_like(adv).uniform_(-epsilon, epsilon)).clamp(0, 1)
    for _ in range(steps):
        with torch.enable_grad():
            adv = adv.detach().requires_grad_(True)
            loss = F.cross_entropy(model(adv), y)
            grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + step_size * grad.sign()
        adv = torch.maximum(torch.minimum(adv, original + epsilon), original - epsilon)
        adv = adv.clamp(0, 1)
    return adv.detach()


def cw_l2(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int = 100,
    binary_search_steps: int = 3,
    learning_rate: float = 0.01,
    initial_const: float = 0.1,
    confidence: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """C&W-style untargeted L2 optimization with per-image constant search.

    Returns the smallest successful adversarial image found per sample and a
    success mask. Unsuccessful samples retain their original image.
    """
    if steps < 1 or binary_search_steps < 1:
        raise ValueError("optimization step counts must be positive")
    original = x.detach()
    best_adv = original.clone()
    best_l2 = torch.full((x.shape[0],), float("inf"), device=x.device)
    lower = torch.zeros_like(best_l2)
    upper = torch.full_like(best_l2, float("inf"))
    constant = torch.full_like(best_l2, initial_const)
    start = (original.clamp(1e-6, 1 - 1e-6) * 2 - 1).atanh().detach()

    for _ in range(binary_search_steps):
        w = start.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([w], lr=learning_rate)
        found_this_round = torch.zeros_like(best_l2, dtype=torch.bool)
        for _ in range(steps):
            with torch.enable_grad():
                adv = (w.tanh() + 1) / 2
                logits = model(adv)
                true_logits = logits.gather(1, y[:, None]).squeeze(1)
                other_logits = logits.scatter(1, y[:, None], -1e9).max(dim=1).values
                margin = (true_logits - other_logits + confidence).clamp_min(0)
                l2 = (adv - original).flatten(1).square().sum(1)
                loss = (l2 + constant * margin).sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                success = logits.argmax(1).ne(y)
                improved = success & (l2 < best_l2)
                best_l2 = torch.where(improved, l2, best_l2)
                best_adv[improved] = adv.detach()[improved]
                found_this_round |= success
        upper = torch.where(found_this_round, torch.minimum(upper, constant), upper)
        lower = torch.where(found_this_round, lower, torch.maximum(lower, constant))
        constant = torch.where(torch.isfinite(upper), (lower + upper) / 2, constant * 10)
    return best_adv.detach(), torch.isfinite(best_l2)
