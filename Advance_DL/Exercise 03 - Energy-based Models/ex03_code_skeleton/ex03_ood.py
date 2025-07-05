import torch
import pytorch_lightning as pl


def grad_norm(model: pl.LightningModule, x: torch.Tensor, y: torch.Tensor = None):
    had_gradients_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    x_k = torch.autograd.Variable(x, requires_grad=True)
    f_prime = torch.autograd.grad(model.cnn(x_k, y=y).sum(), [x_k], retain_graph=True)[0]
    grad = f_prime.view(x.size(0), -1)
    torch.set_grad_enabled(had_gradients_enabled)
    return grad.norm(p=2, dim=1)


def score_fn(model: pl.LightningModule, x: torch.Tensor, y: torch.Tensor = None, score: str = "px"):
    if score == "px":
        return -model.cnn(x, y).detach().cpu()
    elif score == "py":
        return torch.nn.functional.softmax(model.cnn.get_logits(x), dim=1).max(1)[0].detach().cpu()
    elif score == "mass":
        return -grad_norm(x, y).detach().cpu()
    else:
        raise ValueError("Provided score function is not valid.")