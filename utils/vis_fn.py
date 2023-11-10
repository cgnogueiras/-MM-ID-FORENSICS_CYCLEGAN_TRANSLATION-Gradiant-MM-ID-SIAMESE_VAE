import os
import torch
import torchvision.transforms as TF
import torchvision.utils as vutils


def save_reconstructed_images(
    model, x, y, x2, a, color=1, results_path="results", epoch="", n=8
):
    model.eval()
    with torch.no_grad():
        x_hat, _, _, _, x_hat2, _, _, _ = model(x.to(model.device), x2.to(model.device))
        x_hat, x_hat2 = x_hat.detach().cpu(), x_hat2.detach().cpu()
        x_hat = x_hat.reshape(-1, color, 200, 500)
        x_hat2 = x_hat2.reshape(-1, color, 200, 500)

    grid_tensors = []
    for chunk in [x, y, x_hat, x2, a, x_hat2]:
        grid_tensor = vutils.make_grid(chunk, nrow=n)
        grid_tensors.append(grid_tensor)
    grid = torch.cat(grid_tensors, dim=1)

    pil_image = TF.ToPILImage()(grid)
    pil_image.save(os.path.join(results_path, f"reconstructed_{epoch}.png"))

    model.train()
