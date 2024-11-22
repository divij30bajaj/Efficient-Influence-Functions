from random import sample
import torch
import tqdm
from einops import einops


def dataset_sample(dataset, n_samples):
    indices = sample(range(len(dataset)), n_samples)
    return [dataset[i] for i in indices]


def autoregressive_loss(output, target, is_query=False):
    output = einops.rearrange(output, "b s v -> (b s) v")
    target = einops.rearrange(target, "b s -> (b s)")
    if not is_query:
        return torch.nn.functional.cross_entropy(output, target)

    loss = torch.nn.functional.cross_entropy(output, target, reduction='none')
    return loss[-1]


def get_grads(model, tokenizer, dataset, mlp_blocks, device, is_query=False):
    grads = [[] for _ in range(len(mlp_blocks))]
    for data, target in tqdm.tqdm(dataset):
        model.zero_grad()
        data = data.to(device)
        target = target.to(device)

        eos = tokenizer.encode(tokenizer.eos_token)
        eos_index = torch.where(torch.tensor(data == eos[0]))[0]
        if eos_index.shape[0] > 0:
            data = data[:eos_index[0].item()]
            target = target[:eos_index[0].item()]

        if len(data.shape) == 1:
            data = data.unsqueeze(0)
            target = target.unsqueeze(0)
        output = model(data)
        loss = autoregressive_loss(output.logits, target, is_query)
        loss.backward()
        for i, block in enumerate(mlp_blocks):
            grads[i].append(block.get_d_w_l())
    return grads


def get_activation_gradients(model, tokenizer, dataset, mlp_blocks, device, layer_index):
    grads = [[] for _ in range(len(mlp_blocks))]
    new_grads = [[] for _ in range(len(mlp_blocks))]
    for data, target in tqdm.tqdm(dataset):
        model.zero_grad()
        data = data.to(device)
        target = target.to(device)

        eos = tokenizer.encode(tokenizer.eos_token)
        eos_index = torch.where(torch.tensor(data == eos[0]))[0]
        if eos_index.shape[0] > 0:
            data = data[:eos_index[0].item()]
            target = target[:eos_index[0].item()]

        if len(data.shape) == 1:
            data = data.unsqueeze(0)
            target = target.unsqueeze(0)

        activation = None

        def hook(module, input, output):
            nonlocal activation
            activation = output

        handle = list(model.modules())[layer_index].register_forward_hook(hook)
        output = model(data)
        handle.remove()

        activation = activation.last_hidden_state
        for i in tqdm.tqdm(range(activation.size(-1))):
            grad_outputs = torch.zeros_like(activation)
            grad_outputs[..., i] = 1.0

            for j, block in enumerate(mlp_blocks):
                w_grad = torch.autograd.grad(activation, block.mlp.c_fc.weight, grad_outputs=grad_outputs,
                                           retain_graph=True, allow_unused=True)[0]
                b_grad = torch.autograd.grad(activation, block.mlp.c_fc.bias, grad_outputs=grad_outputs,
                                           retain_graph=True, allow_unused=True)[0]

                full_grad = torch.cat([w_grad, b_grad.unsqueeze(0)], dim=0).T
                grads[j].append(full_grad.clone().detach().unsqueeze(0))

        for i in range(len(grads)):
            new_grads[i].append(torch.cat(grads[i], dim=0).cpu())

    return new_grads


def compute_lambda_ii(train_grads, q_a, q_s):
    """Compute Lambda_ii values for a block."""
    n_examples = len(train_grads)
    squared_projections_sum = 0.0
    for j in range(n_examples):
        dtheta = train_grads[j]
        result = (q_s @ dtheta @ q_a.T).view(-1)
        squared_projections_sum += result**2
    lambda_ii_avg = squared_projections_sum / n_examples
    return lambda_ii_avg