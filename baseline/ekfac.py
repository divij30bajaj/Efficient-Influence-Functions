import torch as t
import tqdm
from einops import einops

from baseline.utils import autoregressive_loss, compute_lambda_ii, get_grads

'''
Calculates IHVP (vector product between Inverse of Hessian and the training gradients)
Input:
    - model
    - dataset (sampled from the training dataset)
    - mlp_blocks
    - device
Output:
    - IHVP
'''


def get_ekfac_factors_and_pseudo_grads(model, dataset, tokenizer, device, mlp_blocks):
    kfac_input_covs = [
        t.zeros((b.get_dims()[0] + 1, b.get_dims()[0] + 1)).to(device)
        for b in mlp_blocks
    ]
    kfac_grad_covs = [
        t.zeros((b.get_dims()[1], b.get_dims()[1])).to(device) for b in mlp_blocks
    ]
    pseudo_grads = [[] for _ in range(len(mlp_blocks))]
    tot = 0
    for data, target in tqdm.tqdm(dataset):
        model.zero_grad()
        data = data.to(device)
        target = target.to(device)

        eos = tokenizer.encode(tokenizer.eos_token)
        eos_index = t.where(t.tensor(data == eos[0]))[0]
        if eos_index.shape[0] > 0:
            data = data[:eos_index[0].item()]
            target = target[:eos_index[0].item()]

        if len(data.shape) == 1:
            data = data.unsqueeze(0)
            target = target.unsqueeze(0)
        output = model(data)
        loss = autoregressive_loss(output.logits, target)

        for i, block in enumerate(mlp_blocks):
            a_l_minus_1 = block.get_a_l_minus_1()
            input_covs = t.einsum("...ti,...tj->tij", a_l_minus_1, a_l_minus_1)
            kfac_input_covs[i] += input_covs.mean(dim=0)
        loss.backward()

        for i, block in enumerate(mlp_blocks):
            d_s_l = block.get_d_s_l()
            grad_cov = t.einsum("...ti,...tj->tij", d_s_l, d_s_l)
            kfac_grad_covs[i] += grad_cov.mean(dim=0)
            pseudo_grads[i].append(block.get_d_w_l())
        tot += 1

    kfac_input_covs = [A / tot for A in kfac_input_covs]
    kfac_grad_covs = [S / tot for S in kfac_grad_covs]

    search_grads = get_grads(model, tokenizer, dataset, mlp_blocks, device)

    return kfac_input_covs, kfac_grad_covs, pseudo_grads, search_grads


def get_ekfac_ihvp(
    kfac_input_covs, kfac_grad_covs, pseudo_grads, search_grads, damping=0.001
):
    ihvp = []
    print("Computing IHVP...")
    for i in tqdm.tqdm(range(len(search_grads))):
        V = search_grads[i]
        stacked = t.stack(V)
        q_a, _, q_a_t = t.svd(kfac_input_covs[i])
        q_s, _, q_s_t = t.svd(kfac_grad_covs[i])
        lambda_ii = compute_lambda_ii(pseudo_grads[i], q_a, q_s)
        ekfacDiag_damped_inv = 1.0 / (lambda_ii + damping)
        ekfacDiag_damped_inv = ekfacDiag_damped_inv.reshape(
            (stacked.shape[-2], stacked.shape[-1])
        )
        intermediate_result = t.einsum("bij,jk->bik", stacked, q_a_t)
        intermediate_result = t.einsum("ji,bik->bjk", q_s, intermediate_result)
        result = intermediate_result / ekfacDiag_damped_inv.unsqueeze(0)
        ihvp_component = t.einsum("bij,jk->bik", result, q_a)
        ihvp_component = t.einsum("ji,bik->bjk", q_s_t, ihvp_component)
        ihvp_component = einops.rearrange(ihvp_component, "b j k -> b (j k)")
        ihvp.append(ihvp_component)
    return t.cat(ihvp, dim=-1)


# Caller function
def get_ihvp(model, tokenizer, dataset, mlp_blocks, device):
    kfac_input_covs, kfac_grad_covs, pseudo_grads, search_grads = get_ekfac_factors_and_pseudo_grads(
        model, dataset, tokenizer, device, mlp_blocks
    )
    ihvp = get_ekfac_ihvp(kfac_input_covs, kfac_grad_covs, pseudo_grads, search_grads)
    return ihvp