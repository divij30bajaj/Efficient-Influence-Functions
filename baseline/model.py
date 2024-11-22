import torch
from transformers import GPT2LMHeadModel
from torch import nn


class GPT2MLPWrapper(nn.Module):
    def __init__(self, mlp_block, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input = None
        self.mlp = mlp_block

        def hook_fn(module, grad_input, grad_output):
            self.d_s_l = grad_output[0]

        self.mlp.c_fc.register_full_backward_hook(hook_fn)

    def forward(self, x):
        self.input = x
        return self.mlp(x)

    def get_dims(self):
        return self.mlp.c_fc.weight.shape

    def get_d_s_l(self):
        return self.d_s_l.clone().detach()

    def get_d_w_l(self):
        w_grad = self.mlp.c_fc.weight.grad
        b_grad = self.mlp.c_fc.bias.grad.unsqueeze(0)
        full_grad = torch.cat([w_grad, b_grad], dim=0).T
        return full_grad.clone().detach()

    def get_a_l_minus_1(self):
        return (
            torch.cat([
                    self.input,
                    torch.ones((self.input.shape[0], self.input.shape[1], 1)).to(self.input.device),
                ], dim=-1)
                .clone()
                .detach()
        )


class GPT2Wrapper(nn.Module):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

    def get_mlp_blocks(self):
        num_layers = len(self.model.transformer.h)
        mlp_blocks = []
        for i in range(num_layers):
            mlp = self.model.transformer.h[i].mlp
            mlp_block = GPT2MLPWrapper(mlp)
            self.model.transformer.h[i].mlp = mlp_block
            mlp_blocks.append(mlp_block)
        return mlp_blocks

    def forward(self, x):
        return self.model(x)


def load_model():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return GPT2Wrapper(model)

