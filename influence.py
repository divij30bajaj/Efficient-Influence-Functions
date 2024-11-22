import random

import torch

from baseline.ekfac import get_ihvp
from baseline.model import load_model
from baseline.utils import get_grads
from data.dataloader import load_train_data, decode, load_test_data


class InfluenceCalc:
    def __init__(self, model, tokenizer, mlp_blocks, dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.mlp_blocks = mlp_blocks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.model.to(self.device)

    def calculate_influence(self, queries):
        all_top_training_samples = []
        all_top_influences = []

        print("Entering Inverse-Hessian Vector Product (IHVP) computation...")
        ihvp = get_ihvp(self.model, self.tokenizer, self.dataset, self.mlp_blocks, self.device)
        for query in queries:
            grads = get_grads(self.model, self.tokenizer, [query], self.mlp_blocks, self.device, is_query=True)
            query_grad = torch.cat([q[0].reshape(-1) for q in grads])

            top_influences = -1 * torch.einsum("ij,j->i", ihvp, query_grad)
            print(top_influences)
            top_influences, top_samples = torch.topk(top_influences, topk)
            print(top_influences, top_samples)
            all_top_training_samples.append(top_samples)
            all_top_influences.append(top_influences)
        return all_top_influences, all_top_training_samples


def main():
    model = load_model()
    mlp_blocks = model.get_mlp_blocks()
    tokenizer, dataset = load_train_data(sample_size=sample_size, max_length=max_length)
    queries = load_test_data(1)

    print("Model and data loaded...")

    model.eval()
    inf = InfluenceCalc(model, tokenizer, mlp_blocks, dataset)

    print("Calculating influence scores...")
    all_top_influences, all_top_training_samples = inf.calculate_influence(queries)

    for i, (top_influences, top_samples) in enumerate(zip(all_top_influences, all_top_training_samples)):
        print(f"Query: {decode(queries[i][0][0])}{decode(queries[i][1])}")
        print(f"Top {topk} training samples and their influences:")
        for s, i in zip(top_samples, top_influences):
            s = s.item()
            print(
                f"{decode(dataset[s][0][0])}{decode(dataset[s][1])} Influence: {i}"
            )


if __name__ == '__main__':
    ### HYPERPARAMETERS ###
    topk = 1
    max_length = 64
    sample_size = 100
    #######################
    main()
