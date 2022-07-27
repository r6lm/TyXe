from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_model(params, return_instance=True):
    """Acts as lookup table of all the models that are implemented in 
    the module."""

    if params["alias"] == "MF":
        return (MF(
            params["n_users"], params["n_items"],
            params["n_latents"])
            if return_instance else MF)
    elif params["alias"] == "SP":
        return (SingleParam() if return_instance else SingleParam)
    elif params["alias"] == "UP":
        return (UserParam(params["n_users"]) if return_instance else UserParam)
    elif params["alias"] == "IP":
        return (ItemParam(params["n_items"]) if return_instance else ItemParam)


class MF(nn.Module):
    def __init__(self, n_users, n_items, n_latent):

        # initialize variables
        super(MF, self).__init__()
        self.user_bias_emb = nn.Embedding(n_users, 1)
        self.item_bias_emb = nn.Embedding(n_items, 1)
        self.user_latent_emb = nn.Embedding(n_users, n_latent)
        self.item_latent_emb = nn.Embedding(n_items, n_latent)

        # set bias initialization to zero
        nn.init.zeros_(self.user_bias_emb.weight)
        nn.init.zeros_(self.item_bias_emb.weight)

        # register instance params
        self.n_users = n_users
        self.n_items = n_items
        self.n_latent = n_latent

    def forward(self, x):
        user_ids = x[:, 0].squeeze().long()
        item_ids = x[:, 1].squeeze().long()
        user_bias = self.user_bias_emb(user_ids).squeeze()
        item_bias = self.item_bias_emb(item_ids).squeeze()
        user_latent = self.user_latent_emb(user_ids)
        item_latent = self.item_latent_emb(item_ids)
        interaction = torch.mul(user_latent, item_latent).sum(-1)
        # replace for nn.Sigmoid
        return torch.sigmoid(user_bias + item_bias + interaction)

    def reset_parameters(self):
        # by default initialized with standard Gaussian
        self.user_latent_emb.reset_parameters()
        self.item_latent_emb.reset_parameters()

        # set bias initialization to zero
        nn.init.zeros_(self.user_bias_emb.weight)
        nn.init.zeros_(self.item_bias_emb.weight)


class SingleParam(nn.Module):
    def __init__(self):
        super(SingleParam, self).__init__()
        self.param = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, user_ids, item_ids):
        return torch.sigmoid(self.param).expand(user_ids.size())

    def reset_parameters(self):
        self.param = nn.Parameter(torch.zeros(1))


class UserParam(nn.Module):
    def __init__(self, n_users):
        super(UserParam, self).__init__()
        self.user_emb = nn.Embedding(n_users, 1)

        # initialize weights at zero
        nn.init.zeros_(self.user_emb.weight)

    def forward(self, user_ids, item_ids):
        return torch.sigmoid(self.user_emb(user_ids).squeeze())

    def reset_parameters(self):
        nn.init.zeros_(self.user_emb.weight)


class ItemParam(nn.Module):
    def __init__(self, n_items):
        super(ItemParam, self).__init__()
        self.item_emb = nn.Embedding(n_items, 1)

        # initialize weights at zero
        nn.init.zeros_(self.item_emb.weight)

    def forward(self, user_ids, item_ids):
        return torch.sigmoid(self.item_emb(item_ids).squeeze())

    def reset_parameters(self):
        nn.init.zeros_(self.item_emb.weight)
