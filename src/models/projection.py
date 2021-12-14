import torch
import torch.nn as nn


class ProjectionHead(nn.Module):

    def __init__(
        self,
        embedding_dim,
        projection_dim,
    ):
        super().__init__()
        self.projection = nn.Parameter(
            torch.empty(embedding_dim, projection_dim))
        nn.init.normal_(self.projection)

    def forward(self, x):
        return x @ self.projection
