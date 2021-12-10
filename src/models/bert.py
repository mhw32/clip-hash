import torch.nn as nn
from DeBERTa import deberta
from transformers.activations import ACT2FN
from transformers.models.deberta.modeling_deberta import StableDropout
from src.models.projection import ProjectionHead


class DebertaV3ForSSL(nn.Module):

    def __init__(self, low_dim=128, model='base'):
        super().__init__()
        self.deberta = deberta.DeBERTa(pre_trained=model)
        self.deberta.apply_state()  #initialize
        self.pooler = ContextPooler(self.deberta.hidden_size)
        self.projection = ProjectionHead(self.deberta.hidden_size, low_dim)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
    ):
        outputs = self.deberta.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_all_encoded_layers=False,
        )
        encoder_layer = outputs[0]
        embedding = self.pooler(encoder_layer)
        projection = self.projection(embedding)
        return embedding, projection


class ContextPooler(nn.Module):

    def __init__(self, pooler_hidden_size, pooler_dropout = 0, pooler_hidden_act = 'gelu'):
        super().__init__()
        self.dense = nn.Linear(pooler_hidden_size, pooler_hidden_size)
        self.dropout = StableDropout(pooler_dropout)
        self.pooler_hidden_act = pooler_hidden_act

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.pooler_hidden_act](pooled_output)
        return pooled_output
