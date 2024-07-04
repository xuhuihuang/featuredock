"""
This script is adapted from Huggingface BERT Model, Huggingface ViT Model, 
and UW-Madison CS769 NLP assignment2 (https://github.com/JunjieHu/cs769-assignments/tree/main/assignment2)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transform(self, x, linear_layer):
        # the corresponding linear_layer of k, v, q are used to project the hidden_state (x)
        bs, seq_len = x.shape[:2]
        proj = linear_layer(x)
        # next, we need to produce multiple heads for the proj
        # this is done by spliting the hidden state to self.num_attention_heads, each of size self.attention_head_size
        proj = proj.view(bs, seq_len, self.num_attention_heads,
                                         self.attention_head_size)
        # by proper transpose, we have proj of [bs, num_attention_heads, seq_len, attention_head_size]
        proj = proj.transpose(1, 2)
        return proj

    def attention(self, key, query, value):
        # each attention is calculated following eq (1) of https://arxiv.org/pdf/1706.03762.pdf
        # attention scores are calculated by multiply query and key
        # and get back a score matrix S of [bs, num_attention_heads, seq_len, seq_len]
        # S[*, i, j, k] represents the (unnormalized)attention score between the j-th and k-th token, given by i-th attention head
        # before normalizing the scores, use the attention mask to mask out the padding token scores
        # Note again: in the attention_mask non-padding tokens with 0 and padding tokens with a large negative number

        # [bs, num_attention_heads, seq_len, attention_head_size]
        bs, num_attn_heads, seq_length, attn_head_size = key.shape
        # [bs, num_attention_heads, seq_len, seq_len]
        S = query @ key.transpose(-1, -2)
        S = S / math.sqrt(attn_head_size) # [bs, num_attention_heads, seq_len, seq_len]
        # normalize the scores
        S = F.softmax(S, dim=-1)    # [bs, num_attention_heads, seq_len, seq_len]
        S = self.dropout(S)
        # multiply the attention scores to the value and get back V'
        V = S @ value    # [bs, num_attention_heads, seq_len, attention_head_size]
        # [bs, seq_len, num_attention_heads, attention_head_size]
        V = V.permute(0, 2, 1, 3)
        # next, we need to concat multi-heads and recover the original shape [bs, seq_len, num_attention_heads * attention_head_size = hidden_size]
        # [bs, seq_len, all_head_size]
        V = V.reshape((bs, seq_length, self.all_head_size))
        return V, S

    def forward(self, hidden_states):
        """
        hidden_states: [bs, seq_len, hidden_state]
        output: [bs, seq_len, hidden_state]
        """
        # first, we have to generate the key, value, query for each token for multi-head attention w/ transform (more details inside the function)
        # of *_layers are of [bs, num_attention_heads, seq_len, attention_head_size]
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)
        # calculate the multi-head attention
        context, attn = self.attention(
                key_layer, query_layer, value_layer)
        return context, attn

class BertLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, layer_norm_eps, hidden_dropout_prob, \
                             num_attention_heads, attention_probs_dropout_prob):
        super().__init__()
        # self attention
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.self_attention = BertSelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.attention_dense = nn.Linear(self.all_head_size, hidden_size)
        self.attention_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attention_dropout = nn.Dropout(hidden_dropout_prob)
        # feed forward
        self.interm_dense = nn.Linear(hidden_size, intermediate_size)
        self.interm_af = F.gelu
        # layer out
        self.out_dense = nn.Linear(intermediate_size, hidden_size)
        self.out_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.attention = None

    def add_residual(self, input, output, dense_layer, dropout):
        """
        output: the input that requires the Sublayer to transform
        dense_layer, dropput: the Sublayer 
        This function computes ``input + Sublayer(output)``, where sublayer is a dense_layer followed by dropout.
        """
        # todo
        h = dense_layer(output)
        h = dropout(h)
        h = input+h
        return h

    def forward(self, hidden_states):
        """
        hidden_states: either from the embedding layer (first bert layer) or from the previous bert layer
        as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf 
        each block consists of 
        1. a multi-head attention layer (BertSelfAttention)
        2. a add-norm that takes the output of BertSelfAttention and the input of BertSelfAttention
        3. a feed forward layer
        4. a add-norm that takes the output of feed forward layer and the input of feed forward layer
        """
        ## vit
        h = self.attention_layer_norm(hidden_states)
        context, attn = self.self_attention(h)
        self.attention = attn
        h = self.add_residual(hidden_states, context, self.attention_dense, self.attention_dropout)
        # feed forward
        h2 = self.out_layer_norm(h)
        h2 = self.interm_af(self.interm_dense(h2))
        h = self.add_residual(h, h2, self.out_dense, self.out_dropout)
        return h


class BertModel(nn.Module):
    """
    the bert model returns the final embeddings for each token in a sentence
    it consists
    1. embedding (used in self.embed)
    2. a stack of n bert layers (used in self.encode)
    3. a linear transformation layer for [CLS] token (used in self.forward, as given)
    """
    def __init__(self, num_shells, feature_per_shell, hidden_size, intermediate_size, 
                            num_hidden_layers, num_attention_heads, \
                            # number of shells + 1 class token
                            max_position_embeddings=100, \
                            layer_norm_eps=1e-12, hidden_dropout_prob=0.5, attention_probs_dropout_prob=0.1):
        super().__init__()

        # embedding
        self.num_shells, self.feature_per_shell = num_shells, feature_per_shell
        self.norm_layer = nn.BatchNorm2d(self.feature_per_shell)
        self.cls_embedding = nn.Embedding(1, hidden_size)
        self.word2dense = nn.Linear(num_shells, hidden_size)
        self.pos_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.embed_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.embed_dropout = nn.Dropout(hidden_dropout_prob)
        # position_ids (1, len position emb) is a constant, register to buffer
        position_ids = torch.arange(max_position_embeddings).unsqueeze(0)
        self.register_buffer('position_ids', position_ids)

        # bert encoder
        self.bert_layers = nn.ModuleList([BertLayer(hidden_size, intermediate_size, layer_norm_eps, \
                                        hidden_dropout_prob, num_attention_heads, attention_probs_dropout_prob) \
                    for _ in range(num_hidden_layers)])

    def embed(self, feature_batch):
        device = feature_batch.device
        input_shape = feature_batch.size() # batch_size, n_words, input_features
        seq_length = input_shape[1]
        # class token
        cls_embeds = self.cls_embedding(torch.zeros((input_shape[0], 1)).to(device).long()) # batch_size, 1, hidden_size
        inputs_embeds = self.word2dense(feature_batch) # batch_size, n_words, hidden_size
        inputs_embeds = torch.cat((cls_embeds, inputs_embeds), 1)
        # get position index and position embedding from self.pos_embedding
        pos_ids = self.position_ids[:, :seq_length+1].to(device)
        pos_embeds = self.pos_embedding(pos_ids)
        # add embeddings together
        embeds = inputs_embeds + pos_embeds
        # layer norm and dropout
        embeds = self.embed_layer_norm(embeds)
        embeds = self.embed_dropout(embeds)

        return embeds

    def encode(self, hidden_states):
        """
        hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]
        """
        # pass the hidden states through the encoder layers
        for i, layer_module in enumerate(self.bert_layers):
            # feed the encoding from the last bert_layer to the next
            hidden_states = layer_module(hidden_states)

        return hidden_states

    def forward(self, feature_batch):
        """
        input_ids: [batch_size, seq_len], seq_len is the max length of the batch
        attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
        """
        h = feature_batch.view((-1, self.num_shells, self.feature_per_shell)) # (N, 6, 80)
        h = h.permute((0, 2, 1)) # (N, 80, 6)
        embedding_output = self.embed(h) # (N, 81, hidden_size)

        # feed to a transformer (a stack of BertLayers)
        sequence_output = self.encode(embedding_output)

        # get cls token hidden state
        first_tk = sequence_output[:, 0]
        return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}


class BertSentClassifier(nn.Module):
    def __init__(self, n_class, num_shells, feature_per_shell, hidden_size, intermediate_size, 
                            num_hidden_layers=5, num_attention_heads=8, \
                            max_position_embeddings=7, layer_norm_eps=1e-12, hidden_dropout_prob=0.5, \
                            attention_probs_dropout_prob=0.1, option='finetune'):
        super(BertSentClassifier, self).__init__()
        self.bert = BertModel(num_shells, feature_per_shell, hidden_size, intermediate_size, 
                    num_hidden_layers, num_attention_heads, \
                    max_position_embeddings, layer_norm_eps, hidden_dropout_prob, attention_probs_dropout_prob)

        # pretrain mode does not require updating bert paramters.
        for param in self.bert.parameters():
            if option == 'pretrain':
                param.requires_grad = False
            elif option == 'finetune':
                param.requires_grad = True

        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.linear = torch.nn.Linear(hidden_size, n_class)

    def forward(self, feature_batch):
        result = self.bert(feature_batch) # (batch_size, 480)
        h = result['pooler_output'] # (batch_size, hidden_dim)
        h = self.linear(self.dropout(h)) # (batch_size, n_class)
        return h