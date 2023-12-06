# -*- coding: utf-8 -*-
import numpy as np
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import warnings

warnings.filterwarnings("ignore")


class MLP(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout=0, output_layer=False):
        super(MLP, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            # layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))  # 输出下一个节点的eta
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        return self.mlp(x.float())


class Decoder(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 seq_len,
                 tanh_exploration,
                 use_tanh,
                 n_glimpses=1,
                 mask_glimpses=True,
                 mask_logits=True,
                 geo_vocab_size=10):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_glimpses = n_glimpses
        self.mask_glimpses = mask_glimpses
        self.mask_logits = mask_logits
        self.use_tanh = use_tanh
        self.tanh_exploration = tanh_exploration
        self.decode_type = 'greedy'
        self.n_gaussians = 5
        self.node_dim = 8

        self.lstm = nn.LSTMCell(embedding_dim, hidden_dim)
        self.lstm_eta = nn.LSTMCell(10 + self.embedding_dim, hidden_dim)
        self.pointer = Attention(hidden_dim, use_tanh=use_tanh, C=tanh_exploration)
        self.glimpse = Attention(hidden_dim, use_tanh=False)
        self.sm = nn.Softmax(dim=1)
        self.eta_mlp = MLP(hidden_dim * 3 + 5 + 10 * 2, (hidden_dim,), dropout=0.1)
        self.sigma_mlp = MLP(hidden_dim * 3 + 5 + 10 * 2, (hidden_dim,), dropout=0.1)
        self.eta_mlp_mu = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, self.n_gaussians),
        )
        self.eta_mlp_sigma = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, self.n_gaussians),
        )
        self.eta_mlp_pi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_gaussians),
        )

        self.sigma_dist_mu = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, self.n_gaussians),
        )
        self.sigma_dist_sigma = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, self.n_gaussians),
        )
        self.sigma_dist_pi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_gaussians),
        )
        self.gate =  nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )


    def check_mask(self, mask_):
        def mask_modify(mask):
            all_true = mask.all(1)  # 一条路线中，不能再继续走了，全为True，此时mask中该路线返回值为True
            mask_mask = torch.zeros_like(mask)  # mask_mask 初始化时全为false
            mask_mask[:,
            -1] = all_true  # 如果该路线走完了，该路线对应的all_true值为true, mask_mask[: , -1]=true 否则为false，mask_mask[: , -1]=flase
            return mask.masked_fill(mask_mask, False)

        return mask_modify(mask_)

    def update_mask(self, mask, selected):
        def mask_modify(mask):
            all_true = mask.all(1)
            mask_mask = torch.zeros_like(mask)
            mask_mask[:, -1] = all_true
            return mask.masked_fill(mask_mask, False)

        result_mask = mask.clone().scatter_(1, selected.unsqueeze(-1), True)
        return mask_modify(result_mask)

    def recurrence_eta(self, h_in, step, next_node_idxs, V_mask, start_fea, context, last_node, decode_type):
        B = next_node_idxs.shape[0]
        current_node = torch.gather(V_mask.permute(1, 0, 2).contiguous(), 0,
                                    next_node_idxs.view(1, B, 1).expand(1, B, V_mask.shape[2])).squeeze(0)
        current_node_emb = torch.gather(context, 0,
                                        next_node_idxs.view(1, B, 1).expand(1, B, context.shape[2])).squeeze(0)
        current_node_input = torch.cat([current_node, current_node_emb], dim=1)

        hy, cy = self.lstm_eta(current_node_input.float(), h_in)
        g_l, h_out = hy, (hy, cy)
        route_state = torch.cat([start_fea.reshape(B, -1).float(), last_node, current_node_input.float(), g_l], dim=1)
        eta_h = self.eta_mlp(route_state)
        mu = self.eta_mlp_mu(eta_h)
        sigma = torch.exp(self.eta_mlp_sigma(eta_h))
        pi = F.softmax(self.eta_mlp_pi(eta_h), -1)
        m = torch.distributions.Normal(loc=mu, scale=sigma)
        eta = torch.sum(m.mean * pi, dim=1)
        if decode_type == 'rl':
            return h_out, m, pi, route_state, current_node_input
        else:
            return h_out, eta, m, mu, pi, sigma, route_state, current_node_input

    def recurrence(self, x, h_in, prev_mask, step, prev_idxs, context, V_mask, start_fea, decode_type):
        logit_mask = self.update_mask(prev_mask, prev_idxs) if prev_idxs is not None else prev_mask
        if prev_idxs == None:
            logit_mask = self.check_mask(logit_mask)

        logits, h_out = self.calc_logits(x, h_in, logit_mask, context, V_mask, start_fea, step, prev_idxs, decode_type)

        # Calculate log_softmax for better numerical stability
        log_p = torch.log_softmax(logits, dim=1)
        probs = log_p.exp()

        return h_out, log_p, probs, logit_mask

    def calc_logits(self, x, h_in, logit_mask, context, V_mask, start_fea, step, prev_idxs, decode_type):

        hy, cy = self.lstm(x, h_in)
        g_l, h_out = hy, (hy, cy)

        ref, logits = self.glimpse(g_l, context)
        logits[logit_mask] = -np.inf
        g_l = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
        _, logits = self.pointer(g_l, context)

        logits[logit_mask] = -np.inf

        return logits, h_out

    def recurrent_sigma(self, route_state, decode_type):

        B = route_state.shape[0]
        similar_routes_label = torch.zeros([B, 1]).to(route_state.device)
        sigma_h = self.sigma_mlp(route_state)
        sigma_dist_mu = self.sigma_dist_mu(sigma_h)
        sigma_dist_sigma = torch.exp(self.sigma_dist_sigma(sigma_h))
        sigma_dist_pi = F.softmax(self.sigma_dist_pi(sigma_h), -1)
        sigma_dist = torch.distributions.Normal(loc=sigma_dist_mu, scale=sigma_dist_sigma)
        sigma_mu = torch.sum(sigma_dist.mean * sigma_dist_pi, dim=1)
        if decode_type == 'mle':
            return sigma_mu, similar_routes_label

        elif decode_type == 'rl':
            component_sample_prob = sigma_dist.sample()
            sigma_log_prob = sigma_dist.log_prob(
                component_sample_prob)  # sampled value may be negative, change to log distribution
            sigma_log_prob = torch.sum(sigma_log_prob * sigma_dist_pi, dim=1)
            sigma_sample = torch.sum(component_sample_prob * sigma_dist_pi, dim=1)
            return sigma_sample, sigma_log_prob

    def gated_fusion(self, eta, sigma_argmax, current_eta):
        f_gate = self.gate(torch.cat([eta.unsqueeze(1), sigma_argmax.unsqueeze(1), current_eta.unsqueeze(1)], dim=1)).squeeze()
        eta = eta + f_gate * sigma_argmax
        sigma_argmax = eta - eta * sigma_argmax
        return eta, sigma_argmax

    def forward(self, start_fea, decoder_input, embedded_inputs, hidden, context, V_reach_mask, V_mask, pred_len,
                first_node_input, decode_type):

        batch_size = context.size(1)
        outputs = []
        selections = []
        eta_selections = []
        sigma_selections = []
        eta_resort = torch.zeros([decoder_input.shape[0], V_mask.shape[1]]).to(decoder_input.device)
        sigma_resort = torch.zeros([decoder_input.shape[0], V_mask.shape[1]]).to(decoder_input.device)
        sigma_label_resort = torch.zeros([decoder_input.shape[0], V_mask.shape[1]]).to(decoder_input.device)

        steps = range(embedded_inputs.size(0))
        idxs = None

        mask = Variable(V_reach_mask, requires_grad=False)
        hidden_eta = (hidden[0].clone(), hidden[1].clone())

        hy, cy = self.lstm_eta(first_node_input.float(), hidden_eta)
        route_state, hidden_eta = hy, (hy, cy)
        save_mu = []
        save_sigma = []
        save_pi = []
        last_node = first_node_input.float()
        current_eta = 0
        current_eta_for_input = 0

        if decode_type == 'mle':
            for i in steps:
                hidden, log_p, probs, mask = self.recurrence(decoder_input, hidden, mask, i, idxs, context, V_mask,
                                                             start_fea, decode_type)

                idxs, log_prob = self.decode(
                    probs,
                    mask,
                    decode_type
                )
                hidden_eta, eta_duration_pred, m, eta_mu, eta_pi, eta_sigma, route_state, last_node = self.recurrence_eta(
                    hidden_eta, i, idxs, V_mask, start_fea, context, last_node,
                    decode_type)
                sigma_argmax, similar_routes_label = self.recurrent_sigma(route_state, decode_type)

                if i == 0:
                    current_eta_for_input = current_eta_for_input  +  eta_duration_pred.clone()
                else:
                    current_eta_for_input = current_eta.clone() + eta_duration_pred.clone()
                eta_duration_pred, sigma_argmax = self.gated_fusion(eta_duration_pred, sigma_argmax, current_eta_for_input)
                current_eta = current_eta + eta_duration_pred
                save_mu.append(eta_mu)
                save_pi.append(eta_pi)
                save_sigma.append(eta_sigma)
                eta_resort[[x for x in range(0, idxs.shape[0])], idxs.tolist()] = current_eta
                sigma_resort[[x for x in range(0, idxs.shape[0])], idxs.tolist()] = sigma_argmax
                sigma_label_resort[[x for x in range(0, idxs.shape[0])], idxs.tolist()] = similar_routes_label.squeeze(
                    1).float()

                idxs = idxs.detach()

                decoder_input = torch.gather(
                    embedded_inputs,
                    0,
                    idxs.contiguous().view(1, batch_size, 1).expand(1, batch_size, *embedded_inputs.size()[2:])
                ).squeeze(0)

                outputs.append(log_p)
                selections.append(idxs)
                eta_selections.append(current_eta)
                sigma_selections.append(sigma_argmax)

            return (torch.stack(outputs, 1), torch.stack(selections, 1), eta_resort, torch.stack(eta_selections, 1),
                    sigma_resort,
                    torch.stack(sigma_selections, 1), sigma_label_resort,
                    torch.stack(save_mu, 1), torch.stack(save_sigma, 1), torch.stack(save_pi, 1))

        else:
            order_log_probs = []
            eta_log_probs = []
            sigma_log_probs = []
            rt_log_probs = []
            for i in steps:
                hidden, log_p, probs, mask = self.recurrence(decoder_input, hidden, mask, i, idxs, context, V_mask,
                                                             start_fea, decode_type)

                idxs, order_log_prob = self.decode(
                    probs,
                    mask,
                    decode_type
                )

                hidden_eta, m, pi, route_state, last_node = self.recurrence_eta(hidden_eta, i, idxs, V_mask, start_fea,
                                                                                context, last_node,
                                                                                decode_type)
                sigma_sample, sigma_log_prob = self.recurrent_sigma(route_state, decode_type)

                eta_sample_duration, eta_log_prob = self.decode_eta(
                    m,
                    pi
                )
                if i == 0:
                    current_eta_for_input = current_eta_for_input + eta_sample_duration
                else:
                    current_eta_for_input = current_eta.clone() + eta_sample_duration
                eta_sample_duration, sigma_sample = self.gated_fusion(eta_sample_duration, sigma_sample, current_eta_for_input.clone())
                current_eta = current_eta + eta_sample_duration
                order_log_probs.append(order_log_prob)
                eta_log_probs.append(eta_log_prob)
                sigma_log_probs.append(sigma_log_prob)
                rt_log_probs.append(order_log_prob)
                rt_log_probs.append(eta_log_prob)
                rt_log_probs.append(sigma_log_prob)

                idxs = idxs.detach()

                decoder_input = torch.gather(
                    embedded_inputs,
                    0,
                    idxs.contiguous().view(1, batch_size, 1).expand(1, batch_size, *embedded_inputs.size()[2:])
                ).squeeze(0)

                selections.append(idxs)
                eta_selections.append(current_eta)  # 起点到终点的eta
                sigma_selections.append(sigma_sample)  # 起点到终点的sigma

            return (torch.stack(selections, 1), torch.stack(eta_selections, 1), torch.stack(sigma_selections, 1),
                    torch.stack(order_log_probs, dim=1), torch.stack(eta_log_probs, dim=1),
                    torch.stack(sigma_log_probs, dim=1), torch.stack(rt_log_probs, dim=1))

    def decode(self, probs, mask, decode_type):
        log_prob = torch.tensor([0])
        if decode_type == 'mle':  # greedy
            _, idxs = probs.max(1)
            if mask.gather(1, idxs.unsqueeze(-1)).data.any():
                assert False, "wrong decoding"

        elif decode_type == 'rl':
            multi_dist = Categorical(probs)
            idxs = multi_dist.sample()
            log_prob = multi_dist.log_prob(idxs)

            # Check if sampling went OK, can go wrong due to bug on GPU
            while mask.gather(1, idxs.unsqueeze(-1)).data.any():
                print(' [!] resampling due to race condition')
                idxs = probs.multinomial().squeeze(1)
        else:
            assert False, "Unknown decode type"
        return idxs, log_prob

    def decode_eta(self, m, pi):
        component_sample_prob = m.sample()
        eta_log_prob = m.log_prob(component_sample_prob)  # sampled value may be negative, change to log distribution
        eta_log_prob = torch.sum(eta_log_prob * pi, dim=1)
        eta_sample = torch.sum(component_sample_prob * pi, dim=1)

        return eta_sample, eta_log_prob


class Attention(nn.Module):
    """A generic attention module for a decoder in seq2seq"""

    def __init__(self, dim, use_tanh=False, C=10):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(dim, dim)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1)
        self.C = C  # tanh exploration
        self.tanh = nn.Tanh()

        self.v = nn.Parameter(torch.FloatTensor(dim))
        self.v.data.uniform_(-(1. / math.sqrt(dim)), 1. / math.sqrt(dim))

    def forward(self, query, ref):
        """
        Args:
            query: is the hidden state of the decoder at the current
                time step. batch x dim
            ref: the set of hidden states from the encoder.
                sourceL x batch x hidden_dim
        """
        # ref is now [batch_size x hidden_dim x sourceL]
        ref = ref.permute(1, 2, 0).contiguous()
        q = self.project_query(query).unsqueeze(2)  # batch x dim x 1
        e = self.project_ref(ref)  # batch_size x hidden_dim x sourceL
        # expand the query by sourceL
        # batch x dim x sourceL
        expanded_q = q.repeat(1, 1, e.size(2))
        # batch x 1 x hidden_dim
        v_view = self.v.unsqueeze(0).expand(
            expanded_q.size(0), len(self.v)).unsqueeze(1)
        # [batch_size x 1 x hidden_dim] * [batch_size x hidden_dim x sourceL]
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)
        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u
        return e, logits


class SkipConnection(nn.Module):

    def __init__(self, module, is_mask=False):
        super(SkipConnection, self).__init__()
        self.module = module
        self.is_mask = is_mask

    def forward(self, input):
        if self.is_mask:
            old_input, h, mask = input
            new_input, h, mask = self.module(input)
            new_input = old_input + new_input
            return (new_input, h, mask)
        else:  # 应对 线形层作为 module的时候
            old_input, h, mask = input
            new_input = self.module(old_input)
            new_input = old_input + new_input
            return (new_input, h, mask)


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        q, h, mask = input
        mask = mask.bool()
        old_mask = mask.clone()
        if h is None:
            h = q  # compute self-attention

        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility).bool()
            compatibility.masked_fill_(mask, value=-np.inf)

        attn = torch.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)
        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        # return out
        return (out, None, old_mask)


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        input, h, mask = input
        mask = mask.bool()
        if isinstance(self.normalizer, nn.BatchNorm1d):
            input = self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
            return (input, h, mask)

        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            input = self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
            return (input, h, mask)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return (input, h, mask)


class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                ),
                is_mask=True,
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim),
                is_mask=False,
            ),
            Normalization(embed_dim, normalization)
        )


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(TransformerEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

    def forward(self, x, mask):
        mask = mask.bool()
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x
        h, _, mask = self.layers((h, None, mask))
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )




class PolicyNetwork(nn.Module):
    def __init__(self, args={}):
        super(PolicyNetwork, self).__init__()

        # network parameters
        self.hidden_size = args['hidden_size']
        self.sort_x_size = args['sort_x_size']
        self.args = args
        self.t_size = 2

        self.sort_encoder = TransformerEncoder(node_dim=self.hidden_size, embed_dim=self.hidden_size,
                                               n_heads=8, n_layers=2,
                                               normalization='batch')

        self.sort_x_embedding = nn.Linear(in_features=self.sort_x_size + self.t_size, out_features=self.hidden_size, bias=False)
        self.first_node_embedding = nn.Linear(in_features=self.sort_x_size + self.t_size, out_features=self.hidden_size,
                                              bias=False)
        self.mlp_eta = MLP(args['max_task_num'] * 10 + 5, (self.hidden_size,), dropout=0)
        self.start_emb = nn.Sequential(
            nn.Linear(5, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        tanh_clipping = 10
        mask_inner = True
        mask_logits = True
        self.decoder = Decoder(
            self.hidden_size,
            self.hidden_size,
            seq_len=args['max_task_num'],
            tanh_exploration=tanh_clipping,
            use_tanh=tanh_clipping > 0,
            n_glimpses=1,
            mask_glimpses=mask_inner,
            mask_logits=mask_logits,
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def enc_sort_emb(self, sort_emb, batch_size, max_seq_len, mask_index):
        """
        Encode the sort emb and prepare the input for Decoder
        """
        mask_indices = torch.nonzero(mask_index + 0)  # 指定行，列元素为0
        attn_mask = (mask_index + 0).repeat_interleave(max_seq_len).reshape(batch_size, max_seq_len,
                                                                            max_seq_len).permute(0, 2, 1).contiguous()
        attn_mask = attn_mask.to(sort_emb.device)
        attn_mask[mask_indices[:, 0], mask_indices[:, 1], :] = 1
        sort_encoder_outputs, emb = self.sort_encoder(sort_emb, attn_mask)
        dec_init_state = (emb, emb)
        inputs = sort_encoder_outputs.permute(1, 0, 2).contiguous()
        enc_h = sort_encoder_outputs.permute(1, 0, 2).contiguous()  # (seq_len, batch_size, hidden)
        return inputs, dec_init_state, enc_h

    def forward(self, V, V_reach_mask, start_fea, cou_fea, pred_len, first_node, start_geo, V_geo, decode_type):

        B = V_reach_mask.size(0)
        N = V_reach_mask.size(1)
        mask_index = V.reshape(-1, N, V.shape[-1])[:, :, 0] == 0
        V_dis = V.reshape(B, N, -1)[:, :, [-3, -4]]
        cou_speed = cou_fea.reshape(B, -1)[:, -1]
        V_avg_t = V_dis / cou_speed.unsqueeze(1).unsqueeze(1).repeat(1, N, 1)
        mask = (mask_index + 0).reshape(B, N, 1)
        V = torch.cat([V.reshape(B, N, -1), V_avg_t.reshape(B, N, -1)], dim=2)
        first_node = torch.cat([first_node.reshape(B, -1), torch.zeros([B, 2]).to(V.device)], dim=1)

        sort_x_emb = self.sort_x_embedding(V.float())
        first_node_emb = self.first_node_embedding(first_node.float())
        first_node_input = torch.cat([first_node, first_node_emb], dim=1)
        inputs, dec_init_state, enc_h = self.enc_sort_emb(sort_x_emb, B, N, mask_index)
        decoder_input = self.start_emb(start_fea.float())
        if decode_type == 'mle':
            (pointer_log_scores, order_arg, eta_resort, eta_select, sigma_resort, sigma_select, sigma_label_resort, m,
             sigma, pi) = self.decoder(start_fea, decoder_input, inputs, dec_init_state, enc_h,
                                       V_reach_mask.reshape(-1, N), V * mask, pred_len, first_node_input, decode_type)
            pointer_scores = pointer_log_scores.exp()
            return pointer_scores, order_arg, eta_resort, sigma_resort, eta_select, sigma_select, sigma_label_resort
        elif decode_type == 'rl':
            (order_samples, eta_samples, sigma_samples, order_log_probs, eta_log_probs, sigma_log_probs,
             rt_log_probs) = self.decoder(start_fea, decoder_input, inputs, dec_init_state, enc_h,
                                          V_reach_mask.reshape(-1, N), V * mask, pred_len, first_node_input,decode_type)
            return order_samples, eta_samples, sigma_samples, order_log_probs, eta_log_probs, sigma_log_probs, rt_log_probs

    def model_file_name(self):
        file_name = '+'.join([f'{k}-{self.args[k]}' for k in ['hidden_size']])
        file_name = f'{file_name}-{time.time()}.csv'
        return file_name


# -------------------------------------------------------------------------------------------------------------------------#



