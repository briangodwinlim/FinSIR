import torch
from torch import nn
from dgl import function as fn
import torch.nn.functional as F
from dgl.utils import expand_as_pair


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0, activation=nn.ReLU()):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.drop = nn.Dropout(dropout)
        
        self.linears = nn.ModuleList()
        for i in range(num_layers):
            _input_dim = hidden_dim if i > 0 else input_dim
            _output_dim = hidden_dim if i < num_layers - 1 else output_dim
            self.linears.append(nn.Linear(_input_dim, _output_dim))
    
    def forward(self, feats):
        for i in range(self.num_layers):
            feats = self.linears[i](feats)
            feats = self.activation(feats)
            if i < self.num_layers - 1:
                feats = self.drop(feats)

        return feats


class SIREConv(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim, activation, dropout=0, bias=True, agg_type='sum'):
        super(SIREConv, self).__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.linear_query = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.linear_key = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.linear_edge = nn.Linear(edge_dim, hidden_dim, bias=bias)
        self.linear_relation = nn.Linear(hidden_dim, (1 + int(agg_type == 'gated')) * output_dim, bias=bias)

        self._agg_type = agg_type
        self._agg_func = fn.sum if agg_type in ['sym', 'gated'] else getattr(fn, agg_type)
    
    def message_func(self, edges):
        if self._agg_type in ['sum', 'mean', 'sym']:
            return {'m': edges.src['norm'] * edges.dst['norm'] * self.activation(edges.dst['eq'] + edges.src['ek'] + edges.data['e'])}
        elif self._agg_type in ['max', 'min']:
            return {'m': self.linear_relation(self.activation(edges.dst['eq'] + edges.src['ek'] + edges.data['e']))}
        elif self._agg_type == 'gated':
            return {'m': F.glu(self.linear_relation(self.activation(edges.dst['eq'] + edges.src['ek'] + edges.data['e'])))}
    
    def forward(self, graph, nfeat, efeat):
        with graph.local_scope():
            degs = graph.in_degrees().float().clamp(min=1).to(graph.device)
            norm = torch.pow(degs, -0.5) if self._agg_type == 'sym' else torch.ones(graph.num_nodes(), device=graph.device)
            norm = norm.reshape((graph.num_nodes(),) + (1,) * (nfeat.dim() - 1))
            graph.ndata['norm'] = norm
 
            nfeat_key, nfeat_query = expand_as_pair(nfeat, graph)
            graph.ndata['ek'] = self.dropout(self.linear_key(nfeat_key))
            graph.ndata['eq'] = self.dropout(self.linear_query(nfeat_query))
            graph.edata['e'] = self.dropout(self.linear_edge(efeat))

            graph.update_all(self.message_func, self._agg_func('m', 'ft'))
            rst = graph.ndata.pop('ft')
            rst = self.linear_relation(rst) if self._agg_type in ['sum', 'mean', 'sym'] else rst
            
            return rst


class SIREConvBase(nn.Module):
    def __init__(self, message_func, agg_type='sum'):
        super(SIREConvBase, self).__init__()
        self._agg_type = agg_type
        self._message_func = message_func
        self._agg_func = fn.sum if agg_type in ['sym', 'gated'] else getattr(fn, agg_type)
    
    def message_func(self, edges):
        message = torch.cat([edges.dst['eq'], edges.src['ek']], dim=-1)     # Concatenate node features
        message = message + edges.data['e'].unsqueeze(dim=-2)               # Add edge embedding
        message = self._message_func(message)                               # Recurrent message function
        message = F.glu(message) if self._agg_type == 'gated' else message  # Gating before aggregation
        return {'m': edges.src['norm'] * edges.dst['norm'] * message}
        
    def forward(self, graph, nfeat, efeat):
        with graph.local_scope():
            degs = graph.in_degrees().float().clamp(min=1).to(graph.device)
            norm = torch.pow(degs, -0.5) if self._agg_type == 'sym' else torch.ones(graph.num_nodes(), device=graph.device)
            norm = norm.reshape((graph.num_nodes(), 1, 1))
            graph.ndata['norm'] = norm
 
            nfeat_key, nfeat_query = expand_as_pair(nfeat, graph)
            graph.ndata['ek'] = nfeat_key
            graph.ndata['eq'] = nfeat_query
            graph.edata['e'] = efeat

            graph.update_all(self.message_func, self._agg_func('m', 'ft'))
            rst = graph.ndata.pop('ft')
            
            return rst


class MessageFunction(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=1, dropout=0, **kwargs):
        super(MessageFunction, self).__init__()
        self.message_func = nn.LSTM(input_dim, output_dim, num_layers, batch_first=True, dropout=dropout, **kwargs)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, feats):
        feats, _ = self.message_func(feats)
        feats = self.drop(feats)
        return feats


class TemporalAttention(nn.Module):
    def __init__(self, input_dim):
        super(TemporalAttention, self).__init__()
        self.attn = nn.Linear(input_dim, 1)
    
    def forward(self, feats):
        attn_weights = F.softmax(self.attn(feats), dim=-2)
        attn_feats = torch.sum(feats * attn_weights, dim=-2)
        return attn_feats


class FinSIRModel(nn.Module):
    def __init__(self, input_dim, wiki_dim, industry_dim, correlation_dim, hidden_dim, output_dim, 
                 recurrent_layers, recurrent_dropout, relational_agg, relational_dropout, readout_layers, readout_dropout):
        super(FinSIRModel, self).__init__()
        _hidden_dim = (1 + int(relational_agg == 'gated')) * hidden_dim 
        self.wiki_encoder = nn.Linear(wiki_dim, 2 * input_dim, bias=False)
        self.wiki_message = SIREConvBase(MessageFunction(2 * input_dim, _hidden_dim, recurrent_layers, recurrent_dropout), relational_agg)
        self.industry_encoder = nn.Linear(industry_dim, 2 * input_dim, bias=False)
        self.industry_message = SIREConvBase(MessageFunction(2 * input_dim, _hidden_dim, recurrent_layers, recurrent_dropout), relational_agg)
        self.correlation_encoder = nn.Linear(correlation_dim, 2 * input_dim, bias=False)
        self.correlation_message = SIREConvBase(MessageFunction(2 * input_dim, _hidden_dim, recurrent_layers, recurrent_dropout), relational_agg)
        # self.residual_func = MessageFunction(input_dim + 3 * hidden_dim, 3 * hidden_dim, recurrent_layers, recurrent_dropout)
        self.temporal_attn = TemporalAttention(3 * hidden_dim)
        self.readout = MLP(3 * hidden_dim, hidden_dim, output_dim, readout_layers, readout_dropout, nn.LeakyReLU(0.2, inplace=True))
        
    def forward(self, wiki_graph, industry_graph, correlation_graph, stock_features):
        stock_features = stock_features / torch.mean(stock_features, dim=1, keepdim=True)
        wiki_returns = self.wiki_message(wiki_graph, stock_features, self.wiki_encoder(wiki_graph.edata['feat']))
        industry_returns = self.industry_message(industry_graph, stock_features, self.industry_encoder(industry_graph.edata['feat']))
        correlation_returns = self.correlation_message(correlation_graph, stock_features, self.correlation_encoder(correlation_graph.edata['feat']))
        stock_returns = torch.cat([wiki_returns, industry_returns, correlation_returns], dim=-1)
        # stock_returns = self.residual_func(stock_returns) # concat stock_features
        stock_returns = self.temporal_attn(stock_returns)
        stock_returns = self.readout(stock_returns)
        stock_returns = stock_returns / stock_features[:, -1, 0].unsqueeze(dim=-1) - 1   # Calculate return with last Close
        
        return stock_returns


class RecurrentFinSIRModel(nn.Module):
    def __init__(self, input_dim, wiki_dim, industry_dim, correlation_dim, hidden_dim, output_dim, 
                 recurrent_layers, recurrent_dropout, relational_agg, relational_dropout, readout_layers, readout_dropout):
        super(RecurrentFinSIRModel, self).__init__()
        self.message_func = MessageFunction(input_dim, hidden_dim, recurrent_layers, recurrent_dropout)
        self.wiki_message = SIREConv(hidden_dim, wiki_dim, hidden_dim, hidden_dim, nn.LeakyReLU(0.2, inplace=True), relational_dropout, agg_type=relational_agg)
        self.industry_message = SIREConv(hidden_dim, industry_dim, hidden_dim, hidden_dim, nn.LeakyReLU(0.2, inplace=True), relational_dropout, agg_type=relational_agg)
        self.correlation_message = SIREConv(hidden_dim, correlation_dim, hidden_dim, hidden_dim, nn.LeakyReLU(0.2, inplace=True), relational_dropout, agg_type=relational_agg)
        # self.residual_func = MessageFunction(4 * hidden_dim, 4 * hidden_dim, recurrent_layers, recurrent_dropout)
        self.temporal_attn = TemporalAttention(4 * hidden_dim)
        self.readout = MLP(4 * hidden_dim, hidden_dim, output_dim, readout_layers, readout_dropout, nn.LeakyReLU(0.2, inplace=True))
        
    def forward(self, wiki_graph, industry_graph, correlation_graph, stock_features):
        stock_features = stock_features / torch.mean(stock_features, dim=1, keepdim=True)
        stock_returns = self.message_func(stock_features)
        wiki_returns = torch.stack([self.wiki_message(wiki_graph, stock_returns[:, t, :], wiki_graph.edata['feat']) for t in range(stock_returns.shape[1])], dim=1)
        industry_returns = torch.stack([self.industry_message(industry_graph, stock_returns[:, t, :], industry_graph.edata['feat']) for t in range(stock_returns.shape[1])], dim=1)
        correlation_returns = torch.stack([self.correlation_message(correlation_graph, stock_returns[:, t, :], correlation_graph.edata['feat']) for t in range(stock_returns.shape[1])], dim=1)
        stock_returns = torch.cat([stock_returns, wiki_returns, industry_returns, correlation_returns], dim=-1)
        # stock_returns = self.residual_func(stock_returns)
        stock_returns = self.temporal_attn(stock_returns)
        stock_returns = self.readout(stock_returns)
        stock_returns = stock_returns / stock_features[:, -1, 0].unsqueeze(dim=-1) - 1   # Calculate return with last Close
        
        return stock_returns


class SimpleFinSIRModel(nn.Module):
    def __init__(self, input_dim, wiki_dim, industry_dim, correlation_dim, hidden_dim, output_dim, 
                 recurrent_layers, recurrent_dropout, relational_agg, relational_dropout, readout_layers, readout_dropout):
        super(SimpleFinSIRModel, self).__init__()
        self.message_func = MessageFunction(input_dim, hidden_dim, recurrent_layers, recurrent_dropout)
        self.wiki_message = SIREConv(hidden_dim, wiki_dim, hidden_dim, hidden_dim, nn.LeakyReLU(0.2, inplace=True), relational_dropout, agg_type=relational_agg)
        self.industry_message = SIREConv(hidden_dim, industry_dim, hidden_dim, hidden_dim, nn.LeakyReLU(0.2, inplace=True), relational_dropout, agg_type=relational_agg)
        self.correlation_message = SIREConv(hidden_dim, correlation_dim, hidden_dim, hidden_dim, nn.LeakyReLU(0.2, inplace=True), relational_dropout, agg_type=relational_agg)
        self.readout = MLP(4 * hidden_dim, hidden_dim, output_dim, readout_layers, readout_dropout, nn.LeakyReLU(0.2, inplace=True))
        
    def forward(self, wiki_graph, industry_graph, correlation_graph, stock_features):
        stock_features = stock_features / torch.mean(stock_features, dim=1, keepdim=True)
        stock_returns = self.message_func(stock_features)[:, -1, :]
        wiki_returns = self.wiki_message(wiki_graph, stock_returns, wiki_graph.edata['feat'])
        industry_returns = self.industry_message(industry_graph, stock_returns, industry_graph.edata['feat'])
        correlation_returns = self.correlation_message(correlation_graph, stock_returns, correlation_graph.edata['feat'])
        stock_returns = torch.cat([stock_returns, wiki_returns, industry_returns, correlation_returns], dim=-1)
        stock_returns = self.readout(stock_returns)
        stock_returns = stock_returns / stock_features[:, -1, 0].unsqueeze(dim=-1) - 1   # Calculate return with last Close
        
        return stock_returns


class BaselineModel(nn.Module):
    def __init__(self, input_dim, wiki_dim, industry_dim, correlation_dim, hidden_dim, output_dim, 
                 recurrent_layers, recurrent_dropout, relational_agg, relational_dropout, readout_layers, readout_dropout):
        super(BaselineModel, self).__init__()
        self.message_func = MessageFunction(input_dim, hidden_dim, recurrent_layers, recurrent_dropout)
        self.readout = MLP(hidden_dim, hidden_dim, output_dim, readout_layers, readout_dropout, nn.LeakyReLU(0.2, inplace=True))
        
    def forward(self, wiki_graph, industry_graph, correlation_graph, stock_features):
        stock_features = stock_features / torch.mean(stock_features, dim=1, keepdim=True)
        stock_returns = self.message_func(stock_features)[:, -1, :]
        stock_returns = self.readout(stock_returns)
        stock_returns = stock_returns / stock_features[:, -1, 0].unsqueeze(dim=-1) - 1   # Calculate return with last Close
        
        return stock_returns
