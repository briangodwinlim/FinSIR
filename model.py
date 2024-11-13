import torch
from torch import nn
from dgl import function as fn
from dgl.utils import expand_as_pair
from dgl.nn.functional import edge_softmax


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


class TemporalModule(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=1, dropout=0, **kwargs):
        super(TemporalModule, self).__init__()
        self.temporal_module = nn.LSTM(input_dim, output_dim, num_layers, batch_first=True, dropout=dropout, **kwargs)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, feats):
        feats, _ = self.temporal_module(feats)
        feats = self.drop(feats)
        return feats


class SIREConv(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim, activation, dropout=0, inner_bias=True, outer_bias=True, agg_type='sum'):
        super(SIREConv, self).__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.linear_query = nn.Linear(input_dim, hidden_dim, bias=inner_bias)
        self.linear_key = nn.Linear(input_dim, hidden_dim, bias=False)
        self.linear_edge = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.linear_relation = nn.Linear(hidden_dim, output_dim, bias=outer_bias)

        self._agg_type = agg_type
        self._agg_func = fn.sum if agg_type == 'sym' else getattr(fn, agg_type)
    
    def message_func(self, edges):
        if self._agg_type in ['sum', 'mean', 'sym']:
            return {'m': edges.src['norm'] * edges.dst['norm'] * self.activation(edges.dst['eq'] + edges.src['ek'] + edges.data['e'])}
        else:
            return {'m': self.linear_relation(self.activation(edges.dst['eq'] + edges.src['ek'] + edges.data['e']))}
    
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


class FinSIRModel(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim, recurrent_layers, recurrent_dropout, 
                 relational_agg, relational_dropout, readout_layers, readout_dropout, **kwargs):
        super(FinSIRModel, self).__init__()
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.temporal_module_pre = TemporalModule(input_dim, hidden_dim, recurrent_layers, recurrent_dropout)
        self.relational_module = SIREConv(hidden_dim, edge_dim, hidden_dim, hidden_dim, self.activation, relational_dropout, agg_type=relational_agg)
        self.temporal_module_post = TemporalModule(2 * hidden_dim, hidden_dim, recurrent_layers, recurrent_dropout)
        self.readout = MLP(hidden_dim, hidden_dim, output_dim, readout_layers, readout_dropout, self.activation)
        
    def forward(self, relational_graph, stock_features):
        stock_features = stock_features / torch.mean(stock_features, dim=1, keepdim=True)
        stock_returns = self.temporal_module_pre(stock_features)
        relational_returns = torch.stack([self.relational_module(relational_graph, stock_returns[:, t, :], relational_graph.edata['feat']) 
                                          for t in range(stock_returns.shape[1])], dim=1)
        relational_returns = self.activation(relational_returns)
        stock_returns = torch.cat([stock_returns, relational_returns], dim=-1)
        stock_returns = self.temporal_module_post(stock_returns)[:, -1, :]
        stock_returns = self.readout(stock_returns)
        stock_returns = stock_returns / stock_features[:, -1, 0].unsqueeze(dim=-1) - 1   # Calculate return with last Close
        
        return stock_returns


class SimpleFinSIRModel(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim, recurrent_layers, recurrent_dropout, 
                 relational_agg, relational_dropout, readout_layers, readout_dropout, **kwargs):
        super(SimpleFinSIRModel, self).__init__()
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.temporal_module = TemporalModule(input_dim, hidden_dim, recurrent_layers, recurrent_dropout)
        self.relational_module = SIREConv(hidden_dim, edge_dim, hidden_dim, hidden_dim, self.activation, relational_dropout, agg_type=relational_agg)
        self.readout = MLP(2 * hidden_dim, hidden_dim, output_dim, readout_layers, readout_dropout, self.activation)
        
    def forward(self, relational_graph, stock_features):
        stock_features = stock_features / torch.mean(stock_features, dim=1, keepdim=True)
        stock_returns = self.temporal_module(stock_features)[:, -1, :]
        relational_returns = self.relational_module(relational_graph, stock_returns, relational_graph.edata['feat'])
        relational_returns = self.activation(relational_returns)
        stock_returns = torch.cat([stock_returns, relational_returns], dim=-1)
        stock_returns = self.readout(stock_returns)
        stock_returns = stock_returns / stock_features[:, -1, 0].unsqueeze(dim=-1) - 1   # Calculate return with last Close
        
        return stock_returns


class TemporalGraphConv(nn.Module):
    def __init__(self, input_dim, edge_dim, activation, dropout=0, agg_type='implicit'):
        super(TemporalGraphConv, self).__init__()
        self.agg_type = agg_type
        self.linear_src = nn.Sequential(nn.Linear(input_dim, 1), activation, nn.Dropout(dropout)) if agg_type == 'implicit' else None
        self.linear_dst = nn.Sequential(nn.Linear(input_dim, 1), activation, nn.Dropout(dropout)) if agg_type == 'implicit' else None
        self.linear_edge = nn.Sequential(nn.Linear(edge_dim, 1), activation, nn.Dropout(dropout))

    def weight_func(self, edges):
        if self.agg_type == 'implicit':
            return {'w': self.linear_dst(edges.dst['h']) + self.linear_src(edges.src['h']) + self.linear_edge(edges.data['e'])}
        else:
            return {'w': torch.sum(edges.dst['h'] * edges.src['h'], dim=-1, keepdim=True) * self.linear_edge(edges.data['e'])}

    def forward(self, graph, nfeat, efeat):
        with graph.local_scope():
            graph.ndata['h'] = nfeat
            graph.edata['e'] = efeat
            graph.apply_edges(self.weight_func)
            graph.edata['w_norm'] = edge_softmax(graph, graph.edata.pop('w'))
            graph.update_all(fn.u_mul_e('h', 'w_norm', 'm'), fn.sum('m', 'ft'))
            rst = graph.ndata.pop('ft')
            
            return rst


class RelationalStockRankingModel(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim, recurrent_layers, recurrent_dropout, 
                 relational_agg, relational_dropout, readout_layers, readout_dropout, **kwargs):
        super(RelationalStockRankingModel, self).__init__()
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.temporal_module = TemporalModule(input_dim, hidden_dim, recurrent_layers, recurrent_dropout)
        self.relational_module = TemporalGraphConv(hidden_dim, edge_dim, self.activation, relational_dropout, relational_agg)
        self.readout = MLP(2 * hidden_dim, hidden_dim, output_dim, readout_layers, readout_dropout, self.activation)
        
    def forward(self, relational_graph, stock_features):
        stock_features = stock_features / torch.mean(stock_features, dim=1, keepdim=True)
        stock_returns = self.temporal_module(stock_features)[:, -1, :]
        relational_returns = self.relational_module(relational_graph, stock_returns, relational_graph.edata['feat'])
        stock_returns = torch.cat([stock_returns, relational_returns], dim=-1)
        stock_returns = self.readout(stock_returns)
        stock_returns = stock_returns / stock_features[:, -1, 0].unsqueeze(dim=-1) - 1   # Calculate return with last Close
        
        return stock_returns


class RankLSTMModel(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim, recurrent_layers, recurrent_dropout, 
                 relational_agg, relational_dropout, readout_layers, readout_dropout, **kwargs):
        super(RankLSTMModel, self).__init__()
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.temporal_module = TemporalModule(input_dim, hidden_dim, recurrent_layers, recurrent_dropout)
        self.readout = MLP(hidden_dim, hidden_dim, output_dim, readout_layers, readout_dropout, self.activation)
        
    def forward(self, relational_graph, stock_features):
        stock_features = stock_features / torch.mean(stock_features, dim=1, keepdim=True)
        stock_returns = self.temporal_module(stock_features)[:, -1, :]
        stock_returns = self.readout(stock_returns)
        stock_returns = stock_returns / stock_features[:, -1, 0].unsqueeze(dim=-1) - 1   # Calculate return with last Close
        
        return stock_returns
