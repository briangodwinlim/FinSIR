import os
import dgl
import torch
import pickle
import functools
import numpy as np
import pandas as pd
from tqdm import tqdm
from dgl.data import DGLDataset


# Transform arguments of dataset.correlation_graph into hashable types
def args_transform(func):
    def wrapper(dataset, sample_idx, periods, thresh):
        sample_idx = sample_idx.item() if type(sample_idx) == torch.Tensor else sample_idx
        periods = tuple(periods) if type(periods) == list else periods
        return func(dataset, sample_idx, periods, thresh)
    return wrapper


class StockGraph(DGLDataset):
    def __init__(self, market, start_date, end_date, sequence_length, self_loop=True, return_period=1,
                 data_path='dataset/processed/', cache_path='dataset/cached/', force_reload=False):
        super(StockGraph, self).__init__(name='StockGraph')
        self.market = market
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length
        self.self_loop = self_loop
        self.return_period = return_period
        self.data_path = data_path
        self.cache_path = cache_path
        
        # Load preprocessed data
        self.stocks = [line.strip() for line in open(os.path.join(data_path, f'{market}_stocks.txt'), 'r')]
        self.stock_price = {stock: pd.read_csv(os.path.join(data_path, f'price_data/{market}_{stock}.csv'), index_col=0, parse_dates=True) for stock in self.stocks}
        self.time_idx = self.stock_price[self.stocks[0]].index
        self.filtered_index = self.time_idx[(self.time_idx >= self.start_date) & (self.time_idx <= self.end_date)]
        
        # Process/load features
        cache_file = f'{market}_{start_date.strftime("%Y-%m-%d")}_{end_date.strftime("%Y-%m-%d")}_s{sequence_length}_r{return_period}.pkl'
        if force_reload or not os.path.exists(os.path.join(cache_path, cache_file)):
            self.sample_idx, self.stock_features, self.stock_returns = self.process_features()
            os.makedirs(cache_path, exist_ok=True)
            with open(os.path.join(cache_path, cache_file), 'wb') as file:
                pickle.dump((self.sample_idx, self.stock_features, self.stock_returns), file)
        else:
            with open(os.path.join(cache_path, cache_file), 'rb') as file:
                self.sample_idx, self.stock_features, self.stock_returns = pickle.load(file) 
            
    def process_features(self):
        sample_idx, stock_features, stock_returns = [], [], []
        for i in tqdm(range(len(self.filtered_index) - self.sequence_length), desc='Processing features'):
            stock_feature, stock_return = [], []
            for _, price in self.stock_price.items():
                price['Return'] = price['Close'] / price['Close'].shift(periods=self.return_period) - 1
                price = price.loc[self.start_date : self.end_date]
                stock_feature.append(price.iloc[i : i + self.sequence_length, price.columns.get_indexer(['Close'] + [col for col in price.columns if col.startswith('Close MA')])].values)
                stock_return.append(price.iloc[i + self.sequence_length, price.columns.get_indexer(['Return'])].values)
            sample_idx.append(self.time_idx.get_loc(self.filtered_index[i + self.sequence_length - 1]))
            stock_features.append(stock_feature)
            stock_returns.append(stock_return)
        sample_idx = torch.from_numpy(np.array(sample_idx)).long()
        stock_features = torch.from_numpy(np.array(stock_features)).float()
        stock_returns = torch.from_numpy(np.array(stock_returns)).float()
        
        return sample_idx, stock_features, stock_returns
        
    @functools.cached_property
    def wiki_graph(self):
        wiki_features = torch.from_numpy(np.load(os.path.join(self.data_path, f'{self.market}_wiki_features.npy')))
        wiki_graph = dgl.graph(torch.nonzero(wiki_features.sum(dim=-1), as_tuple=True), num_nodes=len(self.stocks))
        wiki_graph.edata['feat'] = wiki_features[wiki_graph.edges()].float()
        wiki_graph = wiki_graph if self.self_loop else dgl.remove_self_loop(wiki_graph)
        return wiki_graph
    
    @functools.cached_property
    def industry_graph(self):
        industry_features = torch.from_numpy(np.load(os.path.join(self.data_path, f'{self.market}_industry_features.npy')))
        industry_graph = dgl.graph(torch.nonzero(industry_features.sum(dim=-1), as_tuple=True), num_nodes=len(self.stocks))
        industry_graph.edata['feat'] = industry_features[industry_graph.edges()].float()
        industry_graph = industry_graph if self.self_loop else dgl.remove_self_loop(industry_graph)
        return industry_graph
    
    @args_transform
    @functools.lru_cache(maxsize=None)
    def correlation_graph(self, sample_idx, periods, thresh):
        stock_close = torch.stack([torch.from_numpy(price['Close'].values) for _, price in self.stock_price.items()], dim=0)
        close_correlation = torch.stack([torch.corrcoef(stock_close[:, max(0, sample_idx + 1 - period) : sample_idx + 1]) for period in periods], dim=-1)
        close_correlation = torch.cat([close_correlation, torch.eye(len(self.stocks)).unsqueeze(dim=-1)], dim=-1)   # Self-loop indicator
        close_correlation = (torch.abs(close_correlation) > thresh).int()
        correlation_graph = dgl.graph(torch.nonzero(close_correlation.sum(dim=-1), as_tuple=True), num_nodes=len(self.stocks))
        # correlation_graph = dgl.graph(torch.nonzero((torch.abs(close_correlation) > thresh).sum(dim=-1), as_tuple=True), num_nodes=len(self.stocks))
        correlation_graph.edata['feat'] = close_correlation[correlation_graph.edges()].float()
        correlation_graph = correlation_graph if self.self_loop else dgl.remove_self_loop(correlation_graph)
        return correlation_graph
        
    def __getitem__(self, i):
        return self.sample_idx[i], self.stock_features[i], self.stock_returns[i]
    
    def __len__(self):
        return len(self.filtered_index) - self.sequence_length

    def __hash__(self):     # Custom hash for functools.lru_cache
        return hash((self.market, self.start_date, self.end_date, 
                     self.sequence_length, self.self_loop, self.return_period))
    
    def __eq__(self, other):
        return ((self.market == other.market) and 
                (self.start_date == other.start_date) and 
                (self.end_date == other.end_date) and 
                (self.sequence_length == other.sequence_length) and 
                (self.self_loop == other.self_loop) and 
                (self.return_period == other.return_period))
