import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm


def process_wiki_relations(ticker_hash, wiki_relations, selected_wiki_relations):
    # Process ticker hash
    ticker_hash = np.genfromtxt(ticker_hash, dtype=str, delimiter=',', skip_header=False)
    hash_ticker = {hash: ticker for ticker, hash in ticker_hash if hash != 'unknown'}
    
    # Load wiki relations
    with open(wiki_relations, 'r') as file:
        wiki_relations = json.load(file)
    
    # Load selected wiki relations
    selected_wiki_relations = set(np.genfromtxt(selected_wiki_relations, dtype=str, delimiter=' ', skip_header=False)[:, 0])
    
    # Clean wiki relations
    cleaned_wiki_relations = dict()
    for stockA_hash, wiki_relation in wiki_relations.items():
        for stockB_hash, relation in wiki_relation.items():
            stockA = hash_ticker[stockA_hash]
            stockB = hash_ticker[stockB_hash]
            for wiki in relation:
                wiki = '_'.join(wiki)
                if wiki in selected_wiki_relations:
                    if stockA not in cleaned_wiki_relations:
                        cleaned_wiki_relations[stockA] = dict()
                    if stockB not in cleaned_wiki_relations[stockA]:
                        cleaned_wiki_relations[stockA][stockB] = []
                    cleaned_wiki_relations[stockA][stockB].append(wiki)
                    
    return cleaned_wiki_relations


def process_industry_relations(industry_relations, min_stocks=2):
    # Load industry relations
    with open(industry_relations, 'r') as file:
        industry_relations = json.load(file)
    
    # Clean industry relations
    cleaned_industry_relations = {'n/a': []}
    for industry, stocks in industry_relations.items():
        if len(stocks) >= min_stocks and industry != 'n/a':
            cleaned_industry_relations[industry] = stocks
        else:   # Cluster small industries to n/a
            cleaned_industry_relations['n/a'] += stocks
        
    return cleaned_industry_relations


def process_relations(market, price_path, wiki_path, industry_path, fillna=True, save_path=False):
    selected_tickers = sorted(set(np.genfromtxt(os.path.join(price_path, f'{market}_tickers_qualify_dr-0.98_min-5_smooth.csv'), dtype=str, delimiter='\t', skip_header=False)))
    trading_dates = pd.to_datetime(np.genfromtxt(os.path.join(price_path, f'{market}_aver_line_dates.csv'), dtype=str, delimiter=',', skip_header=False), utc=True).tz_localize(None).normalize()
    wiki_relations = process_wiki_relations(os.path.join(wiki_path, f'{market}_wiki.csv'), os.path.join(wiki_path, f'{market}_connections.json'), os.path.join(wiki_path, 'selected_wiki_connections.csv'))
    industry_relations = process_industry_relations(os.path.join(industry_path, f'{market}_industry_ticker.json'), min_stocks=2)
    
    # Process price data
    price_data = dict()
    for stock in tqdm(selected_tickers, desc=f'Processing {market} price data'):
        df = pd.read_csv(os.path.join(price_path, f'{market}_{stock}_30Y.csv'), index_col=0)
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None).normalize()
        
        # Feature extraction
        df = df.reindex(trading_dates).ffill().bfill() if fillna else df.reindex(trading_dates)
        for window in [5, 10, 20, 30]: # MA windows
            df[f'Close MA{window}'] = np.where(np.isnan(df['Close']), np.nan, df['Close'].rolling(window, min_periods=1).mean())
            
        price_data[stock] = df
        
    # Filter wiki_relations
    filtered_wiki_relations = dict()
    for stockA, wiki_relation in wiki_relations.items():
        if stockA in selected_tickers:
            if stockA not in filtered_wiki_relations:
                filtered_wiki_relations[stockA] = dict()
            
            for stockB, relation in wiki_relation.items():
                if stockB in selected_tickers:
                    filtered_wiki_relations[stockA][stockB] = relation
    
    # Filter industry_relations
    filtered_industry_relations = dict()
    for industry, stocks in industry_relations.items():
        filtered_industry_relations[industry] = [stock for stock in stocks if stock in selected_tickers]
    
    # Save files
    if save_path:
        save_price_path = 'price_data' if fillna else 'price_data_nan'
        os.makedirs(os.path.join(save_path, save_price_path), exist_ok=True)
        for stock in tqdm(selected_tickers, desc=f'Saving {market} price data'):
            price_data[stock].to_csv(os.path.join(save_path, save_price_path, f'{market}_{stock}.csv'))
        with open(os.path.join(save_path, f'{market}_wiki.json'), 'w') as file:
            json.dump(filtered_wiki_relations, file, indent=4)
        with open(os.path.join(save_path, f'{market}_industry.json'), 'w') as file:
            json.dump(filtered_industry_relations, file, indent=4)
        with open(os.path.join(save_path, f'{market}_stocks.txt'), 'w') as file:
            for stock in selected_tickers:
                file.write(stock + '\n')
    
    return selected_tickers, price_data, filtered_wiki_relations, filtered_industry_relations


def process_edge_features(market, ticker_index, wiki_relations, wiki_index, industry_relations, industry_index, save_path=False):
    wiki_features = np.zeros((len(ticker_index), len(ticker_index), len(wiki_index) + 1), dtype=int)
    for stockA, wiki_relation in wiki_relations.items():
        for stockB, relation in wiki_relation.items():
            for wiki in relation:
                wiki_features[ticker_index[stockA]][ticker_index[stockB]][wiki_index[wiki]] = 1

    industry_features = np.zeros((len(ticker_index), len(ticker_index), len(industry_index) + 1), dtype=int)
    for industry, stocks in industry_relations.items():
        for stockA in stocks:
            for stockB in stocks:
                industry_features[ticker_index[stockA]][ticker_index[stockB]][industry_index[industry]] = 1
    
    for i in range(len(ticker_index)):
        wiki_features[i][i][-1] = 1
        industry_features[i][i][-1] = 1
                
    # Save files
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, f'{market}_wiki_features'), wiki_features)
        np.save(os.path.join(save_path, f'{market}_industry_features'), industry_features)
    
    return wiki_features, industry_features


if __name__ == '__main__':
    for market in ['NYSE', 'NASDAQ']:
        tickers, price_data, wiki_relations, industry_relations = process_relations(market, price_path='raw/google_finance/', wiki_path='raw/wikidata/', industry_path='raw/sector_industry/', save_path='processed/')
        ticker_index = {stock: idx for idx, stock in enumerate(tickers)}
        
        unique_wiki_relations = set([wiki for wiki_relation in wiki_relations.values() for relation in wiki_relation.values() for wiki in relation])
        wiki_index = {wiki: idx for idx, wiki in enumerate(sorted(unique_wiki_relations))}

        unique_industry_relations = set([industry for industry in industry_relations])
        industry_index = {industry: idx for idx, industry in enumerate(sorted(unique_industry_relations))}

        wiki_features, industry_features = process_edge_features(market, ticker_index, wiki_relations, wiki_index, industry_relations, industry_index, save_path='processed/')
    