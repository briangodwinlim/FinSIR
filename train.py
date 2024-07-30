import os
import dgl
import torch
import random
import argparse
import numpy as np
import datetime as dt
from pprint import pprint
from torchinfo import summary
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import StockGraph
from model import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def warmup_lr(optimizer, lr, epoch, size):
    if epoch <= size:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * epoch / size


def load_dataset(market, args):
    train_dataset = StockGraph(market, dt.datetime(2013,1,2), dt.datetime(2015,12,31), args.sequence_length, args.add_self_loop, args.return_period)
    val_dataset = StockGraph(market, dt.datetime(2016,1,4), dt.datetime(2016,12,30), args.sequence_length, args.add_self_loop, args.return_period)
    test_dataset = StockGraph(market, dt.datetime(2017,1,3), dt.datetime(2017,12,8), args.sequence_length, args.add_self_loop, args.return_period)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=args.nworkers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.nworkers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.nworkers)
    
    return train_dataset, train_loader, val_loader, test_loader


def regularizer(model, args):
    l1 = torch.sum(torch.stack([torch.sum(torch.abs(param)) for param in model.parameters()]))
    l2 = torch.sum(torch.stack([torch.sum(torch.pow(param, 2)) for param in model.parameters()]))
    return (args.l1 * l1 + args.l2 * l2) * int(model.training)

def loss_fn(preds, targets, model, args):
    assert preds.shape == targets.shape
    mse_loss = F.mse_loss(preds, targets)
    rank_loss = torch.mean(F.relu(- (preds - preds.T) * (targets - targets.T)))
    return mse_loss + args.alpha * rank_loss + regularizer(model, args)

def mrr_fn(preds, targets, k=1):
    assert preds.shape == targets.shape
    preds_rank = torch.sort(preds, descending=True, dim=0)[1]
    targets_rank = torch.sort(targets, descending=True, dim=0)[1]
    reciprocal_rank = 1 / (torch.nonzero(preds_rank == targets_rank.T, as_tuple=True)[1] + 1)
    return torch.mean(reciprocal_rank[:k])

def irr_fn(preds, targets, k=1):
    assert preds.shape == targets.shape
    preds_rank = torch.sort(preds, descending=True, dim=0)[1].squeeze()
    investment_return = torch.mean(targets[preds_rank[:k]].squeeze())
    return investment_return

def train(model, dataset, train_loader, device, optimizer, scaler, args):
    model.train()
    
    total_loss, total = 0, 0
    for sample_idx, stock_features, stock_returns in train_loader:
        stock_features = stock_features.squeeze(dim=0).to(device)
        stock_returns = stock_returns.squeeze(dim=0).to(device)
        wiki_graph = dataset.wiki_graph.to(device)
        industry_graph = dataset.industry_graph.to(device)
        correlation_graph = dataset.correlation_graph(sample_idx, args.corr_graph_periods, args.corr_graph_thresh).to(device)

        optimizer.zero_grad()
        # torch.cuda.empty_cache()

        with torch.autocast(device_type=device.type, enabled=args.use_amp):
            pred_stock_returns = model(wiki_graph, industry_graph, correlation_graph, stock_features)
            loss = loss_fn(pred_stock_returns, stock_returns, model, args)
        
        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total = total + 1
        total_loss = total_loss + loss.item()
        
    return total_loss / total - regularizer(model, args).item()

@torch.no_grad()
def evaluate(model, dataset, dataloader, device, args):
    model.eval()
    
    total_mse, total = 0, 0
    total_mrr = {k: 0 for k in args.k_list}
    total_irr = {k: 0 for k in args.k_list}
    for sample_idx, stock_features, stock_returns in dataloader:
        stock_features = stock_features.squeeze(dim=0).to(device)
        stock_returns = stock_returns.squeeze(dim=0).to(device)
        wiki_graph = dataset.wiki_graph.to(device)
        industry_graph = dataset.industry_graph.to(device)
        correlation_graph = dataset.correlation_graph(sample_idx, args.corr_graph_periods, args.corr_graph_thresh).to(device)

        with torch.autocast(device_type=device.type, enabled=args.use_amp):
            pred_stock_returns = model(wiki_graph, industry_graph, correlation_graph, stock_features)
            
            total = total + 1
            total_mse = total_mse + F.mse_loss(pred_stock_returns, stock_returns).item()
            total_mrr = {k: total_mrr[k] + mrr_fn(pred_stock_returns, stock_returns, k).item() for k in args.k_list}
            total_irr = {k: total_irr[k] + irr_fn(pred_stock_returns, stock_returns, k).item() for k in args.k_list}
    
    total_mse = total_mse / total
    total_mrr = {k: total_mrr[k] / total for k in args.k_list}
    
    return total_mse, total_mrr, total_irr


def run(model, dataset, train_loader, val_loader, test_loader, device, args, iter):
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience)
    best_val_irr_1 = 0
    
    # train_opt = torch.compile(train, mode='reduce-overhead')
    # evaluate_opt = torch.compile(evaluate, mode='reduce-overhead')
    
    for epoch in range(args.epochs):
        warmup_lr(optimizer, args.lr, epoch + 1, 10)
        loss = train(model, dataset, train_loader, device, optimizer, scaler, args)
        mse, mrr, irr = evaluate(model, dataset, train_loader, device, args)
        val_mse, val_mrr, val_irr = evaluate(model, dataset, val_loader, device, args)
        test_mse, test_mrr, test_irr = evaluate(model, dataset, test_loader, device, args)
        scheduler.step(loss)

        if val_irr[1] > best_val_irr_1:
            best_val_irr_1 = val_irr[1]
            result = {
                'val_mse': val_mse,
                'val_mrr': val_mrr,
                'val_irr': val_irr,
                'test_mse': test_mse,
                'test_mrr': test_mrr,
                'test_irr': test_irr,
            }

        if (epoch + 1) == args.epochs or (epoch + 1) % args.log_every == 0:
            print(f'Epoch {epoch+1:04d} | mse: {mse:.4f} | ' + ' | '.join([f'mrr_{k}: {mrr_k:.4f}' for k, mrr_k in mrr.items()] + [f'irr_{k}: {irr_k:.4f}' for k, irr_k in irr.items()]))
            print(f'val_mse: {val_mse:.4f} | ' + ' | '.join([f'val_mrr_{k}: {mrr_k:.4f}' for k, mrr_k in val_mrr.items()] + [f'val_irr_{k}: {irr_k:.4f}' for k, irr_k in val_irr.items()]))
            print(f'test_mse: {test_mse:.4f} | ' + ' | '.join([f'test_mrr_{k}: {mrr_k:.4f}' for k, mrr_k in test_mrr.items()] + [f'test_irr_{k}: {irr_k:.4f}' for k, irr_k in test_irr.items()]))

    return result


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        'FinSIR implementation on StockGraph',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument('--cpu', action='store_true', help='CPU mode')
    argparser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    argparser.add_argument('--seed', type=int, default=0, help='seed')
    argparser.add_argument('--nworkers', type=int, default=0, help='number of workers')
    argparser.add_argument('--use-amp', action='store_true', help='use automatic mixed precision')
    
    argparser.add_argument('--market', type=str, default='NASDAQ', help='market name', choices=['NASDAQ', 'NYSE'])
    argparser.add_argument('--sequence-length', type=int, default=16, help='sample sequence length')
    argparser.add_argument('--return-period', type=int, default=1, help='target return period')
    argparser.add_argument('--corr-graph-periods', type=str, default='5 10 20 30', help='list of periods for correlation graph') 
    argparser.add_argument('--corr-graph-thresh', type=float, default=0.9, help='threshold for correlation graph') 
    argparser.add_argument('--add-self-loop', action='store_true', help='add self-loop to relational graphs')
    
    argparser.add_argument('--model', type=str, default='FinSIR', help='model name', choices=['FinSIR', 'Recurrent', 'Simple', 'Baseline']) 
    argparser.add_argument('--nhidden', type=int, default=16, help='number of hidden units')
    argparser.add_argument('--recurrent-layers', type=int, default=1, help='number of layers for recurrent message function')
    argparser.add_argument('--recurrent-dropout', type=float, default=0, help='dropout rate for recurrent message function')
    argparser.add_argument('--relational-agg', type=str, default='sum', help='aggregation type for relational convolution', choices=['sum', 'max', 'mean', 'sym', 'gated'])
    argparser.add_argument('--relational-dropout', type=float, default=0, help='dropout rate for relational convolution')
    argparser.add_argument('--readout-layers', type=int, default=1, help='number of layers for readout function')
    argparser.add_argument('--readout-dropout', type=float, default=0, help='dropout rate for readout function')
    
    argparser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    argparser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    argparser.add_argument('--wd', type=float, default=0, help='weight decay')
    argparser.add_argument('--l1', type=float, default=0, help='weight for L1 regularization')
    argparser.add_argument('--l2', type=float, default=0, help='weight for L2 regularization')
    argparser.add_argument('--factor', type=float, default=0.5, help='factor for learning rate decay')
    argparser.add_argument('--patience', type=int, default=10, help='patience for learning rate decay')
    
    argparser.add_argument('--alpha', type=float, default=1.0, help='weight for rank-aware loss')
    argparser.add_argument('--k-list', type=str, default='1 5 10', help='list of k values for MRR and IRR evaluation')
    
    argparser.add_argument('--nruns', type=int, default=10, help='number of runs')
    argparser.add_argument('--log-every', type=int, default=10, help='log every LOG_EVERY epochs')
    args = argparser.parse_args()
    
    # Process string list inputs
    args.corr_graph_periods = [int(p) for p in args.corr_graph_periods.split(' ')]
    args.k_list = [int(k) for k in args.k_list.split(' ')]

    val_mses, val_mrrs, val_irrs = [], [], [] 
    test_mses, test_mrrs, test_irrs = [], [], []
    for i in range(args.nruns):
        # Set seed
        set_seed(args.seed + i)

        # Load dataset
        device = torch.device('cpu') if args.cpu else torch.device(f'cuda:{args.gpu}')
        dataset, train_loader, val_loader, test_loader = load_dataset(args.market, args)
        
        # Extract input shapes
        input_dim = dataset[0][1].shape[-1]
        wiki_dim = dataset.wiki_graph.edata['feat'].shape[-1]
        industry_dim = dataset.industry_graph.edata['feat'].shape[-1]
        correlation_dim = len(args.corr_graph_periods) + 1
        
        # Load model
        Model = {'FinSIR': FinSIRModel, 'Recurrent': RecurrentFinSIRModel, 'Simple': SimpleFinSIRModel, 'Baseline': BaselineModel}
        model = Model[args.model](input_dim, wiki_dim, industry_dim, correlation_dim, args.nhidden, 1, 
                                  args.recurrent_layers, args.recurrent_dropout, args.relational_agg, args.relational_dropout,
                                  args.readout_layers, args.readout_dropout).to(device)
        summary(model)

        # Training
        result = run(model, dataset, train_loader, val_loader, test_loader, device, args, i)
        val_mses.append(result['val_mse'])
        val_mrrs.append(result['val_mrr'])
        val_irrs.append(result['val_irr'])
        test_mses.append(result['test_mse'])
        test_mrrs.append(result['test_mrr'])
        test_irrs.append(result['test_irr'])
    
    print(args)
    print(f'Runned {args.nruns} times')
    
    process_results = lambda metrics: {k: [metrics[i][k] for i in range(args.nruns)] for k in args.k_list}
    val_mrrs = process_results(val_mrrs)
    val_irrs = process_results(val_irrs)
    test_mrrs = process_results(test_mrrs)
    test_irrs = process_results(test_irrs)
    pprint({'Val MSE': val_mses,
            'Val MRR': val_mrrs,
            'Val IRR': val_irrs,
            'Test MSE': test_mses,
            'Test MRR': test_mrrs,
            'Test IRR': test_irrs})
    
    process_results = lambda metrics: {k: f'{np.mean(metrics[k]):.6f} ± {np.std(metrics[k]):.6f}' for k in args.k_list}    
    val_mrrs = process_results(val_mrrs)
    val_irrs = process_results(val_irrs)
    test_mrrs = process_results(test_mrrs)
    test_irrs = process_results(test_irrs)
    pprint({'Average val MSE': f'{np.mean(val_mses):.6f} ± {np.std(val_mses):.6f}',
            'Average val MRR': val_mrrs,
            'Average val IRR': val_irrs,
            'Average test MSE': f'{np.mean(test_mses):.6f} ± {np.std(test_mses):.6f}',
            'Average test MRR': test_mrrs,
            'Average test IRR': test_irrs})
