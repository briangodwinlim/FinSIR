import os
import torch
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import ASHAScheduler

import train as tr
from model import FinSIRModel as Model


class Arguments:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def run(config):
    args = Arguments(**config, nworkers=4, use_amp=True, market=market, return_period=1, l1=0, l2=0, k_list=[1, 5, 10])
    args.corr_graph_periods = [int(p) for p in args.corr_graph_periods.split(' ')]    
    
    # Change directory
    os.chdir(os.path.join(base_path, 'FinSIR/'))
    
    # Set seed
    tr.set_seed(0)

    # Load dataset
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dataset, train_loader, val_loader, _ = tr.load_dataset(args.market, args)
    
    # Extract input shapes
    input_dim = dataset[0][1].shape[-1]
    wiki_dim = dataset.wiki_graph.edata['feat'].shape[-1]
    industry_dim = dataset.industry_graph.edata['feat'].shape[-1]
    correlation_dim = len(args.corr_graph_periods) + 1
        
    # Load model
    model = Model(input_dim, wiki_dim, industry_dim, correlation_dim, args.nhidden, 1, 
                  args.recurrent_layers, args.recurrent_dropout, args.relational_agg, args.relational_dropout,
                  args.readout_layers, args.readout_dropout).to(device)
    
    # Scaler + optimizer + scheduler
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience)
    best_val_irr_1 = 0

    # Restore checkpoint
    loaded_checkpoint = train.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, scaler_state, optimizer_state, scheduler_state = torch.load(os.path.join(loaded_checkpoint_dir, 'checkpoint.pt'))
            model.load_state_dict(model_state)
            scaler.load_state_dict(scaler_state)
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)

    for epoch in range(50):
        tr.warmup_lr(optimizer, args.lr, epoch + 1, 10)
        loss = tr.train(model, dataset, train_loader, device, optimizer, scaler, args)
        mse, mrr, irr = tr.evaluate(model, dataset, train_loader, device, args)
        val_mse, val_mrr, val_irr = tr.evaluate(model, dataset, val_loader, device, args)
        scheduler.step(loss)

        if val_irr[1] > best_val_irr_1:
            best_val_irr_1 = val_irr[1]
            result = {
                'val_mse': val_mse,
                'val_mrr': val_mrr,
                'val_irr': val_irr,
            }

        os.makedirs('logs/models', exist_ok=True)
        torch.save((model.state_dict(), scaler.state_dict(), optimizer.state_dict(), scheduler.state_dict()), 'logs/models/checkpoint.pt')
        checkpoint = train.Checkpoint.from_directory('logs/models')
        train.report({'loss': loss, 'mse': mse, 'mrr_1': mrr[1], 'mrr_5': mrr[5], 'mrr_10': mrr[10], 'irr_1': irr[1], 'irr_5': irr[5], 'irr_10': irr[10],
                      'val_mse': val_mse, 'val_mrr_1': val_mrr[1], 'val_mrr_5': val_mrr[5], 'val_mrr_10': val_mrr[10], 'val_irr_1': val_irr[1], 'val_irr_5': val_irr[5], 'val_irr_10': val_irr[10],
                      'best_val_irr_1': result['val_irr'][1], 'best_val_irr_5': result['val_irr'][5], 'best_val_irr_10': result['val_irr'][10]}, checkpoint=checkpoint)


if __name__ == '__main__':
    param_space = {
        'sequence_length': tune.choice([2, 4, 8, 16]),
        'corr_graph_periods': tune.choice(['5 20', '5 30', '10 20', '10 30']),
        'corr_graph_thresh': tune.choice([0.8, 0.9]),
        'add_self_loop': tune.choice([True, False]),
        'nhidden': tune.choice([4, 8, 16, 32]),
        'recurrent_layers': tune.choice([1, 2]),
        'recurrent_dropout': tune.choice([0.0, 0.1, 0.2, 0.4]),
        'relational_agg': tune.choice(['sum', 'mean', 'sym', 'max']),
        'relational_dropout': tune.choice([0.0]),
        'readout_layers': tune.choice([1, 2]),
        'readout_dropout': tune.choice([0.0, 0.1, 0.2, 0.4]),
        'lr': tune.choice([1e-4, 1e-3]),
        'wd': tune.choice([1e-8, 1e-7, 1e-6]),
        'factor': tune.choice([0.5]),
        'patience': tune.choice([10]),
        'alpha': tune.choice([0.1, 1.0, 10.0]),
    }
    
    hyperparams = [
    ]
    
    market = 'NASDAQ'
    base_path = '/home/'
    directory = os.path.join(base_path, 'FinSIR/logs/')
    exp_name = f'{market}_{Model.__name__}_binary'
    
    search_alg = OptunaSearch(points_to_evaluate=hyperparams)
    # search_alg.restore_from_dir(os.path.join(directory, exp_name))
    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=1)
    scheduler = ASHAScheduler(max_t=50, grace_period=20)
    
    if tune.Tuner.can_restore(os.path.join(directory, exp_name)):
        tuner = tune.Tuner.restore(
            path=os.path.join(directory, exp_name), 
            trainable=tune.with_resources(tune.with_parameters(run), resources={'cpu': 24, 'gpu': 1}),
            resume_unfinished=True,
        )
    else:
        tuner = tune.Tuner(
            trainable=tune.with_resources(tune.with_parameters(run), resources={'cpu': 24, 'gpu': 1}),
            tune_config=tune.TuneConfig(mode='max', metric='best_val_irr_1', search_alg=search_alg, scheduler=scheduler, num_samples=100),
            run_config=train.RunConfig(name=exp_name, storage_path=directory, failure_config=train.FailureConfig(max_failures=2), 
                                       checkpoint_config=train.CheckpointConfig(num_to_keep=1)),
            param_space=param_space,
        )
    results = tuner.fit()
    best_result = results.get_best_result()

    print(f'Best trial config: {best_result.config}')
    print(f'Best trial val irr_1: {best_result.metrics["best_val_irr_1"]}')
