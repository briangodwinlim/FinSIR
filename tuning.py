import os
import torch
import tempfile
from ray import train, tune
from ray.tune.search.basic_variant import BasicVariantGenerator
# from ray.tune.search.optuna import OptunaSearch
# from ray.tune.search import ConcurrencyLimiter
# from ray.tune.schedulers import ASHAScheduler

import train as tr
from model import FinSIRModel as Model


class Arguments:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def run(config):    
    args = Arguments(**config, nworkers=2, use_amp=True, return_period=1, add_self_loop=True,
                     recurrent_layers=1, recurrent_dropout=0, relational_agg='sym', relational_dropout=0, 
                     readout_layers=1, readout_dropout=0, lr=1e-3, wd=1e-5, l1=0, l2=0, factor=0.5, patience=50, k_list=[1, 5])
    # args.corr_graph_periods = [int(p) for p in args.corr_graph_periods.split(' ')]
    
    # Change directory
    os.chdir(base_path)
    
    # Set seed
    tr.set_seed(0)

    # Load dataset
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dataset, train_loader, val_loader, test_loader = tr.load_dataset(args.market, args)
    
    # Extract input shapes
    input_dim = dataset[0][1].shape[-1]
    edge_dim = {'wiki': dataset.wiki_graph.edata['feat'].shape[-1], 
                'industry': dataset.industry_graph.edata['feat'].shape[-1]}[args.relational_graph]
                # 'correlation': len(args.corr_graph_periods) + 1
    
    # Load model
    model = Model(input_dim, edge_dim, args.nhidden, 1, args.recurrent_layers, args.recurrent_dropout, 
                  args.relational_agg, args.relational_dropout, args.readout_layers, args.readout_dropout).to(device)
    # model = torch.compile(model, mode='default')
    
    # Scaler + optimizer + scheduler
    scaler = torch.amp.GradScaler(device=device.type, enabled=args.use_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience)
    best_val_loss = 1e10

    # Restore checkpoint
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            model_state, scaler_state, optimizer_state, scheduler_state = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pt'))
            model.load_state_dict(model_state)
            scaler.load_state_dict(scaler_state)
            optimizer.load_state_dict(optimizer_state)
            scheduler.load_state_dict(scheduler_state)

    for epoch in range(50):
        tr.warmup_lr(optimizer, args.lr, epoch + 1, 10)
        loss = tr.train(model, dataset, train_loader, device, optimizer, scaler, args)
        mse, loss, mrr, irr = tr.evaluate(model, dataset, train_loader, device, args)
        val_mse, val_loss, val_mrr, val_irr = tr.evaluate(model, dataset, val_loader, device, args)
        test_mse, test_loss, test_mrr, test_irr = tr.evaluate(model, dataset, test_loader, device, args)
        scheduler.step(loss)

        # Process irr
        irr = {k: sum(irr_k) for k, irr_k in irr.items()}
        val_irr = {k: sum(irr_k) for k, irr_k in val_irr.items()}
        test_irr = {k: sum(irr_k) for k, irr_k in test_irr.items()}

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            result = {
                'val_mse': val_mse,
                'val_mrr': val_mrr,
                'val_irr': val_irr,
                'val_loss': val_loss,
                'test_mse': test_mse,
                'test_mrr': test_mrr,
                'test_irr': test_irr,
                'test_loss': test_loss,
            }

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            torch.save((model.state_dict(), scaler.state_dict(), optimizer.state_dict(), scheduler.state_dict()), os.path.join(checkpoint_dir, 'checkpoint.pt'))
            checkpoint = train.Checkpoint.from_directory(checkpoint_dir)
            train.report({'mse': mse, 'loss': loss, 'mrr_1': mrr[1], 'mrr_5': mrr[5], 'irr_1': irr[1], 'irr_5': irr[5],
                          'val_mse': val_mse, 'val_loss': val_loss, 'val_mrr_1': val_mrr[1], 'val_mrr_5': val_mrr[5], 'val_irr_1': val_irr[1], 'val_irr_5': val_irr[5],
                          'test_mse': test_mse, 'test_loss': test_loss, 'test_mrr_1': test_mrr[1], 'test_mrr_5': test_mrr[5], 'test_irr_1': test_irr[1], 'test_irr_5': test_irr[5],
                          'best_val_mse': result['val_mse'], 'best_val_loss': result['val_loss'], 'best_val_irr_1': result['val_irr'][1], 'best_val_irr_5': result['val_irr'][5],
                          'best_test_mse': result['test_mse'], 'best_test_loss': result['test_loss'], 'best_test_irr_1': result['test_irr'][1], 'best_test_irr_5': result['test_irr'][5]}, checkpoint=checkpoint)


if __name__ == '__main__':
    market = 'NYSE'
    relational_graph = 'wiki'
    param_space = {
        'market': tune.grid_search([market]),                       # FIXED
        'relational_graph': tune.grid_search([relational_graph]),   # FIXED
        'sequence_length': tune.grid_search([2, 4, 8, 16]),
        'nhidden': tune.grid_search([8, 16, 32, 64]),
        'alpha': tune.grid_search([0.1, 1.0, 10.0]),
    }
    
    hyperparams = [
    ]
    
    base_path = '/home/FinSIR/'
    directory = os.path.join(base_path, 'logs/tune/')
    exp_name = f'{market}_{Model.__name__}_{relational_graph}'
    
    search_alg = BasicVariantGenerator(points_to_evaluate=hyperparams, max_concurrent=4)
    scheduler = None
    # search_alg = OptunaSearch(points_to_evaluate=hyperparams)
    # search_alg.restore_from_dir(os.path.join(directory, exp_name))
    # search_alg = ConcurrencyLimiter(search_alg, max_concurrent=6)
    # scheduler = ASHAScheduler(max_t=50, grace_period=50)
    
    if tune.Tuner.can_restore(os.path.join(directory, exp_name)):
        tuner = tune.Tuner.restore(
            path=os.path.join(directory, exp_name), 
            trainable=tune.with_resources(tune.with_parameters(run), resources={'cpu': 6, 'gpu': 1/4, 'accelerator_type:RTX': 1/4}),
            resume_unfinished=True,
        )
    else:
        tuner = tune.Tuner(
            trainable=tune.with_resources(tune.with_parameters(run), resources={'cpu': 6, 'gpu': 1/4, 'accelerator_type:RTX': 1/4}),
            tune_config=tune.TuneConfig(mode='max', metric='best_val_irr_1', search_alg=search_alg, scheduler=scheduler, num_samples=1),
            run_config=train.RunConfig(name=exp_name, storage_path=directory, failure_config=train.FailureConfig(max_failures=2), 
                                       checkpoint_config=train.CheckpointConfig(num_to_keep=1)),
            param_space=param_space,
        )
    results = tuner.fit()
    best_result = results.get_best_result()

    print(f'Best trial config: {best_result.config}')
    print(f'Best trial test irr_1: {best_result.metrics["best_test_irr_1"]}')
