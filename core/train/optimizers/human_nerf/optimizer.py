# import torch.optim as optim

# from configs import cfg

# _optimizers = {
#     'adam': optim.Adam
# }

# def get_customized_lr_names():
#     return [k[3:] for k in cfg.train.keys() if k.startswith('lr_')]

# def get_optimizer(network):
#     optimizer = _optimizers[cfg.train.optimizer]

#     cus_lr_names = get_customized_lr_names()
#     params = []
#     print('\n\n********** learnable parameters **********\n')
#     for key, value in network.named_parameters():
#         if not value.requires_grad:
#             continue

#         is_assigned_lr = False
#         for lr_name in cus_lr_names:
#             if lr_name in key:
#                 params += [{"params": [value], 
#                             "lr": cfg.train[f'lr_{lr_name}'],
#                             "name": lr_name}]
#                 print(f"{key}: lr = {cfg.train[f'lr_{lr_name}']}")
#                 is_assigned_lr = True

#         if not is_assigned_lr:
#             params += [{"params": [value], 
#                         "name": key}]
#             print(f"{key}: lr = {cfg.train.lr}")

#     print('\n******************************************\n\n')

#     if cfg.train.optimizer == 'adam':
#         optimizer = optimizer(params, lr=cfg.train.lr, betas=(0.9, 0.999))
#     else:
#         assert False, "Unsupported Optimizer."
        
#     return optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from configs import cfg

_optimizers = {
    'adam': optim.Adam
}

def get_customized_lr_names():
    return [k[3:] for k in cfg.train.keys() if k.startswith('lr_')]

def get_optimizer(network):
    optimizer = _optimizers[cfg.train.optimizer]

    cus_lr_names = get_customized_lr_names()
    params = []
    print('\n\n********** learnable parameters **********\n')
    for key, value in network.named_parameters():
        if not value.requires_grad:
            continue

        is_assigned_lr = False
        for lr_name in cus_lr_names:
            if lr_name in key:
                params += [{"params": [value],
                    "lr": cfg.train[f'lr_{lr_name}'],
                    "initial_lr": cfg.train[f'lr_{lr_name}'],  # Added this line
                    "name": lr_name}]
                print(f"{key}: lr = {cfg.train[f'lr_{lr_name}']}")
                is_assigned_lr = True
        if not is_assigned_lr:
            params += [{"params": [value],
                "initial_lr": cfg.train.lr,  # Added this line
                "name": key}]
            print(f"{key}: lr = {cfg.train.lr}")

    print('\n******************************************\n\n')

    if cfg.train.optimizer == 'adam':
        optimizer = optimizer(params, lr=cfg.train.lr, betas=(0.9, 0.999))
    else:
        assert False, "Unsupported Optimizer."
        
    # Create scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=cfg.train.get('scheduler_patience', 10),
        verbose=True,
        min_lr=cfg.train.get('min_lr', 0.00001)
    )
    
    return optimizer, scheduler  