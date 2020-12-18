from importlib import import_module
from utils import build_dataset
import torch
import numpy as np
from train import train


# model_name = args.model  # bert
if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    model_name = "model"
    x = import_module('models.' + model_name)
    config = x.Config()

    train_samples, dev_samples, test_samples = build_dataset(config)

    from utils import build_iterator
    train_iteration = build_iterator(train_samples, config)
    dev_iteration = build_iterator(dev_samples, config)
    test_samples = build_iterator(test_samples, config)

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iteration, dev_iteration, test_samples)

    print('FINISHED')

