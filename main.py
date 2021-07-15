import argparse
from utils import load_config, get_log_name, set_seed, save_results, \
                    plot_results, get_test_acc, print_config
from datasets import cifar_dataloader
import algorithms
import numpy as np
import nni

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, default='./configs/standardCE.py',
                    help='The path of config file.')
args = parser.parse_args()

def main():
    tuner_params = nni.get_next_parameter()
    config = load_config(args.config, _print=False)
    config.update(tuner_params)
    print_config(config)
    set_seed(config['seed'])
    
    if config['algorithm'] == 'colearning':
        model = algorithms.Colearning(config, input_channel=config['input_channel'], num_classes=config['num_classes'])
        train_mode = 'train'
    else:
        model = algorithms.__dict__[config['algorithm']](config, input_channel=config['input_channel'], num_classes=config['num_classes'])
        train_mode = 'train_single'

    dataloaders = cifar_dataloader(cifar_type=config['dataset'], root=config['root'], batch_size=config['batch_size'], 
                        num_workers=config['num_workers'], noise_type=config['noise_type'], percent=config['percent'])

    trainloader, testloader = dataloaders.run(mode=train_mode), dataloaders.run(mode='test')
    num_test_images = len(testloader.dataset)

    epoch = 0
    # evaluate models with random weights
    test_acc = get_test_acc(model.evaluate(testloader))
    print('Epoch [%d/%d] Test Accuracy on the %s test images: %.4f' % (
            epoch + 1, config['epochs'], num_test_images, test_acc))

    acc_list, acc_all_list = [], []
    best_acc, best_epoch = 0.0, 0
    
    for epoch in range(1, config['epochs']):
        # train
        model.train(trainloader, epoch)
        # evaluate 
        test_acc = get_test_acc(model.evaluate(testloader))
        nni.report_intermediate_result(test_acc)
        if best_acc < test_acc:
            best_acc, best_epoch = test_acc, epoch

        print('Epoch [%d/%d] Test Accuracy on the %s test images: %.4f %%' % (
                epoch + 1, config['epochs'], num_test_images, test_acc))

        if epoch >= config['epochs'] - 10:
            acc_list.extend([test_acc])
        acc_all_list.extend([test_acc])

    if config['save_result']:
        acc_np = np.array(acc_list)
        nni.report_final_result(acc_np.mean())
        jsonfile = get_log_name(args.config, config)
        np.save(jsonfile.replace('.json', '.npy'), np.array(acc_all_list))
        save_results(config=config, last_ten=acc_np, best_acc=best_acc, best_epoch=best_epoch, jsonfile=jsonfile)
        plot_results(epochs=config['epochs'], test_acc=acc_all_list, plotfile=jsonfile.replace('.json', '.png'))
    
if __name__ == '__main__':
    main()