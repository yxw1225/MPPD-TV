import argparse
import os
import sys
from models.VGG import VGG
from models.WideResNet import WideResNet
from models.resnet import ResNet
import data_loaders
from functions import *
from utils import val
import attack
import copy
import torch
import json
from data_loaders import cifar10, cifar100, mnist, tiny_imagenet
from tqdm import tqdm
import models.layers as layers
import io
import torch.nn as nn
from autoattack import AutoAttack


parser = argparse.ArgumentParser()
# just use default setting
parser.add_argument('-j','--workers',default=8, type=int,metavar='N',help='number of data loading workers')
parser.add_argument('-b','--batch_size',default=64, type=int,metavar='N',help='mini-batch size')
parser.add_argument('-sd', '--seed',default=42,type=int,help='seed for initializing training.')
parser.add_argument('-suffix','--suffix',default='', type=str,help='suffix')

# model configuration
parser.add_argument('-data', '--dataset', default='tiny-imagenet',type=str,help='dataset')
parser.add_argument('-arch','--model', default='vgg11', type=str,help='model')
parser.add_argument('-T','--time', default=8, type=int, metavar='N',help='snn simulation time')
parser.add_argument('-id', '--identifier', type=str, help='model statedict identifier')
parser.add_argument('-config', '--config', default='', type=str,help='test configuration file')

# training configuration
parser.add_argument('-dev','--device',default='0',type=str,help='device')
parser.add_argument('-ntype','--neuron_type',default='DLIFSpike',type=str)
parser.add_argument('-vth','--vth',default=1., type=float)
parser.add_argument('-tau','--tau',default=0.99, type=float)
parser.add_argument('-gama','--gama',default=1, type=float)

# adv atk configuration
parser.add_argument('-atk','--attack',default='cw', type=str,help='attack')
parser.add_argument('-eps','--eps',default=2, type=float, metavar='N',help='attack eps')
parser.add_argument('-atk_m','--attack_mode',default='', type=str,help='attack mode')

# only pgd
parser.add_argument('-alpha','--alpha',default=2.55/1,type=float,metavar='N',help='pgd attack alpha')
parser.add_argument('-steps','--steps',default=10,type=int,metavar='N',help='pgd attack steps')
parser.add_argument('-atk_ls', '--attack_loss', default='ce', type=str, metavar='N', help='attack loss')

# 新增CW攻击参数
parser.add_argument('-c', '--c', default=0.5, type=float, help='c for CW attack')
parser.add_argument('-kappa', '--kappa', default=0, type=float, help='kappa for CW attack')
parser.add_argument('-lr', '--lr', default=0.01, type=float, help='learning rate for CW attack')
parser.add_argument('-steps_cw','--steps_cw',default=10,type=int,help='cw attack steps')

args = parser.parse_args()



args.dt = 1
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device("cuda:%s" % args.device)


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

class SNNWrapper(nn.Module):
    def __init__(self, snn_model, T):
        super().__init__()
        self.snn_model = snn_model
        self.T = T

    def forward(self, images):
        return self.snn_model(images).mean(0)

def forward_function(model, image, T):
    output = model(image).mean(0)
    return output


def val(model, test_loader, device, atk):
    if atk is not None and not hasattr(atk, 'set_training_mode'):
        atk.set_training_mode = atk.set_model_training_mode

    correct = 0
    total = 0
    
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
        inputs = inputs.to(device)
        targets = targets.to(device)

        if atk is not None:
            atk.set_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
            inputs = atk(inputs, targets)

        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs).mean(0)

        _, predicted = outputs.cpu().max(1)

        current_correct = float(predicted.eq(targets.cpu()).sum().item())

        current_batch_size = float(targets.size(0))

        total += current_batch_size
        correct += current_correct

        print(f"Batch {batch_idx}: "
              f"BatchAcc={100 * current_correct / current_batch_size:.3f}% "
              f"(Correct: {current_correct:.0f} / BatchSize: {current_batch_size:.0f}) | "
              f"TotalAcc={100 * correct / total:.3f}%")

    final_acc = 100 * correct / total
    return final_acc



def main():
    global args
    if args.dataset.lower() == 'cifar10':
        num_labels = 10
        input_type = 'img'
        in_h = 32
        in_w = 32
        init_c = 3
        train_dataset, val_dataset, znorm = cifar10()
    elif args.dataset.lower() == 'cifar100':
        num_labels = 100
        input_type = 'img'
        in_h = 32
        in_w = 32
        init_c = 3
        train_dataset, val_dataset, znorm = cifar100()
    elif args.dataset.lower() == 'mnist':
        num_labels = 10
        input_type = 'img'
        in_h = 28
        in_w = 28
        init_c = 1
        train_dataset, val_dataset, znorm = mnist()
        znorm = None
    elif args.dataset.lower() == 'tiny-imagenet':
        num_labels = 200
        input_type = 'img'
        in_h = 64
        in_w = 64
        init_c = 3

        train_dataset, val_dataset, znorm = tiny_imagenet(dataroot='tiny-imagenet-200')

    log_dir = '%s-results_l1'% (args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    model_dir = '%s-checkpoints_l1'% (args.dataset)
    if not os.path.exists(os.path.join(model_dir, args.identifier + '.pth')):
        model_dir = '%s-staticcheckpoints'% (args.dataset)
    if not os.path.exists(os.path.join(model_dir, args.identifier + '.pth')):
        print('error')
        exit(0)
    
    log_name = '%s.log'%(args.identifier+args.suffix)
    if len(args.config)!=0:
        log_name = '[%s]' % args.config + log_name
    logger = get_logger(os.path.join(log_dir, log_name))
    logger.info('start testing!')

    seed_all(args.seed)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)

    NeuronFunction = layers.build_neuron_function(type=args.neuron_type, T=args.time, dt=args.dt,
                 thresh=args.vth, tau=args.tau, gama=args.gama)

    if 'vgg' in args.model.lower():
        model = VGG(vgg_name=args.model, T=args.time, dt=args.dt, num_classes=num_labels, norm=znorm,
                    neuron_module=NeuronFunction,
                    init_c=init_c, in_h=in_h, in_w=in_w, input_type=input_type)
    elif 'wrn' in args.model.lower():
        model = WideResNet(name=args.model, T=args.time, dt=args.dt, num_classes=num_labels, norm=znorm,
              neuron_module=NeuronFunction, 
              init_c=init_c, in_h=in_h, in_w=in_w, input_type=input_type)
    elif 'res' in args.model.lower():
        model = ResNet(resnet_name=args.model, T=args.time, dt=args.dt, num_classes=num_labels, norm=znorm,
              neuron_module=NeuronFunction, 
              init_c=init_c, in_h=in_h, in_w=in_w, input_type=input_type)
    else:
        raise AssertionError("model not supported")

    model.to(device)



    if len(args.config) > 0:
        with open(args.config+'.json', 'r') as f:
            config = json.load(f)
    else:
        config = [{}]
    for atk_config in config:
        for arg in atk_config.keys():
            setattr(args, arg, atk_config[arg])

        atkmodel = model

        ff = forward_function

        if args.attack.lower() == 'fgsm':
            atk = attack.FGSM(model, forward_function=ff, eps=args.eps / 255, T=args.time)
            atk.targeted = False
            atk._targeted = False
        elif args.attack.lower() == 'pgd':
            atk = attack.PGD(atkmodel, forward_function=ff, eps=args.eps / 255, alpha=args.alpha / 255, steps=args.steps, T=args.time)
            atk.targeted = False
            atk._targeted = False
        elif args.attack.lower() == 'apgd':
            atk = attack.APGD(atkmodel, forward_function=ff, eps=args.eps / 255, steps=args.steps, T=args.time, loss=args.attack_loss)
        elif args.attack.lower() == 'cw':
            atk = attack.CW(atkmodel, forward_function=ff, T=args.time, c=args.c, kappa=args.kappa, steps=args.steps_cw,
                            lr=args.lr)
            atk.targeted = False
            atk._targeted = False
        elif args.attack.lower() == 'autoattack':
            snn_wrapper = SNNWrapper(atkmodel, T=args.time)
            atk = AutoAttack(snn_wrapper, norm='Linf', eps=args.eps / 255.0, version='standard')
            atk.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
        else:
            atk = None
        try:
            current_out_features = model.classifier[0].out_features
            logger.info("Fixing classifier layer mismatch...")
            logger.info(f"Original classifier[0]: {model.classifier[0]}")

            model.classifier[0] = nn.Linear(in_features=512, out_features=current_out_features)
        
            logger.info(f"Replaced classifier[0]: {model.classifier[0]}")

            model.classifier[0].to(device)
            logger.info("Classifier fix applied successfully.")
        
        except Exception as e:
            logger.error(f"Failed to fix classifier: {e}")
            logger.error("Please check VGG model structure (`model.classifier[0]`)")
            pass
        checkpoint = torch.load(os.path.join(model_dir, args.identifier + '.pth'), map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        if args.attack.lower() == 'autoattack':
            logger.info(f"Preparing full validation set for AutoAttack...")

            x_test = torch.cat([x for x, y in test_loader], 0)
            y_test = torch.cat([y for x, y in test_loader], 0)

            logger.info(f"Running AutoAttack on the full validation set of {len(y_test)} images...")

            original_stdout = sys.stdout
            captured_output = io.StringIO()
            sys.stdout = Tee(original_stdout, captured_output)

            try:
                adv_complete = atk.run_standard_evaluation(x_test, y_test, bs=args.batch_size)
            finally:
                sys.stdout = original_stdout

            log_message = captured_output.getvalue()
            logger.info(f"AutoAttack Results for config: {json.dumps(atk_config)}\n{log_message}")

        else:
            acc = val(model, test_loader, device, atk)
            logger.info(json.dumps(atk_config) + ' Test acc={:.3f}'.format(acc))

if __name__ == "__main__":
    main()