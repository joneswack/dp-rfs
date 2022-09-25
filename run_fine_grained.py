import os
import argparse
import time
import copy
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models

from models.kernel_pooling import CNNKernelPooling, extract_random_patches

# from util.load_cub_data import CUB200, CUB200ReLU

import util.data


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset_config', type=str, required=True,
    #                     help='Path to dataset configuration file')
    parser.add_argument('--model_name', type=str, required=False, default='kernel_pooling_model',
                            help='Name of the model to be saved')
    parser.add_argument('--bs', type=int, required=False, default=32,
                            help='Batch size')
    parser.add_argument('--finetune_epochs', type=int, required=False, default=100,
                            help='Number of epochs for finetuning')
    parser.add_argument('--pretrain_epochs', type=int, required=False, default=50,
                            help='Number of epochs for classifier pretraining')
    parser.add_argument('--pretrain_lr', type=float, required=False, default=0.01,
                            help='Learning rate for classifier pretraining')
    parser.add_argument('--num_samples', type=int, required=False, default=1000,
                            help='Number of samples for lengthscale, feature estimation')
    parser.add_argument('--ard', dest='ard', action='store_true')
    parser.set_defaults(ard=False)
    parser.add_argument('--use_gpu', dest='use_gpu', action='store_true')
    parser.set_defaults(use_gpu=False)

    args = parser.parse_args()

    return args

def makeDefaultTransforms(img_crop_size=448):
    data_transforms = {
        'train': transforms.Compose([
            # shorter edge is resized and aspect ratio is kept
            # TODO: we may consider changing this for torchvision pretrained transforms
            # transforms.RandomResizedCrop(size=448, scale=(0.8, 1.0)),
            
            transforms.Resize(img_crop_size),
            # extracts a square crop
            transforms.RandomCrop(img_crop_size, padding=0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Normalize([0.485, 0.456, 0.406], [1.0, 1.0, 1.0])
        ]),
        'test': transforms.Compose([
            # transforms.Resize(size=(448, 448)),
            
            transforms.Resize(img_crop_size),
            # horizontal flip is left out for TTA
            transforms.CenterCrop(img_crop_size),
            transforms.ToTensor(),
            # the values are obtained from the training set using test transforms
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Normalize([0.485, 0.456, 0.406], [1.0, 1.0, 1.0])
        ]),
    }

    return data_transforms



if __name__ == '__main__':
    configurations = [

    ]

    args = parse_args()

    log_handler = util.data.Log_Handler('kernel_pooling', 'closed_form_benchmark_test')
    csv_handler = util.data.DF_Handler('kernel_pooling', 'closed_form_benchmark_test')

    # print('Comparing approximations...')

    # images_root = '/mnt/workspace/blinear-cnn-faster/data/cub200'
    data_dir = '../datasets/export/CUB_200_2011/images'

    # # Get data transforms
    data_transforms = makeDefaultTransforms(img_crop_size=448)

    pre_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms['test'])
                    for x in ['train', 'test']}
    pre_dataloaders = {x: torch.utils.data.DataLoader(pre_datasets[x], batch_size=args.bs,
                                                shuffle=False, num_workers=0)
                for x in ['train', 'test']}
    # ft_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    #                 for x in ['train', 'test']}
    # ft_dataloaders = {
    #     'train': torch.utils.data.DataLoader(ft_datasets['train'], batch_size=args.bs,
    #                                             shuffle=True, num_workers=0),
    #     'test': torch.utils.data.DataLoader(ft_datasets['test'], batch_size=args.bs,
    #                                             shuffle=False, num_workers=0)
    # }
    dataset_sizes = {x: len(pre_datasets[x]) for x in ['train', 'test']}
    # class_names = pre_datasets['train'].classes

    print('Number of data')
    print('========================================')
    for dataset in dataset_sizes.keys():
        print(dataset,' size:: ', dataset_sizes[dataset],' images')

    print('')
    # print('Number of classes:: ', len(class_names))
    print('========================================')

    ## Extract conv features
    # conv_feature_path = os.path.join('saved_models', '{}_conv53.pth'.format(args.model_name))
    # conv_feature_path = os.path.join('saved_models', 'full_model_new_conv53.pth')
    # TODO: remove slim model
    conv_feature_path = os.path.join('saved_models', 'full_model_conv53_slim.pth')
    # pool_sketch_path = os.path.join('saved_models', '{}_poolsketch.pth'.format(args.model_name))

    if not os.path.isfile(conv_feature_path):
        outputs = {'train': [[], []], 'test': [[], []]}
        
        for phase in ['train', 'test']:
            print('Extracting {} features...'.format(phase))
            for i, (inputs, labels) in enumerate(pre_dataloaders[phase]):
                print('Processing batch {} / {}'.format(i+1, len(pre_dataloaders[phase])))
                if args.use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                with torch.no_grad():
                    outputs[phase][0].append(model.extract_pool_features(inputs).cpu())
                    outputs[phase][1].append(labels.cpu())

            outputs[phase][0] = torch.cat(outputs[phase][0], dim=0)
            outputs[phase][1] = torch.cat(outputs[phase][1], dim=0)

        torch.save(outputs, conv_feature_path)
        # save random projection from poolsketch
        # torch.save(
        #     model.pool_sketch.cpu().state_dict(),
        #     pool_sketch_path
        # )
        print('Extraction successful!')

    precomputed_features = torch.load(conv_feature_path, map_location='cpu')
    random_patches = extract_random_patches(precomputed_features['train'][0], args.num_samples)

    for config in configurations:
        for D in [512, 1024, 2048, 4096]:
            for seed in range(1):
            # for seed in range(5):
                torch.manual_seed(seed)
                np.random.seed(seed)

                model = CNNKernelPooling(
                    models.vgg16(pretrained=True), d=512, D=D,
                    degree=config['degree'], n_classes=200,
                    estimate_lengthscale=config['lengthscale'],
                    method=config['method'], fit_lengthscale=args.ard,
                    proj=config['proj'], hierarchical=config['hierarchical'],
                    bias=config['bias'], sqrt=config['sqrt'], norm=config['norm'],
                    complex_weights=config['comp']
                )

                if args.use_gpu:
                    model.cuda()
                    # model.pool_sketch.move_submodules_to_cuda()
                    random_patches = random_patches.cuda()

                classifier = torch.nn.Linear(4096, 200)
                torch.nn.init.constant_(classifier.bias, val=0.0)
                criterion = nn.CrossEntropyLoss()
                
                with torch.no_grad():
                    model.estimate_lengthscale_and_features(random_patches)
                    model.pool_sketch.resample()
                
                if args.use_gpu:
                    model.cuda()
                    if config['method'] != 'rff':
                        model.pool_sketch.move_submodules_to_cuda()
                    classifier.cuda()
                    criterion.cuda()


                ###### Closed form solution
                # train_labels = precomputed_features['train'][1].view(-1).type(torch.LongTensor)
                # test_labels = precomputed_features['test'][1].view(-1).type(torch.LongTensor)
                # train_labels = torch.nn.functional.one_hot(train_labels).type(torch.FloatTensor)
                # test_labels = torch.nn.functional.one_hot(test_labels).type(torch.FloatTensor)
                    
                # # transform 0 into -1
                # train_labels[train_labels == 0] = -1
                # test_labels[test_labels == 0] = -1

                # # get the projections
                # # train
                # # model.log_lengthscale.data = torch.ones_like(model.log_lengthscale.data) * log_lengthscale
                # # print('Log-Lengthscale: {}'.format(log_lengthscale))
                # with torch.no_grad():
                #     train_features = []
                #     test_features = []
                #     for x, y in torch.utils.data.TensorDataset(*precomputed_features['train']):
                #         x = x.unsqueeze(0)
                #         if args.use_gpu:
                #             proj = model.forward(x.cuda(), finetuning=False)
                #         else:
                #             proj = model.forward(x, finetuning=False)
                #         train_features.append(proj)
                #     for x, y in torch.utils.data.TensorDataset(*precomputed_features['test']):
                #         x = x.unsqueeze(0)
                #         if args.use_gpu:
                #             proj = model.forward(x.cuda(), finetuning=False)
                #         else:
                #             proj = model.forward(x, finetuning=False)
                #         test_features.append(proj)

                #     train_features = torch.cat(train_features, dim=0)
                #     test_features = torch.cat(test_features, dim=0)

                #     if args.use_gpu:
                #         train_labels = train_labels.cuda()
                #         test_labels = test_labels.cuda()

                #     for wd in [10**i for i in range(-6, 2)]:
                #         try:
                #             accuracy, _ = solve_linear_regression(
                #                 train_features, test_features, train_labels, test_labels,
                #                 penalty=wd, complex_weights=config['comp'])
                #             # print('Weight decay: {}, Accuracy: {}'.format(wd, accuracy))
                #         except Exception as e:
                #             log_handler.append('Weight decay: {}: Oops! Exception occured! {}'.format(wd, e.__class__))
                #             log_handler.append('NaN? {}'.format(train_features.norm().item()))
                #             # evs, _ = torch.eig(train_features.t() @ train_features, eigenvectors=False)
                #             # log_handler.append('Min ev: {}, Max ev: {}'.format(evs[:, 0].min().item(), evs[:, 0].max().item()))
                #             accuracy = 0

                #         config['weight_decay'] = wd
                #         config['D'] = D
                #         config['reg_acc'] = accuracy
                #         config['seed'] = seed
                #         log_handler.append(str(config))
                #         csv_handler.append(config)
                #         csv_handler.save()

    # exit()

    clf_dataloaders = {
        x: torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(*precomputed_features[x]),
                batch_size=args.bs,
                shuffle=True, num_workers=0
        ) for x in ['train', 'test']}

    # # Observe that all parameters are being optimized
    # fine-tuning starts with lr=0.001
    # pre_optimizer = optim.SGD(model.parameters(), lr=1.0, momentum=0.9, weight_decay=1e-5) # 0.1
    # pre_optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
    pre_optimizer = optim.Adam(classifier.parameters(), lr=args.pretrain_lr, weight_decay=1e-5) # , weight_decay=1e-5
    # pre_optimizer = optim.SGD(model.parameters(), lr=1.0, momentum=0.9, weight_decay=1e-5)
    # we use 0.01 instead of 0.001 because normalization is different
    # fine_optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
    fine_optimizer = optim.Adam(model.parameters(), lr=0.1 * args.pretrain_lr, weight_decay=1e-5)
    # fine_optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)

    # # Decay LR by a factor of 0.1 every 7 epochs
    # scheduler = lr_scheduler.MultiStepLR(
    #     fine_optimizer,
    #     # we divide the learning rate by 10 after the pretraining and after every 30 epochs thereafter
    #     [i*30 for i in range(1, 10)], # [args.pretrain_epochs] + args.pretrain_epochs+
    #     gamma=0.1, last_epoch=-1)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    pre_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        pre_optimizer, mode='min', factor=0.1, patience=8, verbose=True, threshold=1e-4)
    fine_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        fine_optimizer, mode='min', factor=0.1, patience=8, verbose=True, threshold=1e-4)

    # # Train the model
    since = time.time()

    best_acc = 0.0
    
    history = {'epoch' : [], 'train_loss' : [], 'test_loss' : [], 'train_acc' : [], 'test_acc' : []}

    total_epochs = args.pretrain_epochs + args.finetune_epochs
    for epoch in range(total_epochs):
        print('Epoch {}/{}'.format(epoch + 1, total_epochs))
        print('-' * 10)
        epoch_time = time.time()

        if (epoch + 1 > args.pretrain_epochs):
            finetuning = True
            optimizer = fine_optimizer
            if model.pool_sketch.log_bias is not None:
                model.pool_sketch.log_bias.requires_grad = False
            model.pool_sketch.log_lengthscale.requires_grad = False
            print('Phase: Finetuning')
        else:
            finetuning = False
            optimizer = pre_optimizer
            if model.pool_sketch.log_bias is not None:
                model.pool_sketch.log_bias.requires_grad = args.ard
            model.pool_sketch.log_lengthscale.requires_grad = args.ard
            print('Phase: Pretraining')

        print('LR: {}'.format(optimizer.param_groups[0]["lr"]))

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            epoch_losses = []
            num_correct = 0
            num_total = 0

            # Iterate over data.
            dl = ft_dataloaders[phase] if finetuning else clf_dataloaders[phase]
            for inputs, labels in dl:
                if args.use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # tic = time.time()
                    outputs = model.forward(inputs, finetuning=finetuning)
                    outputs = classifier.forward(outputs)
                    # print('Forward time: {}'.format(time.time() - tic))
                    if phase == 'test' and finetuning:
                        # TTA
                        # flip along width of [batch_size, channels, height, width]
                        # TODO: is it better to place TTA after the softmax?
                        # averaging RELATIVE confidence might be better than ABSOLUTE confidence
                        outputs2 = torch.flip(inputs, [3])
                        outputs2 = model.forward(outputs2, finetuning=finetuning)
                        outputs2 = classifier.forward(outputs2)
                        outputs = (
                            torch.nn.functional.softmax(outputs, dim=1) + \
                            torch.nn.functional.softmax(outputs2, dim=1)
                        ) / 2.
                        loss = torch.nn.functional.nll_loss(torch.log(outputs), labels)
                        # outputs = (outputs + outputs2) / 2.
                    else:
                        loss = criterion(outputs, labels)
                    epoch_losses.append(loss.item())
                    preds = torch.argmax(outputs, dim=1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        # tic = time.time()
                        loss.backward()
                        # print('Backward time: {}'.format(time.time() - tic))
                        # torch.nn.utils.clip_grad_value_(model.parameters(), 1)
                        optimizer.step()

                # statistics
                num_total += labels.size(0)
                num_correct += torch.sum(preds == labels).item()

            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            epoch_acc = num_correct / num_total

            if phase == 'train':
                pre_scheduler.step(epoch_loss)
                if finetuning:
                    fine_scheduler.step(epoch_loss)
            
            history['epoch'].append(epoch)
            history[phase+'_loss'].append(epoch_loss)
            history[phase+'_acc'].append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'saved_models/{}.pth'.format(args.model_name))
                # best_model_wts = copy.deepcopy(model.state_dict())

        
        print('Epoch time: {:2f}'.format(time.time() - epoch_time))

        with open('logs/kernel_pooling/{}.pkl'.format(args.model_name),'wb') as f:
            pickle.dump(history, f)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # # load best model weights
    # print('Returning object of best model.')
    # model.load_state_dict(best_model_wts)

    # # TODO: pickle best model
    # torch.save(model.state_dict(), 'saved_models/{}.pth'.format(args.model_name))