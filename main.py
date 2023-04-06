import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch import nn
from models.SSRNET import SSRNET
from models.SingleCNN import SpatCNN, SpecCNN
from models.TFNet import TFNet, ResTFNet
from models.SSFCNN import SSFCNN, ConSSFCNN
from models.MSDCNN import MSDCNN
from models.MCT import MCT
from utils import *
from data_loader import build_datasets
from validate import validate
from train import train
import pdb
import args_parser
from torch.nn import functional as F


args = args_parser.args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print (args)


def main():
    # Custom dataloader
    train_list, test_list = build_datasets(args.root,
                                           args.dataset,
                                           args.image_size,
                                           args.n_select_bands,
                                           args.scale_ratio)
    if args.dataset == 'PaviaU':
      args.n_bands = 103
    elif args.dataset == 'Pavia':
      args.n_bands = 102
    elif args.dataset == 'Botswana':
      args.n_bands = 145
    elif args.dataset == 'KSC':
      args.n_bands = 176
    elif args.dataset == 'Urban':
      args.n_bands = 162
    elif args.dataset == 'IndianP':
      args.n_bands = 200
    elif args.dataset == 'Washington':
      args.n_bands = 191
    elif args.dataset == 'MUUFL_HSI':
      args.n_bands = 64
    elif args.dataset == 'salinas_corrected':
      args.n_bands = 204
    elif args.dataset == 'Houston_HSI':
      args.n_bands = 144
    # Build the models
    if args.arch == 'SSFCNN':
      model = SSFCNN(args.scale_ratio,
                     args.n_select_bands,
                     args.n_bands).cuda()
    elif args.arch == 'ConSSFCNN':
      model = ConSSFCNN(args.scale_ratio,
                        args.n_select_bands,
                        args.n_bands).cuda()
    elif args.arch == 'TFNet':
      model = TFNet(args.scale_ratio,
                    args.n_select_bands,
                    args.n_bands).cuda()
    elif args.arch == 'ResTFNet':
      model = ResTFNet(args.scale_ratio,
                       args.n_select_bands,
                       args.n_bands).cuda()
    elif args.arch == 'MSDCNN':
      model = MSDCNN(args.scale_ratio,
                     args.n_select_bands,
                     args.n_bands).cuda()
    elif args.arch == 'SSRNET' or args.arch == 'SpatRNET' or args.arch == 'SpecRNET':
      model = SSRNET(args.arch,
                     args.scale_ratio,
                     args.n_select_bands,
                     args.n_bands,
                     ).cuda()
    elif args.arch == 'SpatCNN':
      model = SpatCNN(args.scale_ratio,
                     args.n_select_bands,
                     args.n_bands).cuda()
    elif args.arch == 'SpecCNN':
      model = SpecCNN(args.scale_ratio,
                     args.n_select_bands,
                     args.n_bands).cuda()
    elif args.arch == 'MCT' :
      model = MCT(args.arch,
                     args.scale_ratio,
                     args.n_select_bands,
                     args.n_bands,
                     ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    parameter_nums = sum(p.numel() for p in model.parameters())
    print("Model size:", str(float(parameter_nums / 1e6)) + 'M')
    # Load the trained model parameters
    model_path = args.model_path.replace('dataset', args.dataset) \
                                .replace('arch', args.arch)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
        print ('Load the chekpoint of {}'.format(model_path))
        recent_psnr = validate(test_list,
                                args.arch,
                                model,
                                0,
                                args.n_epochs)
        print ('psnr: ', recent_psnr)

    # Loss and Optimizer
    criterion = nn.MSELoss().cuda()


    best_psnr = 0
    best_psnr = validate(test_list,
                          args.arch,
                          model,
                          0,
                          args.n_epochs)
    print ('psnr: ', best_psnr)

    # Epochs
    print ('Start Training: ')
    best_epoch = 0
    for epoch in range(args.n_epochs):
        # One epoch's traininginceptionv3
        print ('Train_Epoch_{}: '.format(epoch))
        train(train_list,
              args.image_size,
              args.scale_ratio,
              args.n_bands,
              args.arch,
              model,
              optimizer,
              criterion,
              epoch,
              args.n_epochs)

        # One epoch's validation
        print ('Val_Epoch_{}: '.format(epoch))
        recent_psnr = validate(test_list,
                                args.arch,
                                model,
                                epoch,
                                args.n_epochs)
        print ('psnr: ', recent_psnr)

        # # save model
        is_best = recent_psnr > best_psnr
        best_psnr = max(recent_psnr, best_psnr)
        # if epoch > 9000 and epoch % 50 == 0:
        #     model_path_ = model_path.split('.pkl')[0] + 'ep' + str(epoch) + '.pkl'
        #     print(model_path_)
        #     torch.save(model.state_dict(), model_path_)
        if is_best:
          best_epoch=epoch
          if best_psnr > 0:
            torch.save(model.state_dict(), model_path)
            print ('Saved!')
            print ('')
        print('best psnr:', best_psnr, 'at epoch:', best_epoch)

    print ('best_psnr: ', best_psnr)

if __name__ == '__main__':
    main()
