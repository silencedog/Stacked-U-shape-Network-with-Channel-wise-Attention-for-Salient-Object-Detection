import argparse
import os
from dataset import get_loader
from solver import Solver

def main(config):                                                
    
    if config.mode == 'train':
        train_loader = get_loader(config.train_path, config.label_path, config.img_size, config.batch_size, mode='train',
                                  filename=config.train_file, num_thread=config.num_thread)
        if config.val:
            val_loader = get_loader(config.val_path, config.val_label, config.img_size, config.test_size, mode='train',
                                    filename=config.val_file, num_thread=config.num_thread)
        run = 0
        while os.path.exists("%s/run-%d" % (config.save_fold, run)): run += 1
        os.mkdir("%s/run-%d" % (config.save_fold, run))
        os.mkdir("%s/run-%d/logs" % (config.save_fold, run))
        # os.mkdir("%s/run-%d/images" % (config.save_fold, run))
        os.mkdir("%s/run-%d/models" % (config.save_fold, run))
        config.save_fold = "%s/run-%d" % (config.save_fold, run)
        if config.val:
            train = Solver(train_loader, val_loader, None, config)
            #(train_loader, val_loader, test_dataset, config)
        else:
            train = Solver(train_loader, None, None, config)
        train.train()
    elif config.mode == 'test':
        test_loader = get_loader(config.test_path, config.test_label, config.img_size, config.test_size, mode='test',
                                 filename=config.test_file, num_thread=config.num_thread)
        if not os.path.exists(config.test_fold): os.mkdir(config.test_fold)
        test = Solver(None, None, test_loader, config)
        test.test(100, use_crf=config.use_crf)
    else:
        raise IOError("illegal input!!!")


if __name__   == '__main__':
    data_root = os.path.join(os.path.expanduser('~'), '/home/panzefeng/All_code/BDCN_salient_detection')   #目录位置
    #vgg_path  = '/home/panzefeng/All_code/BDCN_salient_detection/results/run-10/models/best.pth'           #pre-training pth
   
    # # -----dataset-----
    #DUTS-TE ECSSD PASCAL-S SOD HKU-IS MSRA10K MSRA-B  MSRA10K-TR   train DUTS-TR-AUG DUTS-TR-4
    dataname   = os.path.join('show')
    image_path = os.path.join(data_root, 'DATASET', dataname, 'images/')
    label_path = os.path.join(data_root, 'DATASET', dataname, 'annotation/')
    train_file = os.path.join(data_root, 'DATASET/DUTS-TR/annotation_path.txt') 
    valid_file = os.path.join(data_root, 'DATASET/DUTS-TR/valid_annotation_path.txt')
    test_file  = os.path.join(data_root, 'DATASET', dataname, 'test.txt')

    parser     = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color',       type=int,   default=3)
    parser.add_argument('--img_size',      type=int,   default=256)  # 256
    parser.add_argument('--lr',            type=float, default=5e-5) #8e-5 1e-6 5e-5 6e-5
    parser.add_argument('--wd',            type=float, default=0.0005)  # 8e-5 1e-6 5e-5 6e-5
    parser.add_argument('--clip_gradient', type=float, default=1.0)
    parser.add_argument('--cuda',          type=bool,  default=True)

    # Training settings
    #parser.add_argument('--vgg',         type=str, default=vgg_path)
    parser.add_argument('--train_path',  type=str, default=image_path)
    parser.add_argument('--label_path',  type=str, default=label_path)
    parser.add_argument('--train_file',  type=str, default=train_file)
    parser.add_argument('--epoch',       type=int, default=100)     #500
    parser.add_argument('--batch_size',  type=int, default=29)      # 8
    parser.add_argument('--test_size',   type=int, default=1)     
    parser.add_argument('--val',         type=bool,default=True)
    parser.add_argument('--val_path',    type=str, default=image_path)
    parser.add_argument('--val_label',   type=str, default=label_path)
    parser.add_argument('--val_file',    type=str, default=valid_file)
    parser.add_argument('--num_thread',  type=int, default=0)
    parser.add_argument('--load',        type=str, default='')
    parser.add_argument('--save_fold',   type=str, default='./results')
    parser.add_argument('--epoch_val',   type=int, default=1)   #10
    parser.add_argument('--epoch_save',  type=int, default=20)   #20
    parser.add_argument('--epoch_show',  type=int, default=1)
    parser.add_argument('--pre_trained', type=str, default=None)    #None
    # Testing settings
    parser.add_argument('--test_path',  type=str, default=image_path)
    parser.add_argument('--test_label', type=str, default=label_path)
    parser.add_argument('--test_file',  type=str, default=test_file)
    parser.add_argument('--model',      type=str, default='./results/这个版本的result比较好/models/best.pth')
    parser.add_argument('--test_fold',  type=str, default='./results/test')
    parser.add_argument('--pre_map',    type=str, default='./results/show')
    parser.add_argument('--use_crf',    type=bool,default=False)#FalseTrue

    # Misc
    parser.add_argument('--mode',   type=str,  default='test', choices=['train', 'test'])

    config = parser.parse_args()#config = parser.parse_args()  config.xxx  xxx就是索引名（‘红字’）
    
    if not os.path.exists(config.save_fold): os.mkdir(config.save_fold)
    main(config)
