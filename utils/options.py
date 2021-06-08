import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=50, help="rounds of training")
    parser.add_argument('--factor', type=int, default=1, help="rounds of training")
    parser.add_argument('--ep_lr_decay', type=int, default=3, help="rounds of training")
    parser.add_argument('--cuda_device', type=str, default="8", help="cuda_device")
    parser.add_argument('--imshow', type=bool, default=False, help="show some training dataset")
    parser.add_argument('--model', type=str, default='ShuffleNet_K5', help='model name')
    parser.add_argument('--path', type=str, default='/home/liuwei/code/carnet/weights/ShuffleNet_K5CNR-EXTPKLot.pth', help='trained model path')
    parser.add_argument('--train_img', type=str, default='/home/liuwei/code/PKLot/PKLotSegmented',
                        help="path to training set images")
    parser.add_argument('--train_lab', type=str, default='/home/liuwei/code/splits/PKLot/train_36.txt',
                        help="path to training set labels")
    # parser.add_argument('--test_img', type=str, default='/home/liuwei/code/PATCHES',
    #                     help="path to test set images")
    # parser.add_argument('--test_lab', type=str, default='/home/liuwei/code/splits/CNRPark-EXT/all.txt',
    #                     help="path to test set labels")
    # parser.add_argument('--train_img', type=str, default='/home/liuwei/code/PATCHES',
    #                     help="path to training set images")
    # parser.add_argument('--train_lab', type=str, default='/home/liuwei/code/splits/CNRPark-EXT/all.txt',
    #                     help="path to training set labels")
    parser.add_argument('--test_img', type=str, default='/home/liuwei/code/PATCHES',
                        help="path to test set images")
    parser.add_argument('--test_lab', type=str, default='/home/liuwei/code/splits/CNRPark-EXT/all.txt',
                        help="path to test set labels")





    parser.add_argument('--img_size', type=int, default=224,
                        help="carnet :54  malexnet=224")
    parser.add_argument('--split_path', type=str, default='/home/liuwei/code/carnet/split',
                        help="path to training set labels")
    parser.add_argument('--local_rank', type=int, default=-1, help="path to training set labels")
    parser.add_argument('--eval_data', type=str,default='mobilenet_small_test_33',help="path to training set labels")
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('--train_data',type=str,default='CNR-EXT')
    parser.add_argument('--test_data',type=str,default='PKLot')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')

    args = parser.parse_args()
    return args
