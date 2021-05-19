import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=60, help="rounds of training")
    parser.add_argument('--factor', type=int, default=1, help="rounds of training")
    parser.add_argument('--ep_lr_decay', type=int, default=3, help="rounds of training")
    parser.add_argument('--cuda_device', type=int, default=2, help="cuda_device")
    parser.add_argument('--imshow', type=bool, default=False, help="show some training dataset")
    parser.add_argument('--model', type=str, default='carnet', help='model name')
    parser.add_argument('--path', type=str, default='/Users/julyc/PycharmProjects/parking_lot_occupancy_detection/weights/carnet_60_0.1.pth', help='trained model path')
    parser.add_argument('--train_img', type=str, default='/home/zengweijia/.jupyter/cnrpark/PKLot/PKLotSegmented',
                        help="path to training set images")
    parser.add_argument('--train_lab', type=str, default='/home/zengweijia/.jupyter/cnrpark/splits/PKLot/train_36.txt',
                        help="path to training set labels")
    parser.add_argument('--test_img', type=str, default='/home/zengweijia/.jupyter/cnrpark/PATCHES',
                        help="path to test set images")
    parser.add_argument('--test_lab', type=str, default='/home/zengweijia/.jupyter/cnrpark/splits/CNRPark-EXT/test.txt',
                        help="path to test set labels")
    # parser.add_argument('--train_img', type=str, default='/Users/julyc/Downloads/CNR-EXT-Patches-150x150/PATCHES',
    #                     help="path to training set images")
    # parser.add_argument('--train_lab', type=str, default='/Users/julyc/Downloads/splits/CNRPark-EXT/train.txt',
    #                     help="path to training set labels")
    # parser.add_argument('--test_img', type=str, default='/Users/julyc/Downloads/CNR-EXT-Patches-150x150/PATCHES',
    #                     help="path to test set images")
    # parser.add_argument('--test_lab', type=str, default='/Users/julyc/Downloads/splits/CNRPark-EXT/test.txt',
    #                     help="path to test set labels")
    parser.add_argument('--img_size', type=int, default=54,
                        help="carnet :54  malexnet=224")
    parser.add_argument('--split_path', type=str, default='/Users/julyc/PycharmProjects/parking_lot_occupancy_detection/data/carnet/pklot_trained',
                        help="path to training set labels")
    parser.add_argument('--local_rank', type=int, default=-1, help="path to training set labels")
    parser.add_argument('--eval_data', type=str,default='test',help="path to training set labels")
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')

    args = parser.parse_args()
    return args
