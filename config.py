from argparse import ArgumentParser

class config(object):
    def __init__(self, args):
        self.HEIGHT=96
        self.WIDTH=96
        self.CHANNEL=3
        self.BATCH_SIZE=32
        self.scale=2

        self.TF_RECORD_PATH=args.tfrecord
        self.GPU=args.gpu

parser=ArgumentParser()

parser.add_argument('--gpu', dest='gpu', type=str, default='0')
parser.add_argument('--tfrecord', dest='tfrecord', type=str, default='train_SR_X2.tfrecord')

args=parser.parse_args()