import argparse


parser = argparse.ArgumentParser(description="SSDAN")


# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='2')

# Data specifications
parser.add_argument('--data_root', type=str, default='/home/dataset-local/wbf/CVPR2026data/NITRE2022', help='dataset directory')
parser.add_argument('--data_name', type=str, default='ARAD', help='dataset directory')


# Saving specifications
parser.add_argument('--outf', type=str, default='./exp/', help='saving_path')
parser.add_argument('--model_name', type=str, default='DAMP', help='model name')

# Training specifications
parser.add_argument('--batch_size', type=int, default=2, help='the number of HSIs per batch')
parser.add_argument("--max_epoch", type=int, default=10000, help='total epoch')
parser.add_argument("--scheduler", type=str, default='CosineAnnealingLR', help='MultiStepLR or CosineAnnealingLR')
parser.add_argument("--milestones", type=int, default=[50,100,150,200,250], help='milestones for MultiStepLR')
parser.add_argument("--gamma", type=float, default=0.5, help='learning rate decay for MultiStepLR')
parser.add_argument("--learning_rate", type=float, default=0.0001)

# Model specifications
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')
parser.add_argument('--dimention', type=int, default=64, help='feature dimention')
parser.add_argument('--num_blocks', nargs='+', type=int, default=[1, 2, 2, 3])
parser.add_argument('--num_dec_blocks', nargs='+', type=int, default=[1, 2, 2])
parser.add_argument('--num_exp_blocks', type=int, default=4) 
parser.add_argument('--num_refinement_blocks', type=int, default=2)
parser.add_argument('--heads', nargs='+', type=int, default=[2, 2, 2, 4])
parser.add_argument('--with_complexity', action="store_true")
parser.add_argument('--complexity_scale', type=str, default="max")
parser.add_argument('--topk', type=int, default=1)
parser.add_argument('--expansion_factor', type=int, default=4)
parser.add_argument('--LayerNorm_type', type=str, default="WithBias")
parser.add_argument('--emb_dim', type=int, default=128)
parser.add_argument('--lambda_diversity', type=float, default=0.0001)
parser.add_argument('--crop_size', type=int, default=128)
parser.add_argument('--deg_index', type=int, default=-1)
parser.add_argument('--resume', type=int, default=1)

opt = parser.parse_args()

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False
