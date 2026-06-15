# Import utility modules
from utils import *
from options import opt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
import torch
import time
import numpy as np
import datetime

from loss import HybridLoss
# Data loader
from dataset import ImageTransformDataset, ImageTransformDataset_test
from torch.utils.data import DataLoader
from tqdm import tqdm

import json
from DAMP import DAMP
 


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')
 
          
train_set = ImageTransformDataset(root=opt.data_root + "/Train_hsi", mode='train', crop_size = opt.crop_size, deg_index=opt.deg_index)
train_loader = DataLoader(train_set, batch_size=opt.batch_size, drop_last=True, pin_memory=True,  num_workers=8, persistent_workers=True, shuffle = True)


test_loaders = []

if opt.deg_index >=0:# and opt.deg_index < len(train_set.deg_type):
    print(f"Using fixed degradation type for testing: {train_set.deg_type[0]}")
    test_set = ImageTransformDataset_test(root=opt.data_root + "/Valid_hsi", mode='test',deg_idx=opt.deg_index)
    test_loader = DataLoader(test_set, batch_size=1, drop_last=False, pin_memory=True,  num_workers=8, persistent_workers=True, shuffle = False)
    test_loaders.append(test_loader)
else:
    print("Using all degradation types for testing.")
    deg_nums = len(train_set.deg_type)
    for i in range(deg_nums):
        test_set = ImageTransformDataset_test(root=opt.data_root + "/Valid_hsi", mode='test',deg_idx=i)
        test_loader = DataLoader(test_set, batch_size=1, drop_last=False, pin_memory=True,  num_workers=8, persistent_workers=True, shuffle = False)
        test_loaders.append(test_loader)


# Set save path
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
model_path = opt.outf + opt.model_name + date_time + '/'
if not os.path.exists(model_path):
    os.makedirs(model_path)


config_dict = vars(opt)  # Convert Namespace to dictionary
config_path = os.path.join(model_path, 'config.json')

# Ensure the model directory exists
os.makedirs(model_path, exist_ok=True)

with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config_dict, f, indent=4, sort_keys=True)
print(f"Configuration saved to {config_path}")


model = DAMP(
            inp_channels=31, 
            out_channels=31, 
            dim = opt.dimention,
            levels=len(opt.num_blocks),
            heads=opt.heads,
            num_blocks=opt.num_blocks,
            num_dec_blocks=opt.num_dec_blocks,
            ffn_expansion_factor=opt.expansion_factor,
            num_refinement_blocks=opt.num_refinement_blocks,
            LayerNorm_type=opt.LayerNorm_type, ## Other option 'BiasFree'
            bias=False,
            num_experts=opt.num_exp_blocks,
            topk=opt.topk,
            with_complexity=opt.with_complexity,
            complexity_scale=opt.complexity_scale,
            emb_dim=opt.emb_dim
            ).cuda()

# Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate , betas=(0.9, 0.999))
if opt.scheduler=='MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
elif opt.scheduler=='CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=1e-6)


hy_loss = HybridLoss().cuda()



def train(epoch, logger, train_loader, model):
    model.train()  # Set model to training mode
    epoch_loss = 0
    begin = time.time()

    # Get total batch number from the shorter dataloader
    total_batches = len(train_loader)
    # Iterate over dataloader
    with tqdm(total=total_batches, desc=f"Epoch {epoch} Train", unit="batch") as pbar:
        for _, hsi, gt, rgb, deg_label in train_loader:
            # Unpack batch data
            hsi, gt, rgb, deg_label = hsi.cuda(), gt.cuda(), rgb.cuda(), deg_label.cuda()
            optimizer.zero_grad()
            # Forward propagation
            model_out = model(hsi)
            loss = hy_loss(model_out, gt)

            loss = torch.mean(loss) + 0.01 * model.total_loss

            # Accumulate loss
            epoch_loss += loss.item()

            # Backward propagation and parameter update
            loss.backward()
            optimizer.step()

            # Update progress bar
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            pbar.update(1)

    end = time.time()
    avg_loss = epoch_loss / total_batches
    logger.info("===> Epoch {} Complete: Avg. Loss: {:.6f}, Time: {:.2f}s".format(epoch, avg_loss, (end - begin)))

# Test function
def test(epoch, logger, test_loader, model):
    psnr_list, ssim_list, sam_list, rmse_list, ergas_list = [], [], [], [], []
    model.eval()
    begin = time.time()

    for _, hsi, gt, rgb, deg_label in test_loader:
        hsi, gt, rgb, deg_label = hsi.cuda(), gt.cuda(), rgb.cuda(), deg_label.cuda()
        with torch.no_grad():
            model_out = model(hsi)
        
        psnr_val = torch_psnr(model_out[0, :, :, :], gt[0, :, :, :])
        ssim_val = torch_ssim(model_out[0, :, :, :], gt[0, :, :, :])
        sam_val = torch_sam(model_out[0, :, :, :], gt[0, :, :, :])
        rmse_val = torch_rmse(model_out[0, :, :, :], gt[0, :, :, :])
        ergas_val = torch_ergas(model_out[0, :, :, :], gt[0, :, :, :])

        psnr_list.append(psnr_val.detach().cpu().numpy())
        ssim_list.append(ssim_val.detach().cpu().numpy())
        sam_list.append(sam_val.detach().cpu().numpy())
        rmse_list.append(rmse_val.detach().cpu().numpy())
        ergas_list.append(ergas_val.detach().cpu().numpy())
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    sam_mean = np.mean(np.asarray(sam_list))
    rmse_mean = np.mean(np.asarray(rmse_list))
    ergas_mean = np.mean(np.asarray(ergas_list))

    end = time.time()
    logger.info('===> Epoch {}: testing psnr = {:.2f}, ssim = {:.3f}, sam = {:.3f}, rmse = {:.4f}, ergas = {:.3f}, time: {:.2f}'
                .format(epoch, psnr_mean, ssim_mean,sam_mean,rmse_mean,ergas_mean,(end - begin)))
    model.train()
    return psnr_list, ssim_list, psnr_mean, ssim_mean

def main():
    logger = gen_log(model_path)
    logger.info("Learning rate:{}, batch_size:{}, dim:{}.\n".format(opt.learning_rate, opt.batch_size, opt.dimention))
    logger.info("Trainset:{}, Testset:{}.\n".format(len(train_set), len(test_set)))

    # Initialize best PSNR values and model save list
    best_psnr_list = [0.0] * 5  # best_psnr1 ~ best_psnr4 (for each test set)
    saved_models_list = [[] for _ in range(5)]  # saved_models1 ~ saved_models4

    # New: Record the best average PSNR and corresponding model path
    best_avg_psnr = 0.0
    best_avg_psnr_epoch = 0
    best_avg_model_path = os.path.join(model_path, 'best_avg_psnr_model.pth')
    # End of new code

    for epoch in range(1, opt.max_epoch + 1):
        train(epoch, logger, train_loader, model)
        psnrs = []

        # Iterate over each test dataloader
        for idx, test_loader in enumerate(test_loaders):
            result = test(epoch, logger, test_loader, model)
            psnr_mean = result[2]
            psnrs.append(psnr_mean)

            # Check and save the best model for current test set
            if psnr_mean > best_psnr_list[idx]:
                best_psnr_list[idx] = psnr_mean
                psnr_val, ssim_val = result[2], result[3]
                model_name = f'model{idx+1}_epoch_{epoch}_psnr_{psnr_val:.2f}_ssim_{ssim_val:.2f}.pth'
                model_path_save = os.path.join(model_path, model_name)

                torch.save(model.state_dict(), model_path_save)
                logger.info(f"===> Saved new best model{idx+1}: {model_name} with PSNR: {psnr_val:.2f}")

                saved_models_list[idx].append(model_name)

                # Keep at most one latest model
                if len(saved_models_list[idx]) > 1:
                    oldest_model = saved_models_list[idx].pop(0)
                    os.remove(os.path.join(model_path, oldest_model))
                    logger.info(f"===> Removed old model{idx+1}: {oldest_model}")

        # Calculate average PSNR
        avg_psnr = np.mean(np.asarray(psnrs))
        logger.info('===> Epoch {}: average testing psnr = {:.2f}'.format(epoch, avg_psnr))

        # New: Update global best average PSNR model
        if avg_psnr > best_avg_psnr:
            best_avg_psnr = avg_psnr
            best_avg_psnr_epoch = epoch
            # Save model
            torch.save(model.state_dict(), best_avg_model_path)
            logger.info(f"===> Saved new best average PSNR model: epoch {epoch}, avg_psnr = {avg_psnr:.2f}")
        else:
            logger.info(f"===> No improvement in average PSNR. Current best: {best_avg_psnr:.2f} (epoch {best_avg_psnr_epoch})")
        # End of new code

        scheduler.step()



if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main()