import os
import torch
from datetime import datetime
from tqdm import tqdm
# from src import l1_loss, lratio_loss, psnr_torch, ssim_torch, gradient_loss
from torch.nn import BCEWithLogitsLoss
from torch.nn import L1Loss

class Trainer():
    def __init__(self, netG, netD, optG, optD, train_loader, valid_loader, device, config, writer, run_name, resume_path = None, kaggle = None):
        # config
        self.config = config
        self.train_cfg = self.config['train']
        self.model_cfg = self.config['model']
        self.val_step = self.train_cfg['val_step']
        
        # Model
        self.netG = netG
        self.netD = netD
        self.optG = optG
        self.optD = optD
        self.device = device

        # coefficent loss
        self.lambda_l1 = self.train_cfg['lambda_l1']

        # Loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_iter_train = len(self.train_loader)
        self.num_iter_valid = len(self.valid_loader)

        
        # log, checkpoint
        self.writer = writer
        self.resume_path =resume_path
        self.run_name = run_name
        self.checkpoint_dir = os.path.join('checkpoints', self.run_name)
        if kaggle:
            self.checkpoint_dir = '/kaggle/working/' + self.checkpoint_dir

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.start_epoch = 1
        self.best_psnr = 0.0
        self.loss_show = 0.0
        self.total_epochs = self.train_cfg['epochs']
        self.end_epoch = self.start_epoch + self.total_epochs

        if self.resume_path:
            self._load_checkpoint(resume_path)


    def _load_checkpoint(self, resume_path):
        try:
            # Load checkpoint
            checkpoint = torch.load(resume_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.end_epoch = self.start_epoch + self.total_epochs
            self.best_psnr = checkpoint['best_psnr']
        
        except Exception as e:
            print(f'Error loading checkpoint {e}. Double check resume path')
            exit()
    
    
    def _save_checkpoint(self, epoch:int, is_best: bool):
        checkpoint_data = {
            'epoch':epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_psnr': self.best_psnr,
            'config': self.config
        }
        # Save last checkpoint
        last_save_path = os.path.join(self.checkpoint_dir, 'last.pth')
        torch.save(checkpoint_data, last_save_path)

        # Save best checkpoint
        if is_best:
            best_save_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint_data, best_save_path)
            print(f"Epoch {epoch}: New best model saved to {best_save_path}")
    

    def _train_epoch(self, epoch: int):
        """
        Process an training epoch
        """
        self.model.train()
        
        tqdm_train_loader = tqdm(self.train_loader, desc=f'Epoch [{epoch}/{self.end_epoch-1}] Train')


        def denorm(x):
            return (x * 0.5 + 0.5).clamp(0, 1)
        
        for iter, ((s2, lc), s1) in enumerate(tqdm_train_loader):
            current_global_step = (epoch - 1) * self.num_iter_train + iter
            s2, lc = s2.to(self.device), lc.to(self.device)
            s1 = s1.to(self.device)
            # hr= denorm(hr)
            
            # forward pass
            s1_fake = self.model(s2, lc)
            
            # train discriminator
            for param in self.netD.parameters():
                param.requires_grad = True
            # compute loss of D
            # real
            D_real_output = self.netD(s2, lc, s1)
            D_real_loss = BCEWithLogitsLoss(D_real_output, torch.ones_like(D_real_output))
            
            # fake
            D_fake_output = self.netD(s2, lc, s1_fake.detach())
            D_fake_loss = BCEWithLogitsLoss(D_fake_output, torch.zeros_like(D_fake_output))
            
            D_loss = (D_fake_loss + D_real_loss) / 2
            # compute gradients
            self.optD.zero_grad()
            D_loss.backward()
            # update weigts
            self.optD.step()
            
            # Train Generator
            for param in self.netD.parameters():
                param.requires_grad = False
            D_fake_output = self.netD(s2, lc, s1_fake)
            G_fake_loss = BCEWithLogitsLoss(D_fake_output, torch.ones_like(D_fake_output))
            G_l1_loss = L1Loss(s1_fake, s1)
            G_loss = self.lambda_l1 * G_l1_loss + G_fake_loss

            # compute gradients
            self.optG.zero_grad()
            G_loss.backward()
            self.optG.step()


            total_ratio_loss = 0.0
            total_l1_loss = 0.0
            total_grad_loss = 0.0
            

           
            
            
            loss = (self.theta_1 * total_ratio_loss) + (self.theta_2 * total_l1_loss) + (self.theta_3 * total_grad_loss)

            self.loss_show += loss.item()
       
            if iter % 100 == 0:
                print(f"\nRatio Loss: {self.theta_1 * total_ratio_loss.item():.4f}, L1 Loss: {self.theta_2 * total_l1_loss.item():.4f}, Gradient Loss: {self.theta_3 * total_grad_loss.item():.4f}")
            # backward pass
            # Update tqdm
            tqdm_train_loader.set_postfix(Loss=f'{loss.item():.5f}')
            

            # Validation by step
            if (current_global_step + 1) % self.val_step == 0:
                # Log loss each step
                avg_loss = self.loss_show/self.val_step
                print(f"Step [{(current_global_step+1)}], Average Train Loss: {avg_loss:.5f}")
                self.loss_show = 0.0
                self.writer.add_scalar(tag='Loss/Train_Step', scalar_value=avg_loss, global_step=current_global_step)

                print(f"\n[Step {(current_global_step + 1)}] Start Validating...")
                # Validate
                avg_psnr, avg_ssim = self._validate_epoch(epoch)
                # Log Metrics by Step
                self.writer.add_scalar(tag='Metrics/PSNR', scalar_value=avg_psnr, global_step=current_global_step)
                self.writer.add_scalar(tag='Metrics/SSIM', scalar_value=avg_ssim, global_step=current_global_step)

                # Check best & Save Checkpoint
                is_best = avg_psnr > self.best_psnr
                if is_best:
                    self.best_psnr = avg_psnr
                    print(f"New Best PSNR: {self.best_psnr:.3f}")
                
                self._save_checkpoint(epoch, is_best)
                self.model.train()

       


    def _validate_epoch(self, epoch: int):
        """
        Process an validation epoch
        """
        self.netG.eval()
        psnr_sum = 0
        ssim_sum = 0
        tqdm_valid_loader = tqdm(self.valid_loader, desc=f'Epoch [{epoch}/{self.end_epoch-1}] Valid')
        with torch.no_grad():
            for iter, ((s2, lc), s1) in enumerate(tqdm_valid_loader):
                s2, lc = s2.to(self.device), lc.to(self.device)
                s1= s1.to(self.device)
                # forward pass
                s1_fake = self.netG(s2, lc)

                # calculate metrics
                psnr_batch = psnr_torch(sr_fusion, hr).item()
                ssim_batch = ssim_torch(sr_fusion, hr).item()
                psnr_sum += psnr_batch
                ssim_sum += ssim_batch

            avg_psnr = psnr_sum/self.num_iter_valid
            avg_ssim = ssim_sum/self.num_iter_valid
            print(f'Epoch [{epoch}/{self.end_epoch-1}] Valid\nPSNR: {avg_psnr:.3f} dB\nSSIM: {avg_ssim:.3f}')
        
        
        return avg_psnr, avg_ssim
    

    def run(self):
        if not self.resume_path:
            print(f"""--------------------
                \nStarting new run: {self.run_name}
                """)
        else:
            print(f"""------------------
                  \nResuming run '{self.run_name}' from epoch {self.start_epoch}. Best PSNR: {self.best_psnr:.3f}
                """)
        
        for epoch in range(self.start_epoch, self.end_epoch):
            torch.cuda.empty_cache()
            
            # training
            self._train_epoch(epoch)
            
        self.writer.close()
        print("Training finished.")
            

            



        

