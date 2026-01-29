import os
import torch
import sys
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
from torch.nn import L1Loss
from ignite.metrics import PSNR, SSIM, Loss
from ignite.engine import Engine
from torchvision.utils import make_grid
from pathlib import Path


def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)

class Trainer():
    def __init__(self, netG, netD, optG, optD, train_loader, valid_loader, device, config, writer, run_name, resume_path = None, kaggle = None, strict_netG = True, strict_netD = True):
        # Config
        self.config = config
        self.train_cfg = self.config['train']
        # self.model_cfg = self.config['model']
        self.val_step = self.train_cfg['val_step']
        self.total_epochs = self.train_cfg['total_epochs']

        # Model
        self.device = device
        self.netG = netG.to(self.device)
        self.netD = netD.to(self.device)
        self.optG = optG
        self.optD = optD
        
        # l1's coefficent 
        self.lambda_l1 = self.train_cfg['lambda_l1']

        # Loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_iter_train = len(self.train_loader)
        self.num_iter_valid = len(self.valid_loader)

        self.total_steps = self.total_epochs * self.num_iter_train
        
        # Log, checkpoint
        self.strict_netG = strict_netG
        self.strict_netD = strict_netD
        self.writer = writer
        self.resume_path =resume_path
        self.run_name = run_name
        self.checkpoint_dir = os.path.join('checkpoints', self.run_name)
        if kaggle:
            self.checkpoint_dir = '/kaggle/working/' + self.checkpoint_dir

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.current_step = 0
        self.best_ssim = 0.0

        # losses
        self.criterion_GAN = BCEWithLogitsLoss()
        self.criterion_L1 = L1Loss()
        self.loss_G = 0.0
        self.loss_l1 = 0.0
        self.loss_D = 0.0
        
        # Initialize Valid Engine
        def eval_step(engine, batch):
            self.netG.eval()
            with torch.no_grad():
                (s2, lc), s1 = batch
                s2, lc, s1 = s2.to(self.device), lc.to(self.device), s1.to(self.device)
                s1_fake = self.netG(s2, lc)

            return s1_fake, s1, s2, lc
        
        # Compute metrics of valid set
        self.evaluator = Engine(eval_step)
        Loss(self.criterion_L1, output_transform=lambda x: (x[0], x[1])).attach(self.evaluator, 'l1')
        PSNR(data_range=1.0, output_transform=lambda x: (denorm(x[0]), denorm(x[1]))).attach(self.evaluator, 'psnr')
        SSIM(data_range=1.0, output_transform=lambda x: (denorm(x[0]), denorm(x[1]))).attach(self.evaluator, 'ssim')

        if self.resume_path:
            self._load_checkpoint(resume_path)


    def _load_checkpoint(self, resume_path):
        try:
            # Load checkpoint
            checkpoint = torch.load(Path(resume_path))
            self.netG.load_state_dict(checkpoint['netG_state_dict'], strict=self.strict_netG)
            self.netD.load_state_dict(checkpoint['netD_state_dict'], strict=self.strict_netD)
            
            # Only load optimizer state if models were loaded with strict=True
            # When strict=False, model architecture may differ, causing optimizer mismatch
            if self.strict_netG and self.strict_netD:
                try:
                    self.optG.load_state_dict(checkpoint['optG_state_dict'])
                    self.optD.load_state_dict(checkpoint['optD_state_dict'])
                except Exception as opt_e:
                    print(f"Warning: Could not load optimizer state: {opt_e}.")
                    sys.exit(1)
            else:
                print("Note: Skipping optimizer state loading because model was loaded with strict=False")

            self.current_step = checkpoint['step']
            self.best_ssim = checkpoint.get('best_ssim', 0.0)

            print(f"Resumed from step {self.current_step}. Best SSIM so far: {self.best_ssim}")

        except Exception as e:
            print(f'Error loading checkpoint {e}. Double check resume path')
            sys.exit(1)
    
    
    def _save_checkpoint(self, step:int, is_best: bool):
        checkpoint_data = {
            'step': step,
            'netG_state_dict': self.netG.state_dict(),
            'netD_state_dict': self.netD.state_dict(),
            'optG_state_dict': self.optG.state_dict(),
            'optD_state_dict': self.optD.state_dict(),
            'best_ssim': self.best_ssim,
            'config': self.config
        }
        # Save last checkpoint
        last_save_path = os.path.join(self.checkpoint_dir, 'last.pth')
        torch.save(checkpoint_data, last_save_path)
        step_save_path = os.path.join(self.checkpoint_dir, f'{step}.pth')
        torch.save(checkpoint_data, step_save_path)

        # Save best checkpoint
        if is_best:
            best_save_path = os.path.join(self.checkpoint_dir, f'{step}-ssim_{self.best_ssim:.4f}.pth')
            torch.save(checkpoint_data, best_save_path)
            
            # Save as fixed name 'best.pth' for easier loading in inference
            best_fixed_path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint_data, best_fixed_path)
            
            print(f"Step {step}: New best model saved to {best_save_path}")
    


    def _validate_step(self, current_step):
        """
        Process an validation by val_step
        """
        # Compute loss of train set
        lossG_avg = self.loss_G / self.val_step
        lossL1_avg = self.loss_l1 / self.val_step
        lossD_avg = self.loss_D / self.val_step

        print(f"""Step [{current_step}/{self.total_steps}]
{20 * '-'}
Average Train L1 Loss: {lossL1_avg:.3f}
Average Train G Loss: {lossG_avg:.3f}
Average Train D Loss: {lossD_avg:.3f}
{20 * '-'}""")
        print(f'Start Validating...')
        
        
        self.evaluator.run(tqdm(self.valid_loader, desc="Validating", leave=False))
        l1_avg = self.evaluator.state.metrics['l1']
        psnr_avg = self.evaluator.state.metrics['psnr']
        ssim_avg = self.evaluator.state.metrics['ssim']
        print(f"""{20*'-'}
L1_val: {l1_avg:.3f}\nPSNR: {psnr_avg:.3f}db\nSSIM: {ssim_avg:.3f}
{20*'-'}""")

        # log
        self.writer.add_scalar(tag = 'L1 Loss/Train_Step', scalar_value = lossL1_avg, global_step = current_step)
        self.writer.add_scalar(tag = 'G Loss/Train_Step', scalar_value = lossG_avg, global_step = current_step)
        self.writer.add_scalar(tag = 'D Loss/Train_Step', scalar_value = lossD_avg, global_step = current_step)
        self.writer.add_scalar(tag='Metrics/L1', scalar_value = l1_avg, global_step = current_step)
        self.writer.add_scalar(tag='Metrics/PSNR', scalar_value = psnr_avg, global_step = current_step)
        self.writer.add_scalar(tag='Metrics/SSIM', scalar_value = ssim_avg, global_step = current_step)

        # Log images
        fake, real, s2, lc = self.evaluator.state.output
        n_imgs = min(8, fake.size(0))
        
        # def to_vis(x):
        #     if x.shape[1] == 2: return x[:, :1] # Take first channel if 2 channels (e.g. VV)
        #     if x.shape[1] > 3: return x[:, :3]  # Take first 3 channels if > 3 (e.g. RGB bands)
        #     return x

        self.writer.add_image('Images/Fake', make_grid((denorm(fake))[:n_imgs], nrow=4), current_step)
        self.writer.add_image('Images/Real', make_grid((denorm(real))[:n_imgs], nrow=4), current_step)
        self.writer.add_image('Images/S2', make_grid((denorm(s2))[:n_imgs], nrow=4), current_step)
        self.writer.add_image('Images/LC', make_grid((denorm(lc))[:n_imgs], nrow=4), current_step)

        # Check best ssim and save checkpoint
        is_best_ssim = ssim_avg > self.best_ssim
        if is_best_ssim:
            self.best_ssim = ssim_avg
            print(f'New best SSIM: {self.best_ssim:.3f}')
        else:
            print(f'Latest checkpoint saved!')
        
        self._save_checkpoint(current_step, is_best_ssim)

        # Reset losses
        self.loss_D = 0.0
        self.loss_G = 0.0
        self.loss_l1 = 0.0


    def run(self):
        if not self.resume_path:
            print(f"""--------------------
                \nStarting new run: {self.run_name}
                """)
        else:
            print(f"""------------------
                  \nResuming run '{self.run_name}' from step {self.current_step}.
                """)
        

        train_iter = iter(self.train_loader)

        pbar = tqdm(total=self.total_steps, initial=self.current_step, desc='Training')
            
        #-----------------MAIN--------------------------
        while self.current_step < self.total_steps:
            
            try:
                (s2, lc), s1= next(train_iter)

            except StopIteration:
                train_iter = iter(self.train_loader)
                (s2, lc), s1 = next(train_iter)

            s2, lc, s1 = s2.to(self.device), lc.to(self.device), s1.to(self.device)
            
            #---------Start Training---------------
            self.netG.train()
            self.netD.train()
            
            # generate S1 fake
            s1_fake = self.netG(s2, lc)


            #------------Train Discriminator--------------
            for param in self.netD.parameters():
                param.requires_grad = True
            
            self.optD.zero_grad()
            # Loss of fake
            D_fake_output = self.netD(s2, lc, s1_fake.detach())
            D_fake_loss = self.criterion_GAN(D_fake_output, torch.zeros_like(D_fake_output))
            
            # Loss of real
            D_real_output = self.netD(s2, lc, s1)
            D_real_loss = self.criterion_GAN(D_real_output, torch.ones_like(D_real_output))

            D_losses = (D_fake_loss + D_real_loss) / 2 # Losses average
            D_losses.backward() # Compute gradients
            self.optD.step() # Update weights


            #--------------Train Generator-----------------
            for param in self.netD.parameters():
                param.requires_grad = False
            
            self.optG.zero_grad()
            # Loss by D
            D_fake_output = self.netD(s2, lc, s1_fake)
            G_gan_loss = self.criterion_GAN(D_fake_output, torch.ones_like(D_fake_output))

            # Loss by L1
            G_l1_loss = self.criterion_L1(s1_fake, s1)

            # Losses sum
            G_total_loss = self.lambda_l1 * G_l1_loss + G_gan_loss
            
            # Update
            G_total_loss.backward()
            self.optG.step()
            #----End 1 iter/step train


            #---------------Logging and Update-----------------------
            self.loss_G += G_gan_loss.item()
            self.loss_l1 += G_l1_loss.item()
            self.loss_D += D_losses.item()

            pbar.set_postfix({
                'D_GAN': f'{D_losses.item():.3f}',
                'G_GAN': f'{G_gan_loss.item():.3f}',
                'L1': f'{G_l1_loss.item():.3f}'
            })

            self.current_step += 1
            pbar.update(1)


            #---------------Validate----------------------
            if self.current_step % self.val_step == 0:
                self._validate_step(self.current_step)

        self.writer.close()
        print("Training finished.")
            

            



        
