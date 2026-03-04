import os
import torch
import sys
from tqdm import tqdm
from torch.nn import MSELoss, BCEWithLogitsLoss, L1Loss
from torchvision.utils import make_grid
from pathlib import Path
import math

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def denorm(x):
    """Inverse of Normalize(mean=0.5, std=0.5): maps [-1,1] -> [0,1]"""
    return (x * 0.5 + 0.5).clamp(0, 1)


class Trainer():
    def __init__(self, netG, netD, optG, optD, train_loader, valid_loader, device, config, writer, run_name, 
                 resume_path=None, kaggle=None, strict_netG=True, strict_netD=True, use_wandb=False):
        # Config
        self.config = config
        self.train_cfg = self.config['train']
        self.val_step = self.train_cfg['val_step']
        self.total_epochs = self.train_cfg['total_epochs']

        # Model
        self.device = device
        self.netG = netG.to(self.device)
        self.netD = netD.to(self.device)
        self.optG = optG
        self.optD = optD
        
        # L1 coefficient
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
        self.use_wandb = use_wandb and HAS_WANDB
        self.resume_path = resume_path
        self.run_name = run_name
        self.checkpoint_dir = os.path.join('checkpoints', self.run_name)
        if kaggle:
            self.checkpoint_dir = '/kaggle/working/' + self.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.current_step = 0
        self.best_ssim = 0.0
        self.best_lpips = float('inf')  # Lower is better

        # ===== GAN Loss (configurable: 'mse' for LSGAN or 'bce' for vanilla) =====
        self.gan_loss_type = self.train_cfg.get('gan_loss', 'mse').lower()
        if self.gan_loss_type == 'bce':
            self.criterion_GAN = BCEWithLogitsLoss()
            print('GAN Loss: BCEWithLogitsLoss (vanilla GAN)')
        else:
            self.criterion_GAN = MSELoss()
            print('GAN Loss: MSELoss (LSGAN)')
        self.criterion_L1 = L1Loss()
        self.loss_G = 0.0
        self.loss_l1 = 0.0
        self.loss_D = 0.0

        # ===== AMP =====
        self.scaler = torch.amp.GradScaler('cuda')
        self.grad_clip_norm = self.train_cfg.get('grad_clip_norm', 1.0)

        # ===== LPIPS (lazy-loaded during validation to save VRAM) =====
        self.lpips_fn = None

        if self.resume_path:
            self._load_checkpoint(resume_path)


    def _load_checkpoint(self, resume_path):
        try:
            checkpoint = torch.load(Path(resume_path))
            self.netG.load_state_dict(checkpoint['netG_state_dict'], strict=self.strict_netG)
            self.netD.load_state_dict(checkpoint['netD_state_dict'], strict=self.strict_netD)
            
            # Only load optimizer state if models were loaded with strict=True
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
            self.best_lpips = checkpoint.get('best_lpips', float('inf'))

            print(f"Resumed from step {self.current_step}. Best SSIM: {self.best_ssim:.4f} | Best LPIPS: {self.best_lpips:.4f}")

        except Exception as e:
            print(f'Error loading checkpoint {e}. Double check resume path')
            sys.exit(1)
    
    
    def _save_checkpoint(self, step: int, is_best_ssim: bool, is_best_lpips: bool):
        checkpoint_data = {
            'step': step,
            'netG_state_dict': self.netG.state_dict(),
            'netD_state_dict': self.netD.state_dict(),
            'optG_state_dict': self.optG.state_dict(),
            'optD_state_dict': self.optD.state_dict(),
            'best_ssim': self.best_ssim,
            'best_lpips': self.best_lpips,
            'config': self.config
        }
        # Always save last checkpoint
        last_save_path = os.path.join(self.checkpoint_dir, 'last.pth')
        torch.save(checkpoint_data, last_save_path)
        step_save_path = os.path.join(self.checkpoint_dir, f'{step}.pth')
        torch.save(checkpoint_data, step_save_path)

        # Save best SSIM checkpoint
        if is_best_ssim:
            best_ssim_path = os.path.join(self.checkpoint_dir, 'best_ssim.pth')
            torch.save(checkpoint_data, best_ssim_path)
            named_path = os.path.join(self.checkpoint_dir, f'{step}-ssim_{self.best_ssim:.4f}.pth')
            torch.save(checkpoint_data, named_path)
            print(f"  ✅ New best SSIM model saved: {named_path}")

        # Save best LPIPS checkpoint
        if is_best_lpips:
            best_lpips_path = os.path.join(self.checkpoint_dir, 'best_lpips.pth')
            torch.save(checkpoint_data, best_lpips_path)
            named_path = os.path.join(self.checkpoint_dir, f'{step}-lpips_{self.best_lpips:.4f}.pth')
            torch.save(checkpoint_data, named_path)
            print(f"  ✅ New best LPIPS model saved: {named_path}")
        
        if not is_best_ssim and not is_best_lpips:
            print(f"  💾 Latest checkpoint saved at step {step}")


    def _validate_step(self, current_step):
        """
        Single-pass validation: compute L1, PSNR, SSIM, LPIPS in one loop.
        LPIPS model is lazy-loaded to CPU/GPU only during validation, then freed.
        """
        # Average train losses since last validation
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

        self.netG.eval()
        
        # Lazy-load LPIPS model
        import lpips as lpips_lib
        lpips_fn = lpips_lib.LPIPS(net='alex').to(self.device)
        lpips_fn.eval()

        # Accumulators
        l1_total = 0.0
        psnr_total = 0.0
        ssim_total = 0.0
        lpips_total = 0.0
        num_samples = 0
        last_fake = None
        last_real = None
        last_s2 = None
        last_lc = None

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                for batch in tqdm(self.valid_loader, desc="Validating", leave=False):
                    (s2, lc), s1 = batch
                    s2, lc, s1 = s2.to(self.device), lc.to(self.device), s1.to(self.device)
                    s1_fake = self.netG(s2, lc)
                    
                    bs = s1.size(0)

                    # --- L1 ---
                    l1_val = self.criterion_L1(s1_fake, s1).item()
                    l1_total += l1_val * bs

                    # --- PSNR & SSIM (on denormed [0,1]) ---
                    fake_dn = denorm(s1_fake)
                    real_dn = denorm(s1)
                    
                    # PSNR: per-sample, then average
                    mse_per_sample = ((fake_dn - real_dn) ** 2).view(bs, -1).mean(dim=1)
                    psnr_per_sample = 10.0 * torch.log10(1.0 / (mse_per_sample + 1e-8))
                    psnr_total += psnr_per_sample.sum().item()

                    # SSIM: structural similarity (simplified per-batch)
                    ssim_val = self._compute_ssim_batch(fake_dn, real_dn)
                    ssim_total += ssim_val * bs

                    # --- LPIPS (needs 3ch, [-1,1]) ---
                    s1_3ch = s1.repeat(1, 3, 1, 1)
                    s1_fake_3ch = s1_fake.repeat(1, 3, 1, 1)
                    lpips_val = lpips_fn(s1_3ch, s1_fake_3ch).mean().item()
                    lpips_total += lpips_val * bs

                    num_samples += bs
                    
                    # Keep last batch for image logging (move to CPU immediately)
                    last_fake = s1_fake[:8].cpu()
                    last_real = s1[:8].cpu()
                    last_s2 = s2[:8].cpu()
                    last_lc = lc[:8].cpu()

        # Free LPIPS model immediately after validation
        del lpips_fn
        torch.cuda.empty_cache()

        l1_avg = l1_total / num_samples
        psnr_avg = psnr_total / num_samples
        ssim_avg = ssim_total / num_samples
        lpips_avg = lpips_total / num_samples

        print(f"""{20*'-'}
L1_val: {l1_avg:.3f}
PSNR: {psnr_avg:.3f}db
SSIM: {ssim_avg:.3f}
LPIPS: {lpips_avg:.4f}
{20*'-'}""")

        # ===== TensorBoard Logging =====
        self.writer.add_scalar(tag='L1 Loss/Train_Step', scalar_value=lossL1_avg, global_step=current_step)
        self.writer.add_scalar(tag='G Loss/Train_Step', scalar_value=lossG_avg, global_step=current_step)
        self.writer.add_scalar(tag='D Loss/Train_Step', scalar_value=lossD_avg, global_step=current_step)
        self.writer.add_scalar(tag='Metrics/L1', scalar_value=l1_avg, global_step=current_step)
        self.writer.add_scalar(tag='Metrics/PSNR', scalar_value=psnr_avg, global_step=current_step)
        self.writer.add_scalar(tag='Metrics/SSIM', scalar_value=ssim_avg, global_step=current_step)
        self.writer.add_scalar(tag='Metrics/LPIPS', scalar_value=lpips_avg, global_step=current_step)

        # Log images to TensorBoard
        n_imgs = min(8, last_fake.size(0))
        self.writer.add_image('Images/Fake', make_grid(denorm(last_fake)[:n_imgs], nrow=4), current_step)
        self.writer.add_image('Images/Real', make_grid(denorm(last_real)[:n_imgs], nrow=4), current_step)
        self.writer.add_image('Images/S2', make_grid(denorm(last_s2)[:n_imgs], nrow=4), current_step)
        self.writer.add_image('Images/LC', make_grid(denorm(last_lc)[:n_imgs], nrow=4), current_step)

        # ===== WandB Logging =====
        if self.use_wandb:
            wandb.log({
                'train/loss_G': lossG_avg,
                'train/loss_D': lossD_avg,
                'train/loss_L1': lossL1_avg,
                'val/L1': l1_avg,
                'val/PSNR': psnr_avg,
                'val/SSIM': ssim_avg,
                'val/LPIPS': lpips_avg,
            }, step=current_step)
            # Log images to WandB (already on CPU)
            wandb.log({
                'images': [
                    wandb.Image(make_grid(denorm(last_fake)[:n_imgs], nrow=4), caption='Fake S1'),
                    wandb.Image(make_grid(denorm(last_real)[:n_imgs], nrow=4), caption='Real S1'),
                    wandb.Image(make_grid(denorm(last_s2)[:n_imgs], nrow=4), caption='S2 Input'),
                    wandb.Image(make_grid(denorm(last_lc)[:n_imgs], nrow=4), caption='LC Input'),
                ]
            }, step=current_step)

        # ===== Dual-Best Checkpointing =====
        is_best_ssim = ssim_avg > self.best_ssim
        is_best_lpips = lpips_avg < self.best_lpips

        if is_best_ssim:
            self.best_ssim = ssim_avg
        if is_best_lpips:
            self.best_lpips = lpips_avg
        
        self._save_checkpoint(current_step, is_best_ssim, is_best_lpips)

        # Reset accumulated losses
        self.loss_D = 0.0
        self.loss_G = 0.0
        self.loss_l1 = 0.0


    @staticmethod
    def _compute_ssim_batch(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
        """Compute SSIM between two batches of images. Returns average SSIM."""
        # Gaussian window
        channels = img1.size(1)
        coords = torch.arange(window_size, dtype=torch.float32, device=img1.device) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
        g = g / g.sum()
        window = g.unsqueeze(0) * g.unsqueeze(1)  # 2D gaussian
        window = window.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        
        pad = window_size // 2
        mu1 = torch.nn.functional.conv2d(img1, window, padding=pad, groups=channels)
        mu2 = torch.nn.functional.conv2d(img2, window, padding=pad, groups=channels)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2
        
        sigma1_sq = torch.nn.functional.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
        sigma2_sq = torch.nn.functional.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
        sigma12 = torch.nn.functional.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12
        
        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean().item()


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
            
        #-----------------MAIN TRAINING LOOP--------------------------
        try:
            while self.current_step < self.total_steps:
                
                try:
                    (s2, lc), s1 = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    (s2, lc), s1 = next(train_iter)

                s2, lc, s1 = s2.to(self.device), lc.to(self.device), s1.to(self.device)
                
                self.netG.train()
                self.netD.train()

                # ============ Train Discriminator ============
                for param in self.netD.parameters():
                    param.requires_grad = True
                
                self.optD.zero_grad()

                with torch.amp.autocast('cuda'):
                    # Generate fake
                    s1_fake = self.netG(s2, lc)
                    
                    # D on fake
                    D_fake_output = self.netD(s2, lc, s1_fake.detach()).float()
                    D_fake_loss = self.criterion_GAN(D_fake_output, torch.zeros_like(D_fake_output))
                    
                    # D on real
                    D_real_output = self.netD(s2, lc, s1).float()
                    D_real_loss = self.criterion_GAN(D_real_output, torch.ones_like(D_real_output))

                    D_losses = (D_fake_loss + D_real_loss) / 2

                self.scaler.scale(D_losses).backward()
                self.scaler.unscale_(self.optD)
                torch.nn.utils.clip_grad_norm_(self.netD.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optD)


                # ============ Train Generator ============
                for param in self.netD.parameters():
                    param.requires_grad = False
                
                self.optG.zero_grad()

                with torch.amp.autocast('cuda'):
                    D_fake_output = self.netD(s2, lc, s1_fake).float()
                    G_gan_loss = self.criterion_GAN(D_fake_output, torch.ones_like(D_fake_output))
                    G_l1_loss = self.criterion_L1(s1_fake.float(), s1.float())
                    G_total_loss = self.lambda_l1 * G_l1_loss + G_gan_loss

                self.scaler.scale(G_total_loss).backward()
                self.scaler.unscale_(self.optG)
                torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optG)

                # Update scaler once per iteration (after both D and G steps)
                self.scaler.update()


                # ============ Logging ============
                self.loss_G += G_gan_loss.item()
                self.loss_l1 += G_l1_loss.item()
                self.loss_D += D_losses.item()

                pbar.set_postfix({
                    'D': f'{D_losses.item():.3f}',
                    'G': f'{G_gan_loss.item():.3f}',
                    'L1': f'{G_l1_loss.item():.3f}'
                })
                
                # Step-level WandB logging (every 10 steps)
                if self.use_wandb and self.current_step % 10 == 0:
                    wandb.log({
                        'train_step/loss_G': G_gan_loss.item(),
                        'train_step/loss_D': D_losses.item(),
                        'train_step/loss_L1': G_l1_loss.item(),
                    }, step=self.current_step)

                self.current_step += 1
                pbar.update(1)


                # ============ Validate ============
                if self.current_step % self.val_step == 0:
                    torch.cuda.empty_cache()
                    self._validate_step(self.current_step)
                    torch.cuda.empty_cache()

        except Exception as e:
            print(f"\nCRITICAL ERROR in training loop at step {self.current_step}: {e}")
            import traceback
            traceback.print_exc()
            # Emergency checkpoint save
            try:
                self._save_checkpoint(self.current_step, False, False)
                print("Emergency checkpoint saved.")
            except:
                pass
            sys.exit(1)

        finally:
            pbar.close()

        self.writer.close()
        if self.use_wandb:
            wandb.finish()
        print("Training finished.")
