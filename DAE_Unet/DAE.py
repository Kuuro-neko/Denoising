import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
#import torch_directml

import csv
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import collections
from tqdm import tqdm

# ============= Configuration du device (AMD/CPU) =============
def get_device():
    """Détecte le meilleur device disponible"""
    if torch.cuda.is_available():
        print(f"Device détecté: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("Device détecté: CPU")
        return torch.device("cpu")

device = get_device()
#device = torch_directml.device()


# ============= Dataset personnalisé =============
class DenoisingDataset(Dataset):
    def __init__(self, clean_dir, noise_types=['gauss', 'poisson', 'sap', 'speckle'], 
                 transform=None, num_images=512):
        """
        Args:
            clean_dir: Répertoire contenant les images propres et bruitées
            noise_types: Liste des préfixes de bruit
            transform: Transformations à appliquer
            num_images: Nombre d'images propres (1 à num_images)
        """
        self.clean_dir = Path(clean_dir)
        self.noise_types = noise_types
        self.transform = transform
        self.num_images = num_images
        
        # Créer la liste de toutes les paires (image bruitée, image propre)
        # ATTENTION : les versions bruitées des patchs sont cherchées dans le même dossier et reconnues uniquement avec leur préfixe
        # Il faudra modifier le script de découpage/bruitage pour qu'il les génère de la même manière
        self.samples = []
        for i in range(1, num_images + 1):
            clean_name = f"{i:06d}.jpg"
            clean_path = self.clean_dir / clean_name
            
            if clean_path.exists():
                for noise in noise_types:
                    noisy_name = f"{noise}_{i:06d}.jpg"
                    noisy_path = self.clean_dir / noisy_name
                    
                    if noisy_path.exists():
                        self.samples.append((noisy_path, clean_path, noise))
        
        print(f"Dataset chargé: {len(self.samples)} paires d'images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        noisy_path, clean_path, noise_type = self.samples[idx]
        
        # Charger les images
        noisy_img = Image.open(noisy_path).convert('RGB')
        clean_img = Image.open(clean_path).convert('RGB')
        
        # Appliquer les transformations
        if self.transform:
            noisy_img = self.transform(noisy_img)
            clean_img = self.transform(clean_img)
        
        return noisy_img, clean_img, noise_type

def calculate_metrics_on_batch(clean_batch, denoised_batch):
    """
    Calcule PSNR et SSIM moyens sur un batch.
    Suppose des tenseurs PyTorch en entrée [B, C, H, W] normalisés.
    """
    # On passe sur CPU et en Numpy
    clean_np = clean_batch.detach().cpu().numpy()
    denoised_np = denoised_batch.detach().cpu().numpy()
        
    # Clip pour être sûr d'être dans [0, 1]
    clean_np = np.clip(clean_np, 0, 1)
    denoised_np = np.clip(denoised_np, 0, 1)
    
    psnr_val = 0.0
    ssim_val = 0.0
    batch_size = clean_np.shape[0]
    
    for i in range(batch_size):
        p = psnr(clean_np[i], denoised_np[i], data_range=1.0)
        s = ssim(clean_np[i], denoised_np[i], data_range=1.0, channel_axis=0)
        psnr_val += p
        ssim_val += s
        
    return psnr_val / batch_size, ssim_val / batch_size

def calculate_metrics_individual(clean_batch, denoised_batch):
    """
    Retourne des LISTES de métriques (une valeur par image du batch)
    au lieu de la moyenne globale.
    """
    clean_np = clean_batch.detach().cpu().numpy()
    denoised_np = denoised_batch.detach().cpu().numpy()
    
    # Clip [0, 1]
    clean_np = np.clip(clean_np, 0, 1)
    denoised_np = np.clip(denoised_np, 0, 1)
    
    psnr_list = []
    ssim_list = []
    batch_size = clean_np.shape[0]
    
    for i in range(batch_size):
        # Data range 1.0 car images normalisées entre 0 et 1
        p = psnr(clean_np[i], denoised_np[i], data_range=1.0)
        s = ssim(clean_np[i], denoised_np[i], data_range=1.0, channel_axis=0)
        psnr_list.append(p)
        ssim_list.append(s)
        
    return psnr_list, ssim_list

def log_to_csv(filepath, data, header=None, mode='a'):
    """Écrit une ligne dans un CSV. Crée le fichier avec header si nouveau."""
    file_exists = os.path.isfile(filepath)
    
    # Si on reprend l'entraînement (mode='a') mais que le fichier n'existe pas, 
    # on le crée comme si c'était 'w'
    if not file_exists and mode == 'a':
        mode = 'w'
        
    with open(filepath, mode, newline='') as f:
        writer = csv.writer(f)
        if not file_exists and header is not None:
            writer.writerow(header)
        writer.writerow(data)


#============Fonction de perte ===============

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Charger VGG19 pré-entraîné sur ImageNet
        vgg = models.vgg19(pretrained=True).features
        
        # On garde les couches jusqu'à la 35ème
        # Cela capture les structures de haut niveau.
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:35]).to(device).eval()
        
        # Geler les paramètres (on ne veut pas entraîner VGG, juste l'utiliser)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # Normalisation standard ImageNet (nécessaire car VGG a appris avec ces valeurs)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, input_img, target_img):
        # On s'assure que l'entrée est dans la bonne range pour la normalisation
        
        input_norm = (input_img - self.mean) / self.std
        target_norm = (target_img - self.mean) / self.std
        
        input_features = self.feature_extractor(input_norm)
        target_features = self.feature_extractor(target_norm)
        
        # On calcule la distance L1 entre les features
        return nn.functional.l1_loss(input_features, target_features)

class GeneratorLoss(nn.Module):
    def __init__(self, device, lambda_pixel=1.0, lambda_percep=0.1, lambda_adv=0.001):
        super().__init__()
        self.vgg_loss = VGGPerceptualLoss(device)
        self.pixel_loss = nn.L1Loss()
        
        # Poids des différentes pertes (Hyperparamètres à ajuster)
        self.w_pixel = lambda_pixel
        self.w_percep = lambda_percep
        self.w_adv = lambda_adv

    def forward(self, fake_img, real_img, discriminator_pred=None):
        """
        fake_img: Image sortie du générateur (U-Net)
        real_img: Image cible (Ground Truth)
        discriminator_pred: pour implémenter le GAN
        """
        
        # 1. Perte Pixel (contenu bas niveau)
        l_pixel = self.pixel_loss(fake_img, real_img)
        
        # 2. Perte Perceptuelle (contenu haut niveau / Texture)
        l_percep = self.vgg_loss(fake_img, real_img)
        
        # 3. Perte Adversariale
        l_adv = 0.0
        if discriminator_pred is not None:
            # On veut que le discriminateur prédise 1 (Vrai) pour notre fausse image
            l_adv = nn.functional.binary_cross_entropy_with_logits(
                discriminator_pred, torch.ones_like(discriminator_pred)
            )

        # Somme pondérée
        total_loss = (self.w_pixel * l_pixel) + (self.w_percep * l_percep) + (self.w_adv * l_adv)
        
        return total_loss, l_pixel, l_percep, l_adv


# ============= Architecture U-Net =============
class DoubleConv(nn.Module):
    """Bloc de deux convolutions avec BatchNorm et ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False  ),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """Architecture U-Net pour le débruitage"""
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Partie descendante (encodeur)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Partie ascendante (décodeur)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))
        
        # Goulot d'étranglement
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Couche finale
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        # Descente
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Montée
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]
            
            # Gérer les dimensions impaires
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            
            concat = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](concat)
        
        return self.final_conv(x)

class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64, normalization=False), # Pas de BN sur la 1ere couche
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            #nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, stride=1, padding=1, bias=False) 
            # La sortie n'est pas une seule valeur, mais une carte de caractéristiques (ex: 16x16 valeurs)
        )

    def forward(self, img):
        return self.model(img)

def train_gan_model(generator, discriminator, train_loader, val_loader, 
                    epochs=50, lr=1e-3, 
                    save_path_G='unet_denoiser.pth', 
                    save_path_D='discriminator.pth',
                    device='cuda'):
    
    # 1. Configuration des pertes
    criterion_G = GeneratorLoss(device=device, lambda_pixel=1.0, lambda_percep=0.1, lambda_adv=0.01).to(device)
    criterion_D = nn.BCEWithLogitsLoss().to(device)
    
    # 2. Optimiseurs séparés
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', patience=5, factor=0.5)
    
    train_losses_G = []
    val_losses_G = []
    train_losses_D = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{epochs}")
        print(f"{'='*60}")
        
        # --- Phase d'entraînement ---
        generator.train()
        discriminator.train()
        
        running_loss_G = 0.0
        running_loss_D = 0.0
        pixel_loss_acc = 0.0 
        vgg_loss_acc = 0.0   
        
        print("Entraînement GAN en cours...")
        for batch_idx, (noisy, clean) in enumerate(train_loader):
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            # ==================================================================
            #  ÉTAPE 1 : Entraîner le Discriminateur (D)
            # ==================================================================
            optimizer_D.zero_grad()
            
            # A. Sur les images réelles
            pred_real = discriminator(clean)
            target_real = torch.ones_like(pred_real) 
            loss_D_real = criterion_D(pred_real, target_real)
            
            # B. Sur les images fausses
            fake_images = generator(noisy)
            # IMPORTANT : .detach() pour ne pas modifier le générateur quand on entraîne le discriminateur
            pred_fake = discriminator(fake_images.detach())
            target_fake = torch.zeros_like(pred_fake)
            loss_D_fake = criterion_D(pred_fake, target_fake)
            
            # Moyenne des deux pertes
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            
            running_loss_D += loss_D.item()

            # ==================================================================
            #  ÉTAPE 2 : Entraîner le Générateur (G)
            # ==================================================================
            optimizer_G.zero_grad()
            
            # On ré-évalue le discriminateur sur les fausses images, mais cette fois
            # sans .detach(), car on veut que l'erreur remonte vers le générateur
            pred_fake_for_G = discriminator(fake_images)
            
            # Calcul de la Loss combinée (Pixel + VGG + Adversarial)
            # Le but du G est que D dise "Vrai" (donc on vise 1 pour pred_fake_for_G)
            loss_G, l_pixel, l_percep, l_adv = criterion_G(fake_images, clean, discriminator_pred=pred_fake_for_G)
            
            loss_G.backward()
            optimizer_G.step()
            
            running_loss_G += loss_G.item()
            pixel_loss_acc += l_pixel.item()
            vgg_loss_acc += l_percep.item()
            
            # Affichage progression
            if (batch_idx + 1) % max(1, len(train_loader) // 10) == 0:
                progress = (batch_idx + 1) / len(train_loader) * 100
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] ({progress:.0f}%) "
                      f"| Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f} "
                      f"(Pix:{l_pixel.item():.3f} VGG:{l_percep.item():.3f})")
        
        # Moyennes de l'époque
        epoch_loss_G = running_loss_G / len(train_loader)
        epoch_loss_D = running_loss_D / len(train_loader)
        train_losses_G.append(epoch_loss_G)
        train_losses_D.append(epoch_loss_D)
        
        # --- Phase de validation ---
        # On valide principalement le Générateur (qualité de l'image)
        generator.eval()
        val_loss = 0.0
        
        print("\nValidation en cours...")
        with torch.no_grad():
            for batch_idx, (noisy, clean) in enumerate(val_loader):
                noisy, clean = noisy.to(device), clean.to(device)
                fake_val = generator(noisy)
                loss, _, _, _ = criterion_G(fake_val, clean, discriminator_pred=None)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses_G.append(val_loss)
        
        # Scheduler update
        scheduler_G.step(val_loss)
        current_lr = optimizer_G.param_groups[0]['lr']
        
        print(f"\n{'─'*60}")
        print(f"Résumé Epoch {epoch+1}:")
        print(f"  Train Loss D: {epoch_loss_D:.6f}")
        print(f"  Train Loss G: {epoch_loss_G:.6f}")
        print(f"  Val Loss G:   {val_loss:.6f}")
        print(f"  LR Generator: {current_lr:.2e}")
        
        # Sauvegarde du meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(generator.state_dict(), save_path_G)
            torch.save(discriminator.state_dict(), save_path_D)
            print(f"  Checkpoint sauvegardé (Best Val: {best_val_loss:.6f})")
        
        print(f"{'─'*60}")
    
    return train_losses_G, train_losses_D, val_losses_G

def train_gan_model_advanced(generator, discriminator, train_loader, val_loader, 
                             epochs=50, lr=1e-3, 
                             resume=False,
                             checkpoint_dir='checkpoints',
                             device='cuda'):
    
    # --- 1. Préparation ---
    os.makedirs(checkpoint_dir, exist_ok=True)
    loss_csv_path = os.path.join(checkpoint_dir, 'training_losses.csv')
    metrics_csv_path = os.path.join(checkpoint_dir, 'quality_metrics.csv')
    
    csv_mode = 'a' if resume else 'w'
    
    # Headers Loss (Fixes)
    loss_header = ['Epoch', 'Loss_D', 'Loss_G', 'Pixel_L1', 'VGG', 'Adv', 'Val_Loss']

    # --- 2. Initialisation ---
    criterion_G = GeneratorLoss(device=device, lambda_pixel=1.0, lambda_percep=0.1, lambda_adv=0.01).to(device)
    criterion_D = nn.BCEWithLogitsLoss().to(device)
    
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    last_n_checkpoints = collections.deque(maxlen=5) 
    start_epoch = 0
    best_val_loss = float('inf')
    best_model_path = os.path.join(checkpoint_dir, 'best_unet.pth')
    
    # --- 3. Reprise ---
    if resume and os.path.exists(best_model_path):
        print(f"Reprise depuis {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        if 'model_state' in checkpoint:
            generator.load_state_dict(checkpoint['model_state'])
            discriminator.load_state_dict(checkpoint['discriminator_state'])
            best_val_loss = checkpoint.get('best_loss', float('inf'))
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f" -> Reprise Epoch {start_epoch}")
        else:
            generator.load_state_dict(checkpoint)

    # --- 4. Boucle ---
    for epoch in range(start_epoch, start_epoch + epochs):
        print(f"\nEPOCH {epoch+1}")
        
        # --- TRAIN ---
        generator.train(); discriminator.train()
        running_loss_G = 0.0; running_loss_D = 0.0
        pixel_acc = 0.0; vgg_acc = 0.0; adv_acc = 0.0

        loop = tqdm(train_loader, leave=True)
        
        for batch_idx, (noisy, clean, _) in enumerate(loop):
            noisy, clean = noisy.to(device), clean.to(device)
            
            # Train D
            optimizer_D.zero_grad()
            pred_real = discriminator(clean)
            loss_d_real = criterion_D(pred_real, torch.ones_like(pred_real))
            pred_fake = discriminator(generator(noisy).detach())
            loss_d_fake = criterion_D(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_d_real + loss_d_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            running_loss_D += loss_D.item()
            
            # Train G
            optimizer_G.zero_grad()
            fake = generator(noisy)
            pred_fake_G = discriminator(fake)
            loss_G, l_pix, l_vgg, l_adv = criterion_G(fake, clean, pred_fake_G)
            loss_G.backward()
            optimizer_G.step()
            
            running_loss_G += loss_G.item()
            pixel_acc += l_pix.item(); vgg_acc += l_vgg.item(); adv_acc += l_adv.item()

            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss_d=loss_D.item(), loss_g=loss_G.item())


        # Moyennes Train
        avg_loss_D = running_loss_D / len(train_loader)
        avg_loss_G = running_loss_G / len(train_loader)
        avg_pix = pixel_acc / len(train_loader)
        avg_vgg = vgg_acc / len(train_loader)
        avg_adv = adv_acc / len(train_loader)

        # --- VALIDATION ---
        generator.eval()
        val_loss = 0.0
        
        # Stockage : {'gauss': {'psnr': [], 'ssim': []}, ...}
        results_per_type = collections.defaultdict(lambda: {'psnr': [], 'ssim': []})
        
        print("Validation...")

        loop = tqdm(val_loader, leave=True)
        with torch.no_grad():
            for noisy, clean, noise_types in loop:
                noisy, clean = noisy.to(device), clean.to(device)
                fake = generator(noisy)
                
                # 1. Val Loss
                loss, _, _, _ = criterion_G(fake, clean, None)
                val_loss += loss.item()
                
                # 2. Métriques individuelles
                batch_psnrs, batch_ssims = calculate_metrics_individual(clean, fake)
                
                # 3. Répartition par bruit
                for i, n_type in enumerate(noise_types):
                    results_per_type[n_type]['psnr'].append(batch_psnrs[i])
                    results_per_type[n_type]['ssim'].append(batch_ssims[i])
                loop.set_description(f"Validation Epoch [{epoch+1}/{epochs}]")
                loop.set_postfix(val_loss=val_loss)
        
        avg_val_loss = val_loss / len(val_loader)

        # --- Calculs et Affichage ---
        final_metrics = {}
        global_psnr = []
        global_ssim = []
        
        print(f"Epoch {epoch+1} Results (Val Loss: {avg_val_loss:.4f}):")
        
        # On parcourt les résultats par type
        for n_type, metrics in results_per_type.items():
            avg_p = np.mean(metrics['psnr'])
            avg_s = np.mean(metrics['ssim'])
            final_metrics[n_type] = {'psnr': avg_p, 'ssim': avg_s}
            
            global_psnr.extend(metrics['psnr'])
            global_ssim.extend(metrics['ssim'])
            print(f"  > {n_type:<8} | PSNR: {avg_p:.2f}dB | SSIM: {avg_s:.4f}")
            
        avg_psnr_global = np.mean(global_psnr)
        avg_ssim_global = np.mean(global_ssim)
        print(f"  > GLOBAL   | PSNR: {avg_psnr_global:.2f}dB | SSIM: {avg_ssim_global:.4f}")

        # --- SAUVEGARDE CSV ---
        # 1. Log Pertes
        log_to_csv(loss_csv_path, 
                   [epoch+1, avg_loss_D, avg_loss_G, avg_pix, avg_vgg, avg_adv, avg_val_loss], 
                   header=loss_header, mode=csv_mode)
        
        # 2. Log Métriques
        sorted_types = sorted(final_metrics.keys())
        metrics_header = ['Epoch', 'Global_PSNR', 'Global_SSIM']
        row_values = [epoch+1, avg_psnr_global, avg_ssim_global]
        
        for nt in sorted_types:
            metrics_header.extend([f'PSNR_{nt}', f'SSIM_{nt}'])
            row_values.extend([final_metrics[nt]['psnr'], final_metrics[nt]['ssim']])
            
        is_new = not os.path.exists(metrics_csv_path)
        if is_new:
            log_to_csv(metrics_csv_path, row_values, header=metrics_header, mode='w')
        else:
            log_to_csv(metrics_csv_path, row_values, header=None, mode='a')

        csv_mode = 'a'

        # --- Checkpoints ---
        checkpoint_data = {
            'epoch': epoch,
            'model_state': generator.state_dict(),
            'discriminator_state': discriminator.state_dict(),
            'best_loss': best_val_loss,
            'optimizer_G_state': optimizer_G.state_dict()
        }
        
        # Save epoch checkpoint
        current_ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(checkpoint_data, current_ckpt_path)
        
        # Nettoyage anciens
        all_ckpts = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_")]
        if len(all_ckpts) > 5:
            all_ckpts.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            for f in all_ckpts[:-5]:
                os.remove(os.path.join(checkpoint_dir, f))

        # Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_data['best_loss'] = best_val_loss
            torch.save(checkpoint_data, best_model_path)
            print(f"  Nouveau meilleur modèle sauvegardé")
            
    return


# ============= Visualisation =============
def visualize_results(model, dataset, num_samples=3):
    """Affiche des exemples de débruitage"""
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            noisy, clean, _ = dataset[idx]
            noisy_input = noisy.unsqueeze(0).to(device)
            denoised = model(noisy_input).cpu().squeeze(0)
            
            # Convertir en images affichables
            noisy_img = noisy.permute(1, 2, 0).numpy()
            clean_img = clean.permute(1, 2, 0).numpy()
            denoised_img = denoised.permute(1, 2, 0).numpy()
            
            # Clipper les valeurs
            noisy_img = np.clip(noisy_img, 0, 1)
            clean_img = np.clip(clean_img, 0, 1)
            denoised_img = np.clip(denoised_img, 0, 1)
            
            axes[i, 0].imshow(noisy_img)
            axes[i, 0].set_title('Image bruitée')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(denoised_img)
            axes[i, 1].set_title('Débruitée')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(clean_img)
            axes[i, 2].set_title('Image propre')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('denoising_results.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============= Script principal =============
def main():
    # Configuration
    DATA_DIR = "../image_database/patches"  # À adapter selon le dossier contenant vos patchs
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 2e-4
    TRAIN_SPLIT = 0.8
    RESUME_TRAINING = False  # Mettre True pour continuer un entraînement
    MODEL_PATH_G = 'unet_denoiser.pth'
    MODEL_PATH_D = 'discriminator.pth'
    CHECKPOINT_DIR = './checkpoints_gan'
    
    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Charger le dataset
    full_dataset = DenoisingDataset(
        clean_dir=DATA_DIR,
        noise_types=['gauss', 'poisson', 'sap', 'speckle', 'gauss_weak', 'gauss_strong'],
        transform=transform,
        num_images=49530 # À adapter au nombre de patchs dans le dossier /patches
    )
    
    # Diviser en train/val
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Créer le modèle
    generator = UNet(in_channels=3, out_channels=3).to(device)
    discriminator = Discriminator(input_channels=3).to(device)
    
    # Chargement des poids si on continue un entraînement
    if RESUME_TRAINING:
        print(f"\n=== Tentative de chargement des checkpoints ===")
        
        # Charger Générateur
        if os.path.exists(MODEL_PATH_G):
            generator.load_state_dict(torch.load(MODEL_PATH_G, map_location=device))
            print(f"✓ Générateur chargé: {MODEL_PATH_G}")
        else:
            print(f"x Pas de générateur trouvé, démarrage à zéro.")

        # Charger Discriminateur
        if os.path.exists(MODEL_PATH_D):
            discriminator.load_state_dict(torch.load(MODEL_PATH_D, map_location=device))
            print(f"Discriminateur chargé: {MODEL_PATH_D}")
        else:
            print(f"Pas de discriminateur trouvé, initialisation aléatoire.")
            
        print("Continuation de l'entraînement...\n")

    else:
        print(f"\nModèles créés (G: {sum(p.numel() for p in generator.parameters())} params)")
    
    # Entraîner
    print("\n=== Début de l'entraînement ===\n")

    train_gan_model_advanced(
        generator, discriminator, 
        train_loader, val_loader, 
        epochs=EPOCHS, 
        resume=RESUME_TRAINING, 
        checkpoint_dir=CHECKPOINT_DIR,
        device=device
    )
    
    # --- Récupération des données ---

    print("\n=== Visualisation des résultats ===\n")

    # 1. Chargement du meilleur Générateur
    generator.load_state_dict(torch.load('unet_denoiser.pth', map_location=device))
    generator.eval()

    # 2. Visualisation des images
    visualize_results(generator, full_dataset, num_samples=10)

    # 3. Plot des courbes
    plt.figure(figsize=(15, 6))

if __name__ == "__main__":
    main()