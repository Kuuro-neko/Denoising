import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_logs(checkpoint_dir):
    # 1. Vérification des chemins
    loss_path = os.path.join(checkpoint_dir, 'training_losses.csv')
    metrics_path = os.path.join(checkpoint_dir, 'quality_metrics.csv')
    
    print(f"--- Recherche des logs dans : {os.path.abspath(checkpoint_dir)} ---")

    if not os.path.exists(loss_path) or not os.path.exists(metrics_path):
        print(f"ERREUR : Fichiers introuvables.\n - {loss_path}\n - {metrics_path}")
        print("Vérifiez que le chemin relatif est correct depuis l'endroit où vous lancez le script.")
        return

    # 2. Chargement des données
    try:
        df_loss = pd.read_csv(loss_path)
        df_metrics = pd.read_csv(metrics_path)
        print(f"Données chargées : {len(df_loss)} époques trouvées.")
    except Exception as e:
        print(f"Erreur à la lecture des CSV : {e}")
        return

    # 3. Création de la figure
    plt.figure(figsize=(18, 6))
    
    # --- GRAPHIQUE 1 : Pertes (Losses) ---
    plt.subplot(1, 3, 1)
    # On trace tout ce qui ressemble à une Loss
    if 'Loss_G' in df_loss.columns:
        plt.plot(df_loss['Epoch'], df_loss['Loss_G'], label='Gen Total', linewidth=2)
    if 'Val_Loss' in df_loss.columns:
        plt.plot(df_loss['Epoch'], df_loss['Val_Loss'], label='Validation', linewidth=2, linestyle='--')
    if 'Loss_D' in df_loss.columns:
        plt.plot(df_loss['Epoch'], df_loss['Loss_D'], label='Discrim', alpha=0.5)
        
    plt.title('Evolution des Pertes')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # --- GRAPHIQUE 2 : PSNR (Global + Par type) ---
    plt.subplot(1, 3, 2)
    # Trace la moyenne globale
    if 'Global_PSNR' in df_metrics.columns:
        plt.plot(df_metrics['Epoch'], df_metrics['Global_PSNR'], label='GLOBAL', color='black', linewidth=2)
    
    # Trace dynamiquement chaque colonne commençant par "PSNR_"
    for col in df_metrics.columns:
        if col.startswith('PSNR_'):
            label_name = col.replace('PSNR_', '')
            plt.plot(df_metrics['Epoch'], df_metrics[col], label=label_name, alpha=0.7, linestyle='--')
            
    plt.title('Qualité : PSNR (dB)')
    plt.xlabel('Epoch')
    plt.legend(fontsize='small')
    plt.grid(True, alpha=0.3)

    # --- GRAPHIQUE 3 : SSIM (Global + Par type) ---
    plt.subplot(1, 3, 3)
    if 'Global_SSIM' in df_metrics.columns:
        plt.plot(df_metrics['Epoch'], df_metrics['Global_SSIM'], label='GLOBAL', color='black', linewidth=2)

    for col in df_metrics.columns:
        if col.startswith('SSIM_'):
            label_name = col.replace('SSIM_', '')
            plt.plot(df_metrics['Epoch'], df_metrics[col], label=label_name, alpha=0.7, linestyle='--')

    plt.title('Qualité : SSIM (0-1)')
    plt.xlabel('Epoch')
    # Pas de légende ici si c'est trop chargé, ou : plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 4. Affichage bloquant
    print("Affichage du graphique...")
    plt.show(block=True) 

# Bloc pour tester le fichier directement
if __name__ == "__main__":
    # Remplacez par votre dossier réel pour tester ce fichier seul
    plot_training_logs('./checkpoints_gan')

def main():
    plot_training_logs('./checkpoints_gan')