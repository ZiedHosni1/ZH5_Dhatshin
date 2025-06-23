import os, json, random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem, DataStructs
import selfies
from tqdm import tqdm
from rdkit.Chem import PandasTools

device = torch.device("cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# --------------------
# lr Parse Setup
# --------------------
parser = argparse.ArgumentParser()
parser.add_argument('--lr_g', type=float, required=True)
parser.add_argument('--lr_d', type=float, required=True)
args = parser.parse_args()

lr_g = args.lr_g
lr_d = args.lr_d
# --------------------
# Output Directory Setup
# --------------------
output_dir = f"outputs/lrg_{lr_g}_lrd_{lr_d}"
os.makedirs(output_dir, exist_ok=True)
checkpoint_path = os.path.join(output_dir, "checkpoint_batch.pth")
metrics_path = os.path.join(output_dir, "training_metrics_batch.csv")

# Load QM9 CSV file (make sure qm9.csv is in the same folder)
df_acid = pd.read_csv('qm9.csv')
df_base = pd.read_csv('lewis_acid_candidates.csv')

# Ensure it has a column named 'smiles'
assert 'smiles' in df_acid.columns, "CSV must have a 'smiles' column."
assert 'smiles' in df_base.columns, "CSV must have a 'smiles' column."

# Convert SMILES to RDKit Mol objects
PandasTools.AddMoleculeColumnToFrame(df_acid, smilesCol='smiles')
PandasTools.AddMoleculeColumnToFrame(df_base, smilesCol='smiles')

def generate_flp_pairs(df_acid, df_base, acid_col='smiles', base_col='smiles'):
    """
    Generate all unique Lewis acid–base pairs as FLP SMILES strings.

    Args:
        df_acid (pd.DataFrame): DataFrame with Lewis acids (must include 'smiles' column).
        df_base (pd.DataFrame): DataFrame with Lewis bases (must include 'smiles' column).
        acid_col (str): Column name for acid SMILES.
        base_col (str): Column name for base SMILES.

    Returns:
        pd.DataFrame: DataFrame with 'flp_smiles' and 'type' columns.
    """
    assert acid_col in df_acid.columns, f"Acid DataFrame must contain column '{acid_col}'"
    assert base_col in df_base.columns, f"Base DataFrame must contain column '{base_col}'"

    acids = df_acid[acid_col].unique()
    bases = df_base[base_col].unique()

    print(f"🔬 Generating all pairwise combinations: {len(acids)} acids × {len(bases)} bases")

    flp_data = [{'flp_smiles': f"{b}.{a}", 'type': 'inter'} for a, b in product(acids, bases)]
    flp_df = pd.DataFrame(flp_data)

    print(f"✅ Generated {len(flp_df)} FLP pairs.")

    return flp_df

# Example use:
print("Extracting all FLP candidates (intra + inter)...")
flp_all_df = generate_flp_pairs(df_acid, df_base)
print(flp_all_df.head())

def extract_flp_features(flp_smiles):
    features = {
        'charge_diff': None,
        'tpsa': None,
        'rotatable_bonds': None,
        'num_rings': None
    }

    try:
        if '.' in flp_smiles:
            # Intermolecular FLP pair: split into base + acid
            base_smiles, acid_smiles = flp_smiles.split('.')
            base_mol = Chem.MolFromSmiles(base_smiles)
            acid_mol = Chem.MolFromSmiles(acid_smiles)
            if base_mol is None or acid_mol is None:
                return features

            # Add Hs + charges
            base = Chem.AddHs(base_mol)
            acid = Chem.AddHs(acid_mol)
            ComputeGasteigerCharges(base)
            ComputeGasteigerCharges(acid)

            base_charges = [float(atom.GetProp('_GasteigerCharge')) for atom in base.GetAtoms()]
            acid_charges = [float(atom.GetProp('_GasteigerCharge')) for atom in acid.GetAtoms()]
            features['charge_diff'] = abs(max(base_charges) - min(acid_charges))

            # Sum properties
            features['tpsa'] = Descriptors.TPSA(base) + Descriptors.TPSA(acid)
            features['rotatable_bonds'] = Descriptors.NumRotatableBonds(base) + Descriptors.NumRotatableBonds(acid)
            features['num_rings'] = base.GetRingInfo().NumRings() + acid.GetRingInfo().NumRings()

        else:
            # Intramolecular FLP: single molecule
            mol = Chem.MolFromSmiles(flp_smiles)
            if mol is None:
                return features

            mol = Chem.AddHs(mol)
            ComputeGasteigerCharges(mol)
            charges = [float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms()]
            # Charge diff: max minus min in same molecule
            features['charge_diff'] = abs(max(charges) - min(charges))

            # Single-molecule descriptors
            features['tpsa'] = Descriptors.TPSA(mol)
            features['rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
            features['num_rings'] = mol.GetRingInfo().NumRings()

    except Exception as e:
        print("⚠️ Feature extraction failed for:", flp_smiles, "| Error:", e)

    return features

# Apply to flp_all_df
flp_feature_dicts = flp_all_df['flp_smiles'].apply(extract_flp_features)
flp_features_df = pd.DataFrame(flp_feature_dicts.tolist())

# Combine with original dataframe
flp_all_features_df = pd.concat([flp_all_df, flp_features_df], axis=1)

print(flp_all_features_df.head())

# --------------------
# Data Preprocessing: Convert SMILES to tokenized SELFIES
# --------------------

class MolecularSELFIESDataset(Dataset):
    def __init__(self, smiles_list):
        self.smiles = smiles_list

        # Convert each SMILES to its tokenized SELFIES representation
        self.selfies_tokens_list = [self.smiles_to_selfies_tokens(smi) for smi in self.smiles]

        # Build vocabulary from all tokens (reserve index 0 for padding)
        all_tokens = [token for tokens in self.selfies_tokens_list for token in tokens]
        unique_tokens = sorted(set(all_tokens))
        self.token_to_idx = {token: idx + 1 for idx, token in enumerate(unique_tokens)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}

        # Determine maximum sequence length for padding
        self.max_seq_len = max(len(tokens) for tokens in self.selfies_tokens_list)
        self.vocab_size = len(self.token_to_idx) + 1  # +1 for padding

        # Convert token sequences to fixed-length one-hot encoded vectors (flattened)
        self.encoded_data = []
        for tokens in self.selfies_tokens_list:
            token_indices = [self.token_to_idx[token] for token in tokens]
            padded = token_indices + [0] * (self.max_seq_len - len(token_indices))
            one_hot = np.eye(self.vocab_size)[padded]
            one_hot_flat = one_hot.flatten()
            self.encoded_data.append(one_hot_flat)

    def smiles_to_selfies_tokens(self, smiles):
        """Convert SMILES to SELFIES tokens"""
        try:
            selfies_str = selfies.encoder(smiles)
            tokens = list(selfies.split_selfies(selfies_str))
            return tokens
        except Exception:
            return []

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        return torch.tensor(self.encoded_data[idx], dtype=torch.float32)


# Extract SMILES list and create the dataset and dataloader
# ✅ NEW: Use the 'flp_smiles' column (includes both intra & inter)
smiles_list = flp_all_df['flp_smiles'].dropna().tolist()

dataset = MolecularSELFIESDataset(smiles_list)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


print(f"✅ Dataset successfully loaded with {len(dataset)} molecules for training.")
print(f"Vocabulary size: {dataset.vocab_size}, Sequence length: {dataset.max_seq_len}")

# Data dimension: each molecule is represented as a flattened vector
data_dim = dataset.max_seq_len * dataset.vocab_size

# --------------------
# WGAN Architecture
# --------------------
latent_dim = 128

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )
    def forward(self, z):
        return self.model(z)
    
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    def forward(self, x):
        return self.model(x)

# Initialize models
generator = Generator(latent_dim=latent_dim, output_dim=data_dim).to(device)
discriminator = Discriminator(input_dim=data_dim).to(device)

@torch.no_grad()
def compute_validation_loss(generator, discriminator, val_loader, device):
    generator.eval()
    discriminator.eval()
    val_loss = 0.0
    num_batches = 0

    for real_samples in val_loader:
        real_samples = real_samples.float().to(device)
        batch_size = real_samples.size(0)

        z = torch.randn(batch_size, latent_dim).to(device)
        fake_samples = generator(z)
        
        # Use the same logic as GAN loss
        loss = -torch.mean(discriminator(fake_samples))
        val_loss += loss.item()
        num_batches += 1

    generator.train()
    discriminator.train()
    return val_loss / num_batches



def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1)
    alpha = alpha.expand_as(real_samples)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)
    d_interpolates = D(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def selfies_grammar_penalty(selfies_str, max_len):
    try:
        # Check if it's valid SELFIES first
        if not selfies.is_valid_selfies(selfies_str):
            return 1.0  # max penalty

        # Check for suspiciously short strings
        if len(selfies_str) < max_len * 4:
            return 0.5  # likely padding/truncated

        return 0.0  # All good
    except:
        return 1.0

def reward_geometric_frustration(features):
    """
    Higher score = more frustration (desired)
    """
    if features['charge_diff'] is None:
        return 0.0  # Invalid mol

    score = 0.0

    # Charge separation (encouraged)
    score += min(features['charge_diff'], 2.0) / 2.0  # scaled [0, 1]

    # Rings (steric crowding = frustration)
    score += min(features['num_rings'], 3) / 3.0      # scaled [0, 1]

    # Penalize too many rotatable bonds (less rigid = less frustrated)
    score += max(0, 1.0 - min(features['rotatable_bonds'], 5) / 5.0)

    return score / 3.0  # normalized to [0, 1]

def reward_homo_lumo_proxy(features):
    """
    Proxy for HOMO-LUMO gap: ideal if charge_diff is moderate (~0.3–1.0)
    """
    if features['charge_diff'] is None:
        return 0.0

    gap = features['charge_diff']
    # Target charge_diff ≈ 0.6 for moderate HOMO-LUMO
    ideal = 0.6
    reward = max(0, 1.0 - abs(gap - ideal) / ideal)  # triangle shape
    return reward

LEWIS_BASE_SMARTS = [
    "[#7;H2,H1;!$(NC=O)]",  # primary or secondary amines
    "[O-]",                 # negative oxygen (e.g., carboxylate)
    "[nH]",                 # aromatic N-H
    "[#8;H1]"               # alcohol OH
]

LEWIS_ACID_SMARTS = [
    "[B]",                  # boron
    "[P+]",                 # phosphonium
    "[Al]",                 # aluminium
    "[Sn]",                 # tin
    "[Si]",                 # silicon
]

def contains_substructure(mol, smarts_list):
    for smarts in smarts_list:
        pattern = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(pattern):
            return True
    return False

def reward_lewis_acid_base(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    has_base = contains_substructure(mol, LEWIS_BASE_SMARTS)
    has_acid = contains_substructure(mol, LEWIS_ACID_SMARTS)
    return 1.0 if has_base and has_acid else 0.0


# --------------------
# Reconstructing Molecular Representations
# --------------------
def reconstruct_selfies(gen_output, max_seq_len, vocab_size, idx_to_token):
    """
    Convert a GAN-generated flattened vector into a SELFIES string.
    
    Args:
        gen_output (np.array): Flattened output vector from the generator.
        max_seq_len (int): Maximum sequence length used in the dataset.
        vocab_size (int): Vocabulary size used in one-hot encoding.
        idx_to_token (dict): Mapping from indices to SELFIES tokens.
    
    Returns:
        str: Reconstructed SELFIES string.
    """
    # Reshape the output vector into [max_seq_len, vocab_size]
    reshaped = gen_output.reshape(max_seq_len, vocab_size)
    # Discretize: choose the token with maximum value in each position
    token_indices = np.argmax(reshaped, axis=1)
    # Remove padding tokens (assumed index 0)
    token_indices = [idx for idx in token_indices if idx != 0]
    # Map indices back to tokens
    tokens = [idx_to_token.get(idx, '') for idx in token_indices]
    selfies_str = "".join(tokens)
    return selfies_str

# --------------------
# Training Loop
# --------------------
optimizer_G = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

start_epoch = 0
metrics = {
    "epoch": [],
    "loss_G": [],
    "loss_D": [],
    "val_loss": [],
    "reward_total": [],
    "reward_frustration": [],
    "reward_homo_lumo": [],
    "reward_acid_base": []
}

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint["G"])
    discriminator.load_state_dict(checkpoint["D"])
    optimizer_G.load_state_dict(checkpoint["opt_G"])
    optimizer_D.load_state_dict(checkpoint["opt_D"])
    start_epoch = checkpoint["epoch"] + 1
    if os.path.exists(metrics_path):
        metrics = pd.read_csv(metrics_path).to_dict(orient="list")
    print(f"🔁 Resuming from epoch {start_epoch}")


n_epochs = 1000  # Adjust number of epochs as needed
lambda_gp = 10
losses_G, losses_D = [], []

def plot_losses():
    plt.figure(figsize=(10, 5))
    plt.plot(losses_G, label='Generator Loss')
    plt.plot(losses_D, label='Discriminator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GAN Training Loss')
    plt.show()

# Reward tracking
total_rewards = []
frustration_scores = []
homo_lumo_scores = []
acid_base_scores = []
lr_G_log = []
lr_D_log = []
best_loss = float("inf")        # guarantees first loss is lower
no_improvement = 0
delta = 0.0
min_epochs = 50


for epoch in range(n_epochs):
    for real_samples in data_loader:
        real_samples = real_samples.float().to(device)
        batch_size = real_samples.size(0)
        
        # Train Discriminator
        optimizer_D.zero_grad()
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_samples = generator(z).detach()
        loss_D_real = -torch.mean(discriminator(real_samples))
        loss_D_fake = torch.mean(discriminator(fake_samples))
        gradient_penalty = compute_gradient_penalty(discriminator, real_samples, fake_samples)
        loss_D = loss_D_real + loss_D_fake + lambda_gp * gradient_penalty
        loss_D.backward()
        optimizer_D.step()

        # Train Generator with RL-internal reward
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim)
        logits = generator(z)  # shape: [batch_size, seq_len * vocab_size] or [batch_size, seq_len, vocab_size]

        # Reshape if needed: assumes generator outputs flat vector
        logits = logits.view(batch_size, dataset.max_seq_len, dataset.vocab_size)

        # Sample token indices using softmax sampling (stochastic)
        token_probs = torch.softmax(logits, dim=-1)  # [B, L, V]
        token_indices = torch.multinomial(token_probs.view(-1, dataset.vocab_size), num_samples=1)  # [B*L, 1]
        token_indices = token_indices.view(batch_size, dataset.max_seq_len)

        # Decode SELFIES strings
        selfies_batch = []
        for i in range(batch_size):
            tokens = [dataset.idx_to_token[idx.item()] for idx in token_indices[i] if idx.item() != 0]
            selfies_str = ''.join(tokens)
            selfies_batch.append(selfies_str)

        # Compute reward using geometric frustration + HOMO-LUMO proxy
        rewards = []
        frustration_vals = []
        homo_lumo_vals = []
        acid_base_vals = []
        for selfies_str in selfies_batch:
            try:
                smiles_str = selfies.decoder(selfies_str)
                features = extract_flp_features(smiles_str)

                frustration = reward_geometric_frustration(features)
                homo_lumo = reward_homo_lumo_proxy(features)
                acid_base = reward_lewis_acid_base(smiles_str)

                # Weighted reward: 0.2 weight for Lewis acid/base check
                reward = 0.4 * frustration + 0.4 * homo_lumo + 0.2 * acid_base
                rewards.append(reward)
                frustration_vals.append(frustration)
                homo_lumo_vals.append(homo_lumo)
                acid_base_vals.append(acid_base)
            except:
                rewards.append(0.0)
                frustration_vals.append(0.0)
                homo_lumo_vals.append(0.0)
                acid_base_vals.append(0.0)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(z.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        # Compute log_probs of sampled tokens
        gathered_log_probs = torch.gather(token_probs, 2, token_indices.unsqueeze(-1)).squeeze(-1)  # [B, L]
        log_probs = torch.log(gathered_log_probs + 1e-8)  # prevent log(0)

        # REINFORCE-style loss: reward-weighted log-prob
        rl_loss = -(log_probs.sum(dim=1) * rewards).mean()

        # GAN loss
        gan_loss = -torch.mean(discriminator(logits.view(batch_size, -1)))

        # Total loss
        λ = 0.5
        total_loss_G = gan_loss + λ * rl_loss
        total_loss_G.backward()
        optimizer_G.step()


    # Log average reward values for this epoch
    total_rewards.append(torch.tensor(rewards).mean().item())
    frustration_scores.append(torch.tensor(frustration_vals).mean().item())
    homo_lumo_scores.append(torch.tensor(homo_lumo_vals).mean().item())
    acid_base_scores.append(torch.tensor(acid_base_vals).mean().item())    
    losses_G.append(total_loss_G.item())
    losses_D.append(loss_D.item())
    val_loss = compute_validation_loss(generator, discriminator, val_loader, device)
    # lr_G_log.append(lr_scheduler_G.get_last_lr()[0])
    # lr_D_log.append(lr_scheduler_D.get_last_lr()[0])

    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss D: {loss_D.item()}, Loss G: {total_loss_G.item()}")
        plot_losses()
    
    # Save checkpoint
    torch.save({
        "G": generator.state_dict(),
        "D": discriminator.state_dict(),
        "opt_G": optimizer_G.state_dict(),
        "opt_D": optimizer_D.state_dict(),
        "epoch": epoch
    }, checkpoint_path)

    # Save metrics
    metrics["epoch"].append(epoch)
    metrics["loss_G"].append(total_loss_G.item())
    metrics["loss_D"].append(loss_D.item())
    metrics["val_loss"].append(val_loss)
    metrics["reward_total"].append(total_rewards[-1])
    metrics["reward_frustration"].append(frustration_scores[-1])
    metrics["reward_homo_lumo"].append(homo_lumo_scores[-1])
    metrics["reward_acid_base"].append(acid_base_scores[-1])
    # metrics["lr_G"].append(lr_G_log[-1])
    # metrics["lr_D"].append(lr_D_log[-1])
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)

    current_loss = total_loss_G.item()

    if current_loss < best_loss - delta:
        best_loss = current_loss
        no_improvement = 0
    else:
        no_improvement += 1

    if epoch >= min_epochs and no_improvement >= patience:
        print(f"Early stopping at epoch {epoch} due to no improvement in generator loss.")
        break


plt.figure()
plt.plot(metrics["epoch"], metrics["loss_G"], label="Generator")
plt.plot(metrics["epoch"], metrics["loss_D"], label="Discriminator")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Losses")
plt.savefig(os.path.join(output_dir, "loss_plot.png"))