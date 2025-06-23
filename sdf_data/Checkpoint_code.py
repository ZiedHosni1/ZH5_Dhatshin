import os, json, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem, DataStructs
import selfies
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# --------------------
# Output Directory Setup
# --------------------
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
checkpoint_path = os.path.join(output_dir, "checkpoint.pth")
metrics_path = os.path.join(output_dir, "training_metrics.csv")

# Load QM9 CSV file (make sure qm9.csv is in the same folder)
df = pd.read_csv('qm9.csv')

# Ensure it has a column named 'smiles'
assert 'smiles' in df.columns, "CSV must have a 'smiles' column."

# Convert SMILES to RDKit Mol objects
PandasTools.AddMoleculeColumnToFrame(df, smilesCol='smiles')

def generate_unique_pairs(base_smiles_list, acid_smiles_list, max_pairs):
    seen = set()
    unique_pairs = []

    while len(unique_pairs) < max_pairs:
        b = random.choice(base_smiles_list)
        a = random.choice(acid_smiles_list)
        pair_key = (b, a)

        if pair_key not in seen:
            seen.add(pair_key)
            unique_pairs.append({'flp_smiles': f"{b}.{a}", 'type': 'inter'})

        # Optional: break early if we've exhausted all possible unique pairs
        if len(seen) >= len(base_smiles_list) * len(acid_smiles_list):
            break

    return unique_pairs

def has_carbon_oxygen(mol):
    """Returns True if the molecule has at least one C and one O."""
    if mol is None:
        return False
    carbon = oxygen = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:
            carbon += 1
        elif atom.GetAtomicNum() == 8:
            oxygen += 1
    return carbon >= 1 and oxygen >= 1

def is_lewis_base(mol):
    """Checks if molecule has a Lewis base site (N or P with lone pair)."""
    if mol is None or not has_carbon_oxygen(mol):
        return False
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in [7, 15]:  # N, P
            if atom.GetTotalDegree() >= 1 and atom.GetFormalCharge() <= 0:
                return True
    return False

def is_lewis_acid(mol):
    """Checks if molecule has a Lewis acid site (Boron, carbonyl C, or cation)."""
    if mol is None or not has_carbon_oxygen(mol):
        return False
    for atom in mol.GetAtoms():
        num = atom.GetAtomicNum()
        if num == 5 and atom.GetTotalDegree() <= 3:
            return True  # Boron
        if num == 6:
            neighbors = [nbr.GetAtomicNum() for nbr in atom.GetNeighbors()]
            if atom.GetHybridization() == Chem.HybridizationType.SP2 and 8 in neighbors:
                return True  # C=O carbon
        if atom.GetFormalCharge() > 0:
            return True  # Positively charged site
    return False

def is_intramolecular_flp(mol):
    """Returns True if molecule has both base and acid sites (intramolecular FLP)."""
    return is_lewis_base(mol) and is_lewis_acid(mol)

def extract_all_flps(df, max_pairs=None):
    """
    Extracts both intramolecular FLPs and intermolecular FLP pairs (carbon + oxygen required).

    Args:
        df (pd.DataFrame): QM9 DataFrame with 'ROMol' and 'smiles' columns.
        max_pairs (int, optional): Max number of intermolecular pairs to generate.

    Returns:
        pd.DataFrame: Combined FLP candidates with a 'type' column: 'intra' or 'inter'.
    """
    print("🔎 Filtering intramolecular FLPs...")
    intra_df = df[df['ROMol'].apply(is_intramolecular_flp)].copy()
    intra_df['flp_smiles'] = intra_df['smiles']
    intra_df['type'] = 'intra'

    print(f"✅ Found {len(intra_df)} intramolecular FLPs.")

    print("🔎 Filtering Lewis bases and acids for intermolecular FLPs...")
    bases_df = df[df['ROMol'].apply(is_lewis_base)].copy()
    acids_df = df[df['ROMol'].apply(is_lewis_acid)].copy()

    print(f"✅ Found {len(bases_df)} bases and {len(acids_df)} acids for pairing.")

    # Build intermolecular pairs
    inter_data = []
    if not bases_df.empty and not acids_df.empty:
        base_smiles = bases_df['smiles'].tolist()
        acid_smiles = acids_df['smiles'].tolist()
        if max_pairs:
            seen = set()
            while len(inter_data) < max_pairs:
                b = random.choice(base_smiles)
                a = random.choice(acid_smiles)
                key = (b, a)
                if key not in seen:
                    seen.add(key)
                    inter_data.append({'flp_smiles': f"{b}.{a}", 'type': 'inter'})

                if len(seen) >= len(base_smiles) * len(acid_smiles):
                    break  # all unique pairs exhausted

        else:
            seen = set()
            for b in base_smiles:
                for a in acid_smiles:
                    pair_key = (b, a)
                    if pair_key not in seen:
                        seen.add(pair_key)
                        inter_data.append({'flp_smiles': f"{b}.{a}", 'type': 'inter'})

    inter_df = pd.DataFrame(inter_data)

    print(f"✅ Generated {len(inter_df)} intermolecular FLP pairs.")

    # Combine both
    combined_df = pd.concat([intra_df[['flp_smiles', 'type']], inter_df], ignore_index=True)
    print(f"🔗 Total FLP candidates: {len(combined_df)}")

    return combined_df

# Example use:
print("Extracting all FLP candidates (intra + inter)...")
flp_all_df = extract_all_flps(df, max_pairs=60000)
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
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

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

start_epoch = 0
metrics = {"epoch": [], "loss_G": [], "loss_D": []}

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

# --------------------
# Training Loop
# --------------------
optimizer_G = optim.Adam(generator.parameters(), lr=0.0006, betas=(0.5, 0.9))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0004, betas=(0.5, 0.9))

n_epochs = 5  # Adjust number of epochs as needed
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
            tokens = [dataset.idx_to_token[idx.item()] for idx in token_indices[i]]
            selfies_str = ''.join(tokens)
            selfies_batch.append(selfies_str)

        # Compute reward using geometric frustration + HOMO-LUMO proxy
        rewards = []
        for selfies_str in selfies_batch:
            try:
                smiles_str = selfies.decoder(selfies_str)
                features = extract_flp_features(smiles_str)

                # Combine reward components
                frustration = reward_geometric_frustration(features)
                homo_lumo = reward_homo_lumo_proxy(features)
                reward = 0.5 * frustration + 0.5 * homo_lumo
                rewards.append(reward)
            except:
                rewards.append(0.0)  # fallback if decoding fails

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



        
    losses_G.append(loss_G.item())
    losses_D.append(loss_D.item())
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss D: {loss_D.item()}, Loss G: {loss_G.item()}")
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
    metrics["loss_G"].append(loss_G.item())
    metrics["loss_D"].append(loss_D.item())
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)

plt.figure()
plt.plot(metrics["epoch"], metrics["loss_G"], label="Generator")
plt.plot(metrics["epoch"], metrics["loss_D"], label="Discriminator")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Losses")
plt.savefig(os.path.join(output_dir, "loss_plot.png"))