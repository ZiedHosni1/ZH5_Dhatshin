# %%
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
from selfies import encoder, split_selfies
from tqdm import tqdm
from rdkit.Chem import PandasTools
from itertools import product
from pathlib import Path
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# %%
print(torch.version.cuda)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# %%
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# %%
# --------------------
# Output Directory Setup
# --------------------
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
checkpoint_path = os.path.join(output_dir, "checkpoint.pth")
metrics_path = os.path.join(output_dir, "training_metrics.csv")

# %%
# Load QM9 CSV file (make sure qm9.csv is in the same folder)
acid = pd.read_csv('acid.csv')
acid = acid.sample(frac=0.1, random_state=42)
base = pd.read_csv('zinc.csv')

# %%
# Ensure it has a column named 'smiles'
assert 'smiles' in acid.columns, "CSV must have a 'smiles' column."
assert 'smiles' in base.columns, "CSV must have a 'smiles' column."

# %%
# Convert SMILES to RDKit Mol objects
PandasTools.AddMoleculeColumnToFrame(acid, smilesCol='smiles')
PandasTools.AddMoleculeColumnToFrame(base, smilesCol='smiles')

def generate_flp_pairs(df_acid, df_base, acid_col='smiles', base_col='smiles'):
    assert acid_col in df_acid.columns, f"Acid DataFrame must contain column '{acid_col}'"
    assert base_col in df_base.columns, f"Base DataFrame must contain column '{base_col}'"

    acids = df_acid[acid_col].unique()
    bases = df_base[base_col].unique()

    flp_data = [{'flp_smiles': f"{b}.{a}", 'type': 'inter'} for a, b in product(acids, bases)]
    flp_df = pd.DataFrame(flp_data)

    return flp_df

# %%
# Example use:
print("Extracting all FLP candidates (intra + inter)...")
flp_all_df = generate_flp_pairs(acid, base)
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
            AllChem.ComputeGasteigerCharges(base)
            AllChem.ComputeGasteigerCharges(acid)

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
            AllChem.ComputeGasteigerCharges(mol)
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
# %%

def build_selfies_vocab(smiles_list):
    tokens = set()
    for smi in smiles_list[:50000]:
        try:
            selfies_str = encoder(smi)
            tokens.update(split_selfies(selfies_str))
        except:
            continue
    return sorted(tokens)

# %%
# --------------------
# Data Preprocessing: Convert SMILES to tokenized SELFIES
# --------------------

class MolecularSELFIESDataset(Dataset):
    def __init__(self, smiles_list, max_seq_len=60, save_path="tokenized_dataset.pt"):
        self.max_seq_len = max_seq_len
        self.save_path = save_path
        self.smiles = smiles_list

        # ----------------------------------------
        # Load preprocessed dataset and vocab
        # ----------------------------------------
        if Path(save_path).exists() and Path("vocab.json").exists():
            print(f"? Loading preprocessed token tensors from {save_path}...")
            self.tokenized_tensors = torch.load(save_path)

            with open("vocab.json", "r") as f:
                vocab_data = json.load(f)
            self.token_to_idx = vocab_data["token_to_idx"]
            self.idx_to_token = {int(k): v for k, v in vocab_data["idx_to_token"].items()}
            self.vocab_size = max(self.token_to_idx.values()) + 1
            return

        # ----------------------------------------
        # Build vocab from SMILES
        # ----------------------------------------
        print("?? Building vocabulary...")
        sample_tokens = []
        for smi in self.smiles[:50000]:
            try:
                tokens = list(selfies.split_selfies(selfies.encoder(smi)))
                sample_tokens.extend(tokens)
            except Exception:
                continue

        unique_tokens = sorted(set(sample_tokens))
        self.token_to_idx = {token: idx + 1 for idx, token in enumerate(unique_tokens)}  # 0 = PAD
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        self.vocab_size = max(self.token_to_idx.values()) + 1

        # Save vocab
        vocab_data = {
            "token_to_idx": self.token_to_idx,
            "idx_to_token": self.idx_to_token
        }
        with open("vocab.json", "w") as f:
            json.dump(vocab_data, f)

        # ----------------------------------------
        # Tokenize and pad
        # ----------------------------------------
        print("?? Preprocessing SELFIES to token tensors...")
        self.tokenized_tensors = []
        for smi in tqdm(self.smiles, desc="Tokenizing"):
            try:
                selfies_str = selfies.encoder(smi)
                tokens = list(selfies.split_selfies(selfies_str))
                token_indices = [self.token_to_idx.get(t, 0) for t in tokens]
                padded = token_indices + [0] * (self.max_seq_len - len(token_indices))
                padded = padded[:self.max_seq_len]
                self.tokenized_tensors.append(torch.tensor(padded, dtype=torch.long))
            except:
                self.tokenized_tensors.append(torch.zeros(self.max_seq_len, dtype=torch.long))

        # ----------------------------------------
        # Save to disk
        # ----------------------------------------
        print(f"?? Saving tokenized dataset to {save_path}...")
        torch.save(self.tokenized_tensors, save_path)
    def __getitem__(self, idx):
        return self.tokenized_tensors[idx]

    def __len__(self):
        return len(self.tokenized_tensors)
# %%

def one_hot_encode(tokens, vocab_size):
    return torch.nn.functional.one_hot(tokens, num_classes=vocab_size).float()

# %%
# Extract SMILES list and create the dataset and dataloader
# ✅ NEW: Use the 'flp_smiles' column (includes both intra & inter)
smiles_list = flp_all_df['flp_smiles'].dropna().tolist()

# %%
dataset = MolecularSELFIESDataset(smiles_list)
# Adjust validation size as needed (e.g., 5% of total)
val_size = int(0.05 * len(dataset))
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
batch_size = 128  # or your current batch size

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

batch_size=128

# %%
print(f"✅ Dataset successfully loaded with {len(dataset)} molecules for training.")
print(f"Vocabulary size: {dataset.vocab_size}, Sequence length: {dataset.max_seq_len}")

# %%
# Data dimension: each molecule is represented as a flattened vector
data_dim = dataset.max_seq_len * dataset.vocab_size

# %%
# --------------------
# WGAN Architecture
# --------------------
latent_dim = 128
embed_dim = 128
hidden_dim = 256
seq_len = dataset.max_seq_len
vocab_size = dataset.vocab_size

class Generator(nn.Module):
    def __init__(self, latent_dim, embed_dim, hidden_dim, seq_len, vocab_size):
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.rnn = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.embed_out = nn.Linear(hidden_dim, vocab_size)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim



    def forward(self, z):
        batch_size = z.size(0)
        h0 = torch.tanh(self.fc(z)).unsqueeze(0)  # initial hidden state

        # input tokens = <start> token = zeros
        inputs = torch.zeros(batch_size, self.seq_len, self.token_embedding.embedding_dim, device=z.device)
        output, _ = self.rnn(inputs, h0)
        logits = self.embed_out(output)
        return logits  # shape: [B, L, V]

    
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

# %%
# Initialize models
generator = Generator(latent_dim, embed_dim, hidden_dim, seq_len, vocab_size).to(device)
discriminator = Discriminator(input_dim=data_dim).to(device)

@torch.no_grad()
def compute_validation_loss(generator, discriminator, val_loader, device):
    generator.eval()
    discriminator.eval()

    total_loss = 0.0
    num_batches = 0

    for real_samples in val_loader:
        batch_size = real_samples.size(0)
        
        # Prepare real samples
        real_samples = real_samples.long().to(device)
        real_samples = torch.nn.functional.one_hot(real_samples, num_classes=generator.vocab_size)
        real_samples = real_samples.float().view(batch_size, -1)

        # Generate fake samples
        z = torch.randn(batch_size, generator.latent_dim).to(device)
        fake_samples = generator(z).view(batch_size, -1)

        # Discriminator loss on validation set (WGAN)
        loss_D_real = -torch.mean(discriminator(real_samples))
        loss_D_fake = torch.mean(discriminator(fake_samples))
        loss_D = loss_D_real + loss_D_fake

        total_loss += loss_D.item()
        num_batches += 1

    generator.train()
    discriminator.train()

    return total_loss / num_batches if num_batches > 0 else None


def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, device=real_samples.device)
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

LEWIS_BASE_SMARTS = [
    "[#7;H2,H1;!$(NC=O)]",  # primary or secondary amines
    "[O-]",                 # negative oxygen
    "[nH]",                 # aromatic N-H
    "[#8;H1]"               # alcohol OH
]

LEWIS_ACID_SMARTS = [
    "[B]",                  # boron
    "[P+]",                 # phosphonium
    "[Al]",                 # aluminium
    "[Sn]",                 # tin
    "[Si]"                  # silicon
]

def contains_substructure(mol, smarts_list):
    for smarts in smarts_list:
        patt = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(patt):
            return True
    return False

from rdkit import Chem

# Precompiled SMARTS (reuse your existing lists/patterns)
BASE_SMARTS = [Chem.MolFromSmarts(s) for s in [
    "[#7;H2,H1;!$(NC=O)]","[O-]","[nH]","[#8;H1]"
]]
ACID_SMARTS = [Chem.MolFromSmarts(s) for s in [
    "[B]","[P+]","[Al]","[Sn]","[Si]"
]]

def frag_has(mol, patterns):
    return any(mol.HasSubstructMatch(p) for p in patterns)

def reward_bimolecular_flp(smiles: str,
                           prefer_exactly_two=True,
                           allow_intramolecular=False) -> float:
    """
    Reward 1.0 when the SMILES has exactly two fragments and 
    one fragment is acid-like and the other base-like.
    Optionally allow intramolecular FLP with smaller reward.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0

        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        n = len(frags)

        # Penalize too many or too few fragments
        if prefer_exactly_two:
            if n == 2:
                a_has_acid = frag_has(frags[0], ACID_SMARTS) or frag_has(frags[1], ACID_SMARTS)
                b_has_base = frag_has(frags[0], BASE_SMARTS) or frag_has(frags[1], BASE_SMARTS)
                return 1.0 if (a_has_acid and b_has_base) else 0.0
            elif n == 1 and allow_intramolecular:
                # smaller reward if a single fragment contains both motifs
                has_acid  = frag_has(frags[0], ACID_SMARTS)
                has_base  = frag_has(frags[0], BASE_SMARTS)
                return 0.4 if (has_acid and has_base) else 0.0
            else:
                # >2 frags or 1 frag (when intramolecular not allowed)
                return 0.0
        else:
            # More permissive: any number of frags, but must find acid in one and base in another
            if n >= 2:
                acid_any = any(frag_has(f, ACID_SMARTS) for f in frags)
                base_any = any(frag_has(f, BASE_SMARTS) for f in frags)
                return 1.0 if (acid_any and base_any) else 0.0
            # Fall back to intramolecular if allowed
            if n == 1 and allow_intramolecular:
                has_acid  = frag_has(frags[0], ACID_SMARTS)
                has_base  = frag_has(frags[0], BASE_SMARTS)
                return 0.4 if (has_acid and has_base) else 0.0
            return 0.0
    except:
        return 0.0


def reward_contains_flp(smiles: str) -> float:
    """Reward presence of both Lewis acid and base SMARTS patterns."""
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
        "[Si]"                  # silicon
    ]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    num_bases = sum(mol.HasSubstructMatch(Chem.MolFromSmarts(s)) for s in LEWIS_BASE_SMARTS)
    num_acids = sum(mol.HasSubstructMatch(Chem.MolFromSmarts(s)) for s in LEWIS_ACID_SMARTS)
    
    # Require at least 2 bases or 2 acids for more complexity
    if num_bases >= 1 and num_acids >= 1:
        return 0.5 + 0.05 * min(num_bases, 2) + 0.05 * min(num_acids, 2)  # up to 0.6
    return 0.0

def penalize_simple_molecule(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    num_atoms = mol.GetNumAtoms()
    num_unique_atoms = len(set(atom.GetSymbol() for atom in mol.GetAtoms()))
    return min(0.2, 0.05 * num_unique_atoms)  # encourage at least some diversity



REDUCIBLE_SMARTS = [
    "[C]=[O]",          # carbonyl
    "[C]=[N]",          # imine
    "[N+](=O)[O-]",     # nitro
    "[C]=[C]",          # alkene
    "[N]=[N+]=[N-]",    # azide
]

def reward_hydrogenation_potential(smiles: str) -> float:
    """Reward based on number of reducible groups (scaled to max 1.0)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    matches = 0
    for smarts in REDUCIBLE_SMARTS:
        patt = Chem.MolFromSmarts(smarts)
        matches += len(mol.GetSubstructMatches(patt))
    return min(matches, 3) / 3.0  # reward capped at 1.0

def combined_flp_reward(smiles: str) -> float:
    base_acid = reward_contains_flp(smiles)
    hydrogenation = reward_hydrogenation_potential(smiles)
    return 0.7 * base_acid + 0.3 * hydrogenation  # adjust weights as needed


# %%
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

# %%
# --------------------
# Training Loop
# --------------------
optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0004, betas=(0.5, 0.999))
lr_scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_G, mode='min', factor=0.7, patience=10, min_lr=1e-6
)
lr_scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=20, gamma=0.9)

warmup_epochs = 5
lambda_rl = 0.1

start_epoch = 0
metrics = {
    "epoch": [],
    "loss_G": [],
    "loss_D": [],
    "val_loss": [],
    "reward_total": [],
    "reward_flp": [],
    "reward_simplicity": [],
    "entropy": [],
    "tau": [],
    "entropy_weight": [],
    "lr_G": [],
    "lr_D": []
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


n_epochs = 500  # Adjust number of epochs as needed
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
flp_scores = []
simplicity_penalties = []
frustration_scores = []
homo_lumo_scores = []
acid_base_scores = []
entropy_log = []
lr_G_log = []
lr_D_log = []
best_loss = float("inf")        # guarantees first loss is lower
no_improvement = 0
delta = 0.0
min_epochs = 50
patience = 10  # moderately adaptive
initial_tau = 2.0              # Gumbel-softmax initial temperature
min_tau = 0.5                  # Final annealed temperature
anneal_rate = 0.95             # Multiplicative decay per epoch

initial_entropy_weight = 0.1  # Start with high entropy
min_entropy_weight = 0.0      # Gradually remove it




print(f"starting training loop", flush=True)

#  %%

n_critic = 5
tau = initial_tau
entropy_weight = initial_entropy_weight

for epoch in range(n_epochs):
    for i, real_samples in enumerate(train_loader):
        real_samples = real_samples.long().to(device)
        real_samples = F.one_hot(real_samples, num_classes=dataset.vocab_size).float()
        real_samples = real_samples.view(real_samples.size(0), -1)
        batch_size = real_samples.size(0)

        # --- 1. Train Discriminator ---
        optimizer_D.zero_grad()
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_samples = generator(z).detach().view(batch_size, -1)

        loss_D_real = -torch.mean(discriminator(real_samples))
        loss_D_fake = torch.mean(discriminator(fake_samples))
        gp = compute_gradient_penalty(discriminator, real_samples, fake_samples)
        loss_D = loss_D_real + loss_D_fake + lambda_gp * gp
        loss_D.backward()
        optimizer_D.step()

        # --- 2. Train Generator every n_critic steps ---
        if i % n_critic == 0:
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_dim).to(device)
            logits = generator(z)  # [B, L, V]
            logits = logits.view(batch_size, dataset.max_seq_len, dataset.vocab_size)
            token_probs = torch.softmax(logits, dim=-1)
            entropy = -(token_probs * torch.log(torch.clamp(token_probs, min=1e-8))).sum(dim=-1).mean()
            entropy_bonus = 0.01 * entropy  # small scaling factor

            gumbel_samples = F.gumbel_softmax(logits, tau=tau, hard=True)
            token_indices = gumbel_samples.argmax(dim=-1)  # For reward computation

            selfies_batch = []
            for row in token_indices:
                tokens = [dataset.idx_to_token.get(idx.item(), '') for idx in row if idx.item() != 0]
                selfies_str = ''.join(tokens)
                selfies_batch.append(selfies_str)

            # Compute rewards
            rewards = []
            flp_vals = []
            simplicity_vals = []
            for selfies_str in selfies_batch:
                try:
                    smiles_str = selfies.decoder(selfies_str)
                    r_flp = reward_bimolecular_flp(smiles_str, prefer_exactly_two=True, allow_intramolecular=False)
                    # flp_score = reward_contains_flp(smiles_str)
                    flp_vals.append(r_flp)
                    diversity_penalty = penalize_simple_molecule(smiles_str)
                    simplicity_vals.append(diversity_penalty)
                    reward = 0.9 * r_flp + 0.1 * diversity_penalty
                    rewards.append(reward)
                except:
                    rewards.append(0.0)

            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

            gathered_log_probs = torch.sum(gumbel_samples * torch.log(torch.clamp(token_probs, min=1e-8)), dim=-1)
            log_probs = gathered_log_probs.sum(dim=1)

            rl_loss = -(log_probs * rewards).mean()
            gan_loss = -torch.mean(discriminator(logits.view(batch_size, -1)))


            if epoch >= warmup_epochs:
                total_loss_G = gan_loss + lambda_rl * rl_loss + entropy_weight * entropy
            else:
                total_loss_G = gan_loss


            if not torch.isnan(total_loss_G):
                total_loss_G.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                optimizer_G.step()
            else:
                print("⚠️ Generator loss is NaN — skipping optimizer step.")
            
    # --- Annealing ---
    tau = max(min_tau, tau * anneal_rate)
    entropy_weight = max(min_entropy_weight, entropy_weight * anneal_rate)


    total_rewards.append(torch.tensor(rewards).mean().item())
    flp_scores.append(torch.tensor(flp_vals).mean().item())
    simplicity_penalties.append(torch.tensor(simplicity_vals).mean().item())
    entropy_log.append(entropy.item())

    val_loss = compute_validation_loss(generator, discriminator, val_loader, device)
    lr_scheduler_G.step(val_loss)
    lr_scheduler_D.step()

    losses_D.append(loss_D.item())
    lr_G_log.append(lr_scheduler_G.get_last_lr()[0])
    lr_D_log.append(lr_scheduler_D.get_last_lr()[0])

    # Logging
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss D: {loss_D.item()}, Loss G: {total_loss_G.item() if losses_G else 'N/A'}")
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
    print(f"saving metrics: epoch {epoch}", flush=True)
    metrics["epoch"].append(epoch)
    metrics["loss_G"].append(losses_G[-1] if losses_G else None)
    metrics["loss_D"].append(loss_D.item())
    metrics["val_loss"].append(val_loss)
    metrics["reward_total"].append(total_rewards[-1])
    metrics["reward_flp"].append(flp_scores[-1])
    metrics["reward_simplicity"].append(simplicity_penalties[-1])
    metrics["entropy"].append(entropy_log[-1])
    metrics["lr_G"].append(lr_G_log[-1])
    metrics["lr_D"].append(lr_D_log[-1])
    metrics["tau"].append(tau)
    metrics["entropy_weight"].append(entropy_weight)
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)

    # Early stopping logic
    current_loss = losses_G[-1] if losses_G else float('inf')
    if current_loss < best_loss - delta:
        best_loss = current_loss
        no_improvement = 0
    else:
        no_improvement += 1

    if epoch >= min_epochs and no_improvement >= patience:
        print(f"Early stopping at epoch {epoch} due to no improvement in generator loss.")
        break


# %%
plt.figure()
plt.plot(metrics["epoch"], metrics["loss_G"], label="Generator")
plt.plot(metrics["epoch"], metrics["loss_D"], label="Discriminator")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Losses")
plt.savefig(os.path.join(output_dir, "loss_plot.png"))
# %%
