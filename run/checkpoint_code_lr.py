from __future__ import annotations
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
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--run_id', type=int, default=0)
args = parser.parse_args()

# Use args.run_id to create unique output folders
output_dir = f"outputs/run_{args.run_id}"
os.makedirs(output_dir, exist_ok=True)


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

# ---- Acid validity helpers ----
from rdkit.Chem.rdmolops import GetMolFrags

B_TRICOORD_NEUTRAL = Chem.MolFromSmarts("[B&D3&+0]")  # boron, degree 3, neutral

def _sanitize_or_none(m):
    try:
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None

def acid_fragment_has_valid_boron(acid_mol: Chem.Mol) -> bool:
    m = _sanitize_or_none(Chem.Mol(acid_mol))
    if m is None:
        return False
    return m.HasSubstructMatch(B_TRICOORD_NEUTRAL)

def pick_acid_fragment(frags) -> Chem.Mol | None:
    """Pick the fragment that contains any acid SMARTS. If multiple, return the first."""
    for f in frags:
        if any(f.HasSubstructMatch(p) for p in _ACID_SMARTS):
            return f
    return None

def flp_has_valid_boron_acid(smiles: str) -> bool:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return False
    frags = GetMolFrags(m, asMols=True, sanitizeFrags=False)
    if len(frags) != 2:
        return False  # we only accept bimolecular FLPs here
    acid = pick_acid_fragment(frags)
    if acid is None:
        return False
    return acid_fragment_has_valid_boron(acid)


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

# ---- RL REWARD HELPERS (FLP bimolecular + simplicity shaping) ----
from rdkit import Chem

# Precompile SMARTS once
_BASE_SMARTS = [Chem.MolFromSmarts(s) for s in [
    "[#7;H2,H1;!$(NC=O)]",  # primary/secondary amines
    "[O-]",                 # anionic O
    "[nH]",                 # aromatic N-H
    "[#8;H1]"               # alcohol OH
]]
_ACID_SMARTS = [Chem.MolFromSmarts(s) for s in [
    "[B]", "[P+]", "[Al]", "[Sn]", "[Si]"
]]

def _frag_has(mol, patterns):
    return any(mol.HasSubstructMatch(p) for p in patterns)

def reward_bimolecular_flp(smiles: str,
                           prefer_exactly_two=True,
                           allow_intramolecular=False) -> float:
    """
    Reward 1.0 when the SMILES has exactly two fragments and
    one fragment is acid-like and the other base-like, AND
    the acid fragment has neutral tricoordinate boron.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0

        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        n = len(frags)

        if prefer_exactly_two:
            if n == 2:
                # original acid/base checks
                acid_any = any(frag_has(f, _ACID_SMARTS) for f in frags)
                base_any = any(frag_has(f, _BASE_SMARTS) for f in frags)
                if not (acid_any and base_any):
                    return 0.0
                # NEW: acid validity gate
                if not flp_has_valid_boron_acid(smiles):
                    return 0.0
                return 1.0
            elif n == 1 and allow_intramolecular:
                has_acid  = frag_has(frags[0], _ACID_SMARTS)
                has_base  = frag_has(frags[0], _BASE_SMARTS)
                if not (has_acid and has_base):
                    return 0.0
                # For intramolecular, still require valid boron if an acid is present
                return 0.4 if flp_has_valid_boron_acid(smiles) else 0.0
            else:
                return 0.0
        else:
            if n >= 2:
                acid_any = any(frag_has(f, _ACID_SMARTS) for f in frags)
                base_any = any(frag_has(f, _BASE_SMARTS) for f in frags)
                if not (acid_any and base_any):
                    return 0.0
                return 1.0 if flp_has_valid_boron_acid(smiles) else 0.0
            if n == 1 and allow_intramolecular:
                has_acid  = frag_has(frags[0], _ACID_SMARTS)
                has_base  = frag_has(frags[0], _BASE_SMARTS)
                if not (has_acid and has_base):
                    return 0.0
                return 0.4 if flp_has_valid_boron_acid(smiles) else 0.0
            return 0.0
    except:
        return 0.0

def penalize_simple_molecule(smiles: str) -> float:
    """
    Small shaping bonus that favors a bit of complexity (heteroatoms/rings/size).
    Scaled conservatively so it doesn't dominate the FLP objective.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    num_atoms = mol.GetNumAtoms()
    heteros = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))  # not H or C
    rings = int(Chem.GetSSSR(mol))  # number of rings

    # Cap each term and scale to [0, 0.3] total
    size_term   = min(num_atoms / 25.0, 1.0) * 0.10
    hetero_term = min(heteros   /  5.0, 1.0) * 0.15
    ring_term   = min(rings     /  3.0, 1.0) * 0.05
    return size_term + hetero_term + ring_term


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

# --------------------
# Optimizers & schedulers
# --------------------
optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=4e-4, betas=(0.5, 0.999))

# ReduceLROnPlateau for G (needs val metric); StepLR for D
lr_scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_G, mode='min', factor=0.7, patience=10, min_lr=1e-6
)
lr_scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=20, gamma=0.9)

# --------------------
# Resume checkpoint (if any)
# --------------------
start_epoch = 0
metrics = {
    "epoch": [], "loss_G": [], "loss_D": [], "val_loss": [],
    "reward_total": [], "reward_flp": [], "reward_simplicity": [],
    "entropy": [], "tau": [], "entropy_weight": [],
    "lr_G": [], "lr_D": []
}

if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(ckpt["G"])
    discriminator.load_state_dict(ckpt["D"])
    optimizer_G.load_state_dict(ckpt["opt_G"])
    optimizer_D.load_state_dict(ckpt["opt_D"])
    start_epoch = ckpt["epoch"] + 1
    if os.path.exists(metrics_path):
        metrics = pd.read_csv(metrics_path).to_dict(orient="list")
    print(f"Resuming from epoch {start_epoch}")

# --------------------
# Hyperparams & logs
# --------------------
n_epochs = 500
lambda_gp = 10.0
n_critic = 3                  # critic steps per G step
warmup_epochs = 10            # delay RL
lambda_rl = 0.1               # RL weight

# Annealing
initial_tau = 2.0
min_tau = 0.5
anneal_rate = 0.95

initial_entropy_weight = 0.1
min_entropy_weight = 0.0

tau = initial_tau
entropy_weight = initial_entropy_weight

# Early stopping
patience = 50
best_loss = float("inf")
no_improvement = 0
delta = 0.0
min_epochs = 100

# --------------------
# Trackers
# --------------------
metrics = {
    "epoch": [],
    "loss_D": [],
    "loss_G": [],
    "val_loss": [],
    "reward_total": [],
    "reward_flp": [],
    "reward_simplicity": [],
    "entropy": [],
    "tau": [],
    "entropy_weight": [],
    "lr_G": [],
    "lr_D": [],
    "gp": [],
}

print("Starting training loop", flush=True)

# --------------------
# Training
# --------------------
for epoch in range(start_epoch, n_epochs):

    D_epoch_vals, G_epoch_vals = [], []
    GP_epoch_vals = []
    reward_epoch_vals, flp_epoch_vals, simp_epoch_vals, ent_epoch_vals = [], [], [], []
    D_real_epoch_vals, D_fake_epoch_vals = [], []

    for i, real_tokens in enumerate(train_loader):
        # ----- prepare real -----
        real_tokens = real_tokens.long().to(device)                           # [B, L]
        real = F.one_hot(real_tokens, num_classes=dataset.vocab_size).float() # [B, L, V]
        real = real.view(real.size(0), -1)                                    # [B, L*V]
        batch_size = real.size(0)

        # ----- train D -----
        optimizer_D.zero_grad()
        z = torch.randn(batch_size, latent_dim, device=device)
        # Fake for D uses detach
        logits_fake = generator(z).view(batch_size, dataset.max_seq_len, dataset.vocab_size)  # [B, L, V]
        with torch.no_grad():
            gumbel_fake = F.gumbel_softmax(logits_fake, tau=max(tau, 1.0), hard=True)  # stable temp for D
        fake = gumbel_fake.view(batch_size, -1)  # [B, L*V]

        d_real = discriminator(real)
        d_fake = discriminator(fake)

        loss_D_real = -torch.mean(d_real)
        loss_D_fake =  torch.mean(d_fake)
        gp = compute_gradient_penalty(discriminator, real, fake)
        loss_D = loss_D_real + loss_D_fake + lambda_gp * gp

        loss_D.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
        optimizer_D.step()

        D_epoch_vals.append(loss_D.item())
        GP_epoch_vals.append(gp.item())
        D_real_epoch_vals.append((-loss_D_real).item())  # E[D(real)]
        D_fake_epoch_vals.append((loss_D_fake).item())   # E[D(fake)]

        # ----- train G every n_critic -----
        if i % n_critic == 0:
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_dim, device=device)
            logits = generator(z).view(batch_size, dataset.max_seq_len, dataset.vocab_size)  # [B, L, V]

            # token probs & entropy (for bonus)
            token_probs = torch.softmax(logits, dim=-1)
            entropy = -(token_probs * torch.log(torch.clamp(token_probs, 1e-8, 1.0))).sum(dim=-1).mean()

            # Gumbel-Softmax with annealed temperature
            gumbel_samples = F.gumbel_softmax(logits, tau=tau, hard=True)  # [B, L, V]
            token_indices = gumbel_samples.argmax(dim=-1)  # [B, L]

            # Decode SELFIES -> SMILES in batch
            selfies_batch = [
                "".join([dataset.idx_to_token.get(idx.item(), "") for idx in row if idx.item() != 0])
                for row in token_indices
            ]

            # Compute rewards
            rewards, flp_vals, simp_vals = [], [], []
            for selfies_str in selfies_batch:
                # inside the for selfies_str in selfies_batch loop
                smiles_str = selfies.decoder(selfies_str)

                # HARD FILTER: enforce a valid boron acid before scoring
                if not flp_has_valid_boron_acid(smiles_str):
                    rewards.append(0.0)
                    flp_vals.append(0.0)
                    simp_vals.append(0.0)
                    continue

                r_flp = reward_bimolecular_flp(smiles_str, prefer_exactly_two=True, allow_intramolecular=False)
                flp_vals.append(r_flp)
                diversity_penalty = penalize_simple_molecule(smiles_str)
                simp_vals.append(diversity_penalty)
                reward = 0.9 * r_flp + 0.1 * diversity_penalty
                rewards.append(reward)

            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            # normalize and clamp to stabilize RL gradients
            std = rewards.std()
            if std > 1e-6:
                rewards = (rewards - rewards.mean()) / (std + 1e-6)
            rewards = rewards.clamp(-1.0, 1.0)

            # policy gradient (REINFORCE with gumbel proxy log-probs)
            gathered_log_probs = (gumbel_samples * torch.log(torch.clamp(token_probs, 1e-8, 1.0))).sum(dim=-1)  # [B, L]
            log_probs = gathered_log_probs.sum(dim=1)  # [B]
            rl_loss = -(log_probs * rewards).mean()

            # GAN loss: train in same space as D (flattened one-hot), no detach for G
            gan_input = gumbel_samples.view(batch_size, -1)    # [B, L*V]
            gan_loss = -torch.mean(discriminator(gan_input))

            # total generator loss
            if epoch >= warmup_epochs:
                total_loss_G = gan_loss + lambda_rl * rl_loss + entropy_weight * entropy
            else:
                total_loss_G = gan_loss

            if not torch.isnan(total_loss_G):
                total_loss_G.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                optimizer_G.step()
                G_epoch_vals.append(total_loss_G.item())

            # collect batch stats
            reward_epoch_vals.append(float(rewards.mean().item()))
            flp_epoch_vals.append(float(torch.tensor(flp_vals).mean().item()))
            simp_epoch_vals.append(float(torch.tensor(simp_vals).mean().item()))
            ent_epoch_vals.append(float(entropy.item()))

    # ---- end epoch: averages ----
    val_loss = compute_validation_loss(
        generator, discriminator, val_loader, device)

    # Step schedulers
    lr_scheduler_G.step(val_loss)
    lr_scheduler_D.step()

    loss_D_epoch = float(sum(D_epoch_vals) / max(1, len(D_epoch_vals)))
    loss_G_epoch = float(sum(G_epoch_vals) / len(G_epoch_vals)) if len(G_epoch_vals) else None
    gp_epoch     = float(sum(GP_epoch_vals) / max(1, len(GP_epoch_vals)))

    mean_reward = float(sum(reward_epoch_vals)/len(reward_epoch_vals)) if reward_epoch_vals else None
    mean_flp    = float(sum(flp_epoch_vals)/len(flp_epoch_vals))       if flp_epoch_vals else None
    mean_simp   = float(sum(simp_epoch_vals)/len(simp_epoch_vals))     if simp_epoch_vals else None
    mean_ent    = float(sum(ent_epoch_vals)/len(ent_epoch_vals))       if ent_epoch_vals else None

    # anneal after epoch
    tau = max(min_tau, tau * anneal_rate)
    entropy_weight = max(min_entropy_weight, entropy_weight * anneal_rate)

    # log
    lr_G = optimizer_G.param_groups[0]['lr']
    lr_D = optimizer_D.param_groups[0]['lr']

    metrics["epoch"].append(epoch)
    metrics["loss_D"].append(loss_D_epoch)
    metrics["loss_G"].append(loss_G_epoch)
    metrics["val_loss"].append(val_loss)
    metrics["reward_total"].append(mean_reward)
    metrics["reward_flp"].append(mean_flp)
    metrics["reward_simplicity"].append(mean_simp)
    metrics["entropy"].append(mean_ent)
    metrics["tau"].append(tau)
    metrics["entropy_weight"].append(entropy_weight)
    metrics["lr_G"].append(lr_G)
    metrics["lr_D"].append(lr_D)
    metrics["gp"].append(gp_epoch)

    pd.DataFrame(metrics).to_csv(metrics_path, index=False)

    if epoch % 50 == 0:
        g_print = f"{loss_G_epoch:.4f}" if loss_G_epoch is not None else "N/A"
        print(f"Epoch {epoch} | D {loss_D_epoch:.4f} | G {g_print} | Val {val_loss:.4f} | GP {gp_epoch:.4f} | tau {tau:.3f} | ent_w {entropy_weight:.3f}")

    # save checkpoint
    torch.save({
        "G": generator.state_dict(),
        "D": discriminator.state_dict(),
        "opt_G": optimizer_G.state_dict(),
        "opt_D": optimizer_D.state_dict(),
        "epoch": epoch,
        "tau": tau,
        "entropy_weight": entropy_weight,
        "metrics_last": {k: metrics[k][-1] for k in metrics},
    }, checkpoint_path)

    # early stopping on G
    current_loss = loss_G_epoch if loss_G_epoch is not None else float("inf")
    if current_loss < best_loss - delta:
        best_loss = current_loss
        no_improvement = 0
    else:
        no_improvement += 1
    if epoch >= min_epochs and no_improvement >= patience:
        print(f"Early stopping at epoch {epoch} (no G improvement).")
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
