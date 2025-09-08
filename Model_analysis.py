#%%
#! =========================
#! Cell 0 — Imports & Config
#! =========================
import os, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Descriptors, QED
from rdkit.Chem import rdDistGeom
import selfies

# ---------- User config ----------
ROOT = Path("output")            # expects output/run_0 ... output/run_5
RUN_IDS = list(range(6))         # adjust if you have a different count
VOCAB_PATH = Path("vocab.json")  # shared vocab stored during training
MAX_SEQ_LEN = 60                 # must match training
LATENT_DIM = 128                 # must match training
EMBED_DIM  = 128                 # must match training
HIDDEN_DIM = 256                 # must match training
SAMPLES_PER_RUN_IF_MISSING = 300 # number of samples to generate if file is missing
MIN_VALID_FOR_STATS = 10         # minimum valid molecules to compute metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

#%%
#! ===================================
#! Cell 1 — Load vocab and rebuild GNN
#! ===================================
with open(VOCAB_PATH, "r") as f:
    vocab_json = json.load(f)
token_to_idx = vocab_json["token_to_idx"]
idx_to_token = {int(k): v for k, v in vocab_json["idx_to_token"].items()}
VOCAB_SIZE = max(token_to_idx.values()) + 1

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
        b = z.size(0)
        h0 = torch.tanh(self.fc(z)).unsqueeze(0)  # [1, B, H]
        inputs = torch.zeros(b, self.seq_len, self.token_embedding.embedding_dim, device=z.device)
        out, _ = self.rnn(inputs, h0)             # [B, L, H]
        return self.embed_out(out)                 # [B, L, V]

#%%
#! ===========================================
#! Cell 2 — Load checkpoints & training metrics
#! ===========================================
def load_generator_for_run(run_dir: Path):
    ckpt_path = run_dir / "checkpoint.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    G = Generator(LATENT_DIM, EMBED_DIM, HIDDEN_DIM, MAX_SEQ_LEN, VOCAB_SIZE).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(ckpt["G"])
    G.eval()
    return G, ckpt.get("epoch", None)

def load_metrics_for_run(run_dir: Path) -> pd.DataFrame:
    mpath = run_dir / "training_metrics.csv"
    if not mpath.exists():
        raise FileNotFoundError(f"Missing metrics: {mpath}")
    df = pd.read_csv(mpath)
    df["run"] = run_dir.name
    return df

generators = {}         # { "run_0": (Generator, last_epoch), ... }
metrics_frames = []     # list of per-run metrics DataFrames

for rid in RUN_IDS:
    rd = ROOT / f"run_{rid}"
    try:
        G, last_epoch = load_generator_for_run(rd)
        generators[rd.name] = (G, last_epoch)
        print(f"Loaded {rd.name}: epoch={last_epoch}")
    except Exception as e:
        print(f"[warn] Generator load failed for {rd.name}: {e}")

    try:
        mdf = load_metrics_for_run(rd)
        metrics_frames.append(mdf)
        print(f"Loaded metrics for {rd.name}: {len(mdf)} rows")
    except Exception as e:
        print(f"[warn] Metrics load failed for {rd.name}: {e}")

if metrics_frames:
    all_metrics = pd.concat(metrics_frames, ignore_index=True)
    for col in ["epoch", "val_loss", "loss_D", "loss_G"]:
        if col in all_metrics.columns:
            all_metrics[col] = pd.to_numeric(all_metrics[col], errors="coerce")
    print(f"Combined metrics: {all_metrics.shape[0]} rows from {all_metrics['run'].nunique()} runs")
else:
    all_metrics = pd.DataFrame()
    print("No metrics loaded.")

#%%
#! ======================================================
#! Cell 3 — Decoding, rewards, and sampling if file missing
#! ======================================================
def reconstruct_selfies(logits_lv, max_seq_len, vocab_size, idx_to_token):
    token_idx = np.argmax(logits_lv, axis=1)
    token_idx = [i for i in token_idx if i != 0]  # drop PAD
    tokens = [idx_to_token.get(i, "") for i in token_idx]
    return "".join(tokens)

LEWIS_BASE_SMARTS = ["[#7;H2,H1;!$(NC=O)]", "[O-]", "[nH]", "[#8;H1]"]
LEWIS_ACID_SMARTS = ["[B]", "[P+]", "[Al]", "[Sn]", "[Si]"]
REDUCIBLE_SMARTS  = ["[C]=[O]", "[C]=[N]", "[N+](=O)[O-]", "[C]=[C]", "[N]=[N+]=[N-]"]

def contains_substructure(mol, smarts_list):
    for s in smarts_list:
        patt = Chem.MolFromSmarts(s)
        if mol is not None and mol.HasSubstructMatch(patt):
            return True
    return False

def reward_contains_flp(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return 0.0
    return 1.0 if (contains_substructure(m, LEWIS_BASE_SMARTS) and
                   contains_substructure(m, LEWIS_ACID_SMARTS)) else 0.0

def reward_hydrogenation_potential(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return 0.0
    matches = 0
    for s in REDUCIBLE_SMARTS:
        patt = Chem.MolFromSmarts(s)
        matches += len(m.GetSubstructMatches(patt))
    return min(matches, 3) / 3.0

def combined_flp_reward(smiles):
    return 0.7 * reward_contains_flp(smiles) + 0.3 * reward_hydrogenation_potential(smiles)

def sample_and_save_with_scores(run_dir: Path, G: Generator, n_samples=300):
    smiles_list, scores = [], []
    G.eval()
    with torch.no_grad():
        for _ in range(n_samples):
            z = torch.randn(1, LATENT_DIM).to(device)
            lv = G(z).cpu().numpy()[0]  # [L, V]
            sfs = reconstruct_selfies(lv, MAX_SEQ_LEN, VOCAB_SIZE, idx_to_token)
            try:
                smi = selfies.decoder(sfs)
            except Exception:
                continue
            if Chem.MolFromSmiles(smi):
                smiles_list.append(smi)
                scores.append(combined_flp_reward(smi))
    df = pd.DataFrame({"smiles": smiles_list, "score": scores})
    out = run_dir / "generated_scored.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} molecules to {out}")

# Ensure each run has generated_scored.csv
for run_name, (G, _) in generators.items():
    rdir = ROOT / run_name
    gcsv = rdir / "generated_scored.csv"
    if not gcsv.exists():
        sample_and_save_with_scores(rdir, G, n_samples=SAMPLES_PER_RUN_IF_MISSING)

#%%
#! ========================================================
#! Cell 4 — Build reference set (acid.csv x qm9.csv) for novelty
#! ========================================================
#! =========================================================
#! Cell 4 — FAST reference (component-wise) for novelty/NN
#! =========================================================
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from pathlib import Path
import numpy as np

# 1) Load acids and bases (no cartesian product!)
acid_path = Path("acid.csv")
base_path = Path("qm9.csv")      # user said this is the "base" file name they’re using now

assert acid_path.exists(), "acid.csv not found"
assert base_path.exists(), "qm9.csv not found"

acid_df = pd.read_csv(acid_path)
base_df = pd.read_csv(base_path)

assert "smiles" in acid_df.columns, "acid.csv must have 'smiles'"
assert "smiles" in base_df.columns, "qm9.csv must have 'smiles'"

acid_smiles_set = set(acid_df["smiles"].dropna().tolist())
base_smiles_set = set(base_df["smiles"].dropna().tolist())

#%% 2) Precompute fingerprints for component libraries
def mol_ok(s):
    try:
        m = Chem.MolFromSmiles(s)
        return m is not None
    except Exception:
        return False

acid_mols = [Chem.MolFromSmiles(s) for s in acid_smiles_set if mol_ok(s)]
base_mols = [Chem.MolFromSmiles(s) for s in base_smiles_set if mol_ok(s)]

acid_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in acid_mols]
base_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in base_mols]

# Keep arrays of (smiles) aligned with fps
acid_smiles_lib = [Chem.MolToSmiles(m) for m in acid_mols]
base_smiles_lib = [Chem.MolToSmiles(m) for m in base_mols]

print(f"Reference components: acids={len(acid_fps)}, bases={len(base_fps)}")


#%% 3) Helpers: split generated pairs, novelty flags, NN similarity per component
def split_pair(s):
    # expects "base.acid"; if no dot, treat entire as base and acid=None
    if "." in s:
        b, a = s.split(".", 1)
        return b, a
    return s, None

def component_novelty_flags(base_smi, acid_smi):
    base_novel = (base_smi not in base_smiles_set) if base_smi is not None else True
    acid_novel = (acid_smi not in acid_smiles_set) if acid_smi is not None else True
    return base_novel, acid_novel

def max_nn_similarity_to_lib(query_smi, lib_fps):
    if query_smi is None:
        return np.nan
    m = Chem.MolFromSmiles(query_smi)
    if m is None:
        return np.nan
    qfp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
    # BulkTanimotoSimilarity is fast and vectorized in C++
    sims = DataStructs.BulkTanimotoSimilarity(qfp, lib_fps)
    return float(max(sims)) if sims else np.nan

#%% 4) For each run’s generated set, compute component‑wise novelty and NN sim
#    Also store quick aggregates you can plot (mean ± std) later.
per_run_comp_stats = []

for run_name in generators.keys():
    gen_csv = ROOT / run_name / "generated_scored.csv"
    if not gen_csv.exists():
        print(f"[skip] {gen_csv} not found.")
        continue
    gdf = pd.read_csv(gen_csv)
    if "smiles" not in gdf.columns or gdf.empty:
        print(f"[skip] No smiles in {gen_csv}")
        continue

    # Split pairs
    pairs = gdf["smiles"].astype(str).apply(split_pair)
    gdf["base_smi"] = pairs.apply(lambda x: x[0])
    gdf["acid_smi"] = pairs.apply(lambda x: x[1])

    # Validity check
    gdf["valid"] = gdf["smiles"].apply(lambda s: Chem.MolFromSmiles(s) is not None)
    gdf_v = gdf[gdf["valid"]].copy()
    if gdf_v.empty:
        print(f"[warn] All invalid in {run_name}")
        continue

    # Novelty flags (component-wise)
    nov_flags = gdf_v.apply(lambda r: component_novelty_flags(r["base_smi"], r["acid_smi"]), axis=1)
    gdf_v["base_novel"] = [bn for bn, an in nov_flags]
    gdf_v["acid_novel"] = [an for bn, an in nov_flags]
    gdf_v["pair_novel_any"] = gdf_v["base_novel"] | gdf_v["acid_novel"]
    gdf_v["pair_novel_both"] = gdf_v["base_novel"] & gdf_v["acid_novel"]

    # NN similarity (component-wise)
    # Using BulkTanimotoSimilarity to the precomputed libraries
    gdf_v["nn_sim_base"] = gdf_v["base_smi"].apply(lambda s: max_nn_similarity_to_lib(s, base_fps))
    gdf_v["nn_sim_acid"] = gdf_v["acid_smi"].apply(lambda s: max_nn_similarity_to_lib(s, acid_fps) if s is not None else np.nan)
    # Combine to a single pair similarity summary if desired
    gdf_v["nn_sim_pair_mean"] = gdf_v[["nn_sim_base", "nn_sim_acid"]].mean(axis=1, skipna=True)

    # Save back to the run folder for later reuse
    out_aug = ROOT / run_name / "generated_scored_with_refstats.csv"
    gdf_v.to_csv(out_aug, index=False)
    print(f"Augmented: {out_aug} (n_valid={len(gdf_v)})")

    # Aggregate per-run stats
    per_run_comp_stats.append({
        "run": run_name,
        "n_valid": len(gdf_v),
        "base_novel_frac": gdf_v["base_novel"].mean(),
        "acid_novel_frac": gdf_v["acid_novel"].mean(),
        "pair_novel_any_frac": gdf_v["pair_novel_any"].mean(),
        "pair_novel_both_frac": gdf_v["pair_novel_both"].mean(),
        "nn_sim_base_mean": gdf_v["nn_sim_base"].mean(),
        "nn_sim_base_std":  gdf_v["nn_sim_base"].std(ddof=1),
        "nn_sim_acid_mean": gdf_v["nn_sim_acid"].mean(),
        "nn_sim_acid_std":  gdf_v["nn_sim_acid"].std(ddof=1),
        "nn_sim_pair_mean": gdf_v["nn_sim_pair_mean"].mean(),
        "nn_sim_pair_std":  gdf_v["nn_sim_pair_mean"].std(ddof=1),
    })

per_run_comp_stats = pd.DataFrame(per_run_comp_stats)
display(per_run_comp_stats)

# You can also compute mean ± std across runs for these component metrics:
if not per_run_comp_stats.empty:
    comp_cols = [
        "base_novel_frac","acid_novel_frac","pair_novel_any_frac","pair_novel_both_frac",
        "nn_sim_base_mean","nn_sim_acid_mean","nn_sim_pair_mean"
    ]
    comp_summary = per_run_comp_stats[comp_cols].agg(["mean","std"]).T.reset_index().rename(columns={"index":"metric"})
    print("\nComponent-wise novelty/NN — mean ± std across runs:")
    display(comp_summary)

    # Error-bar plot (example for pair_novel_any_frac and nn_sim_pair_mean)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11,4))

    # Pair novelty (any)
    x = np.arange(1)
    axes[0].bar(x, [comp_summary[comp_summary["metric"]=="pair_novel_any_frac"]["mean"].values[0]],
                yerr=[comp_summary[comp_summary["metric"]=="pair_novel_any_frac"]["std"].values[0]], capsize=3)
    axes[0].set_xticks(x); axes[0].set_xticklabels(["pair_novel_any"])
    axes[0].set_ylabel("Fraction")
    axes[0].set_title("Pair novelty (any) — mean ± std")

    # NN sim (pair mean)
    axes[1].bar(x, [comp_summary[comp_summary["metric"]=="nn_sim_pair_mean"]["mean"].values[0]],
                yerr=[comp_summary[comp_summary["metric"]=="nn_sim_pair_mean"]["std"].values[0]], capsize=3)
    axes[1].set_xticks(x); axes[1].set_xticklabels(["nn_sim_pair_mean"])
    axes[1].set_ylabel("Tanimoto")
    axes[1].set_title("NN similarity (pair mean) — mean ± std")

    plt.tight_layout(); plt.show()

#%%
#! ============================================================
#! Cell 5 — Global Generative Metrics (with novelty & NN-sim)
#! ============================================================
def smiles_validity(smiles_list):
    return [Chem.MolFromSmiles(s) is not None for s in smiles_list]

def compute_uniqueness(valid_smiles):
    return len(set(valid_smiles)) / max(len(valid_smiles), 1)

def morgan_fps(smiles_list, radius=2, nbits=2048):
    fps = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            fps.append(None); continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits))
    return fps

def tanimoto_diversity(fps):
    valid = [fp for fp in fps if fp is not None]
    n = len(valid)
    if n < 2: return np.nan
    dsum, cnt = 0.0, 0
    for i in range(n):
        for j in range(i+1, n):
            sim = DataStructs.TanimotoSimilarity(valid[i], valid[j])
            dsum += (1.0 - sim); cnt += 1
    return dsum / cnt if cnt else np.nan

def scaffold_diversity(smiles_list):
    scaffolds = []
    total = 0
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        total += 1
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
            scaffolds.append(scaf)
        except Exception:
            continue
    return len(set(scaffolds)) / max(total, 1)

def compute_novelty(gen_smiles):
    if reference_df.empty:
        return np.nan
    gen_set = set(gen_smiles)
    ref_set = set(reference_df["smiles"])
    new = gen_set - ref_set
    return len(new) / max(len(gen_set), 1)

def compute_nn_similarity(gen_smiles):
    if not ref_fps:
        return np.nan
    gen_mols = [Chem.MolFromSmiles(s) for s in gen_smiles]
    gen_fps  = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in gen_mols if m is not None]
    sims = []
    for fp in gen_fps:
        max_sim = max(DataStructs.TanimotoSimilarity(fp, rfp) for rfp in ref_fps)
        sims.append(max_sim)
    return float(np.mean(sims)) if sims else np.nan

def compute_global_metrics(df: pd.DataFrame):
    smiles = df["smiles"].dropna().tolist()
    if len(smiles) < MIN_VALID_FOR_STATS:
        return {"validity": np.nan, "uniqueness": np.nan, "diversity": np.nan,
                "scaffold_diversity": np.nan, "novelty": np.nan, "nn_similarity": np.nan,
                "n_samples": len(smiles)}
    vmask = smiles_validity(smiles)
    valid_smiles = [s for s, v in zip(smiles, vmask) if v]
    validity = sum(vmask) / max(len(smiles), 1)
    uniq = compute_uniqueness(valid_smiles)
    fps = morgan_fps(valid_smiles)
    div = tanimoto_diversity(fps)
    scaf_div = scaffold_diversity(valid_smiles)
    novelty = compute_novelty(valid_smiles)
    nn_sim = compute_nn_similarity(valid_smiles)
    return {"validity": validity, "uniqueness": uniq, "diversity": div,
            "scaffold_diversity": scaf_div, "novelty": novelty, "nn_similarity": nn_sim,
            "n_samples": len(smiles)}

# Load/concat generated_scored.csv across runs
gen_frames = []
for run_name in generators.keys():
    p = ROOT / run_name / "generated_scored.csv"
    if p.exists():
        df = pd.read_csv(p)
        df["run"] = run_name
        gen_frames.append(df)
gen_all = pd.concat(gen_frames, ignore_index=True) if gen_frames else pd.DataFrame()

if gen_all.empty:
    print("No generated_scored.csv files available for generative metrics.")
else:
    # Per-run metrics
    rows = []
    for run_name in generators.keys():
        sub = gen_all[gen_all["run"] == run_name]
        if sub.empty: continue
        m = compute_global_metrics(sub)
        m["run"] = run_name
        rows.append(m)
    global_df = pd.DataFrame(rows)
    print("\nPer-run global generative metrics:")
    print(global_df)

    # Mean ± std across runs
    met_cols = ["validity", "uniqueness", "diversity", "scaffold_diversity", "novelty", "nn_similarity"]
    mean_std = global_df[met_cols].agg(["mean", "std"]).T.reset_index().rename(columns={"index": "metric"})
    print("\nGlobal generative metrics (mean ± std):")
    print(mean_std)

    # Plot: mean ± std bars
    plt.figure(figsize=(10,5))
    x = np.arange(len(met_cols))
    plt.bar(x, mean_std["mean"].values, yerr=mean_std["std"].values, capsize=3)
    plt.xticks(x, met_cols, rotation=30, ha="right")
    plt.ylabel("Score")
    plt.title("Global Generative Metrics (mean ± std across runs)")
    plt.tight_layout(); plt.show()

    # Plot: per-run grouped bars
    plt.figure(figsize=(10,5))
    width = 0.8 / max(len(global_df), 1)
    for i, (_, row) in enumerate(global_df.iterrows()):
        vals = [row[c] for c in met_cols]
        plt.bar(x + i*width, vals, width=width, label=row["run"])
    plt.xticks(x + width*(len(global_df)-1)/2, met_cols, rotation=30, ha="right")
    plt.ylabel("Score"); plt.title("Global Generative Metrics per run")
    plt.legend(); plt.tight_layout(); plt.show()

#%%
#! ==================================================
#! Cell 6 — Physicochemical distributions and stats
#! ==================================================
def compute_physchem(df: pd.DataFrame):
    rows = []
    for s in df["smiles"].dropna():
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        try:
            rows.append({
                "smiles": s,
                "QED": QED.qed(m),
                "MW": Descriptors.MolWt(m),
                "logP": Descriptors.MolLogP(m),
            })
        except Exception:
            continue
    props = pd.DataFrame(rows)
    # Merge HOMO/LUMO if present in df to compute gap
    if "homo" in df.columns and "lumo" in df.columns:
        props = props.merge(df[["smiles","homo","lumo"]], on="smiles", how="left")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            props["gap"] = props["lumo"] - props["homo"]
    return props

if gen_all.empty:
    print("No generated data to compute physicochemical distributions.")
else:
    props_frames = []
    for run_name in generators.keys():
        sub = gen_all[gen_all["run"] == run_name]
        if sub.empty: continue
        props = compute_physchem(sub)
        if not props.empty:
            props["run"] = run_name
            props_frames.append(props)

    if not props_frames:
        print("No properties computed; skipping distributions.")
    else:
        props_all = pd.concat(props_frames, ignore_index=True)
        prop_cols = ["QED", "MW", "logP"] + (["gap"] if "gap" in props_all.columns else [])

        # Combined histograms
        for col in prop_cols:
            plt.figure(figsize=(9,5))
            plt.hist(props_all[col].dropna(), bins=30)
            plt.xlabel(col); plt.ylabel("Frequency")
            plt.title(f"Distribution of {col} (combined across runs)")
            plt.tight_layout(); plt.show()

        # Per-run mean ± std
        for col in prop_cols:
            stat = props_all.groupby("run")[col].agg(["mean","std"]).reset_index()
            plt.figure(figsize=(9,5))
            x = np.arange(len(stat))
            plt.bar(x, stat["mean"].values, yerr=stat["std"].values, capsize=3)
            plt.xticks(x, stat["run"].tolist(), rotation=45, ha="right")
            plt.ylabel(f"{col} mean"); plt.title(f"{col} per run (mean ± std)")
            plt.tight_layout(); plt.show()

#%%
#! ======================================
#! Cell 7 — Reward progression over epochs
#! ======================================
if all_metrics is None or all_metrics.empty:
    print("No training metrics; skipping reward progression.")
else:
    reward_cols = [c for c in all_metrics.columns if c.startswith("reward_")]
    reward_cols = [c for c in reward_cols if all_metrics[c].notna().any()]
    if not reward_cols:
        print("No reward_* columns with values; skipping reward plots.")
    else:
        # per-run overlays
        for rc in reward_cols:
            plt.figure(figsize=(9,5))
            for name, sub in all_metrics.groupby("run"):
                ss = sub[["epoch", rc]].dropna()
                if ss.empty: continue
                plt.plot(ss["epoch"], ss[rc], label=name, alpha=0.8)
            plt.xlabel("Epoch"); plt.ylabel(rc.replace("reward_", "").replace("_", " ").title())
            plt.title(f"Reward progression per run: {rc}")
            plt.legend(); plt.tight_layout(); plt.show()

        # mean ± std across runs
        for rc in reward_cols:
            agg = (all_metrics[["epoch", rc]]
                   .dropna()
                   .groupby("epoch")[rc]
                   .agg(["mean","std"]))
            if agg.empty: continue
            plt.figure(figsize=(9,5))
            plt.errorbar(agg.index, agg["mean"], yerr=agg["std"], fmt='-o', capsize=3)
            plt.xlabel("Epoch"); plt.ylabel(rc.replace("reward_", "").replace("_", " ").title())
            plt.title(f"{rc}: mean ± std across runs")
            plt.tight_layout(); plt.show()

#%%
#! ======================================================
#! Cell 8 — Export top candidates (CSV, SDF, Gaussian .gjf)
#! ======================================================
# Combine all runs' generated data (ensure existence)
gen_frames = []
for run_name in generators.keys():
    p = ROOT / run_name / "generated_scored_with_refstats.csv"
    if p.exists():
        df = pd.read_csv(p)
        df["run"] = run_name
        df = df[df["smiles"].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)]
        gen_frames.append(df)
all_gen = pd.concat(gen_frames, ignore_index=True) if gen_frames else pd.DataFrame()

if all_gen.empty:
    print("No generated data to export top candidates.")
else:
    # Top per run & global
    K_PER_RUN = 20
    M_GLOBAL  = 50

    per_run_top = []
    for run_name in generators.keys():
        sub = all_gen[all_gen["run"] == run_name]
        if sub.empty: continue
        per_run_top.append(sub.sort_values("score", ascending=False).head(K_PER_RUN))
    per_run_top_df = pd.concat(per_run_top, ignore_index=True) if per_run_top else pd.DataFrame()

    global_top_df = all_gen.sort_values("score", ascending=False).head(M_GLOBAL)

    per_run_csv = ROOT / "top_candidates_per_run.csv"
    global_csv  = ROOT / "top_candidates_global.csv"
    per_run_top_df.to_csv(per_run_csv, index=False)
    global_top_df.to_csv(global_csv, index=False)
    print("Saved:", per_run_csv)
    print("Saved:", global_csv)

    # SDF with 3D conformers (quick ETKDG + UFF)
    def build_complex_from_pair(smiles_pair, sep=3.0):
        if "." in smiles_pair:
            b_smi, a_smi = smiles_pair.split(".")
            mb = Chem.AddHs(Chem.MolFromSmiles(b_smi))
            ma = Chem.AddHs(Chem.MolFromSmiles(a_smi))
            if mb is None or ma is None: return None
            rdDistGeom.EmbedMolecule(mb, rdDistGeom.ETKDGv3())
            rdDistGeom.EmbedMolecule(ma, rdDistGeom.ETKDGv3())
            confa = ma.GetConformer()
            for i in range(ma.GetNumAtoms()):
                pos = confa.GetAtomPosition(i)
                confa.SetAtomPosition(i, (pos.x + sep, pos.y, pos.z))
            comb = Chem.CombineMols(mb, ma)
            mc = Chem.RWMol(comb).GetMol()
            confc = Chem.Conformer(mc.GetNumAtoms())
            idx = 0
            confb = mb.GetConformer()
            for i in range(mb.GetNumAtoms()):
                pb = confb.GetAtomPosition(i)
                confc.SetAtomPosition(idx, pb); idx += 1
            confa = ma.GetConformer()
            for i in range(ma.GetNumAtoms()):
                pa = confa.GetAtomPosition(i)
                confc.SetAtomPosition(idx, pa); idx += 1
            mc.RemoveAllConformers(); mc.AddConformer(confc, assignId=True)
            return mc
        else:
            m = Chem.AddHs(Chem.MolFromSmiles(smiles_pair))
            if m is None: return None
            rdDistGeom.EmbedMolecule(m, rdDistGeom.ETKDGv3())
            return m

    sdf_path = ROOT / "dft_candidates_top_global.sdf"
    w = Chem.SDWriter(str(sdf_path))
    for _, row in global_top_df.iterrows():
        smi = row["smiles"]
        mol = build_complex_from_pair(smi, sep=3.0)
        if mol is None: continue
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            pass
        mol.SetProp("_Name", f"{smi} | score={row['score']:.3f}")
        w.write(mol)
    w.close()
    print("Wrote SDF:", sdf_path)

    # Optional Gaussian .gjf templates
    def mol_to_gjf(mol, title="FLP candidate", route="# opt freq b3lyp/6-31G(d) scf=tight", charge=0, multiplicity=1):
        conf = mol.GetConformer()
        lines = ["%nprocs=8", "%mem=8GB", route, "", title, "", f"{charge} {multiplicity}"]
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            sym = mol.GetAtomWithIdx(i).GetSymbol()
            lines.append(f"{sym:2s}  {pos.x: .6f}  {pos.y: .6f}  {pos.z: .6f}")
        lines.append("")
        return "\n".join(lines)

    gjf_dir = ROOT / "gjf_top_global"
    gjf_dir.mkdir(exist_ok=True)
    n_written = 0
    for i, row in global_top_df.head(M_GLOBAL).iterrows():
        smi = row["smiles"]
        m = build_complex_from_pair(smi, sep=3.0)
        if m is None: continue
        try:
            AllChem.UFFOptimizeMolecule(m, maxIters=300)
        except Exception:
            pass
        gjf_text = mol_to_gjf(m, title=f"FLP {smi} | score={row['score']:.3f}")
        out_path = gjf_dir / f"candidate_{i:03d}.gjf"
        with open(out_path, "w") as f:
            f.write(gjf_text)
        n_written += 1
    print(f"Wrote {n_written} Gaussian input files to {gjf_dir}")

#%% === FAST Global Generative Metrics per run: mean ± SD (error bars) ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

# ---------------- Config (speed/accuracy trade-offs) ----------------
ROOT = Path("output")
RUN_IDS = list(range(6))         # change if different
GEN_FILE = "generated_scored.csv"

BOOT_N = 30                      # bootstrap resamples per run (was 300)
PAIR_SUBSAMPLE = 150             # pair count per bootstrap for diversity (was 2000)
RNG = np.random.default_rng(0)

# ---------------- Morgan FP generator (fast + no warnings) ----------
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# ---------------- Helpers ------------------------------------------
def validity_fraction(smiles):
    """Fraction of strings that parse as molecules."""
    n = len(smiles)
    if n == 0:
        return np.nan, 0, 0
    valid_mask = []
    for s in smiles:
        try:
            valid_mask.append(Chem.MolFromSmiles(s) is not None)
        except Exception:
            valid_mask.append(False)
    p = float(np.mean(valid_mask))
    n_valid = int(np.sum(valid_mask))
    return p, n, n_valid

def precompute_for_run(valid_smiles):
    """Precompute Mol, Morgan FP, and Bemis–Murcko scaffold once per valid molecule."""
    mols, fps, scaffolds = [], [], []
    for s in valid_smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            mols.append(None); fps.append(None); scaffolds.append(None); continue
        mols.append(m)
        fps.append(morgan_gen.GetFingerprint(m))
        try:
            scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m))
        except Exception:
            scaffolds.append(None)
    return mols, fps, scaffolds

def uniqueness_fraction(sample_smiles):
    if not sample_smiles:
        return np.nan
    return len(set(sample_smiles)) / float(len(sample_smiles))

def tanimoto_diversity_from_fps(sample_idx, fps):
    """Mean Tanimoto distance (1 - sim) over a random subset of pairs."""
    # collect valid fps indices among the sample
    valid_ids = [i for i in sample_idx if fps[i] is not None]
    n = len(valid_ids)
    if n < 2:
        return np.nan
    # sample pair indices
    m = min(PAIR_SUBSAMPLE, n * (n - 1) // 2)
    if m <= 0:
        return np.nan
    # sample two lists of indices and ensure i != j (retry a few times if needed)
    ij = RNG.integers(0, n, size=(m, 2))
    same = ij[:, 0] == ij[:, 1]
    tries = 0
    while same.any() and tries < 5:
        ij[same, 1] = RNG.integers(0, n, size=same.sum())
        same = ij[:, 0] == ij[:, 1]
        tries += 1
    dists = []
    for a, b in ij:
        if a == b: 
            continue
        sim = DataStructs.TanimotoSimilarity(fps[valid_ids[a]], fps[valid_ids[b]])
        dists.append(1.0 - sim)
    return float(np.mean(dists)) if dists else np.nan

def scaffold_diversity_fraction(sample_idx, scaffolds):
    scafs = [scaffolds[i] for i in sample_idx if scaffolds[i] is not None]
    n = len(scafs)
    if n == 0:
        return np.nan
    return len(set(scafs)) / float(n)

def bootstrap_run_metrics(all_smiles):
    """Compute mean±sd of metrics within a run via light bootstrap.
       Validity is computed once; its SD is binomial-based for speed.
    """
    # validity on the full list (don’t redo in every bootstrap)
    p_valid, n_total, n_valid = validity_fraction(all_smiles)
    valid_smiles = [s for s in all_smiles if Chem.MolFromSmiles(s) is not None]
    if n_valid == 0:
        return {
            "validity_mean": p_valid, "validity_sd": np.sqrt(p_valid*(1-p_valid)/max(n_total,1)),
            "uniqueness_mean": np.nan, "uniqueness_sd": np.nan,
            "diversity_mean": np.nan, "diversity_sd": np.nan,
            "scaffold_diversity_mean": np.nan, "scaffold_diversity_sd": np.nan,
            "n_total": n_total, "n_valid": n_valid
        }

    # precompute for speed
    mols, fps, scaffolds = precompute_for_run(valid_smiles)
    vN = len(valid_smiles)
    # bootstrap over the valid set only (validity already measured)
    uniq_vals, div_vals, scaf_vals = [], [], []
    for _ in range(BOOT_N):
        idx = RNG.integers(0, vN, size=vN)  # sample with replacement
        uniq_vals.append(uniqueness_fraction([valid_smiles[i] for i in idx]))
        div_vals.append(tanimoto_diversity_from_fps(idx, fps))
        scaf_vals.append(scaffold_diversity_fraction(idx, scaffolds))

    return {
        "validity_mean": p_valid,
        "validity_sd": np.sqrt(p_valid*(1-p_valid)/max(n_total,1)),
        "uniqueness_mean": float(np.nanmean(uniq_vals)), 
        "uniqueness_sd": float(np.nanstd(uniq_vals, ddof=1)),
        "diversity_mean": float(np.nanmean(div_vals)), 
        "diversity_sd": float(np.nanstd(div_vals, ddof=1)),
        "scaffold_diversity_mean": float(np.nanmean(scaf_vals)), 
        "scaffold_diversity_sd": float(np.nanstd(scaf_vals, ddof=1)),
        "n_total": n_total, "n_valid": n_valid
    }

#%% ---------------- Load runs & compute -------------------
rows = []
missing = []
for rid in RUN_IDS:
    run_name = f"run_{rid}"
    fpath = ROOT / run_name / GEN_FILE
    if not fpath.exists():
        missing.append(run_name); continue
    df = pd.read_csv(fpath)
    if "smiles" not in df.columns or df.empty:
        missing.append(run_name); continue

    smiles = df["smiles"].dropna().astype(str).tolist()
    stats = bootstrap_run_metrics(smiles)

    rows.append({
        "run": run_name,
        "n_generated": len(smiles),
        "n_valid": stats["n_valid"],
        "validity_mean": stats["validity_mean"], "validity_sd": stats["validity_sd"],
        "uniqueness_mean": stats["uniqueness_mean"], "uniqueness_sd": stats["uniqueness_sd"],
        "diversity_mean": stats["diversity_mean"], "diversity_sd": stats["diversity_sd"],
        "scaffold_diversity_mean": stats["scaffold_diversity_mean"], "scaffold_diversity_sd": stats["scaffold_diversity_sd"],
    })

per_run_df = pd.DataFrame(rows).sort_values("run")
display(per_run_df)

if missing:
    print("Skipped (missing/empty):", ", ".join(missing))

#%% ---------------- Plots: mean ± SD per run ----------------
def plot_metric_bars(df, mean_col, sd_col, title, ylabel):
    if df.empty:
        print(f"Skipping plot '{title}' — no data."); return
    x = np.arange(len(df))
    plt.figure(figsize=(10,5))
    plt.bar(x, df[mean_col].values, yerr=df[sd_col].values, capsize=3)
    plt.xticks(x, df["run"].tolist(), rotation=45, ha="right")
    plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); plt.show()

plot_metric_bars(per_run_df, "validity_mean", "validity_sd", "Validity per run (mean ± SD)", "Validity")
plot_metric_bars(per_run_df, "uniqueness_mean", "uniqueness_sd", "Uniqueness per run (mean ± SD)", "Uniqueness")
plot_metric_bars(per_run_df, "diversity_mean", "diversity_sd", "Tanimoto diversity per run (mean ± SD)", "1 − similarity")
plot_metric_bars(per_run_df, "scaffold_diversity_mean", "scaffold_diversity_sd", "Scaffold diversity per run (mean ± SD)", "Fraction unique scaffolds")

#%% -------- Optional: overall mean ± SD across runs --------
if not per_run_df.empty:
    overall = pd.DataFrame({
        "metric": ["validity","uniqueness","diversity","scaffold_diversity"],
        "mean_across_runs": [
            per_run_df["validity_mean"].mean(),
            per_run_df["uniqueness_mean"].mean(),
            per_run_df["diversity_mean"].mean(),
            per_run_df["scaffold_diversity_mean"].mean(),
        ],
        "sd_across_runs": [
            per_run_df["validity_mean"].std(ddof=1),
            per_run_df["uniqueness_mean"].std(ddof=1),
            per_run_df["diversity_mean"].std(ddof=1),
            per_run_df["scaffold_diversity_mean"].std(ddof=1),
        ],
    })
    display(overall)
    x = np.arange(len(overall))
    plt.figure(figsize=(9,5))
    plt.bar(x, overall["mean_across_runs"].values, yerr=overall["sd_across_runs"].values, capsize=3)
    plt.xticks(x, overall["metric"].tolist(), rotation=30, ha="right")
    plt.ylabel("Score"); plt.title("Global Generative Metrics — mean ± SD across runs")
    plt.tight_layout(); plt.show()


# %%
# === Rotatable Bonds & TPSA Analysis per run ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Lipinski, rdMolDescriptors

ROOT = Path("output")
RUN_IDS = list(range(6))  # adjust if needed
GEN_FILE = "generated_scored.csv"

def compute_props(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None, None
    n_rot = Lipinski.NumRotatableBonds(m)
    tpsa = rdMolDescriptors.CalcTPSA(m)
    return n_rot, tpsa

per_run_stats = []

for rid in RUN_IDS:
    run_name = f"run_{rid}"
    fpath = ROOT / run_name / GEN_FILE
    if not fpath.exists():
        print(f"[skip] {fpath} not found.")
        continue
    
    df = pd.read_csv(fpath)
    if "smiles" not in df.columns or df.empty:
        print(f"[skip] No smiles in {fpath}")
        continue
    
    props = [compute_props(s) for s in df["smiles"].dropna()]
    props = [(r, t) for r, t in props if r is not None]
    if not props:
        print(f"[warn] All invalid in {run_name}")
        continue

    rots, tpsas = zip(*props)
    per_run_stats.append({
        "run": run_name,
        "rot_mean": np.mean(rots), "rot_sd": np.std(rots, ddof=1),
        "tpsa_mean": np.mean(tpsas), "tpsa_sd": np.std(tpsas, ddof=1),
        "rot_values": rots, "tpsa_values": tpsas
    })

# Convert to DataFrame
stats_df = pd.DataFrame(per_run_stats).sort_values("run")
display(stats_df[["run","rot_mean","rot_sd","tpsa_mean","tpsa_sd"]])

# --- Plot distributions per run ---
def plot_distributions(stats_df, value_key, title, xlabel, bins=20):
    plt.figure(figsize=(10,5))
    for _, row in stats_df.iterrows():
        plt.hist(row[value_key], bins=bins, alpha=0.5, label=row["run"])
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_distributions(stats_df, "rot_values", "Distribution of Rotatable Bonds per run", "Rotatable Bonds")
plot_distributions(stats_df, "tpsa_values", "Distribution of TPSA per run", "TPSA (Å²)")

# --- Plot mean ± SD per run ---
def plot_means(stats_df, mean_key, sd_key, title, ylabel):
    x = np.arange(len(stats_df))
    plt.figure(figsize=(9,5))
    plt.bar(x, stats_df[mean_key], yerr=stats_df[sd_key], capsize=3)
    plt.xticks(x, stats_df["run"], rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_means(stats_df, "rot_mean", "rot_sd", "Mean Rotatable Bonds per run (± SD)", "Rotatable Bonds")
plot_means(stats_df, "tpsa_mean", "tpsa_sd", "Mean TPSA per run (± SD)", "TPSA (Å²)")

# %%
# === Top 20 FLPs Visualisation across all runs ===
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
from IPython.display import display


ROOT = Path("output")
RUN_IDS = list(range(6))  # adjust if needed
GEN_FILE = "generated_scored.csv"

all_mols = []

for rid in RUN_IDS:
    run_name = f"run_{rid}"
    fpath = ROOT / run_name / GEN_FILE
    if not fpath.exists():
        print(f"[skip] {fpath} not found.")
        continue
    
    df = pd.read_csv(fpath)
    if "smiles" not in df.columns or "score" not in df.columns:
        print(f"[skip] No 'smiles' or 'score' in {fpath}")
        continue
    
    for _, row in df.iterrows():
        smi = row["smiles"]
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            all_mols.append({
                "smiles": smi,
                "score": row["score"],
                "run": run_name
            })

# Convert to DataFrame
all_df = pd.DataFrame(all_mols)

# Sort by score and take top 20
top20_df = all_df.sort_values("score", ascending=False).head(20).reset_index(drop=True)
top20_mols = [Chem.MolFromSmiles(s) for s in top20_df["smiles"]]

# Draw grid of top 20
img = Draw.MolsToGridImage(
    top20_mols,
    molsPerRow=5,
    subImgSize=(300,300),
    legends=[f"{i+1}. {row['smiles']}\nScore: {row['score']:.2f}" for i, row in top20_df.iterrows()]
)

# Display inline (if in notebook)
display(img)

# === Filter to likely FLPs, then apply CO2 proxy and combine with old score ===

FLP_SCORE_THRESH = 1   # if you trust the old score as 'FLP-ness', gate here

def is_likely_flp_structural(smiles):
    """Check FLP by SMARTS:
       - Intermolecular 'base.acid': base fragment must match any base SMARTS and acid fragment any acid SMARTS
       - Intramolecular: single fragment must contain BOTH a base and an acid SMARTS
    """
    if "." in smiles:
        b_smi, a_smi = smiles.split(".", 1)
        bmol = Chem.MolFromSmiles(b_smi)
        amol = Chem.MolFromSmiles(a_smi)
        if bmol is None or amol is None:
            return False
        base_ok = any(bmol.HasSubstructMatch(p) for p in base_patterns.values())
        acid_ok = any(amol.HasSubstructMatch(p) for p in acid_patterns.values())
        return base_ok and acid_ok
    else:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        base_ok = any(mol.HasSubstructMatch(p) for p in base_patterns.values())
        acid_ok = any(mol.HasSubstructMatch(p) for p in acid_patterns.values())
        return base_ok and acid_ok

# 1) Pool all runs (valid SMILES only)
rows = []
for rid in RUN_IDS:
    path = ROOT / f"run_{rid}" / GEN_FILE
    if not path.exists():
        continue
    df = pd.read_csv(path)
    if "smiles" not in df.columns or df.empty:
        continue
    if "score" not in df.columns:
        df["score"] = 0.0  # fallback
    df["run"] = f"run_{rid}"
    # keep valid smiles
    df = df[df["smiles"].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)]
    rows.append(df[["smiles","score","run"]])

if not rows:
    raise RuntimeError("No generated_scored.csv files found or all invalid.")
all_df = pd.concat(rows, ignore_index=True)

# 2) Take top-N by old score (fast prefilter)
pre = all_df.sort_values("score", ascending=False).head(TOP_N_OVERALL).reset_index(drop=True)

# 3) FLP gating: old score threshold OR structural FLP check
pre["is_flp_by_old"] = pre["score"] >= FLP_SCORE_THRESH
pre["is_flp_struct"] = pre["smiles"].apply(is_likely_flp_structural)
flp_df = pre[ pre["is_flp_by_old"] | pre["is_flp_struct"] ].copy()

# 4) Compute CO2 proxy ONLY on likely FLPs
flp_df["co2_proxy"] = flp_df["smiles"].apply(co2_proxy_score)

# 5) Combine with old score (choose ONE of the combos below)

# (A) Gated product (default): keeps final in [0,1] and zeroes weak FLPs automatically
flp_df["final_co2_flp_score"] = flp_df["co2_proxy"] * flp_df["score"]

# (B) Weighted blend (uncomment to use): retains old score ranking but injects proxy info
# OLD_W = 0.4
# flp_df["final_co2_flp_score"] = OLD_W*flp_df["score"] + (1-OLD_W)*flp_df["co2_proxy"]

# 6) Save re-ranked list and proceed with plots as before (just switch column name)
rescored_csv = ROOT / "topN_rescored_CO2_proxy_FLPonly.csv"
flp_df.sort_values("final_co2_flp_score", ascending=False).to_csv(rescored_csv, index=False)
print(f"Saved FLP‑gated rescored list: {rescored_csv} (N={len(flp_df)})")

# === Plots using the FLP‑gated final score ===
per_run = (flp_df
           .groupby("run")["final_co2_flp_score"]
           .agg(["mean","std","count"])
           .reset_index()
           .sort_values("run"))
display(per_run)

# Mean ± SD bar
x = np.arange(len(per_run))
plt.figure(figsize=(9,5))
plt.bar(x, per_run["mean"], yerr=per_run["std"], capsize=3)
plt.xticks(x, per_run["run"], rotation=45, ha="right")
plt.ylabel("Final CO₂–FLP score")
plt.title("CO₂ hydrogenation proxy (FLP‑gated) — mean ± SD per run")
plt.tight_layout(); plt.show()

# Histograms per run
plt.figure(figsize=(10,5))
for _, r in per_run.iterrows():
    vals = flp_df[flp_df["run"]==r["run"]]["final_co2_flp_score"].values
    plt.hist(vals, bins=30, alpha=0.5, label=r["run"])
plt.xlabel("Final CO₂–FLP score"); plt.ylabel("Count")
plt.title("CO₂–FLP score distributions per run (FLP‑gated)")
plt.legend(); plt.tight_layout(); plt.show()

# Top‑20 grid by final score (across all runs)
top20 = flp_df.sort_values("final_co2_flp_score", ascending=False).head(TOP_K_PLOT).reset_index(drop=True)
top20_mols = [Chem.MolFromSmiles(s) for s in top20["smiles"]]
img = Draw.MolsToGridImage(
    top20_mols,
    molsPerRow=5,
    subImgSize=(300,300),
    legends=[f"{i+1}. {row['run']}\nFinal: {row['final_co2_flp_score']:.2f}\nOld: {row['score']:.2f}  CO₂: {row['co2_proxy']:.2f}"
             for i, row in top20.iterrows()]
)
from PIL import Image
display(img)  # display inline if in a notebook

# %%
# --- Thesis-style Matplotlib defaults (minimal, readable, journal-like) ---
import matplotlib as mpl
import matplotlib.pyplot as plt

def apply_thesis_style():
    mpl.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.grid": True,
        "grid.linewidth": 0.3,
        "grid.alpha": 0.25,
        "grid.color": "#9aa0a6",
        "axes.grid.which": "major",
        "legend.frameon": False,
        "figure.autolayout": False,  # we'll always call tight_layout()
    })
apply_thesis_style()

# %%
import numpy as np

def plot_distribution_with_median(datasets, labels, title, xlabel, bins=30, figsize=(9,4.8)):
    """
    datasets: list of 1D arrays (one per run)
    labels:   list of run names
    """
    plt.figure(figsize=figsize)
    # stack all to choose a common binning (keeps shapes comparable)
    all_vals = np.concatenate([np.asarray(x, float) for x in datasets if len(x)>0])
    if all_vals.size == 0:
        print(f"[skip] No data for '{title}'"); return
    hist_range = (np.nanmin(all_vals), np.nanmax(all_vals))
    counts_max = 0
    for vals, lab in zip(datasets, labels):
        vals = np.asarray(vals, float); vals = vals[~np.isnan(vals)]
        if vals.size == 0: 
            continue
        c, edges = np.histogram(vals, bins=bins, range=hist_range)
        counts_max = max(counts_max, c.max())
        centers = 0.5*(edges[1:]+edges[:-1])
        plt.step(centers, c, where="mid", linewidth=1.4, label=lab, alpha=0.9)
        # median & IQR lines on the x-axis
        med = np.median(vals); q1 = np.percentile(vals, 25); q3 = np.percentile(vals, 75)
        y_base = -0.05*counts_max if counts_max>0 else -0.5
        plt.plot([med, med], [0, counts_max*0.08], lw=1.6)   # median tick
        plt.plot([q1, q3], [counts_max*0.04, counts_max*0.04], lw=3, alpha=0.6)  # IQR bar

    plt.xlabel(xlabel); plt.ylabel("Count"); plt.title(title)
    plt.legend(ncol=2, frameon=False)
    plt.tight_layout(); plt.show()

def box_or_violin(datasets, labels, title, ylabel, kind="violin", figsize=(9,4.8)):
    """
    kind: "violin" (with medians) or "box"
    """
    plt.figure(figsize=figsize)
    data = [np.asarray(x, float) for x in datasets]
    # remove NaNs per group
    data = [d[~np.isnan(d)] for d in data]
    if all(len(d)==0 for d in data):
        print(f"[skip] No data for '{title}'"); return

    if kind == "violin":
        vp = plt.violinplot(data, showextrema=False, showmedians=False)
        # lightly color violins
        for b in vp['bodies']:
            b.set_alpha(0.5)
        # add median markers
        meds = [np.median(d) if len(d)>0 else np.nan for d in data]
        plt.scatter(np.arange(1,len(data)+1), meds, marker="o", zorder=3)
    else:
        bp = plt.boxplot(data, showmeans=False, medianprops=dict(linewidth=2))

    plt.xticks(np.arange(1,len(labels)+1), labels, rotation=30, ha="right")
    plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); plt.show()

# %%
# Assuming you still have the raw bootstrap arrays, but if not,
# we can re-sample quickly from the valid molecules per run:

from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs, Scaffolds

ROOT = Path("output")
RUN_IDS = list(range(6))
GEN_FILE = "generated_scored.csv"

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def compute_uniqueness(vals): 
    return len(set(vals)) / max(1,len(vals))

def run_metric_samples(smiles, n_boot=20, pair_subsample=100):
    # Generate a small set of bootstrap samples for plotting distributions
    vals_valid = [s for s in smiles if Chem.MolFromSmiles(s) is not None]
    if len(vals_valid) < 2:
        return {k: np.array([np.nan]) for k in ["validity","uniqueness","diversity","scaffold_diversity"]}

    # Precompute once
    mols = [Chem.MolFromSmiles(s) for s in vals_valid]
    fps  = [morgan_gen.GetFingerprint(m) for m in mols]
    scafs= []
    for m in mols:
        try:
            scafs.append(Scaffolds.MurckoScaffold.MurckoScaffoldSmiles(mol=m))
        except Exception:
            scafs.append(None)

    validity_full = np.mean([Chem.MolFromSmiles(s) is not None for s in smiles])

    uniq_list, div_list, scaf_list, val_list = [], [], [], []
    N = len(vals_valid)
    rng = np.random.default_rng(0)
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boot_smiles = [vals_valid[i] for i in idx]
        val_list.append(validity_full)  # keep same validity on full set (stable)
        uniq_list.append(compute_uniqueness(boot_smiles))

        # diversity
        ids = rng.integers(0, N, size=(min(pair_subsample, max(1, N*(N-1)//2)), 2))
        sims = []
        for a,b in ids:
            if a == b: continue
            sims.append(1.0 - DataStructs.TanimotoSimilarity(fps[a], fps[b]))
        div_list.append(np.mean(sims) if sims else np.nan)

        # scaffold diversity
        sc = [scafs[i] for i in idx if scafs[i] is not None]
        scaf_list.append(len(set(sc))/max(1,len(sc)) if sc else np.nan)

    return {
        "validity": np.array(val_list, float),
        "uniqueness": np.array(uniq_list, float),
        "diversity": np.array(div_list, float),
        "scaffold_diversity": np.array(scaf_list, float),
    }

# Collect per-run bootstrap arrays
metric_samples = { "validity":[], "uniqueness":[], "diversity":[], "scaffold_diversity": [] }
run_labels = []
for rid in RUN_IDS:
    path = ROOT / f"run_{rid}" / GEN_FILE
    if not path.exists(): 
        continue
    df = pd.read_csv(path)
    if "smiles" not in df.columns or df.empty:
        continue
    smiles = df["smiles"].dropna().astype(str).tolist()
    samples = run_metric_samples(smiles, n_boot=40, pair_subsample=150)
    for k in metric_samples:
        metric_samples[k].append(samples[k])
    run_labels.append(f"run_{rid}")

# Plot distributions with medians (and IQR on histogram line)
for metric, xlabel in [
    ("validity", "Validity"),
    ("uniqueness", "Uniqueness"),
    ("diversity", "Tanimoto distance (1 − similarity)"),
    ("scaffold_diversity", "Scaffold diversity (fraction)"),
]:
    # Hist-like step overlays + median/IQR
    plot_distribution_with_median(
        [arr for arr in metric_samples[metric]],
        run_labels,
        title=f"{metric.replace('_',' ').title()} — bootstrap distribution per run",
        xlabel=xlabel,
        bins=25
    )
    # Violin with median markers
    box_or_violin(
        [arr for arr in metric_samples[metric]],
        run_labels,
        title=f"{metric.replace('_',' ').title()} — violin per run (median shown)",
        ylabel=xlabel,
        kind="violin"
    )

# %%
# --- Rotatable Bonds / TPSA: distributions with medians + violin/box ---
rot_sets  = stats_df["rot_values"].tolist()
tpsa_sets = stats_df["tpsa_values"].tolist()
labels    = stats_df["run"].tolist()

plot_distribution_with_median(rot_sets, labels,
                              title="Rotatable Bonds — distributions per run (medians + IQR)",
                              xlabel="Rotatable Bonds", bins=20)
box_or_violin(rot_sets, labels, title="Rotatable Bonds — violin per run (medians shown)",
              ylabel="Rotatable Bonds", kind="violin")

plot_distribution_with_median(tpsa_sets, labels,
                              title="TPSA — distributions per run (medians + IQR)",
                              xlabel="TPSA (Å²)", bins=25)
box_or_violin(tpsa_sets, labels, title="TPSA — violin per run (medians shown)",
              ylabel="TPSA (Å²)", kind="violin")

# %%
