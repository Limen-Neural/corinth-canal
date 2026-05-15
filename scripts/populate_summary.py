import csv
import statistics
from pathlib import Path

TARGET_MODELS = {"olmoe_baseline", "gemma4_26b_a4b_iq4_nl", "llama_3_2_dark_champion_q5_k_m"}
INDEX_PATH = Path("artifacts/index.csv")
SUMMARY_MD_PATH = Path("artifacts/csv_off_only_summary.md")

if not INDEX_PATH.exists():
    print("index.csv not found!")
    exit(1)

with open(INDEX_PATH, newline="") as f:
    rows = list(csv.DictReader(f))

# Filter
filtered = []
for r in rows:
    if r.get("model_slug") in TARGET_MODELS and \
       r.get("telemetry_source") == "csv_re4_path_tracing_telemetry" and \
       r.get("heartbeat_enabled") == "false":
        filtered.append(r)

# Sort by model, then repeat
filtered.sort(key=lambda x: (x["model_slug"], int(x["repeat_idx"])))

table_rows = []

for r in filtered:
    model = r["model_slug"]
    repeat = r["repeat_idx"]
    status = r["validation_status"]
    run_dir = Path(r["run_dir"])
    latent_csv = run_dir / "latent_telemetry.csv"
    
    if not latent_csv.exists():
        table_rows.append(f"| {model} | {repeat} | {status} (missing latent) | - | - | - | - | - | - | - | - |")
        continue
        
    with open(latent_csv, newline="") as f:
        l_rows = list(csv.DictReader(f))
        
    if not l_rows:
        table_rows.append(f"| {model} | {repeat} | {status} (empty latent) | 0 | - | - | - | - | - | - | - | - |")
        continue

    entropies = [float(x["routing_entropy"]) for x in l_rows if "routing_entropy" in x]
    v15_targets = [float(x["saaq_delta_q_v15_target"]) for x in l_rows if "saaq_delta_q_v15_target" in x]
    legacy_targets = [float(x["saaq_delta_q_legacy_target"]) for x in l_rows if "saaq_delta_q_legacy_target" in x]
    
    num_rows = len(l_rows)
    
    ent_mean = statistics.mean(entropies) if entropies else 0.0
    ent_min = min(entropies) if entropies else 0.0
    ent_max = max(entropies) if entropies else 0.0
    ent_range = ent_max - ent_min
    
    v15_final = v15_targets[-1] if v15_targets else 0.0
    v15_range = (max(v15_targets) - min(v15_targets)) if v15_targets else 0.0
    
    leg_final = legacy_targets[-1] if legacy_targets else 0.0
    leg_range = (max(legacy_targets) - min(legacy_targets)) if legacy_targets else 0.0
    
    # Format
    row_str = f"| {model} | {repeat} | {status} | {num_rows} | {ent_mean:.4f} | {ent_min:.4f} | {ent_max:.4f} | {ent_range:.4f} | {v15_final:.4f} | {v15_range:.4f} | {leg_final:.4f} | {leg_range:.4f} |"
    table_rows.append(row_str)

# Read markdown file
if SUMMARY_MD_PATH.exists():
    with open(SUMMARY_MD_PATH, "r") as f:
        lines = f.readlines()
else:
    print(f"{SUMMARY_MD_PATH} not found!")
    exit(1)

# Find where to insert table rows
header_idx = -1
for i, line in enumerate(lines):
    if line.strip().startswith("|---|---:|"):
        header_idx = i
        break

if header_idx != -1:
    out_lines = lines[:header_idx+1] + [r + "\n" for r in table_rows] + lines[header_idx+1:]
    with open(SUMMARY_MD_PATH, "w") as f:
        f.writelines(out_lines)
    print("Updated markdown file successfully.")
else:
    print("Could not find table header in markdown file!")
    exit(1)
