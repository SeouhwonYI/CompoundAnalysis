"""Cuisine-only molecule cluster analysis runner.

This script replaces the heavyweight notebook workflow with a diff-friendly Python module.
It consumes graph exports from `result/graph/<cuisine>/` and writes deterministic outputs to:

- result/analysis/<cuisine>/
- result/plots/<cuisine>/

Default scope is Thai/Korean for faster iterative setup.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_ROOT = REPO_ROOT / "result" / "graph"
ANALYSIS_ROOT = REPO_ROOT / "result" / "analysis"
PLOTS_ROOT = REPO_ROOT / "result" / "plots"
DEFAULT_TARGET_CUISINES = ["Thai", "Korean"]


def safe_key(name: str) -> str:
    return "_".join(str(name).strip().split()) or "unknown"


def suggest_params_by_recipe_count(n_recipes: int) -> dict:
    """Simple adaptive defaults by cuisine recipe count."""
    if n_recipes < 100:
        return {"min_pair_count": 2, "npmi_threshold": 0.05, "top_edges_per_node": 20}
    if n_recipes < 300:
        return {"min_pair_count": 3, "npmi_threshold": 0.08, "top_edges_per_node": 30}
    return {"min_pair_count": 5, "npmi_threshold": 0.10, "top_edges_per_node": 40}


def load_recipe_molecule_edges(cuisine: str) -> pd.DataFrame:
    csv_path = GRAPH_ROOT / cuisine / "000_recipe_molecule_edges.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing cuisine graph file: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"recipe_id", "molecule", "p"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")

    out = df.copy()
    out["recipe_id"] = out["recipe_id"].astype(int)
    out["molecule"] = out["molecule"].astype(int)
    out["p"] = out["p"].astype(float)
    return out


def build_counts(df_edges: pd.DataFrame, top_k: int | None = 200, min_p: float = 0.0):
    """Build recipe-conditioned co-occurrence counts from recipe->molecule edges."""
    count_m = Counter()
    count_mm = Counter()
    recipe_masses: Dict[int, float] = {}

    grouped = df_edges.sort_values(["recipe_id", "rank" if "rank" in df_edges.columns else "p"], ascending=[True, True]).groupby("recipe_id")

    used_recipe_count = 0
    for rid, grp in grouped:
        g = grp
        if top_k is not None:
            g = g.head(top_k)
        if min_p > 0:
            g = g[g["p"] >= min_p]
        mols = sorted(set(g["molecule"].tolist()))
        if len(mols) < 2:
            continue
        used_recipe_count += 1
        recipe_masses[int(rid)] = float(g["p"].sum())
        count_m.update(mols)
        for i, src in enumerate(mols):
            for dst in mols[i + 1 :]:
                count_mm[(src, dst)] += 1

    return used_recipe_count, count_m, count_mm, recipe_masses


def build_npmi_graph(
    n_used: int,
    count_m: Counter,
    count_mm: Counter,
    min_pair_count: int,
    npmi_threshold: float,
    top_edges_per_node: int,
) -> nx.Graph:
    g = nx.Graph()
    if n_used <= 0:
        return g

    p_m = {m: c / n_used for m, c in count_m.items()}
    edges_by_node: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for (i, j), c_ij in count_mm.items():
        if c_ij < min_pair_count:
            continue
        p_ij = c_ij / n_used
        p_i, p_j = p_m.get(i, 0.0), p_m.get(j, 0.0)
        if min(p_ij, p_i, p_j) <= 0:
            continue
        pmi = math.log(p_ij / (p_i * p_j))
        npmi = pmi / (-math.log(p_ij))
        if npmi < npmi_threshold:
            continue
        edges_by_node[i].append((j, float(npmi)))
        edges_by_node[j].append((i, float(npmi)))

    kept = set()
    for src, cands in edges_by_node.items():
        cands.sort(key=lambda x: x[1], reverse=True)
        for dst, w in cands[:top_edges_per_node]:
            a, b = (src, dst) if src < dst else (dst, src)
            kept.add((a, b, w))

    for a, b, w in kept:
        g.add_edge(a, b, weight=w)
    return g


def detect_communities(g: nx.Graph, resolution: float, seed: int) -> Dict[int, int]:
    """Try python-louvain first and fallback to networkx implementation."""
    if g.number_of_nodes() == 0:
        return {}

    try:
        import community as community_louvain  # type: ignore

        part = community_louvain.best_partition(g, weight="weight", resolution=resolution, random_state=seed)
        return {int(k): int(v) for k, v in part.items()}
    except Exception:
        communities = nx.community.louvain_communities(g, weight="weight", resolution=resolution, seed=seed)
        part = {}
        for cid, nodes in enumerate(communities):
            for n in nodes:
                part[int(n)] = int(cid)
        return part


def export_analysis(cuisine: str, df_edges: pd.DataFrame, part: Dict[int, int], g: nx.Graph, recipe_masses: Dict[int, float]):
    cuisine_key = safe_key(cuisine)
    out_dir = ANALYSIS_ROOT / cuisine_key
    out_dir.mkdir(parents=True, exist_ok=True)

    node_rows = []
    for node in sorted(part.keys()):
        node_rows.append(
            {
                "molecule": int(node),
                "cluster_id": int(part[node]),
                "degree": int(g.degree(node)),
                "weighted_degree": float(g.degree(node, weight="weight")),
            }
        )
    node_df = pd.DataFrame(node_rows).sort_values(["cluster_id", "weighted_degree"], ascending=[True, False])
    node_df.to_csv(out_dir / "001_molecule_cluster_membership.csv", index=False)

    recipe_cluster = (
        df_edges[df_edges["molecule"].isin(part.keys())]
        .assign(cluster_id=lambda d: d["molecule"].map(part))
        .groupby(["recipe_id", "cluster_id"], as_index=False)["p"]
        .sum()
        .rename(columns={"p": "cluster_mass"})
    )
    recipe_cluster["recipe_total_mass"] = recipe_cluster["recipe_id"].map(recipe_masses)
    recipe_cluster["cluster_mass_norm"] = recipe_cluster["cluster_mass"] / recipe_cluster["recipe_total_mass"].replace(0, np.nan)
    recipe_cluster.to_csv(out_dir / "002_recipe_cluster_mass.csv", index=False)

    summary_df = (
        recipe_cluster.groupby("cluster_id", as_index=False)
        .agg(
            recipe_count=("recipe_id", "nunique"),
            mean_cluster_mass=("cluster_mass_norm", "mean"),
        )
        .sort_values("recipe_count", ascending=False)
    )
    summary_df["rank"] = np.arange(1, len(summary_df) + 1)
    summary_df.to_csv(out_dir / "000_cluster_summary.csv", index=False)


def save_plot(cuisine: str, g: nx.Graph, part: Dict[int, int], summary_csv: Path):
    cuisine_key = safe_key(cuisine)
    out_dir = PLOTS_ROOT / cuisine_key
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(summary_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if not summary_df.empty:
        top = summary_df.sort_values("recipe_count", ascending=False).head(8)
        axes[0].bar(top["cluster_id"].astype(str), top["recipe_count"])
        axes[0].set_title(f"{cuisine} top clusters")
        axes[0].set_xlabel("cluster_id")
        axes[0].set_ylabel("recipe_count")
    else:
        axes[0].text(0.5, 0.5, "No cluster summary", ha="center", va="center")
        axes[0].axis("off")

    axes[1].axis("off")
    axes[1].text(
        0.01,
        0.98,
        "\n".join(
            [
                f"cuisine: {cuisine}",
                f"nodes: {g.number_of_nodes()}",
                f"edges: {g.number_of_edges()}",
                f"clusters: {len(set(part.values()))}",
            ]
        ),
        va="top",
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(out_dir / "000_cluster_overview.png", dpi=180)
    plt.close(fig)


def write_checkpoint(cuisine: str, step: str, payload: dict):
    cp_dir = ANALYSIS_ROOT / safe_key(cuisine) / "checkpoints"
    cp_dir.mkdir(parents=True, exist_ok=True)
    with (cp_dir / f"{step}.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_for_cuisine(cuisine: str, args):
    print(f"\n=== [{cuisine}] analysis start ===")

    df_edges = load_recipe_molecule_edges(cuisine)
    write_checkpoint(cuisine, "00_loaded", {"rows": len(df_edges), "recipes": int(df_edges['recipe_id'].nunique())})

    n_used, count_m, count_mm, recipe_masses = build_counts(df_edges, top_k=args.top_k, min_p=args.min_p)
    params = suggest_params_by_recipe_count(n_used)
    params.update(
        {
            "min_pair_count": args.min_pair_count if args.min_pair_count is not None else params["min_pair_count"],
            "npmi_threshold": args.npmi_threshold if args.npmi_threshold is not None else params["npmi_threshold"],
            "top_edges_per_node": args.top_edges_per_node if args.top_edges_per_node is not None else params["top_edges_per_node"],
        }
    )
    write_checkpoint(cuisine, "01_counts", {"N_used": n_used, "unique_molecules": len(count_m), "pairs": len(count_mm), **params})

    g = build_npmi_graph(
        n_used,
        count_m,
        count_mm,
        min_pair_count=params["min_pair_count"],
        npmi_threshold=params["npmi_threshold"],
        top_edges_per_node=params["top_edges_per_node"],
    )
    part = detect_communities(g, resolution=args.louvain_resolution, seed=args.seed)
    write_checkpoint(
        cuisine,
        "02_graph_clustered",
        {
            "graph_nodes": g.number_of_nodes(),
            "graph_edges": g.number_of_edges(),
            "cluster_count": len(set(part.values())),
        },
    )

    export_analysis(cuisine, df_edges, part, g, recipe_masses)
    summary_csv = ANALYSIS_ROOT / safe_key(cuisine) / "000_cluster_summary.csv"
    save_plot(cuisine, g, part, summary_csv)
    write_checkpoint(cuisine, "03_exported", {"analysis_dir": str((ANALYSIS_ROOT / safe_key(cuisine)).relative_to(REPO_ROOT))})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cuisine-only clustering analysis (Python runner)")
    parser.add_argument("--cuisines", nargs="*", default=DEFAULT_TARGET_CUISINES)
    parser.add_argument("--all-cuisines", action="store_true")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--min-pair-count", type=int)
    parser.add_argument("--npmi-threshold", type=float)
    parser.add_argument("--top-edges-per-node", type=int)
    parser.add_argument("--louvain-resolution", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.all_cuisines:
        cuisines = sorted([p.name for p in GRAPH_ROOT.iterdir() if p.is_dir()])
    else:
        cuisines = list(args.cuisines)

    if not cuisines:
        raise ValueError("No cuisines selected. Use --cuisines or --all-cuisines.")

    for cuisine in cuisines:
        run_for_cuisine(cuisine, args)


if __name__ == "__main__":
    main()
