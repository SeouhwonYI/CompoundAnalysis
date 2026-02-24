from __future__ import annotations

import argparse
import ast
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import networkx as nx
except Exception:
    plt = None
    nx = None

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


def norm_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s']", "", s)
    return s.strip()


def safe_key(name: str) -> str:
    return re.sub(r"\W+", "_", str(name)).strip("_") or "unknown"


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def parse_ingredient_ratio(raw):
    if pd.isna(raw):
        return []

    if isinstance(raw, str):
        txt = raw.strip()
        if not txt:
            return []
        try:
            raw = ast.literal_eval(txt)
        except Exception:
            return []

    out = []
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                out.append((norm_text(k), float(v)))
            except Exception:
                continue
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                ing = item.get("ingredient", item.get("name", item.get("ing")))
                val = item.get("ratio", item.get("weight", item.get("w", item.get("gram", 0))))
                try:
                    out.append((norm_text(ing), float(val)))
                except Exception:
                    continue
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    out.append((norm_text(item[0]), float(item[1])))
                except Exception:
                    continue

    return [(i, w) for i, w in out if i and np.isfinite(w) and w > 0]


def parse_cleaned_ingredients(raw):
    if pd.isna(raw):
        return []
    if isinstance(raw, str):
        txt = raw.strip()
        if not txt:
            return []
        try:
            raw = ast.literal_eval(txt)
        except Exception:
            return []
    out = []
    if isinstance(raw, (list, tuple, set)):
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                ing = norm_text(item[0])
            else:
                ing = norm_text(item)
            if ing:
                out.append((ing, 1.0))
    return out


def parse_molecule_ids(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, (set, list, tuple)):
        out = []
        for v in x:
            try:
                out.append(int(v))
            except Exception:
                continue
        return out
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return []
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (set, list, tuple)):
            out = []
            for v in val:
                try:
                    out.append(int(v))
                except Exception:
                    continue
            return out
    except Exception:
        pass
    return [int(v) for v in re.findall(r"\d+", s)]


def build_recipe_to_molset(df_edges: pd.DataFrame):
    d = defaultdict(set)
    for rid, mid in df_edges[["recipe_id", "molecule"]].itertuples(index=False):
        d[int(rid)].add(int(mid))
    return d


def build_molecule_graph(rids, rid2molset, min_cooc=3):
    cooc = Counter()
    for rid in rids:
        mols = sorted(rid2molset.get(int(rid), []))
        for i in range(len(mols)):
            for j in range(i + 1, len(mols)):
                cooc[(mols[i], mols[j])] += 1

    g = nx.Graph()
    for (a, b), w in cooc.items():
        if w >= min_cooc:
            g.add_edge(a, b, weight=w)
    return g


def cluster_graph(g):
    if g.number_of_nodes() == 0:
        return {}
    communities = nx.algorithms.community.louvain_communities(g, weight="weight", seed=42)
    part = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            part[node] = cid
    return part


def export_graph_for_group(df_edges: pd.DataFrame, out_graph_dir: Path, key: str) -> None:
    cuisine_dir = out_graph_dir / key
    cuisine_dir.mkdir(parents=True, exist_ok=True)

    edge_cols = ["recipe_id", "name", "rank", "molecule", "molecule_name", "flavor_profile", "p"]
    edge_cols = [c for c in edge_cols if c in df_edges.columns]
    edge_export = df_edges[edge_cols].sort_values(["recipe_id", "rank"]).copy()
    edge_export.to_csv(cuisine_dir / "000_recipe_molecule_edges.csv", index=False)

    mol_weight = (
        df_edges.groupby("molecule", as_index=False)["p"]
        .sum()
        .rename(columns={"p": "weight"})
    )
    mol_meta_cols = [c for c in ["molecule", "molecule_name", "flavor_profile"] if c in df_edges.columns]
    mol_meta = df_edges[mol_meta_cols].drop_duplicates(subset=["molecule"])
    mol_weight = mol_weight.merge(mol_meta, on="molecule", how="left")
    mol_weight = mol_weight.sort_values("weight", ascending=False).copy()
    mol_weight["rank"] = np.arange(1, len(mol_weight) + 1)
    mol_weight.to_csv(cuisine_dir / "001_molecule_weight.csv", index=False)

    mol_recipe_cols = [c for c in ["molecule", "molecule_name", "flavor_profile", "recipe_id", "name", "p"] if c in df_edges.columns]
    mol_recipe = df_edges[mol_recipe_cols].copy()
    mol_recipe = mol_recipe.sort_values(["molecule", "p"], ascending=[True, False]).copy()
    mol_recipe["rank"] = mol_recipe.groupby("molecule")["p"].rank(method="first", ascending=False).astype(int)
    ordered = [c for c in ["molecule", "molecule_name", "flavor_profile", "rank", "recipe_id", "name", "p"] if c in mol_recipe.columns]
    mol_recipe = mol_recipe[ordered]
    mol_recipe.to_csv(cuisine_dir / "002_molecule_recipe_edges.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cuisine-wise molecule graph inputs from recipes.")
    parser.add_argument("--recipes-path", type=Path, default=Path("./recipes.csv"))
    parser.add_argument("--flavordb-path", type=Path, default=Path("./preprocess/flavordb_alias_in_vocab_only.csv"))
    parser.add_argument("--molecules-path", type=Path, default=Path("./molecules.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("./result"))
    parser.add_argument("--cuisines", nargs="*", default=None, help="Target cuisines. Omit to generate all cuisines and ALL.")
    parser.add_argument("--top-k-molecules-per-recipe", type=int, default=200)
    parser.add_argument("--include-all", action="store_true", help="Also export ALL group when --cuisines is provided.")
    parser.add_argument("--export-cluster-overview-plots", action="store_true")
    parser.add_argument("--cluster-min-cooc", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    start_ts = time.time()
    args = parse_args()

    out_dir = args.out_dir
    graph_dir = out_dir / "graph"
    analysis_dir = out_dir / "analysis"
    plots_dir = out_dir / "plots"
    for d in [out_dir, graph_dir, analysis_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    recipes = pd.read_csv(args.recipes_path)
    flavordb = pd.read_csv(args.flavordb_path)
    molecules = pd.read_csv(args.molecules_path)

    recipe_id_col = find_col(recipes, ["recipe_id", "id", "rid"])
    recipe_name_col = find_col(recipes, ["name", "recipe_name", "title"])
    cuisine_col = find_col(recipes, ["cuisine", "cuisine_name"])
    ratio_col = find_col(recipes, ["ingredients_ratio", "ingredient_ratio", "ingredients_with_ratio"])
    cleaned_col = find_col(recipes, ["cleaned_ingredients", "ingredients"])

    if cuisine_col is None:
        raise KeyError(f"Missing cuisine column. columns={list(recipes.columns)}")
    if ratio_col is None and cleaned_col is None:
        raise KeyError(f"Missing both ratio and cleaned ingredient columns. columns={list(recipes.columns)}")
    if recipe_id_col is None:
        recipes = recipes.copy()
        recipes["recipe_id"] = np.arange(recipes.shape[0], dtype=np.int64)
        recipe_id_col = "recipe_id"
    if recipe_name_col is None:
        recipes = recipes.copy()
        recipes["name"] = recipes[recipe_id_col].map(lambda x: f"recipe_{x}")
        recipe_name_col = "name"

    ing_col = find_col(flavordb, ["alias", "ingredient", "name", "entity_alias_readable", "entity_alias"])
    mol_col = find_col(flavordb, ["molecules", "molecule_ids", "molecule id", "molecule_id"])
    if ing_col is None or mol_col is None:
        raise KeyError(f"FlavorDB columns missing. ing_col={ing_col}, mol_col={mol_col}, all={list(flavordb.columns)}")

    molecule_id_col = find_col(molecules, ["pubchem id", "molecule", "molecule_id", "id"])
    molecule_name_col = find_col(molecules, ["common name", "molecule_name", "name"])
    flavor_profile_col = find_col(molecules, ["flavor profile", "flavor_profile"])
    if molecule_id_col is None:
        raise KeyError(f"Molecule id column missing. all={list(molecules.columns)}")

    selected_cuisines = args.cuisines
    if selected_cuisines:
        recipes = recipes[recipes[cuisine_col].isin(selected_cuisines)].copy()

    flavordb = flavordb.copy()
    flavordb["ingredient_norm"] = flavordb[ing_col].map(norm_text)
    flavordb["molecule_ids"] = flavordb[mol_col].map(parse_molecule_ids)
    tmp = flavordb[["ingredient_norm", "molecule_ids"]].dropna(subset=["ingredient_norm"])
    tmp = tmp.explode("molecule_ids").dropna(subset=["molecule_ids"])
    ing_to_molecules = (
        tmp.groupby("ingredient_norm")["molecule_ids"]
        .apply(lambda x: sorted(set(int(v) for v in x)))
        .to_dict()
    )

    records = []
    for _, row in tqdm(recipes.iterrows(), total=len(recipes), desc="Parse recipes"):
        rid = row[recipe_id_col]
        rname = row[recipe_name_col]
        cuisine = row[cuisine_col]
        if ratio_col is not None:
            pairs = parse_ingredient_ratio(row[ratio_col])
        else:
            pairs = parse_cleaned_ingredients(row[cleaned_col])
        if not pairs:
            continue
        z = sum(w for _, w in pairs)
        if z <= 0:
            continue
        for ing, w in pairs:
            records.append(
                {
                    "recipe_id": int(rid),
                    "name": str(rname),
                    "cuisine": str(cuisine),
                    "ingredient": ing,
                    "w_ri": float(w / z),
                }
            )

    recipes_long = pd.DataFrame(records)
    if recipes_long.empty:
        raise RuntimeError("No parsed recipe rows. Check recipe ingredient format.")

    recipes_long["has_flavordb_match"] = recipes_long["ingredient"].isin(ing_to_molecules)
    matched_rids = set(recipes_long.loc[recipes_long["has_flavordb_match"], "recipe_id"].unique())
    recipes_long = recipes_long[recipes_long["recipe_id"].isin(matched_rids)].copy()
    recipes_long.to_csv(out_dir / "recipes_long_normalized.csv", index=False)

    score_rows = []
    unk_rows = []
    for rid, grp in tqdm(recipes_long.groupby("recipe_id"), desc="Compute S(r,m)"):
        acc = defaultdict(float)
        mass_covered = 0.0
        for ing, w in grp[["ingredient", "w_ri"]].itertuples(index=False):
            mols = ing_to_molecules.get(ing, [])
            if not mols:
                continue
            contrib = w / len(mols)
            for m in mols:
                acc[m] += contrib
            mass_covered += w
        for m, s in acc.items():
            score_rows.append({"recipe_id": int(rid), "molecule": int(m), "S_rm": float(s)})
        unk_rows.append({"recipe_id": int(rid), "u_r": float(max(0.0, 1.0 - mass_covered))})

    recipe_molecule = pd.DataFrame(score_rows)
    recipe_unk = pd.DataFrame(unk_rows)

    rid_meta = recipes_long[["recipe_id", "name", "cuisine"]].drop_duplicates(subset=["recipe_id"])
    mol_meta_cols = [molecule_id_col]
    if molecule_name_col is not None:
        mol_meta_cols.append(molecule_name_col)
    if flavor_profile_col is not None:
        mol_meta_cols.append(flavor_profile_col)
    molecule_meta = molecules[mol_meta_cols].copy()

    rename_map = {molecule_id_col: "molecule"}
    if molecule_name_col is not None:
        rename_map[molecule_name_col] = "molecule_name"
    if flavor_profile_col is not None:
        rename_map[flavor_profile_col] = "flavor_profile"
    molecule_meta = molecule_meta.rename(columns=rename_map)
    molecule_meta["molecule"] = pd.to_numeric(molecule_meta["molecule"], errors="coerce")
    molecule_meta = molecule_meta.dropna(subset=["molecule"]).copy()
    molecule_meta["molecule"] = molecule_meta["molecule"].astype(int)
    molecule_meta = molecule_meta.drop_duplicates(subset=["molecule"])

    recipe_molecule = recipe_molecule.merge(rid_meta, on="recipe_id", how="left")
    recipe_molecule = recipe_molecule.merge(molecule_meta, on="molecule", how="left")
    recipe_molecule = recipe_molecule.sort_values(["recipe_id", "S_rm"], ascending=[True, False]).copy()
    recipe_molecule["rank"] = recipe_molecule.groupby("recipe_id")["S_rm"].rank(method="first", ascending=False).astype(int)
    recipe_molecule = recipe_molecule[recipe_molecule["rank"] <= args.top_k_molecules_per_recipe].copy()
    recipe_molecule["p"] = recipe_molecule["S_rm"] / recipe_molecule.groupby("recipe_id")["S_rm"].transform("sum")

    recipe_molecule.to_csv(graph_dir / "recipe_molecule_edges.csv", index=False)
    recipe_unk.to_csv(graph_dir / "recipe_unk_mass.csv", index=False)

    for cuisine, sub in tqdm(recipe_molecule.groupby("cuisine"), desc="Export cuisine graphs"):
        export_graph_for_group(sub.copy(), graph_dir, safe_key(cuisine))

    include_all = args.include_all or (not selected_cuisines)
    if include_all:
        export_graph_for_group(recipe_molecule.copy(), graph_dir, "ALL")

    analysis_summary = []
    target_groups = sorted(recipe_molecule["cuisine"].dropna().astype(str).unique().tolist())
    if include_all:
        target_groups = ["ALL"] + target_groups

    rid2molset = build_recipe_to_molset(recipe_molecule)
    for group in tqdm(target_groups, desc="Cuisine clustering"):
        if group == "ALL":
            rids = sorted(recipes_long["recipe_id"].unique())
        else:
            rids = sorted(recipes_long.loc[recipes_long["cuisine"] == group, "recipe_id"].unique())

        n_nodes = 0
        n_edges = 0
        n_clusters = 0
        if nx is not None:
            g = build_molecule_graph(rids, rid2molset, min_cooc=args.cluster_min_cooc)
            part = cluster_graph(g)
            n_nodes = g.number_of_nodes()
            n_edges = g.number_of_edges()
            n_clusters = len(set(part.values())) if part else 0

            if args.export_cluster_overview_plots and plt is not None:
                fig = plt.figure(figsize=(8, 6))
                if g.number_of_nodes() > 0:
                    pos = nx.spring_layout(g, seed=42)
                    node_colors = [part.get(n, -1) for n in g.nodes()]
                    nx.draw_networkx(
                        g,
                        pos=pos,
                        node_size=80,
                        with_labels=False,
                        edge_color="lightgray",
                        node_color=node_colors,
                        cmap=plt.cm.tab20,
                    )
                plt.title(f"Molecule graph clusters - {group}")
                plt.axis("off")
                fig.savefig(plots_dir / f"clusters_{safe_key(group)}.png", dpi=200, bbox_inches="tight")
                plt.close(fig)

        analysis_summary.append(
            {
                "group": group,
                "n_recipes": len(rids),
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "n_clusters": n_clusters,
            }
        )

    pd.DataFrame(analysis_summary).to_csv(analysis_dir / "analysis_summary.csv", index=False)
    elapsed = time.time() - start_ts
    print(f"Done. elapsed={elapsed:.1f}s")
    print("Output:")
    print(f" - {graph_dir.resolve()}")
    print(f" - {analysis_dir.resolve()}")
    print(f" - {plots_dir.resolve()}")


if __name__ == "__main__":
    main()
