"""Cuisine-only molecule clustering and plotting runner.

Refactored from the notebook workflow to reproduce:
1) per-cuisine summary panel (pie + top-5 cluster wordcloud + ingredient lift)
2) per-cuisine molecule-space graph visualization

Primary input (notebook-compatible):
- saved_state/p_by_r.pkl
- saved_state/rid2cuisine.pkl
- saved_state/recipes_long.pkl
- data/molecules.csv

Fallback input:
- result/graph/<cuisine>/000_recipe_molecule_edges.csv
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None

try:
    from pyvis.network import Network
except Exception:
    Network = None

REPO_ROOT = Path(__file__).resolve().parents[1]
SAVE_DIR = REPO_ROOT / "saved_state"
DATA_DIR = REPO_ROOT / "data"
GRAPH_ROOT = REPO_ROOT / "result" / "graph"
PLOTS_ROOT = REPO_ROOT / "result" / "plots"
ANALYSIS_ROOT = REPO_ROOT / "result" / "analysis"


def safe_key(name: str) -> str:
    return "_".join(str(name).strip().split()) or "unknown"


def _pick_col(df: pd.DataFrame, cands: Sequence[str]) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    return None


def _parse_ingredient_cell(cell) -> list[str]:
    if cell is None or pd.isna(cell):
        return []

    if isinstance(cell, (list, tuple, set)):
        return [str(x).strip() for x in cell if str(x).strip()]

    sval = str(cell).strip()
    if not sval or sval.lower() == "nan":
        return []

    if (sval.startswith("[") and sval.endswith("]")) or (sval.startswith("{") and sval.endswith("}")):
        try:
            parsed = ast.literal_eval(sval)
            if isinstance(parsed, (list, tuple, set)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

        inner = sval[1:-1]
        toks = [t.strip().strip("'\"") for t in inner.split(",")]
        toks = [t for t in toks if t]
        if toks:
            return toks

    for sep in ["|", ";", ","]:
        if sep in sval:
            toks = [t.strip() for t in sval.split(sep)]
            toks = [t for t in toks if t]
            if len(toks) > 1:
                return toks

    return [sval]


def load_molecule_names() -> Dict[int, str]:
    candidates = [
        DATA_DIR / "molecules.csv",
        REPO_ROOT / "molecules.csv",
    ]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        return {}
    mols = pd.read_csv(csv_path)
    col_id = _pick_col(mols, ["pubchem id", "pubchem_id", "PubChem ID", "id", "molecule_id"])
    col_name = _pick_col(mols, ["common_name", "name", "molecule_name", "Molecule Name"])
    if col_id is None or col_name is None:
        return {}
    mol_meta = (
        mols[[col_id, col_name]]
        .dropna()
        .drop_duplicates(subset=[col_id])
        .rename(columns={col_id: "mol_id", col_name: "mol_name"})
    )
    return dict(zip(mol_meta["mol_id"].astype(int), mol_meta["mol_name"].astype(str)))


def load_saved_state() -> tuple[dict, dict, dict]:
    p_path = SAVE_DIR / "p_by_r.pkl"
    c_path = SAVE_DIR / "rid2cuisine.pkl"
    r_path = SAVE_DIR / "recipes_long.pkl"
    if not (p_path.exists() and c_path.exists() and r_path.exists()):
        return {}, {}, {}

    with p_path.open("rb") as f:
        p_by_r = pickle.load(f)
    with c_path.open("rb") as f:
        rid2cuisine = pickle.load(f)
    recipes_long = pd.read_pickle(r_path)

    rid_col = _pick_col(recipes_long, ["rid", "recipe_id", "recipeid"])
    ing_col = _pick_col(recipes_long, ["ingredient", "ingredients", "ing", "ingredient_name"])
    rid2ings = defaultdict(set)
    if rid_col and ing_col:
        for rid, ing in recipes_long[[rid_col, ing_col]].itertuples(index=False):
            if pd.isna(rid) or pd.isna(ing):
                continue
            for val in _parse_ingredient_cell(ing):
                rid2ings[int(rid)].add(val)

    return p_by_r, rid2cuisine, rid2ings


def _extract_rid2ings_from_df(df: pd.DataFrame) -> dict[int, set[str]]:
    rid_col = _pick_col(df, ["recipe_id", "rid", "recipeid"])
    ing_col = _pick_col(
        df,
        [
            "ingredient",
            "ingredients",
            "ingredient_name",
            "ingredient_name_en",
            "ing",
            "ingredient_clean",
            "ingredient_list",
            "ingredients_list",
            "ingredients_en",
        ],
    )
    rid2ings: dict[int, set[str]] = defaultdict(set)
    if rid_col is None or ing_col is None:
        return rid2ings

    for rid, ing in df[[rid_col, ing_col]].dropna().itertuples(index=False):
        for sval in _parse_ingredient_cell(ing):
            rid2ings[int(rid)].add(sval)
    return rid2ings


def load_from_graph_csv(cuisine: str) -> tuple[dict, list[int], dict[int, set[str]]]:
    csv_path = GRAPH_ROOT / cuisine / "000_recipe_molecule_edges.csv"
    if not csv_path.exists():
        return {}, [], {}
    df = pd.read_csv(csv_path)
    required = {"recipe_id", "molecule", "p"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path} missing required columns: {sorted(required - set(df.columns))}")
    p_by_r: dict[int, dict[int, float]] = defaultdict(dict)
    for rid, mol, p in df[["recipe_id", "molecule", "p"]].itertuples(index=False):
        p_by_r[int(rid)][int(mol)] = float(p)
    rid2ings = _extract_rid2ings_from_df(df)
    return p_by_r, sorted(p_by_r.keys()), rid2ings


def enrich_molecule_names_from_edges(mol_id_to_name: Dict[int, str], cuisine: str) -> Dict[int, str]:
    """Fill missing molecule names from cuisine graph csv columns when available."""
    csv_path = GRAPH_ROOT / cuisine / "000_recipe_molecule_edges.csv"
    if not csv_path.exists():
        return mol_id_to_name
    try:
        df = pd.read_csv(csv_path, nrows=50000)
    except Exception:
        return mol_id_to_name

    mol_col = _pick_col(df, ["molecule", "molecule_id", "mol_id"])
    name_col = _pick_col(df, ["molecule_name", "name", "mol_name", "common_name"])
    if mol_col is None or name_col is None:
        return mol_id_to_name

    updates = 0
    for m, name in df[[mol_col, name_col]].dropna().itertuples(index=False):
        mid = int(m)
        sval = str(name).strip()
        if not sval:
            continue
        if mid not in mol_id_to_name:
            mol_id_to_name[mid] = sval
            updates += 1
    if updates:
        print(f"[{cuisine}] molecule names enriched from graph CSV: +{updates}")
    return mol_id_to_name


def mass_cut_molecules(dist: dict, rho: float, max_keep: int | None) -> list[int]:
    items: list[tuple[int, float]] = []
    for k, v in dist.items():
        if k is None:
            continue
        try:
            items.append((int(k), float(v)))
        except Exception:
            continue
    items.sort(key=lambda x: x[1], reverse=True)
    kept: list[int] = []
    s = 0.0
    for m, p in items:
        kept.append(m)
        s += p
        if max_keep is not None and len(kept) >= max_keep:
            break
        if s >= rho:
            break
    return kept


def cooc_counts_from_recipes(rids: Iterable[int], p_by_r: dict, rho: float, max_keep: int) -> tuple[int, Counter, Counter]:
    count_m = Counter()
    count_mm = Counter()
    n = 0
    for rid in rids:
        dist = p_by_r.get(rid)
        if dist is None:
            continue
        mols = sorted(set(mass_cut_molecules(dist, rho=rho, max_keep=max_keep)))
        if len(mols) < 2:
            continue
        n += 1
        count_m.update(mols)
        for a, b in combinations(mols, 2):
            count_mm[(a, b)] += 1
    return n, count_m, count_mm


def suggest_params_by_N(n: int) -> dict:
    return {
        "rho": 0.9,
        "max_keep_per_recipe": 200,
        "min_pair_count": 20 if n >= 500 else 10,
        "npmi_threshold": 0.5 if n >= 500 else 0.3,
        "top_edges_per_node": 50,
        "louvain_resolution": 1.0,
        "seed": 0,
    }


def build_npmi_graph(n: int, count_m: Counter, count_mm: Counter, min_pair_count: int, npmi_threshold: float, top_edges_per_node: int) -> tuple[nx.Graph, dict]:
    g = nx.Graph()
    if n <= 0:
        return g, {}
    p_m = {m: c / n for m, c in count_m.items()}
    edges_by_node: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (i, j), c in count_mm.items():
        if c < min_pair_count:
            continue
        p_ij = c / n
        p_i = p_m.get(i, 0.0)
        p_j = p_m.get(j, 0.0)
        if min(p_ij, p_i, p_j) <= 0:
            continue
        pmi = math.log(p_ij / (p_i * p_j))
        npmi = pmi / (-math.log(p_ij))
        if npmi < npmi_threshold:
            continue
        edges_by_node[i].append((j, float(npmi)))
        edges_by_node[j].append((i, float(npmi)))

    kept = set()
    for src, arr in edges_by_node.items():
        arr.sort(key=lambda x: x[1], reverse=True)
        for dst, w in arr[:top_edges_per_node]:
            a, b = (src, dst) if src < dst else (dst, src)
            kept.add((a, b, w))
    for a, b, w in kept:
        g.add_edge(a, b, weight=w)
    return g, p_m


def louvain_partition(g: nx.Graph, resolution: float, seed: int) -> tuple[dict[int, int], list[list[int]]]:
    try:
        import community as community_louvain  # type: ignore

        part = community_louvain.best_partition(g, weight="weight", resolution=resolution, random_state=seed)
        comms = defaultdict(list)
        for n, cid in part.items():
            comms[int(cid)].append(int(n))
        clusters = [nodes for _, nodes in sorted(comms.items(), key=lambda x: len(x[1]), reverse=True)]
    except Exception:
        comm_list = nx.community.louvain_communities(g, weight="weight", resolution=resolution, seed=seed)
        clusters = [list(nodes) for nodes in sorted(comm_list, key=len, reverse=True)]
    part2 = {}
    for cid, nodes in enumerate(clusters):
        for n in nodes:
            part2[int(n)] = cid
    return part2, clusters


def cuisine_soft_mass(rids: Iterable[int], p_by_r: dict, clusters: list[list[int]]) -> np.ndarray:
    acc = np.zeros(len(clusters), dtype=float)
    cnt = 0
    for rid in rids:
        dist = p_by_r.get(rid)
        if dist is None:
            continue
        masses = np.array([sum(float(dist.get(m, 0.0)) for m in nodes) for nodes in clusters], dtype=float)
        z = masses.sum()
        if z > 0:
            acc += masses / z
            cnt += 1
    return acc / cnt if cnt > 0 else acc


def ingredient_lift_for_cluster_soft(rids: Iterable[int], rid2w: dict, cid: int, rid2ings: dict, min_df: int = 5) -> list[tuple[str, float, float, float, int]]:
    cuisine_df = Counter()
    n = 0
    for rid in rids:
        ings = rid2ings.get(rid, [])
        if not ings:
            continue
        n += 1
        cuisine_df.update(set(ings))
    if n == 0:
        return []

    ing_w = Counter()
    w_sum = 0.0
    for rid in rids:
        w = rid2w.get(rid)
        if w is None:
            continue
        wr = float(w[cid])
        if wr <= 0:
            continue
        ings = rid2ings.get(rid, [])
        if not ings:
            continue
        w_sum += wr
        for ing in set(ings):
            ing_w[ing] += wr
    if w_sum <= 0:
        return []

    out = []
    for ing, c_w in ing_w.items():
        if cuisine_df[ing] < min_df:
            continue
        p_k = c_w / w_sum
        p_c = cuisine_df[ing] / n
        out.append((ing, float(p_k / max(p_c, 1e-12)), float(p_k), float(p_c), int(cuisine_df[ing])))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def make_wordcloud(weights: dict[str, float]):
    if not weights:
        return None
    if WordCloud is None:
        return None
    return WordCloud(width=600, height=420, background_color="white", max_words=60).generate_from_frequencies(weights)


def node_weights(cluster_nodes: list[int], p_m: dict, mol_id_to_name: dict) -> dict[str, float]:
    out = {}
    for m in cluster_nodes:
        val = p_m.get(int(m), 0.0)
        if val <= 0:
            continue
        out[mol_id_to_name.get(int(m), str(int(m)))] = float(val)
    return out


def save_moleculespace_plot(cuisine: str, g: nx.Graph, part: dict[int, int], mol_id_to_name: dict[int, str], out_path: Path) -> None:
    if g.number_of_nodes() == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(g, weight="weight", seed=0, k=0.35)
    cluster_ids = np.array([part.get(int(n), -1) for n in g.nodes()])
    nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.15, width=0.5)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=16, node_color=cluster_ids, cmap="tab20", alpha=0.9)

    top_nodes = sorted(g.nodes(), key=lambda n: g.degree(n, weight="weight"), reverse=True)[:25]
    labels = {n: mol_id_to_name.get(int(n), f"m{int(n)}") for n in top_nodes}
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=7, ax=ax)
    ax.set_title(f"{cuisine} molecule-space graph")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _parse_flavor_tokens(s: str) -> list[str]:
    if s is None:
        return []
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return []
    s = s.strip("{}")
    toks = [re.sub(r"\s+", " ", p.strip().strip("'\"").lower()).strip() for p in s.split(",")]
    return [t for t in toks if t]


def _community_keywords(cluster_nodes: list[int], mol_id_to_name: dict[int, str], top_n: int = 5) -> str:
    ban = {
        "acid", "methyl", "ethyl", "propyl", "isopropyl", "d", "l", "dl", "cis", "trans",
        "alpha", "beta", "gamma", "oxide", "one", "ol", "ene", "ate", "yl", "di", "tri",
    }
    cnt: Counter = Counter()
    for m in cluster_nodes:
        name = mol_id_to_name.get(int(m), "")
        if not name:
            continue
        toks = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", name)]
        cnt.update(t for t in toks if t not in ban)
    return ", ".join([w for w, _ in cnt.most_common(top_n)])


def save_moleculespace_html(cuisine: str, g: nx.Graph, part: dict[int, int], mol_id_to_name: dict[int, str], out_html: Path) -> None:
    """Interactive molecule-space HTML similar to vocab_alignment notebook style."""
    if Network is None or g.number_of_nodes() == 0:
        return

    wdeg = dict(g.degree(weight="weight"))
    top_nodes = set(sorted(g.nodes(), key=lambda n: wdeg.get(n, 0.0), reverse=True)[:50])
    deg_vals = np.array([max(wdeg.get(n, 0.0), 1e-12) for n in g.nodes()])
    size_map = {n: 4 + 7 * math.log1p(v) for n, v in zip(g.nodes(), deg_vals)}
    palette = [
        "#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1", "#9C755F", "#76B7B2", "#EDC948",
        "#BAB0AC", "#8CD17D", "#A0CBE8", "#FFBE7D",
    ]

    net = Network(height="860px", width="100%", bgcolor="#ffffff", font_color="black", notebook=False, cdn_resources="in_line")

    for n in g.nodes():
        cid = int(part.get(int(n), 0))
        color = palette[cid % len(palette)]
        name = mol_id_to_name.get(int(n), str(int(n)))
        title = (
            f"<b>{name}</b><br>"
            f"Molecule ID: {int(n)}<br>"
            f"Cluster: {cid+1}<br>"
            f"Weighted degree: {wdeg.get(n, 0.0):.3f}"
        )
        net.add_node(
            int(n),
            label=name if n in top_nodes else "",
            title=title,
            size=float(size_map[n]),
            color={"background": color, "border": "rgba(0,0,0,0)", "highlight": {"background": color, "border": "rgba(0,0,0,0)"}},
            group=str(cid),
        )

    weights = [g[u][v].get("weight", 0.0) for u, v in g.edges()]
    if weights:
        w_min, w_max = min(weights), max(weights)
    else:
        w_min, w_max = 0.0, 1.0

    for u, v in g.edges():
        w = float(g[u][v].get("weight", 0.0))
        width = 0.4 + 3.6 * ((w - w_min) / (w_max - w_min + 1e-12))
        net.add_edge(int(u), int(v), value=w, width=float(width), color="rgba(120,120,120,0.45)")

    net.set_options(
        """
        const options = {
          "physics": {
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {"gravitationalConstant": -18, "springLength": 110, "damping": 0.92},
            "minVelocity": 0.02,
            "maxVelocity": 8,
            "timestep": 0.18,
            "stabilization": {"enabled": true, "iterations": 1200, "updateInterval": 50}
          },
          "interaction": {"hover": true, "tooltipDelay": 100},
          "edges": {"smooth": false}
        }
        """
    )
    net.write_html(str(out_html), notebook=False)

    cluster_nodes: dict[int, list[int]] = defaultdict(list)
    for node, cid in part.items():
        cluster_nodes[int(cid)].append(int(node))
    legend_data = [
        {
            "cid": int(cid),
            "label": f"Community {cid}",
            "color": palette[cid % len(palette)],
            "keywords": _community_keywords(nodes, mol_id_to_name, top_n=5),
        }
        for cid, nodes in sorted(cluster_nodes.items(), key=lambda x: x[0])
    ]

    html = out_html.read_text(encoding="utf-8")
    injection = f"""
<style>
#legend-panel {{ position: fixed; right: 18px; top: 18px; width: 360px; max-height: 78vh; overflow: auto; background: rgba(245,245,245,0.95); border-radius: 14px; border: 1px solid #d0d0d0; padding: 14px; font-family: Arial, sans-serif; z-index: 9999; }}
#legend-panel h3 {{ margin: 0 0 6px 0; font-size: 28px; }}
#legend-panel .sub {{ color:#555; margin-bottom: 12px; }}
.legend-item {{ display:flex; align-items:flex-start; gap:10px; margin-bottom:10px; cursor:pointer; }}
.sw {{ width:20px; height:20px; border-radius:4px; margin-top:4px; }}
.item-txt b {{ display:block; font-size: 22px; line-height:1.1; }}
.item-txt small {{ color:#555; font-size: 16px; line-height:1.2; }}
#legend-reset {{ float:right; }}
</style>
<div id="legend-panel">
  <button id="legend-reset">Reset</button>
  <h3>Legend (click to dim others)</h3>
  <div class="sub">Click a community: selected stays exactly legend color; others dim.</div>
  <div id="legend-items"></div>
</div>
<script>
const LEGEND_DATA = {json.dumps(legend_data, ensure_ascii=False)};
const baseNodeColors = new Map();
const nodesDS = network.body.data.nodes;
nodesDS.get().forEach(n => baseNodeColors.set(n.id, n.color));

function resetLegend() {{
  const updates = [];
  nodesDS.get().forEach(n => {{
    updates.push({{id:n.id, color:baseNodeColors.get(n.id), opacity:1, font:{{color:'#222'}}}});
  }});
  nodesDS.update(updates);
}}

function selectCommunity(cid, color) {{
  const updates = [];
  nodesDS.get().forEach(n => {{
    if (String(n.group) === String(cid)) {{
      updates.push({{id:n.id, color:{{background:color,border:'rgba(0,0,0,0)',highlight:{{background:color,border:'rgba(0,0,0,0)'}}}}, opacity:1, font:{{color:'#111'}}}});
    }} else {{
      updates.push({{id:n.id, color:'rgba(210,210,210,0.24)', opacity:0.18, font:{{color:'rgba(120,120,120,0.3)'}}}});
    }}
  }});
  nodesDS.update(updates);
}}

const itemsHost = document.getElementById('legend-items');
LEGEND_DATA.forEach(row => {{
  const item = document.createElement('div');
  item.className = 'legend-item';
  item.innerHTML = `<div class="sw" style="background:${{row.color}}"></div><div class="item-txt"><b>${{row.label}}</b><small>${{row.keywords || ''}}</small></div>`;
  item.onclick = () => selectCommunity(row.cid, row.color);
  itemsHost.appendChild(item);
}});
document.getElementById('legend-reset').onclick = resetLegend;

network.once('stabilizationIterationsDone', function() {{
  network.setOptions({{physics: {{enabled: false}}}});
}});
</script>
"""
    if "</body>" in html:
        html = html.replace("</body>", injection + "\n</body>")
    else:
        html += injection
    out_html.write_text(html, encoding="utf-8")


def plot_cuisine_summary(cuisine: str, rids: list[int], p_by_r: dict, rid2ings: dict, mol_id_to_name: dict[int, str], out_dir: Path, top_k_clusters: int, df_rate_ban_threshold: float, verbose_ban: bool) -> dict | None:
    n, count_m, count_mm = cooc_counts_from_recipes(rids, p_by_r, rho=0.9, max_keep=200)
    if n == 0:
        print(f"[{cuisine}] no usable recipes")
        return None

    df_rate = {m: c / n for m, c in count_m.items()}
    ban_mols = {m for m, r in df_rate.items() if r > df_rate_ban_threshold}
    if verbose_ban:
        print(f"[{cuisine}] banning molecules with df_rate > {df_rate_ban_threshold}: {len(ban_mols)}")
        top_ban = sorted([(m, df_rate[m], count_m[m]) for m in ban_mols], key=lambda x: x[1], reverse=True)[:20]
        for m, r, df in top_ban:
            print(f"  {m:7d}  df_rate={r:6.3f}  df={df:6d}  name={mol_id_to_name.get(m, str(m))}")
    count_m = Counter({m: c for m, c in count_m.items() if m not in ban_mols})
    count_mm = Counter({k: c for k, c in count_mm.items() if k[0] not in ban_mols and k[1] not in ban_mols})

    params = suggest_params_by_N(n)
    g, p_m = build_npmi_graph(
        n,
        count_m,
        count_mm,
        min_pair_count=params["min_pair_count"],
        npmi_threshold=params["npmi_threshold"],
        top_edges_per_node=params["top_edges_per_node"],
    )
    if g.number_of_nodes() == 0:
        print(f"[{cuisine}] empty graph")
        return None

    part, clusters = louvain_partition(g, resolution=params["louvain_resolution"], seed=params["seed"])
    soft = cuisine_soft_mass(rids, p_by_r, clusters)
    top_ids = [int(i) for i in np.argsort(soft)[::-1][:top_k_clusters] if soft[int(i)] > 0]

    rid2w = {}
    for rid in rids:
        dist = p_by_r.get(rid)
        if dist is None:
            continue
        masses = np.array([sum(float(dist.get(m, 0.0)) for m in nodes) for nodes in clusters], dtype=float)
        z = masses.sum()
        rid2w[rid] = masses / z if z > 0 else np.zeros_like(masses)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    ax0 = axes[0, 0]
    pie_vals = [float(v) for v in soft]
    wedges, _ = ax0.pie(pie_vals, labels=None, startangle=90)
    ax0.set_title(f"{cuisine} (all clusters; labels=top {top_k_clusters})")

    for cid in top_ids:
        w = wedges[cid]
        ang = np.deg2rad(0.5 * (w.theta1 + w.theta2))
        x, y = 1.15 * np.cos(ang), 1.15 * np.sin(ang)
        ax0.annotate(
            f"Cluster {cid+1} ({soft[cid]*100:.1f}%)",
            xy=(np.cos(ang), np.sin(ang)),
            xytext=(x, y),
            ha="left" if x >= 0 else "right",
            va="center",
            arrowprops=dict(arrowstyle="-", lw=0.8),
            fontsize=10,
        )

    panels = [axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]
    for ax, cid in zip(panels, top_ids):
        wc = make_wordcloud(node_weights(clusters[cid], p_m, mol_id_to_name))
        ax.set_title(f"Cluster {cid+1}")
        ax.axis("off")
        if wc is not None:
            ax.imshow(wc)
        lifts = ingredient_lift_for_cluster_soft(rids, rid2w, cid, rid2ings, min_df=5)
        txt = "\n".join([f"{i+1}. {ing} (lift={lift:.2f})" for i, (ing, lift, *_rest) in enumerate(lifts[:12])])
        if not txt:
            txt = "(ingredient lift unavailable)"
        ax.text(0.01, 0.01, txt, transform=ax.transAxes, fontsize=10, va="bottom", ha="left", bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    for ax in panels[len(top_ids) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / f"{safe_key(cuisine)}_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    save_moleculespace_plot(cuisine, g, part, mol_id_to_name, out_dir / f"{safe_key(cuisine)}_moleculespace_graph.png")
    save_moleculespace_html(cuisine, g, part, mol_id_to_name, out_dir / f"{safe_key(cuisine)}_moleculespace_graph.html")

    rows = [{"cluster_id": i + 1, "soft_mass": float(soft[i]), "molecule_count": len(nodes)} for i, nodes in enumerate(clusters)]
    pd.DataFrame(rows).sort_values("soft_mass", ascending=False).to_csv(
        (ANALYSIS_ROOT / safe_key(cuisine) / "000_cluster_summary.csv"), index=False
    )

    return {"N_used": n, "cluster_count": len(clusters), "graph_nodes": g.number_of_nodes(), "graph_edges": g.number_of_edges()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cuisine-only cluster summary plot generator")
    parser.add_argument("--cuisines", nargs="*", default=["Japanese"])
    parser.add_argument("--top-k-clusters", type=int, default=5)
    parser.add_argument("--all-cuisines", action="store_true")
    parser.add_argument("--df-rate-ban-threshold", type=float, default=0.75)
    parser.add_argument("--quiet-ban-log", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    p_by_r, rid2cuisine, rid2ings = load_saved_state()
    mol_id_to_name = load_molecule_names()

    if args.all_cuisines and rid2cuisine:
        cuisines = sorted({str(c) for c in rid2cuisine.values()})
    else:
        cuisines = args.cuisines

    for cuisine in cuisines:
        out_plot_dir = PLOTS_ROOT / safe_key(cuisine)
        out_analysis_dir = ANALYSIS_ROOT / safe_key(cuisine)
        out_plot_dir.mkdir(parents=True, exist_ok=True)
        out_analysis_dir.mkdir(parents=True, exist_ok=True)

        if p_by_r and rid2cuisine:
            rids = [int(rid) for rid, c in rid2cuisine.items() if str(c) == cuisine]
            p_source = p_by_r
            rid2ings_source = rid2ings
        else:
            p_source, rids, rid2ings_source = load_from_graph_csv(cuisine)

        mol_id_to_name = enrich_molecule_names_from_edges(mol_id_to_name, cuisine)

        print(f"\n=== {cuisine} ===")
        print(f"recipes: {len(rids)}")
        result = plot_cuisine_summary(
            cuisine=cuisine,
            rids=rids,
            p_by_r=p_source,
            rid2ings=rid2ings_source,
            mol_id_to_name=mol_id_to_name,
            out_dir=out_plot_dir,
            top_k_clusters=args.top_k_clusters,
            df_rate_ban_threshold=args.df_rate_ban_threshold,
            verbose_ban=(not args.quiet_ban_log),
        )
        if result:
            print(result)


if __name__ == "__main__":
    main()
