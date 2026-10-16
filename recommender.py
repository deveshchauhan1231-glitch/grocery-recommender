from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.sparse import load_npz


def load_artifacts(folder="artifacts"):
    X = load_npz(f"{folder}/X_orders.npz").tocsr()
    X_csc = X.tocsc()  # fast column slicing (one-time cost)

    product_ids = np.load(f"{folder}/product_ids.npy")
    order_users = np.load(f"{folder}/order_users.npy")
    product_labels = np.load(f"{folder}/product_labels.npy")

    n_orders, n_products = X.shape
    order_sizes = np.diff(X.indptr).astype(float)
    global_counts = np.asarray(X_csc.sum(axis=0)).ravel().astype(float)

    # user -> row indices (vectorised, no Python loop over orders)
    idx = np.argsort(order_users, kind="stable")
    su = order_users[idx]
    uniq, starts = np.unique(su, return_index=True)
    ends = np.append(starts[1:], len(su))
    user_order_rows = {int(u): idx[s:e] for u, s, e in zip(uniq, starts, ends)}

    return SimpleNamespace(
        X=X, X_csc=X_csc,
        n_orders=n_orders, n_products=n_products,
        order_sizes=order_sizes,
        global_counts=global_counts,
        global_rate=global_counts / max(n_orders, 1),
        product_ids=product_ids,
        product_labels=product_labels,
        labels_lower=np.char.lower(product_labels),
        product_to_idx={int(p): i for i, p in enumerate(product_ids)},
        user_order_rows=user_order_rows,
        all_rows=np.arange(n_orders),
    )


def _overlaps(art, pool_rows, cart_indices, is_global):
    if is_global:
        # CSC column slice: touches only the cart's columns, never copies orders
        sub = art.X_csc[:, cart_indices]
    else:
        sub = art.X[pool_rows][:, cart_indices]  # small: one user's orders
    return np.asarray(sub.sum(axis=1)).ravel().astype(float)


def recommend_for_user(
    art, user_id, cart_product_ids,
    top_k=5, min_user_orders=30, n_similar_orders=10,
    min_similarity=0.0, min_new_items=3,
    w_similar=0.60, w_user=0.25, w_global=0.15,
):
    """
    Original model: similar-order rate + user rate + global popularity.
    Returns (DataFrame, info_string).
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if n_similar_orders < 1:
        raise ValueError("n_similar_orders must be at least 1.")

    # 1. Convert basket product IDs to indices
    cart_indices = list(dict.fromkeys(
        art.product_to_idx[int(p)] for p in cart_product_ids
        if int(p) in art.product_to_idx
    ))
    if not cart_indices:
        return pd.DataFrame(), "No recognized products in the cart."
    cart_set = set(cart_indices)

    # 2. Choose this user's orders or all orders
    user_rows = np.asarray(art.user_order_rows.get(int(user_id), []), dtype=int)

    pools = [(art.all_rows, "global_fallback", True)]
    if len(user_rows) > min_user_orders:
        pools.insert(0, (user_rows, "user_history", False))

    # 3. Find the closest historical orders
    selected_rows, search_mode = None, None
    for pool_rows, mode, is_global in pools:
        overlaps = _overlaps(art, pool_rows, cart_indices, is_global)
        sizes = art.order_sizes[pool_rows]
        denom = np.sqrt(len(cart_indices) * sizes)
        sims = np.divide(overlaps, denom,
                         out=np.zeros(len(pool_rows)), where=denom > 0)
        new_items = sizes - overlaps

        valid = np.flatnonzero(
            (overlaps > 0) & (sims > min_similarity) & (new_items >= min_new_items)
        )
        if len(valid) < min(n_similar_orders, 20):
            continue

        take = min(n_similar_orders, len(valid))
        if take < len(valid):
            chosen = valid[np.argpartition(sims[valid], len(valid) - take)[-take:]]
        else:
            chosen = valid
        chosen = chosen[np.argsort(sims[chosen])[::-1]]

        selected_rows = pool_rows[chosen]
        search_mode = mode
        break

    if selected_rows is None:
        return pd.DataFrame(), "No historical orders match the current cart."

    # 4. Score products found in similar orders
    sim_counts = np.asarray(art.X[selected_rows].sum(axis=0)).ravel().astype(float)
    sim_rate = sim_counts / len(selected_rows)

    # 5. This user's purchase history
    if len(user_rows) > 0:
        user_counts = np.asarray(art.X[user_rows].sum(axis=0)).ravel().astype(float)
        user_rate = user_counts / len(user_rows)
    else:
        user_counts = np.zeros(art.n_products)
        user_rate = np.zeros(art.n_products)

    # 6. Combine the three scoring signals
    scores = w_similar * sim_rate + w_user * user_rate - w_global * art.global_rate

    cand = np.flatnonzero(sim_counts > 0)
    cand = cand[~np.isin(cand, list(cart_set))]
    if len(cand) == 0:
        return pd.DataFrame(), "No new products to recommend."

    ranked = cand[np.argsort(scores[cand])[::-1]][:top_k]

    # 7. Return recommendations
    df = pd.DataFrame({
        "product_id": art.product_ids[ranked],
        "product": art.product_labels[ranked],
        "score": np.round(scores[ranked], 4),
        "similar_orders_containing_item": sim_counts[ranked].astype(int),
        "similar_order_rate": np.round(sim_rate[ranked], 4),
        "user_orders_containing_item": user_counts[ranked].astype(int),
        "global_orders_containing_item": art.global_counts[ranked].astype(int),
        "search_mode": search_mode,
        "similar_orders_used": len(selected_rows),
    })
    info = f"Mode: **{search_mode}** · similar orders used: **{len(selected_rows)}**"
    return df, info