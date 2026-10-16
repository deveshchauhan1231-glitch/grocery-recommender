import numpy as np
import pandas as pd
import streamlit as st
from recommender import load_artifacts, recommend_for_user

st.set_page_config(page_title="Basket Recommender", layout="wide")


@st.cache_resource(show_spinner="Loading data (one-time)...")
def get_art():
    return load_artifacts("artifacts")


@st.cache_data(show_spinner=False, max_entries=500)
def cached_recommend(user_id, cart, top_k, min_user_orders,
                     n_similar, min_new_items, ws, wu, wg):
    return recommend_for_user(
        get_art(), user_id, list(cart),
        top_k=top_k, min_user_orders=min_user_orders,
        n_similar_orders=n_similar, min_new_items=min_new_items,
        w_similar=ws, w_user=wu, w_global=wg,
    )


art = get_art()
if "cart" not in st.session_state:
    st.session_state.cart = []

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    user_id = st.number_input("User ID", min_value=1, value=1, step=1)
    top_k = st.slider("Top K", 1, 20, 5)
    min_user_orders = st.slider("Min user orders (personalize above)", 0, 100, 30)
    n_similar = st.slider("Similar orders to use", 1, 50, 10)
    min_new_items = st.slider("Min new items per similar order", 0, 10, 3)
    with st.expander("Score weights"):
        ws = st.slider("Similar-order rate", 0.0, 1.0, 0.60, 0.05)
        wu = st.slider("User rate", 0.0, 1.0, 0.25, 0.05)
        wg = st.slider("Global popularity", 0.0, 1.0, 0.15, 0.05)

    n_user_orders = len(art.user_order_rows.get(int(user_id), []))
    st.caption(f"User {int(user_id)} has **{n_user_orders}** historical orders → "
               + ("personalized" if n_user_orders > min_user_orders else "global fallback"))

# ---------------- Main ----------------
st.title("🛒 Basket Recommendations")
left, right = st.columns([1, 1.4])

with left:
    st.subheader("Build your cart")
    q = st.text_input("Search products", placeholder="e.g. banana, milk, eggs")
    if q.strip():
        hits = np.flatnonzero(np.char.find(art.labels_lower, q.strip().lower()) >= 0)
        hits = hits[np.argsort(art.global_counts[hits])[::-1]][:50]  # most popular first
        if len(hits):
            choice = st.selectbox("Matches", hits,
                                  format_func=lambda j: art.product_labels[j])
            if st.button("➕ Add to cart"):
                pid = int(art.product_ids[choice])
                if pid not in st.session_state.cart:
                    st.session_state.cart.append(pid)
                st.rerun()
        else:
            st.info("No matches.")

    st.markdown("**Cart**")
    if not st.session_state.cart:
        st.caption("Empty")
    for pid in list(st.session_state.cart):
        c1, c2 = st.columns([5, 1])
        c1.write(art.product_labels[art.product_to_idx[pid]])
        if c2.button("✖", key=f"rm_{pid}"):
            st.session_state.cart.remove(pid)
            st.rerun()
    if st.session_state.cart and st.button("Clear cart"):
        st.session_state.cart = []
        st.rerun()

with right:
    st.subheader("Recommended for you")
    if st.session_state.cart:
        recs, info = cached_recommend(
            int(user_id), tuple(st.session_state.cart), top_k,
            min_user_orders, n_similar, min_new_items, ws, wu, wg,
        )
        if recs.empty:
            st.warning(info)
        else:
            st.caption(info)
            st.dataframe(recs, use_container_width=True, hide_index=True)
    else:
        st.caption("Add items to your cart to see recommendations.")