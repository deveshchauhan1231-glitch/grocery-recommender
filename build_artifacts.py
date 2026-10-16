import os
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, save_npz

OUT = "artifacts"
os.makedirs(OUT, exist_ok=True)

path = "C:\\Users\\Devesh\\Desktop\\grocery"

op = pd.read_csv(f"{path}/order_products__prior.csv",
                 usecols=["order_id", "product_id"])
orders = pd.read_csv(f"{path}/orders.csv", usecols=["order_id", "user_id"])
products = pd.read_csv(f"{path}/products.csv",
                       usecols=["product_id", "product_name"])

# Index orders / products (same idea as your notebook, no big merge needed)
order_codes, order_ids = pd.factorize(op["order_id"])
product_codes, product_ids = pd.factorize(op["product_id"])

X_orders = csr_matrix(
    (np.ones(len(op), dtype=np.uint8), (order_codes, product_codes)),
    shape=(len(order_ids), len(product_ids)),
)

# User for every matrix row
order_users = (
    orders.set_index("order_id")["user_id"].reindex(order_ids).to_numpy()
)

# Product labels aligned with matrix columns
name_map = products.drop_duplicates("product_id").set_index("product_id")["product_name"]
product_labels = np.array(
    [str(name_map.get(pid, f"Product {pid}")) for pid in product_ids], dtype=str
)

save_npz(f"{OUT}/X_orders.npz", X_orders)
np.save(f"{OUT}/product_ids.npy", np.asarray(product_ids))
np.save(f"{OUT}/order_users.npy", order_users)
np.save(f"{OUT}/product_labels.npy", product_labels)

print("Saved:", X_orders.shape)