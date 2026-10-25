"""Deterministic sample sales database (customers, products, orders, order_items) and schema extraction."""
from __future__ import annotations

import hashlib
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

CATEGORIES = {
    "Electronics": [("Wireless Mouse", 24.99, 11.0), ("Mechanical Keyboard", 79.0, 41.0), ("USB-C Hub", 39.5, 18.0),
                    ("Noise-Cancelling Headphones", 149.0, 78.0), ("Webcam 1080p", 59.9, 27.0), ("Portable SSD 1TB", 109.0, 62.0)],
    "Furniture": [("Standing Desk", 389.0, 215.0), ("Ergonomic Chair", 259.0, 140.0), ("Bookshelf", 119.0, 58.0),
                  ("Desk Lamp", 34.0, 14.5), ("Filing Cabinet", 149.0, 80.0), ("Monitor Stand", 45.0, 19.0)],
    "Stationery": [("Notebook A5", 4.5, 1.6), ("Gel Pen Pack", 6.25, 2.1), ("Sticky Notes", 3.2, 1.0),
                   ("Desk Organizer", 18.0, 7.5), ("Whiteboard Markers", 8.9, 3.4), ("Printer Paper Ream", 7.5, 4.2)],
    "Kitchen": [("Coffee Grinder", 42.0, 21.0), ("Electric Kettle", 36.5, 17.0), ("Lunch Box", 14.9, 5.9),
                ("Water Bottle", 19.0, 7.2), ("Blender", 64.0, 33.0), ("Toaster", 29.0, 14.0)],
}
NEVER_ORDERED = {"Filing Cabinet", "Whiteboard Markers", "Toaster"}
CITY_COUNTS = [("Pune", "India", 12), ("Mumbai", "India", 9), ("Berlin", "Germany", 7), ("Austin", "USA", 6),
               ("Nagpur", "India", 5), ("Munich", "Germany", 5), ("Lyon", "France", 5), ("Toronto", "Canada", 4),
               ("Denver", "USA", 4), ("Leeds", "UK", 3)]
FIRST = ["Aarav", "Meera", "Liam", "Sofia", "Noah", "Isha", "Lucas", "Emma", "Rohan", "Zara",
         "Mateo", "Anaya", "Oliver", "Kavya", "Elena", "Arjun", "Chloe", "Dev", "Hanna", "Ishaan"]
LAST = ["Sharma", "Patel", "Müller", "Rossi", "Smith", "Kulkarni", "Dubois", "Brown", "Joshi", "Khan",
        "Weber", "Nair", "Taylor", "Deshmukh", "Martin", "Iyer", "Wilson", "Gupta", "Fischer", "Reddy"]
SEGMENTS = ["Consumer", "Corporate", "Home Office"]

SCHEMA_SQL = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    city TEXT NOT NULL,
    country TEXT NOT NULL,
    segment TEXT NOT NULL,          -- Consumer, Corporate or Home Office
    signup_date TEXT NOT NULL       -- YYYY-MM-DD
);
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_price REAL NOT NULL,
    cost REAL NOT NULL
);
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date TEXT NOT NULL,       -- YYYY-MM-DD
    status TEXT NOT NULL            -- Delivered, Shipped, Cancelled or Returned
);
CREATE TABLE order_items (
    item_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(order_id),
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    quantity INTEGER NOT NULL,
    discount REAL NOT NULL          -- fraction, e.g. 0.1 = 10% off
);
"""


def build_db(path: str | Path, seed: int = 42) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    rng = random.Random(seed)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA_SQL)

    customers = []
    cid = 0
    for city, country, n in CITY_COUNTS:
        for _ in range(n):
            cid += 1
            name = f"{FIRST[(cid - 1) % 20]} {LAST[(7 * cid + (cid - 1) // 20) % 20]}"
            signup = date(2023, 1, 1) + timedelta(days=rng.randrange(0, 365))
            customers.append((cid, name, city, country, rng.choice(SEGMENTS), signup.isoformat()))
    assert len({c[1] for c in customers}) == len(customers), "customer names must be unique"
    conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?)", customers)

    products, pid = [], 0
    for cat, items in CATEGORIES.items():
        for name, price, cost in items:
            pid += 1
            products.append((pid, name, cat, price, cost))
    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?)", products)
    orderable = [p[0] for p in products if p[1] not in NEVER_ORDERED]

    active = rng.sample([c[0] for c in customers], 50)  # 10 customers never order
    oid = iid = 0
    start, span = date(2024, 1, 1), (date(2025, 9, 30) - date(2024, 1, 1)).days
    for _ in range(300):
        oid += 1
        status = rng.choices(["Delivered", "Shipped", "Cancelled", "Returned"], [70, 10, 12, 8])[0]
        conn.execute("INSERT INTO orders VALUES (?,?,?,?)",
                     (oid, rng.choice(active), (start + timedelta(days=rng.randrange(span))).isoformat(), status))
        for p in rng.sample(orderable, rng.randint(1, 4)):
            iid += 1
            conn.execute("INSERT INTO order_items VALUES (?,?,?,?,?)",
                         (iid, oid, p, rng.randint(1, 5), rng.choice([0, 0, 0, 0.05, 0.1, 0.15])))
    conn.commit()
    counts = table_counts(conn)
    conn.close()
    return counts


def table_counts(conn: sqlite3.Connection) -> dict:
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")}


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def extract_schema(path: str | Path, sample_rows: int = 2) -> str:
    """CREATE statements, foreign keys and a few sample rows, as the context given to the LLM."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    parts = []
    for name, sql in conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY rowid"):
        fks = [f"{name}.{r[3]} -> {r[2]}.{r[4]}" for r in conn.execute(f"PRAGMA foreign_key_list({name})")]
        rows = conn.execute(f"SELECT * FROM {name} LIMIT {sample_rows}").fetchall()
        parts.append(sql + (f"\n-- foreign keys: {', '.join(fks)}" if fks else "") + f"\n-- sample rows: {rows}")
    conn.close()
    return "\n\n".join(parts)
