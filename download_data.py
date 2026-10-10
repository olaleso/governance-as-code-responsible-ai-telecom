from pathlib import Path
import hashlib
import requests

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
OUT = Path("data/raw/Telco-Customer-Churn.csv")

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading IBM Telco Customer Churn dataset to {OUT} ...")
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    OUT.write_bytes(r.content)
    print(f"Saved {OUT} ({OUT.stat().st_size:,} bytes)")
    print("SHA-256:", sha256(OUT))

if __name__ == "__main__":
    main()
