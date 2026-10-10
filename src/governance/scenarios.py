import numpy as np
import pandas as pd

PROTECTED = {"customerID","Churn"}

def inject_missingness(df, fraction, seed):
    out = df.copy(deep=True)
    rng = np.random.default_rng(seed)
    cols = [c for c in out.columns if c not in PROTECTED]
    n_cells = len(out)*len(cols)
    for idx in rng.choice(n_cells, size=int(round(fraction*n_cells)), replace=False):
        r, c = divmod(int(idx), len(cols))
        out.iat[r, out.columns.get_loc(cols[c])] = np.nan
    return out

def inject_duplicates(df, fraction, seed):
    rng = np.random.default_rng(seed)
    n = max(1, int(round(fraction*len(df))))
    dup = df.loc[rng.choice(df.index.to_numpy(), size=n, replace=False)].copy()
    return pd.concat([df.copy(), dup], ignore_index=True)

def inject_invalid_categories(df, fraction, seed):
    out = df.copy(deep=True)
    rng = np.random.default_rng(seed)
    n = max(1, int(round(fraction*len(out))))
    rows = rng.choice(out.index.to_numpy(), size=n, replace=False)
    out.loc[rows, "Contract"] = "INVALID_CONTRACT"
    return out

def inject_negative_tenure(df, fraction, seed):
    out = df.copy(deep=True)
    rng = np.random.default_rng(seed)
    n = max(1, int(round(fraction*len(out))))
    rows = rng.choice(out.index.to_numpy(), size=n, replace=False)
    out.loc[rows, "tenure"] = -rng.integers(1, 25, size=n)
    return out

def build_scenarios(df, seed):
    return {
        "E0_baseline": df.copy(deep=True),
        "E1_missing_05pct": inject_missingness(df,0.05,seed+1),
        "E2_missing_10pct": inject_missingness(df,0.10,seed+2),
        "E3_missing_20pct": inject_missingness(df,0.20,seed+3),
        "E4_duplicates_02pct": inject_duplicates(df,0.02,seed+4),
        "E5_duplicates_05pct": inject_duplicates(df,0.05,seed+5),
        "E6_invalid_contract_02pct": inject_invalid_categories(df,0.02,seed+6),
        "E7_negative_tenure_02pct": inject_negative_tenure(df,0.02,seed+7),
    }
