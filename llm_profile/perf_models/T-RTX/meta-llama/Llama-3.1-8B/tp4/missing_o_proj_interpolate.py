import shutil
import pandas as pd

df = pd.read_csv("layers.csv")

# Identify input_len values where o_proj is missing
o_proj_df = df[df["layer_name"] == "o_proj"]
all_inputs = sorted(df["input"].unique())
missing_inputs = sorted(set(all_inputs) - set(o_proj_df["input"].values))
print(f"Missing input_len values for o_proj: {missing_inputs}")

# Linear interpolation over the full input range using existing o_proj rows
interpolated_latency = (
    o_proj_df.set_index("input")[["latency(ns)"]]
    .reindex(all_inputs)
    .interpolate(method="index")
)

# Reconstruct missing rows using metadata from an existing o_proj row
template = o_proj_df.iloc[0]
missing_rows = []
for inp in missing_inputs:
    row = template.copy()
    row["input"] = inp
    row["latency(ns)"] = int(interpolated_latency.loc[inp, "latency(ns)"])
    missing_rows.append(row)

# Backup original and insert interpolated rows in correct position
shutil.copy("layers.csv", "layers_original.csv")
pd.concat([df, pd.DataFrame(missing_rows)]).sort_values(
    "input", kind="stable"
).reset_index(drop=True).to_csv("layers.csv", index=False)
print(f"Interpolated {len(missing_rows)} missing o_proj rows into layers.csv")
