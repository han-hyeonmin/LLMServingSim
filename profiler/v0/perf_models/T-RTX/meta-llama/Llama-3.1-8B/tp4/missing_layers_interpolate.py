import shutil
import pandas as pd

df = pd.read_csv("layers.csv")

all_inputs = sorted(df["input"].unique())
all_layers = df["layer_name"].unique()

missing_rows = []
for layer in all_layers:
    layer_df = df[df["layer_name"] == layer]
    missing_inputs = sorted(set(all_inputs) - set(layer_df["input"].values))

    if not missing_inputs:
        continue

    print(f"[{layer}] Missing input_len values: {missing_inputs}")

    # Linear interpolation over the full input range using existing rows for this layer
    interpolated_latency = (
        layer_df.set_index("input")[["latency(ns)"]]
        .reindex(all_inputs)
        .interpolate(method="index")
    )

    # Reconstruct missing rows using metadata from an existing row of this layer
    template = layer_df.iloc[0]
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
print(
    f"Interpolated {len(missing_rows)} missing rows across all layers into layers.csv"
)
