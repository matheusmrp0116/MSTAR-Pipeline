from pathlib import Path

import pandas as pd

from src.mstar_pipeline.config import (
    RAW_DIR,
    METADATA_DIR,
)

from src.mstar_pipeline.signal import (
    inspect_binary_layout,
)


manifest_path = (
    METADATA_DIR
    / "manifest_enriched.parquet"
)

df = pd.read_parquet(
    manifest_path
)


records = []


for _, row in df.iterrows():

    path = (
        RAW_DIR
        / row["relative_path"]
    )

    record = {
        "source_id": row[
            "source_id"
        ],
        "relative_path": row[
            "relative_path"
        ],
    }

    try:

        (
            header,
            detected_end,
            candidates,
        ) = inspect_binary_layout(
            path
        )

        exact_one = [
            c
            for c in candidates
            if c.exact_one_plane
        ]

        exact_two = [
            c
            for c in candidates
            if c.exact_two_planes
        ]

        record.update(
            {
                "layout_status":
                    "success",

                "rows":
                    header.rows,

                "columns":
                    header.columns,

                "physical_file_size":
                    path.stat().st_size,

                "phoenix_sig_size":
                    header.signal_size,

                "file_minus_sig_size":
                    (
                        path.stat().st_size
                        - header.signal_size
                        if header.signal_size
                        is not None
                        else None
                    ),

                "exact_one_plane":
                    len(exact_one) > 0,

                "exact_two_planes":
                    len(exact_two) > 0,

                "exact_one_count":
                    len(exact_one),

                "exact_two_count":
                    len(exact_two),
            }
        )

    except Exception as exc:

        record.update(
            {
                "layout_status":
                    "error",

                "error_type":
                    type(exc).__name__,

                "error_message":
                    str(exc),
            }
        )

    records.append(
        record
    )


result = pd.DataFrame(
    records
)


output = (
    METADATA_DIR
    / "binary_layout_diagnostic.parquet"
)

result.to_parquet(
    output,
    index=False
)


print(
    "\n=== STATUS ==="
)

print(
    result[
        "layout_status"
    ].value_counts(
        dropna=False
    )
)


print(
    "\n=== ONE PLANE ==="
)

print(
    result[
        "exact_one_plane"
    ].value_counts(
        dropna=False
    )
)


print(
    "\n=== TWO PLANES ==="
)

print(
    result[
        "exact_two_planes"
    ].value_counts(
        dropna=False
    )
)


print(
    "\n=== FILE SIZE - PhoenixSigSize ==="
)

print(
    result[
        "file_minus_sig_size"
    ].value_counts(
        dropna=False
    ).head(20)
)