from pathlib import Path

from mstar_pipeline.phoenix import (
    parse_phoenix_header,
    validate_phoenix_header
)


def test_phoenix_sample():

    path = Path(
        "data/raw/mstar/MSTAR-PublicTargetChips-T72-BMP2-BTR70-SLICY/MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY/TARGETS/TEST/15_DEG/BMP2/SN_9563/HB03333.000"
    )

    header, detected_end = (
        parse_phoenix_header(path)
    )

    assert detected_end > 0

    assert header.rows is not None
    assert header.rows > 0

    assert header.columns is not None
    assert header.columns > 0

    assert header.header_length \
        is not None