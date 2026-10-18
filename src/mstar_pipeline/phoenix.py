from pathlib import Path
from dataclasses import dataclass, field


PHOENIX_END_MARKER = "EndofPhoenixHeader"


@dataclass
class PhoenixHeader:
    """
    Parsed structural information from a Phoenix header.
    """

    version: str | None = None

    header_length: int | None = None
    native_header_length: int | None = None
    signal_size: int | None = None

    rows: int | None = None
    columns: int | None = None

    datatype: str | None = None
    signal_type: str | None = None
    architecture: str | None = None

    raw_fields: dict[str, str] = field(
        default_factory=dict
    )

def read_phoenix_text(
    path: str | Path,
    max_header_bytes: int = 256 * 1024
) -> tuple[list[str], int]:
    """
    Read Phoenix ASCII header line by line.

    Returns
    -------
    lines
        Header lines as decoded strings.

    detected_end_offset
        Byte position immediately after
        the header end marker.
    """

    path = Path(path)

    lines = []

    with path.open("rb") as file:

        while file.tell() < max_header_bytes:

            raw_line = file.readline()

            if not raw_line:
                raise ValueError(
                    "Unexpected end of file before "
                    "Phoenix header terminator."
                )

            line = raw_line.decode(
                "ascii",
                errors="replace"
            ).strip()

            lines.append(line)

            if PHOENIX_END_MARKER in line:

                return lines, file.tell()

    raise ValueError(
        f"Phoenix header terminator not found "
        f"within {max_header_bytes} bytes."
    )

def parse_key_value_lines(
    lines: list[str]
) -> dict[str, str]:

    fields = {}

    for line in lines:

        if "=" not in line:
            continue

        key, value = line.split(
            "=",
            1
        )

        key = key.strip()
        value = value.strip()

        if key:
            fields[key] = value

    return fields

def detect_phoenix_version(
    lines: list[str]
) -> str | None:

    for line in lines:

        if line.startswith(
            "[PhoenixHeaderVer"
        ):
            return (
                line
                .strip("[]")
            )

    return None

def safe_int(
    value: str | None
) -> int | None:

    if value is None:
        return None

    try:
        return int(
            value.strip()
        )

    except (
        TypeError,
        ValueError
    ):
        return None

def parse_phoenix_header(
    path: str | Path
) -> tuple[PhoenixHeader, int]:

    lines, detected_end = (
        read_phoenix_text(path)
    )

    fields = (
        parse_key_value_lines(
            lines
        )
    )

    header = PhoenixHeader(

        version=detect_phoenix_version(
            lines
        ),

        header_length=safe_int(
            fields.get(
                "PhoenixHeaderLength"
            )
        ),

        native_header_length=safe_int(
            fields.get(
                "native_header_length"
            )
        ),

        signal_size=safe_int(
            fields.get("PhoenixSigSize")
        ),

        rows=safe_int(
            fields.get(
                "NumberOfRows"
            )
        ),

        columns=safe_int(
            fields.get(
                "NumberOfColumns"
            )
        ),

        datatype=fields.get(
            "Datatype"
        ),

        signal_type=fields.get(
            "SignalType"
        ),

        architecture=fields.get(
            "Architecture"
        ),

        raw_fields=fields,
    )

    return header, detected_end

@dataclass
class HeaderValidation:
    valid: bool
    issues: list[str]


def validate_phoenix_header(
    header: PhoenixHeader,
    detected_end: int,
    file_size: int
) -> HeaderValidation:

    issues = []

    if header.rows is None:
        issues.append(
            "NumberOfRows missing or invalid"
        )

    elif header.rows <= 0:
        issues.append(
            "NumberOfRows must be positive"
        )

    if header.columns is None:
        issues.append(
            "NumberOfColumns missing or invalid"
        )

    elif header.columns <= 0:
        issues.append(
            "NumberOfColumns must be positive"
        )

    if header.header_length is None:
        issues.append(
            "PhoenixHeaderLength missing or invalid"
        )

    elif header.header_length <= 0:
        issues.append(
            "PhoenixHeaderLength must be positive"
        )

    elif header.header_length >= file_size:
        issues.append(
            "PhoenixHeaderLength exceeds "
            "or equals file size"
        )

    if detected_end >= file_size:
        issues.append(
            "Detected Phoenix header end "
            "exceeds or equals file size"
        )

    return HeaderValidation(
        valid=(len(issues) == 0),
        issues=issues
    )

def inspect_phoenix_file(
    path: str | Path
) -> dict:

    path = Path(path)

    header, detected_end = (
        parse_phoenix_header(path)
    )

    validation = (
        validate_phoenix_header(
            header=header,
            detected_end=detected_end,
            file_size=path.stat().st_size
        )
    )

    return {

        "phoenix_header_version":
            header.version,

        "phoenix_header_length":
            header.header_length,

        "phoenix_detected_end":
            detected_end,

        "native_header_length":
            header.native_header_length,

        "number_of_rows":
            header.rows,

        "number_of_columns":
            header.columns,

        "datatype":
            header.datatype,

        "signal_type":
            header.signal_type,

        "architecture":
            header.architecture,

        "header_valid":
            validation.valid,

        "header_issues":
            " | ".join(
                validation.issues
            ),
    }
