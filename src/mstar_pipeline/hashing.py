from pathlib import Path
import hashlib


def sha256_file(
    path: str | Path,
    chunk_size: int = 1024 * 1024
) -> str:
    """
    Compute SHA-256 checksum without loading
    the entire file into memory.
    """

    path = Path(path)

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:

            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()