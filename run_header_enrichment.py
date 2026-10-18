from src.mstar_pipeline.enrich_manifest \
    import run_header_enrichment


if __name__ == "__main__":

    manifest, errors = (
        run_header_enrichment()
    )

    print()
    print(
        manifest.head()
    )