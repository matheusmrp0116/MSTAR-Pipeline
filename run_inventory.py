from src.mstar_pipeline.inventory \
    import run_inventory


if __name__ == "__main__":

    inventory, errors = (
        run_inventory()
    )

    print()
    print(
        inventory.head()
    )