import alembic.config

if __name__ == "__main__":
    alembic.config.main(["upgrade", "head"])
