"""CLI entry point: python -m deltaloop.storage.init_db"""
import asyncio

from loguru import logger

from deltaloop.storage.repository import init_db


async def main() -> None:
    await init_db()
    logger.info("Database initialized successfully.")


if __name__ == "__main__":
    asyncio.run(main())
