#!/usr/bin/env python3
"""
Board Embedding Worker

This worker processes board embedding jobs asynchronously.
It runs in a loop, processing one job at a time as requested.

Usage:
    python board_embedding_worker.py

Environment Variables:
    BOARD_EMBEDDING_WORKER_INTERVAL: Sleep interval between job checks (default: 30 seconds)
    BOARD_EMBEDDING_WORKER_MAX_JOBS: Maximum jobs to process before exiting (default: unlimited)
"""
import asyncio
import logging
import os
import sys
import signal
from typing import Optional

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from open_webui.env import SRC_LOG_LEVELS
from open_webui.utils.board_embedding_job import process_pending_board_embedding_job

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("BOARD_EMBEDDING_WORKER", logging.INFO))

# Configuration
WORKER_INTERVAL = int(os.getenv("BOARD_EMBEDDING_WORKER_INTERVAL", "30"))  # seconds
MAX_JOBS = int(os.getenv("BOARD_EMBEDDING_WORKER_MAX_JOBS", "0"))  # 0 = unlimited

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    log.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_requested = True


async def run_worker():
    """
    Main worker loop that processes board embedding jobs.
    """
    log.info("Starting Board Embedding Worker")
    log.info(f"Check interval: {WORKER_INTERVAL} seconds")
    log.info(f"Max jobs: {'unlimited' if MAX_JOBS == 0 else MAX_JOBS}")

    jobs_processed = 0

    try:
        while not shutdown_requested:
            try:
                log.info("🔍 Worker checking for pending jobs...")
                # Process one pending job
                job = await process_pending_board_embedding_job()

                if job:
                    jobs_processed += 1
                    log.info(
                        f"Successfully processed job {job.get('id')} for board {job.get('board_id')}"
                    )
                    log.info(f"Total jobs processed: {jobs_processed}")

                    # Check if we've reached the max jobs limit
                    if MAX_JOBS > 0 and jobs_processed >= MAX_JOBS:
                        log.info(
                            f"Reached maximum jobs limit ({MAX_JOBS}), shutting down..."
                        )
                        break
                else:
                    log.info("⚠️ No pending jobs found to process")

            except Exception as e:
                log.error(f"Error processing job: {e}")
                # Continue processing other jobs even if one fails

            # Wait before checking for more jobs
            if not shutdown_requested:
                await asyncio.sleep(WORKER_INTERVAL)

    except Exception as e:
        log.error(f"Fatal error in worker loop: {e}")
        raise

    log.info(f"Board Embedding Worker stopped. Total jobs processed: {jobs_processed}")


async def main():
    """Main entry point."""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await run_worker()
    except KeyboardInterrupt:
        log.info("Worker interrupted by user")
    except Exception as e:
        log.error(f"Worker failed with error: {e}")
        sys.exit(1)

    log.info("Worker shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
