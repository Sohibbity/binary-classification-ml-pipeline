from datetime import datetime

import logging


logger = logging.getLogger(__name__)
def log_stage(stage_name: str, chunk_id: int) -> None:
    """Log completion time for a pipeline stage."""
    logger.info(f"Completed {stage_name} for chunk {chunk_id}: {datetime.now().isoformat()}")

def log_chunk_failure(stage_name: str, chunk_id: int, info_message : str, error_message: str) -> None:
    logger.info(f"{stage_name} for chunk {chunk_id} failed to execute {info_message}"
                f" Error Message: {error_message}")

def log_retry(stage_name: str, chunk_id: int, retry_attempt:int, max_attempts, exception) -> None:
    if retry_attempt == max_attempts:
        logger.info(
            f"{stage_name} for chunk {chunk_id} failed to execute, reached max attempts: {max_attempts}, abandoning processing of this batch")
    else:
        logger.info(f"{stage_name} for chunk {chunk_id} failed to execute, retrying attempt # {retry_attempt}, exceptcion: {exception}")