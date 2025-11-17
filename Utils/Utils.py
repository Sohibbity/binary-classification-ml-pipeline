from datetime import datetime

import logging

logger = logging.getLogger(__name__)
def log_stage(stage_name: str, chunk_id: int) -> None:
    """Log completion time for a pipeline stage."""
    logger.info(f"Completed {stage_name} for chunk {chunk_id}: {datetime.now().isoformat()}")
