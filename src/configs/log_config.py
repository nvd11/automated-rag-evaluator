import sys
from loguru import logger

def setup_logging(app_env: str = "dev"):
    """
    Configures the global loguru logger with beautiful terminal output.
    """
    logger.remove() # Remove default handler
    
    log_level = "DEBUG" if app_env.lower() == "dev" else "INFO"
    
    # Console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        enqueue=True, # Thread-safe logging
    )
    
    logger.info(f"Logging initialized. Level: {log_level}")
