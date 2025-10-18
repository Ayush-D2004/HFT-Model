"""
Logging Configuration Module
===========================

Controls logging levels and output formatting for the HFT system.
Provides different logging modes for development, backtesting, and production.
"""

import sys
from enum import Enum
from typing import Optional
from loguru import logger


class LogLevel(Enum):
    """Logging levels for different system modes"""
    SILENT = "SILENT"           # Only critical errors
    QUIET = "QUIET"             # Errors and warnings only
    NORMAL = "NORMAL"           # Info, warnings, and errors
    VERBOSE = "VERBOSE"         # Debug, info, warnings, and errors
    TRACE = "TRACE"             # All logging including trace


class LogConfig:
    """Logging configuration manager"""
    
    def __init__(self):
        self.current_level = LogLevel.NORMAL
        self._initialized = False
    
    def setup_logging(self, 
                     level: LogLevel = LogLevel.NORMAL,
                     show_backtrace: bool = False,
                     show_diagnose: bool = False) -> None:
        """
        Configure logging for the application
        
        Args:
            level: Logging level to use
            show_backtrace: Show full backtraces on errors
            show_diagnose: Show diagnostic information
        """
        # Remove default logger
        logger.remove()
        
        # Set up level mapping
        level_mapping = {
            LogLevel.SILENT: "CRITICAL",
            LogLevel.QUIET: "WARNING", 
            LogLevel.NORMAL: "INFO",
            LogLevel.VERBOSE: "DEBUG",
            LogLevel.TRACE: "TRACE"
        }
        
        loguru_level = level_mapping[level]
        
        # Configure format based on level
        if level == LogLevel.SILENT:
            # Minimal format for critical errors only
            format_str = "<red><bold>CRITICAL</bold></red> | {message}"
        elif level == LogLevel.QUIET:
            # Simple format for errors and warnings
            format_str = "<level>{level}</level> | {message}"
        elif level == LogLevel.NORMAL:
            # Standard format for production
            format_str = "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | {message}"
        else:
            # Detailed format for debugging
            format_str = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | {message}"
        
        # Add console handler
        logger.add(
            sys.stderr,
            format=format_str,
            level=loguru_level,
            backtrace=show_backtrace,
            diagnose=show_diagnose,
            colorize=True
        )
        
        self.current_level = level
        
        if level != LogLevel.SILENT and not self._initialized:
            logger.info(f"Logging configured: level={level.value}")
        
        self._initialized = True
    
    def set_backtesting_mode(self) -> None:
        """Configure logging for backtesting - minimal output"""
        self.setup_logging(
            level=LogLevel.QUIET,
            show_backtrace=False,
            show_diagnose=False
        )
    
    def set_development_mode(self) -> None:
        """Configure logging for development - full output"""
        self.setup_logging(
            level=LogLevel.VERBOSE,
            show_backtrace=True,
            show_diagnose=True
        )
    
    def set_production_mode(self) -> None:
        """Configure logging for production - balanced output"""
        self.setup_logging(
            level=LogLevel.NORMAL,
            show_backtrace=False,
            show_diagnose=False
        )
    
    def set_silent_mode(self) -> None:
        """Configure logging for silent operation - critical errors only"""
        self.setup_logging(
            level=LogLevel.SILENT,
            show_backtrace=False,
            show_diagnose=False
        )
    
    def add_file_logging(self, 
                        filepath: str,
                        level: LogLevel = LogLevel.VERBOSE,
                        rotation: str = "10 MB",
                        retention: str = "7 days") -> None:
        """
        Add file logging in addition to console
        
        Args:
            filepath: Path to log file
            level: Logging level for file
            rotation: File rotation policy
            retention: Log retention policy
        """
        level_mapping = {
            LogLevel.SILENT: "CRITICAL",
            LogLevel.QUIET: "WARNING",
            LogLevel.NORMAL: "INFO", 
            LogLevel.VERBOSE: "DEBUG",
            LogLevel.TRACE: "TRACE"
        }
        
        # Detailed format for file logging
        file_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}"
        
        logger.add(
            filepath,
            format=file_format,
            level=level_mapping[level],
            rotation=rotation,
            retention=retention,
            backtrace=True,
            diagnose=True
        )
        
        if self.current_level != LogLevel.SILENT:
            logger.info(f"File logging enabled: {filepath}")
    
    def suppress_module_logging(self, modules: list[str]) -> None:
        """
        Suppress logging from specific modules during backtesting
        
        Args:
            modules: List of module names to suppress
        """
        for module in modules:
            logger.disable(module)
        
        if self.current_level != LogLevel.SILENT:
            logger.info(f"Suppressed logging for modules: {modules}")
    
    def enable_module_logging(self, modules: list[str]) -> None:
        """
        Re-enable logging for specific modules
        
        Args:
            modules: List of module names to enable
        """
        for module in modules:
            logger.enable(module)
        
        if self.current_level != LogLevel.SILENT:
            logger.info(f"Enabled logging for modules: {modules}")


# Global log configuration instance
log_config = LogConfig()


def setup_backtesting_logging():
    """Quick setup for backtesting - minimal logging"""
    log_config.set_backtesting_mode()
    
    # Suppress verbose modules during backtesting
    log_config.suppress_module_logging([
        "src.strategy.avellaneda_stoikov",
        "src.data_ingestion.order_book",
        "src.strategy.quote_manager",
        "src.backtesting.fill_simulator"
    ])


def setup_development_logging():
    """Quick setup for development - full logging"""
    log_config.set_development_mode()


def setup_production_logging():
    """Quick setup for production - balanced logging"""
    log_config.set_production_mode()


def setup_silent_logging():
    """Quick setup for silent operation"""
    log_config.set_silent_mode()


def get_logger(name: str):
    """
    Get a logger instance for a module
    
    Args:
        name: Module name
        
    Returns:
        Logger instance
    """
    if not log_config._initialized:
        # Initialize with default settings if not configured
        log_config.setup_logging()
    
    return logger.bind(name=name)


# Initialize default logging on import
if not log_config._initialized:
    log_config.setup_logging(LogLevel.NORMAL)