"""
Analytics module for OceanCVBench.

This module provides tools for analyzing and reporting benchmark results.
"""

from .class_balance_report import CBR, print_cbr_summary

__all__ = ['CBR', 'print_cbr_summary']
