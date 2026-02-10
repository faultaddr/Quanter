"""
A-Share Market Analysis System
Unified interface for stock screening, strategy analysis, and signal generation
"""
import os
from .cli_interface import UnifiedCLIInterface

__version__ = "1.0.0"
__author__ = "A-Share Analysis Team"

def run_analysis_system():
    """
    Run the unified A-Share market analysis system
    
    Returns:
        UnifiedCLIInterface: The main system interface
    """
    # Use default tokens
    tushare_token = "744295f7af6adf63074518f919f5ad5054caf8b84d3c07c066f5c42e"
    
    eastmoney_cookie = {
        'qgqp_b_id': 'b7c0c5065c6db033910b1b3175b7c9bb',
        'st_nvi': 'pr7nepf3axSLFdLauyP5y8deb',
        'websitepoptg_api_time': '1770690681021',
        'st_si': '43191381080720',
        'nid18': '0095a8fdc53e2c9dc00f4d602b3c459e',
        'nid18_create_time': '1770690681336',
        'gviem': '6A44mgyL6Tsg59OPlfAXDd677',
        'gviem_create_time': '1770690681337',
        'p_origin': 'https%3A%2F%2Fpassport2.eastmoney.com',
        'mtp': '1',
        'ct': 'wYdhYQ7SFCReRY7yObWFWJwcS2isXO6R8wHwamkysQRCcR9yEiEaMsskY-1tsHOmajDCrGLWHPVacX0DGd_9HoMFpWjxWtVUZEdR8ibclVermnomP1JWdjUpI3BhaRN2ft3jRsDjazoC6F9O5Jzssk-rkmWM3b3LsGJq5RJDxVM',
        'ut': 'FobyicMgeV5FJnFT189SwEfSo-wAjCKxRGfhgXzug4j9BdKmq4gQdtlHffBaUl7Djr5Ju3CTO3tQqVCOs_Vhp9WUQe_9zHJxPmg__J71QWWtiytGWHR6CUXelUQfxok_geZEOJXcc9bQWieI7LUcRQjQFmB-1bwzaZYU3t525uGbFHwr6SZYdP3PBVz04EfQ796KX06LCuYpITwvNu6laJotFHyE5dflMcANoRBf6d8isLvw34K59yZB985bsVHnckUA0HIycKAoU137ZeAYrEX8rjmONDCZy7QGj-BHcAWyIH9OIF98zmSo71GWwWu_X5FP1R2JqWLg9CMTh9wlVBTitMAXMcc5',
        'pi': '9694097255613200%3Bu9694097255613200%3B%E5%A0%82%E5%A0%82%E6%AD%A3%E6%AD%A3%E7%9A%84%E6%9B%B9%E6%93%8D%3BryhxoVjcWC8PTbi0bFrviFAowUa3asGIsa%2F0auHDuAKp6CJ%2BPVN0UwnSDOaEd7utp5uK4oSJImRgmTF0VD7Nm1Zqq9vnKuG5c1wWVRNZxJmnEN416UgEorQVUQJ5tnsTgIcvWxtVIJHhIll%2F9SIWv6E6wIrLFINK3wF12TZX3gkL7%2FxLaYbHaFQ0YON21YMY%2BZKCiilR%3Bp2dLhWNuZSa0SCigDD%2FOLxaCiti2fW5OSY32vbSSck%2BT1BzvA%2FAQHG2jYCxHc8Httaxt1PRsFPhuwvBF873qXa7Y5muaKZZN0jzerURbzjeerxd31x755Is9mu7LD%2BGWpkI3piLVRUUL5xl2ifRVnekqrax4Yg%3D%3D',
        'uidal': '9694097255613200%e5%a0%82%e5%a0%82%e6%ad%a3%e6%ad%a3%e7%9a%84%e6%9b%b9%e6%93%8d',
        'sid': '',
        'vtpst': '|',
        'st_asi': 'delete',
        'wsc_checkuser_ok': '1',
        'fullscreengg': '1',
        'fullscreengg2': '1',
        'st_pvi': '27562121748759',
        'st_sp': '2025-10-30%2011%3A15%3A42',
        'st_inirUrl': 'https%3A%2F%2Fwww.google.com.hk%2F',
        'st_sn': '5',
        'st_psi': '20260210130257951-111000300841-0487608401'
    }
    
    return UnifiedCLIInterface(tushare_token, eastmoney_cookie)

__all__ = ['UnifiedCLIInterface', 'run_analysis_system']