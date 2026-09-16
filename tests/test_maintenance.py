"""
tests/test_maintenance.py - Unit tests for deterministic maintenance status calculations.
"""
import pytest
from tools.maintenance import calculate_service_status


def test_calculate_service_status_approaching():
    """Test APPROACHING status calculation (within 500 km)."""
    res = calculate_service_status(
        current_mileage=9800,
        last_service_mileage=5000,
        interval_km=5000,
        last_service_date="2024-03-15",
        interval_months=6,
        reference_date="2024-08-01"
    )
    assert res["status"] == "APPROACHING"
    assert res["remaining_km"] == 200


def test_calculate_service_status_due():
    """Test DUE status calculation (remaining km == 0)."""
    res = calculate_service_status(
        current_mileage=10000,
        last_service_mileage=5000,
        interval_km=5000,
        last_service_date="2024-03-15",
        interval_months=6,
        reference_date="2024-08-01"
    )
    assert res["status"] == "DUE"
    assert res["remaining_km"] == 0


def test_calculate_service_status_overdue():
    """Test OVERDUE status calculation (current mileage exceeds interval)."""
    res = calculate_service_status(
        current_mileage=10500,
        last_service_mileage=5000,
        interval_km=5000,
        last_service_date="2024-03-15",
        interval_months=6,
        reference_date="2024-08-01"
    )
    assert res["status"] == "OVERDUE"
    assert res["remaining_km"] == -500


def test_calculate_service_status_not_due():
    """Test NOT_DUE status calculation."""
    res = calculate_service_status(
        current_mileage=6000,
        last_service_mileage=5000,
        interval_km=5000,
        last_service_date="2024-03-15",
        interval_months=6,
        reference_date="2024-04-01"
    )
    assert res["status"] == "NOT_DUE"
    assert res["remaining_km"] == 4000
