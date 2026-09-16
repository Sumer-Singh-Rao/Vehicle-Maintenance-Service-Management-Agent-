"""
tests/test_tools.py - Unit tests for LangChain tool wrappers in agent/tools.py.
"""
import pytest
from agent.tools import (
    ALL_TOOLS,
    get_vehicle_info,
    get_service_history,
    calculate_service_status,
    geocode_location,
    search_service_centers,
    check_availability,
    book_appointment,
    cancel_appointment,
    send_notification,
    add_vehicle,
    delete_vehicle,
    update_vehicle_mileage,
    update_vehicle_service_details,
    update_service_after_completion,
    update_service_cost
)


def test_tool_count_and_names():
    """Verify all 15 LangChain tools are registered with proper names."""
    assert len(ALL_TOOLS) == 15
    tool_names = {t.name for t in ALL_TOOLS}
    expected = {
        "get_vehicle_info",
        "get_service_history",
        "calculate_service_status",
        "geocode_location",
        "search_service_centers",
        "check_availability",
        "book_appointment",
        "cancel_appointment",
        "send_notification",
        "add_vehicle",
        "delete_vehicle",
        "update_vehicle_mileage",
        "update_vehicle_service_details",
        "update_service_after_completion",
        "update_service_cost"
    }
    assert tool_names == expected


def test_get_vehicle_info_tool():
    """Test get_vehicle_info LangChain tool execution."""
    res = get_vehicle_info.invoke({"user_id": 1, "vehicle_name": "Tata Nexon"})
    assert isinstance(res, dict)
    assert "vehicle" in res
    assert "user_vehicles" in res


def test_calculate_service_status_tool():
    """Test calculate_service_status LangChain tool execution."""
    res = calculate_service_status.invoke({
        "current_mileage": 9800,
        "last_service_mileage": 5000,
        "interval_km": 5000,
        "last_service_date": "2024-03-15",
        "interval_months": 6,
        "reference_date": "2024-08-01"
    })
    assert res["status"] == "APPROACHING"
    assert res["remaining_km"] == 200


def test_geocode_location_tool():
    """Test geocode_location LangChain tool execution."""
    res = geocode_location.invoke({"address_or_city": "current location"})
    assert res["status"] == "SUCCESS"
    assert "latitude" in res
    assert "longitude" in res


def test_check_availability_tool():
    """Test check_availability LangChain tool execution."""
    res = check_availability.invoke({"center_id": "osm_101", "date": "2026-10-15"})
    assert isinstance(res, dict)
    assert "available_slots" in res
    assert isinstance(res["available_slots"], list)


def test_notification_tool():
    """Test send_notification LangChain tool execution."""
    res = send_notification.invoke({
        "user_id": 1,
        "appointment_id": None,
        "message": "Test notification message",
        "channel": "EMAIL"
    })
    assert isinstance(res, dict)
    assert "status" in res
