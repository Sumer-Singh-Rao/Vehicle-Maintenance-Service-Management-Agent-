"""
agent/tools.py - LangChain Tool Wrappers.
Exposes existing business functions as LangChain Structured Tools without altering underlying tool logic.
"""
from typing import Dict, List, Any, Optional
from langchain_core.tools import tool

import database
from tools.maintenance import calculate_service_status as _calculate_service_status
from tools.location import geocode_location as _geocode_location, search_service_centers as _search_service_centers
from tools.booking import check_availability as _check_availability, book_appointment as _book_appointment, cancel_appointment as _cancel_appointment
from tools.notification import send_notification as _send_notification


@tool
def get_vehicle_info(user_id: int = 1, vehicle_name: Optional[str] = None) -> Dict[str, Any]:
    """Get vehicle details by user_id and optional name, model, or label."""
    v = database.get_vehicle_info(user_id=user_id, vehicle_name=vehicle_name)
    all_v = database.get_user_vehicles(user_id=user_id)
    return {
        "vehicle": v,
        "user_vehicles": [{"id": x["id"], "name": f"{x['make']} {x['model']}"} for x in all_v]
    }


@tool
def get_service_history(vehicle_id: int = 1) -> Dict[str, Any]:
    """Get past maintenance and repair service history records for a vehicle."""
    return {"service_history": database.get_service_history(vehicle_id=int(vehicle_id))}


@tool
def calculate_service_status(
    current_mileage: int,
    last_service_mileage: int,
    interval_km: int,
    last_service_date: str,
    interval_months: int,
    reference_date: Optional[str] = None
) -> Dict[str, Any]:
    """Calculate maintenance service due status (OVERDUE, DUE, APPROACHING, NOT_DUE) deterministically."""
    return _calculate_service_status(
        current_mileage=int(current_mileage),
        last_service_mileage=int(last_service_mileage),
        interval_km=int(interval_km),
        last_service_date=str(last_service_date),
        interval_months=int(interval_months),
        reference_date=reference_date
    )


@tool
def geocode_location(address_or_city: str) -> Dict[str, Any]:
    """Convert address or city name to GPS coordinates or auto-detect current location."""
    return _geocode_location(address_or_city=str(address_or_city))


@tool
def search_service_centers(
    latitude: float,
    longitude: float,
    radius_km: int = 10,
    brand: Optional[str] = None
) -> Dict[str, Any]:
    """Find nearby authorized service centers by GPS coordinates and vehicle brand."""
    centers = _search_service_centers(
        latitude=float(latitude),
        longitude=float(longitude),
        radius_km=int(radius_km),
        brand=brand
    )
    return {"service_centers": centers}


@tool
def check_availability(center_id: str, date: str) -> Dict[str, Any]:
    """Check available booking time slots for a service center and target date."""
    return _check_availability(center_id=str(center_id), date=str(date))


@tool
def book_appointment(
    vehicle_id: int,
    center_id: str,
    date: str,
    time: str,
    service_type: str = "Periodic Maintenance Service"
) -> Dict[str, Any]:
    """Book a service appointment and generate booking reference code."""
    return _book_appointment(
        vehicle_id=int(vehicle_id),
        center_id=str(center_id),
        date=str(date),
        time=str(time),
        service_type=str(service_type)
    )


@tool
def cancel_appointment(booking_reference: str) -> Dict[str, Any]:
    """Cancel an appointment using its booking reference code."""
    return _cancel_appointment(booking_reference=str(booking_reference))


@tool
def send_notification(
    user_id: int = 1,
    appointment_id: Optional[int] = None,
    message: str = "",
    channel: str = "EMAIL"
) -> Dict[str, Any]:
    """Send and record an appointment confirmation notification via SMS/Email."""
    appt_id = int(appointment_id) if appointment_id else None
    return _send_notification(
        user_id=int(user_id),
        appointment_id=appt_id,
        message=str(message),
        channel=str(channel)
    )


@tool
def add_vehicle(
    user_id: int = 1,
    make: str = "Tata",
    model: str = "Harrier",
    registration_number: Optional[str] = None,
    current_mileage: int = 0,
    label: Optional[str] = None
) -> Dict[str, Any]:
    """Add a new vehicle to the user's garage."""
    return database.add_vehicle(
        user_id=int(user_id),
        make=str(make),
        model=str(model),
        registration_number=registration_number,
        current_mileage=int(current_mileage),
        label=label
    )


@tool
def delete_vehicle(vehicle_identifier: str, user_id: int = 1) -> Dict[str, Any]:
    """Remove a vehicle from user's garage by name, label, or registration number."""
    return database.delete_vehicle(vehicle_identifier=str(vehicle_identifier), user_id=int(user_id))


@tool
def update_vehicle_mileage(vehicle_id: int, new_mileage: int) -> Dict[str, Any]:
    """Update current odometer mileage for a vehicle."""
    return database.update_vehicle_mileage(vehicle_id=int(vehicle_id), new_mileage=int(new_mileage))


@tool
def update_vehicle_service_details(
    vehicle_id: int,
    last_service_date: str,
    last_service_mileage: int,
    current_mileage: Optional[int] = None
) -> Dict[str, Any]:
    """Update last service date and last service mileage for a vehicle."""
    return database.update_vehicle_service_details(
        vehicle_id=int(vehicle_id),
        last_service_date=str(last_service_date),
        last_service_mileage=int(last_service_mileage),
        current_mileage=current_mileage
    )


@tool
def update_service_after_completion(
    vehicle_id: int,
    service_date: str,
    next_service_interval_km: Optional[int] = None,
    service_mileage: Optional[int] = None,
    service_type: str = "Periodic Maintenance Service"
) -> Dict[str, Any]:
    """Update vehicle mileage, maintenance schedule, and service history after completing service."""
    interval = int(next_service_interval_km) if next_service_interval_km else None
    return database.update_service_after_completion(
        vehicle_id=int(vehicle_id),
        service_date=str(service_date),
        next_service_interval_km=interval,
        service_mileage=service_mileage,
        service_type=str(service_type)
    )


@tool
def update_service_cost(
    vehicle_id: int,
    cost: float,
    service_history_id: Optional[int] = None
) -> Dict[str, Any]:
    """Record or update the expense/cost of a completed service in the database."""
    h_id = int(service_history_id) if service_history_id else None
    return database.update_service_cost(
        vehicle_id=int(vehicle_id),
        cost=float(cost),
        service_history_id=h_id
    )


# All 15 LangChain tools exported for LangGraph binding
ALL_TOOLS = [
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
    update_service_cost,
]

TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
