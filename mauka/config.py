

from dataclasses import dataclass

CITIES = {
    "New Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Jaipur": (26.9124, 75.7873),
    "Kota": (25.2138, 75.8648),
    "Lucknow": (26.8467, 80.9462),
    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Pune": (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
    "Surat": (21.1702, 72.8311),
    "Jaipur": (26.9124, 75.7873),
    "Udaipur": (24.5854, 73.7125),
}


@dataclass(frozen=True)
class Thresholds:
    
    max_us_aqi: float = 100.0           # air quality: moderate or better
    max_apparent_temp_c: float = 35.0   # "feels like" temperature, upper limit
    min_apparent_temp_c: float = 8.0    # and a lower limit for cold mornings
    max_precip_mm: float = 0.5          # basically no rain
    first_hour: int = 5                 # we only care about waking hours
    last_hour: int = 21


THRESHOLDS = Thresholds()
