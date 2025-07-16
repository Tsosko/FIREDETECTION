[sys.path.append(os.path.join(os.getcwd(), folder)) for folder in variables.get("dependent_modules_folders").split(",")]
import proactive_helper as ph
import os
import requests
import math
import json
from typing import Tuple, List, Optional


class TaskSelectUsers:
    def __init__(
        self,
        osrm_url: str,
        selection_diameter_km: Optional[float] = None,
        user_profile_selection: str = "driving",
        filter_only_available: bool = True,
        sort_by: str = "distance",
        osrm_timeout_s: int = 5,
        euclidian_filter_km: Optional[float] = None,
    ):
        """
        Initialize the user selection process.
        :param osrm_url: URL of the OSRM service for calculating distances
        :param selection_diameter_km: Maximum diameter (km) to include users
        :param user_profile_selection: Routing profile (driving, walking, cycling)
        :param filter_only_available: Only include users marked available
        :param sort_by: Sort by 'distance' or 'travel_time'
        :param osrm_timeout_s: Timeout for OSRM requests in seconds
        :param euclidian_filter_km: Pre-filter by straight-line distance (km)
        """
        self.osrm_url = osrm_url
        self.selection_diameter_km = selection_diameter_km
        self.user_profile_selection = user_profile_selection
        self.filter_only_available = filter_only_available
        self.sort_by = sort_by
        self.osrm_timeout_s = osrm_timeout_s
        self.euclidian_filter_km = euclidian_filter_km

    def get_all_users(self) -> List[dict]:
        try:
            usersInfo = variables.get("UsersInfo")
            usersInfoFile = usersInfo + "/mock_users.json"
            with open(usersInfoFile, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading mock users: {e}")
            return []

    def haversine_distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        R = 6371.0  # Earth radius in km
        lat1, lon1 = map(math.radians, loc1)
        lat2, lon2 = map(math.radians, loc2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def calculate_distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> Tuple[float, float]:
        """
        Return driving distance (km) and travel time (s) between two coords using OSRM.
        """
        print(f"Calculating distance from {loc1} to {loc2} using OSRM at {self.osrm_url}")
        try:
            url = (
                f"{self.osrm_url}/route/v1/{self.user_profile_selection}/"
                f"{loc1[1]},{loc1[0]};{loc2[1]},{loc2[0]}?overview=full&geometries=geojson"
            )
            resp = requests.get(url, timeout=self.osrm_timeout_s)
            resp.raise_for_status()
            data = resp.json()['routes'][0]
            print(f"OSRM response: {data}")

            distance_km = data['distance'] / 1000.0
            duration_s = data['duration']
            return distance_km, duration_s
        except Exception as e:
            print(f"Error in OSRM request: {e}")
            return float('inf'), float('inf')

    def select_nearest_available_users(
        self,
        alert_location: Tuple[float, float],
        num_users: int = 2
    ) -> List[dict]:
        """
        Select users by combined filters and sorting.
        """
        users = self.get_all_users()
        candidates: List[Tuple[dict, float, float]] = []
        osrm_results_map = {}

        for idx, user in enumerate(users):
            lat = user.get('latitude')
            lon = user.get('longitude')
            if lat is None or lon is None:
                continue
            loc = (lat, lon)
            # Pre-filter by straight-line distance
            if self.euclidian_filter_km is not None:
                if self.haversine_distance(alert_location, loc) > self.euclidian_filter_km:
                    continue
            # Calculate route metrics
            distance_km, duration_s = self.calculate_distance(alert_location, loc)
            # Store OSRM result for this user (by index or user id if available)
            user_key = user.get('id', idx)
            osrm_results_map[user_key] = {'distance_km': distance_km, 'duration_s': duration_s}
            # Filter by diametern
            if self.selection_diameter_km is not None:
                if distance_km > (self.selection_diameter_km / 2):
                    continue
            candidates.append((user, distance_km, duration_s))

        # Sort candidates
        if self.sort_by == 'travel_time':
            sorted_list = sorted(candidates, key=lambda x: x[2])
        else:
            sorted_list = sorted(candidates, key=lambda x: x[1])

        selected: List[dict] = []
        for user, dist, dur in sorted_list:
            if self.filter_only_available and not user.get('available', True):
                continue
            selected.append(user)
            if len(selected) >= num_users:
                break

        return selected, osrm_results_map

if __name__ == '__main__':
    OSRM_URL ='http://89.227.207.199:5002'
    SELECTION_DIAMETER_KM = float(variables.get("selection_diameter_km"))
    USER_PROFILE_SELECTION = variables.get("user_profile_selection")
    FILTER_ONLY_AVAILABLE = bool(variables.get("filter_only_available"))
    SORT_BY = variables.get("sort_by")
    OSRM_TIMEOUT_S = int(variables.get("osrm_timeout_s"))
    EUCLIDIAN_FILTER_KM = float(variables.get("euclidian_filter_km"))
    NUM_USERS_SELECTION = int(variables.get("num_users_selection"))

    selector = TaskSelectUsers(
        osrm_url=OSRM_URL,
        selection_diameter_km=SELECTION_DIAMETER_KM,
        user_profile_selection=USER_PROFILE_SELECTION,
        filter_only_available=FILTER_ONLY_AVAILABLE,
        sort_by=SORT_BY,
        osrm_timeout_s=OSRM_TIMEOUT_S,
        euclidian_filter_km=EUCLIDIAN_FILTER_KM,
    )

    alert_location = (48.79842194845064, 1.9707823090763419)
    selected_users, osrm_results_map = selector.select_nearest_available_users(
        alert_location,
        num_users=NUM_USERS_SELECTION
    )

    resultMap.put("SELECTED_USERS", json.dumps(selected_users))
    resultMap.put("OSRM_RESULTS", json.dumps(osrm_results_map))
