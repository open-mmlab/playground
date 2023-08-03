import json
import math

class NearNeighborRemover:
    def __init__(self,distance_threshold):
        self.distance_threshold = distance_threshold

    def calculate_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def remove_near_neighbors(self, points):
        filtered_points = [points[0]]  # Add the first point to the filtered list
        for i in range(1, len(points)):
            # Calculate the distance between the current point and the last added point
            distance = self.calculate_distance(points[i], filtered_points[-1])
            # If the distance is above the threshold, add the current point to the filtered list
            if distance >= self.distance_threshold:
                filtered_points.append(points[i])
        return filtered_points 



