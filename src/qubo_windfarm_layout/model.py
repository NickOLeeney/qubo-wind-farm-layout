import yaml
import shapely
import numpy as np
from shapely.ops import unary_union
from shapely.geometry import Polygon
from shapely.geometry.multipolygon import MultiPolygon


def get_farm_area(BOUNDARY_FILE):
    with open(BOUNDARY_FILE) as f:
        boundary_data = yaml.safe_load(f)

    region_names = boundary_data['boundaries'].keys()
    polygons = []

    for region_name in region_names:
        region_coords = boundary_data['boundaries'][region_name]
        polygon = Polygon(region_coords)
        polygons.append(polygon)

    # Defining the union of the 5 regions
    farm_area = unary_union(polygons) 
    return polygons, farm_area


def get_mask(farm_area: MultiPolygon, GRID_STEP: int=200):
    """
    GRID_STEP (int): grid step in meters
    """
    # Define the rectangular area including all the polygons
    min_x, min_y, max_x, max_y = farm_area.bounds
    x_coords = np.arange(min_x, max_x + GRID_STEP, GRID_STEP)
    y_coords = np.arange(min_y, max_y + GRID_STEP, GRID_STEP)
    X, Y = np.meshgrid(x_coords, y_coords)

    # Building the mask (i.e. force all the point outside the polygons to have value 0)
    points = shapely.points(X.ravel(), Y.ravel())
    inside = shapely.covers(farm_area, points)
    mask = inside.reshape(X.shape)

    return X, Y, mask

 


