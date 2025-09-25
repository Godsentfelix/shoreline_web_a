#utils.py
import os
import zipfile
import numpy as np
import cv2
import torch
from torchvision import transforms
import geopandas as gpd
import shutil
from shapely import ops
from shapely.geometry import LineString, Point
from shapely.geometry import MultiLineString
from shapely.ops import nearest_points
import leafmap.foliumap as leafmap
from werkzeug.utils import secure_filename
from osgeo import gdal, osr
import networkx as nx
from model import build_unet
from skimage.morphology import skeletonize
import traceback
import uuid
import numpy as np
from datetime import datetime

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path, img_size=(512, 512)):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, img_size)
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, original_image


def get_pixel_size(image_path):
    dataset = gdal.Open(image_path)
    if dataset is None:
        raise FileNotFoundError(f"Unable to open image: {image_path}")
    geotransform = dataset.GetGeoTransform()
    return geotransform[1], abs(geotransform[5])


def pixel_to_world(x, y, image_path):
    dataset = gdal.Open(image_path)
    if not dataset:
        raise FileNotFoundError(f"Could not open image: {image_path}")
    geotransform = dataset.GetGeoTransform()
    if geotransform[1] == 0 or geotransform[5] == 0:
        raise ValueError("Image does not appear to be georeferenced.")
    world_x = geotransform[0] + x * geotransform[1] + y * geotransform[2]
    world_y = geotransform[3] + x * geotransform[4] + y * geotransform[5]
    return world_x, world_y


def get_image_crs(image_path):
    dataset = gdal.Open(image_path)
    if not dataset:
        raise FileNotFoundError(f"Could not open image: {image_path}")
    proj = dataset.GetProjection()
    if not proj:
        raise ValueError("Image is not georeferenced.")
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)
    return srs.ExportToWkt()


def extract_centerline(binary_mask):
    binary_mask = (binary_mask > 0).astype(np.uint8)
    skeleton = skeletonize(binary_mask).astype(np.uint8) * 255
    if skeleton.ndim == 3:
        skeleton = skeleton[..., 0]
    skeleton = cv2.convertScaleAbs(skeleton)
    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("No valid centerline detected.")
    longest_contour = max(contours, key=cv2.contourArea)
    centerline_points = [(point[0][0], point[0][1]) for point in longest_contour]
    return skeleton, centerline_points


def calculate_centerline_length(centerline_points, image_path):
    if len(centerline_points) < 2:
        return 0
    pixel_width, pixel_height = get_pixel_size(image_path)
    world_coordinates = [pixel_to_world(x, y, image_path) for x, y in centerline_points]
    G = nx.Graph()
    for i in range(len(world_coordinates) - 1):
        x1, y1 = world_coordinates[i]
        x2, y2 = world_coordinates[i + 1]
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        G.add_edge(i, i + 1, weight=distance)
    try:
        path = nx.shortest_path(G, source=0, target=len(world_coordinates) - 1, weight="weight")
        total_length = sum(G[i][j]['weight'] for i, j in zip(path[:-1], path[1:]))
    except nx.NetworkXNoPath:
        return 0
    return total_length


def save_shapefile(world_points, output_shapefile, image_crs_wkt):
    line = LineString(world_points)
    gdf = gpd.GeoDataFrame(geometry=[line], crs=image_crs_wkt)
    os.makedirs(os.path.dirname(output_shapefile), exist_ok=True)
    gdf.to_file(output_shapefile, driver="ESRI Shapefile")
    return output_shapefile


def convert_shapefile_to_geojson(shapefile_path):
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs is None:
        raise ValueError("Shapefile has no CRS defined.")
    gdf_4326 = gdf.to_crs(epsg=4326)
    geojson_path = shapefile_path.replace(".shp", ".geojson")
    gdf_4326.to_file(geojson_path, driver="GeoJSON")
    return geojson_path


def zip_shapefile(shapefile_path):
    zip_filename = shapefile_path.replace(".shp", ".zip")
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for ext in [".shp", ".shx", ".dbf", ".prj"]:
            file_path = shapefile_path.replace(".shp", ext)
            if os.path.exists(file_path):
                zipf.write(file_path, os.path.basename(file_path))
    return zip_filename

def sample_line_by_spacing(line, spacing=20):
    from shapely.geometry import LineString, MultiLineString
    from shapely.ops import linemerge

    if isinstance(line, MultiLineString):
        print("🔁 Merging MultiLineString...")
        merged = linemerge(line)
        if isinstance(merged, MultiLineString):
            line = max(merged.geoms, key=lambda g: g.length)
        else:
            line = merged
    elif not isinstance(line, LineString):
        raise ValueError("🚫 Input must be a LineString or MultiLineString.")

    length = line.length

    if length == 0:
        print("⚠️ Cannot sample zero-length line.")
        return []

    distances = np.arange(0, length, spacing).tolist()
    if not np.isclose(distances[-1], length):
        distances.append(length)  # ensure the last point is included

    points = [line.interpolate(d) for d in distances]
    
    # 📐 DEBUG: Print distance between each pair of sampled points
    for i in range(len(points) - 1):
        d = points[i].distance(points[i + 1])

    return points


def compute_normal_transect(point, line, length=90000):
    from shapely.geometry import LineString, MultiLineString
    from shapely.ops import linemerge

    # If input is a MultiLineString, merge into LineString or MultiLineString
    if isinstance(line, MultiLineString):
        merged = linemerge(line)
    else:
        merged = line

    # If merged result is still a MultiLineString, find the closest LineString part
    if isinstance(merged, MultiLineString):
        min_dist = float("inf")
        closest_line = None
        for linestring in merged.geoms:
            dist = linestring.distance(point)
            if dist < min_dist:
                min_dist = dist
                closest_line = linestring
        line = closest_line
    elif isinstance(merged, LineString):
        line = merged
    else:
        raise ValueError("🚫 Merged geometry is not a LineString or MultiLineString.")

    # Now work with a guaranteed LineString
    coords = list(line.coords)
    min_dist = float("inf")
    closest_seg = None
    for i in range(len(coords) - 1):
        seg = LineString([coords[i], coords[i + 1]])
        dist = seg.distance(point)
        if dist < min_dist:
            min_dist = dist
            closest_seg = seg

    if closest_seg is None:
        return None

    dx = closest_seg.coords[1][0] - closest_seg.coords[0][0]
    dy = closest_seg.coords[1][1] - closest_seg.coords[0][1]
    length_seg = (dx**2 + dy**2) ** 0.5
    if length_seg == 0:
        return None

    # Unit perpendicular vector
    perp_dx = -dy / length_seg
    perp_dy = dx / length_seg

    # Create points in both directions along the perpendicular
    x1 = point.x + perp_dx * length / 2
    y1 = point.y + perp_dy * length / 2
    x2 = point.x - perp_dx * length / 2
    y2 = point.y - perp_dy * length / 2

    transect = LineString([(x1, y1), (x2, y2)])
    return transect

def normalize_line(line, name="line"):
    if line.geom_type == "MultiLineString":
        print(f"🔁 Merging {name}")
        merged = ops.linemerge(line)
        if isinstance(merged, MultiLineString):
            print(f"⚠️ Still MultiLineString after merge — selecting longest segment from {name}")
            return max(merged.geoms, key=lambda g: g.length)
        return merged
    elif line.geom_type != "LineString":
        raise ValueError(f"🚫 Unsupported geometry type in {name}: {line.geom_type}")
    return line

def extract_closest_point(geom, ref_pt):
    if geom.is_empty:
        return None
    if geom.geom_type == "Point":
        return geom
    elif geom.geom_type == "MultiPoint":
        return min(geom.geoms, key=lambda g: g.distance(ref_pt))
    elif geom.geom_type in ["LineString", "LinearRing"]:
        return geom.interpolate(geom.project(ref_pt))
    elif geom.geom_type == "GeometryCollection":
        points = [extract_closest_point(g, ref_pt) for g in geom.geoms if not g.is_empty]
        return min(points, key=lambda p: p.distance(ref_pt)) if points else None
    return None