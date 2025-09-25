# app.py
import os
import zipfile
import numpy as np
import cv2
import torch
from torchvision import transforms
from flask import Flask, request, render_template, jsonify, send_file, session, url_for
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
from datetime import datetime
from utils import (
    sample_line_by_spacing,
    compute_normal_transect,
    extract_centerline,
    calculate_centerline_length,
    get_image_crs,
    pixel_to_world,
    save_shapefile,
    convert_shapefile_to_geojson,
    get_pixel_size,
    allowed_file,
    preprocess_image,
    zip_shapefile,
    normalize_line,
    extract_closest_point,
    get_shapefile_crs,
    get_shapefile_bounds,

)

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'static', 'results')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_unet()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()


@app.route('/')
def index():
    if "processed_files" not in session:
        session["processed_files"] = []

    m = leafmap.Map(center=[6.5244, 3.3792], zoom=10)
    m.add_basemap("HYBRID")
    layer_names = []

    for geojson_path in session["processed_files"]:
        if os.path.exists(geojson_path):
            gdf = gpd.read_file(geojson_path)
            if gdf.crs is None or gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
                gdf.to_file(geojson_path, driver="GeoJSON")
            layer_name = os.path.splitext(os.path.basename(geojson_path))[0]
            m.add_geojson(geojson_path, layer_name=layer_name, style={"color": "red", "weight": 2})
            layer_names.append(os.path.basename(geojson_path))  # Just filename for dropdowns

    m.add_layer_control()
    map_html = m._repr_html_()
    return render_template('index.html', map_html=map_html, layer_names=layer_names)

@app.route("/list-files", methods=["GET"])
def list_files():
    try:
        files = os.listdir(app.config['RESULTS_FOLDER'])
        # Only include GeoJSON and Shapefiles for comparison (you can extend this)
        valid_extensions = [".geojson", ".shp"]
        available_files = [f for f in files if os.path.splitext(f)[1].lower() in valid_extensions]

        return jsonify({"available_files": available_files})
    except Exception as e:
        return jsonify({"error": "Failed to list files", "details": str(e)}), 500


@app.route('/process', methods=['POST'])
def process_file():
    if 'files' not in request.files:
        return jsonify({"error": "No file part"}), 400
    files = request.files.getlist('files')
    if not files:
        return jsonify({"error": "No files selected"}), 400

    results = []
    try:
        for file in files:
            if file.filename == '' or not allowed_file(file.filename):
                continue
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            image_tensor, original_image = preprocess_image(image_path)
            image_tensor = image_tensor.to(device)

            with torch.no_grad():
                pred_mask = model(image_tensor)
                pred_mask = torch.sigmoid(pred_mask).squeeze().cpu().numpy()

            if pred_mask.ndim == 3:
                pred_mask = pred_mask[0]
            binary_mask = (pred_mask > 0.5).astype(np.uint8)
            binary_mask = cv2.resize(binary_mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

            centerline, centerline_points = extract_centerline(binary_mask)
            world_points = [pixel_to_world(x, y, image_path) for x, y in centerline_points]
            image_crs_wkt = get_image_crs(image_path)

            base_filename = os.path.splitext(filename)[0]
            shapefile_path = os.path.join(app.config['RESULTS_FOLDER'], f"{base_filename}_shoreline.shp")
            save_shapefile(world_points, shapefile_path, image_crs_wkt)

            shoreline_length = calculate_centerline_length(centerline_points, image_path)
            geojson_path = convert_shapefile_to_geojson(shapefile_path)

            if "processed_files" not in session:
                session["processed_files"] = []
            session["processed_files"].append(geojson_path)
            session.modified = True

            zip_path = zip_shapefile(shapefile_path)

            result_entry = {
                "filename": filename,
                "shapefile": os.path.basename(shapefile_path),
                "geojson": os.path.basename(geojson_path),
                "geojson_url": url_for('static', filename=f"results/{os.path.basename(geojson_path)}"),
                "shoreline_length_meters": shoreline_length,
                "download_link": f"/download/{os.path.basename(zip_path)}",
                "crs": image_crs_wkt
            }
            results.append(result_entry)

        return jsonify({"results": results, "message": "Processing completed successfully!"})

    except Exception as e:
        return jsonify({"error": "Error processing files.", "details": str(e), "trace": traceback.format_exc()}), 500

@app.route("/clear_session", methods=["POST"])
def clear_session():
    session["processed_files"] = []
    session.modified = True
    return jsonify({"status": "cleared"})

@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    return send_file(path, as_attachment=True)


@app.route("/compare", methods=["POST"])
def compare_shorelines():

    data = request.json

    file1 = data.get("file1")  # baseline
    file2 = data.get("file2")  # target

    try:
        path1 = os.path.join(app.config['RESULTS_FOLDER'], file1)
        path2 = os.path.join(app.config['RESULTS_FOLDER'], file2)

        gdf1 = gpd.read_file(path1)
        gdf2 = gpd.read_file(path2)

        if gdf1.crs.is_geographic:
            try:
                print("🌍 Input CRS is geographic. Estimating UTM CRS...")
                utm_crs = gdf1.estimate_utm_crs()
                print(f"📐 Estimated UTM CRS: {utm_crs}")
            except Exception as e:
                print(f"⚠️ Failed to estimate UTM CRS: {e}")
                print("⏪ Falling back to EPSG:3857 for distance calculations")
                utm_crs = "EPSG:3857"

            gdf1 = gdf1.to_crs(utm_crs)
            gdf2 = gdf2.to_crs(utm_crs)

        line1 = gdf1.geometry.unary_union
        line2 = gdf2.geometry.unary_union

        line1 = normalize_line(line1, name="line1")
        line2 = normalize_line(line2, name="line2")

        sampled_points = sample_line_by_spacing(line1, spacing=20)
        print(f"📌 Sampled {len(sampled_points)} points on line1")

        transects = []
        distances = []
        directions = []

        for idx, pt in enumerate(sampled_points):
            transect = compute_normal_transect(pt, line1, length=90000)
            if not transect:
                continue

            intersection = transect.intersection(line2)
            intersection_point = extract_closest_point(intersection, pt)

            if not intersection_point:
                continue

            distance = pt.distance(intersection_point)

            pt_to_intersection = [pt, intersection_point]
            signed_distance = distance

            if hasattr(intersection_point, 'x') and hasattr(pt, 'x'):
                dx = intersection_point.x - pt.x
                dy = intersection_point.y - pt.y
                transeg = compute_normal_transect(pt, line1, length=2)
                tdx = transeg.coords[1][0] - transeg.coords[0][0]
                tdy = transeg.coords[1][1] - transeg.coords[0][1]
                dot = dx * tdx + dy * tdy
                signed_distance = distance if dot > 0 else -distance

            transects.append(LineString(pt_to_intersection))
            distances.append(signed_distance)
            directions.append("advance" if signed_distance > 0 else "retreat")

        if not transects:
            print("❌ No valid transects found")
            return jsonify({"error": "No valid transects found."}), 400

        diff_gdf_projected = gpd.GeoDataFrame({
            "geometry": transects,
            "distance_m": distances,
            "direction": directions
        }, crs=gdf1.crs)
        
        diff_gdf = diff_gdf_projected.to_crs(epsg=4326)
        
        dist_array = np.array(distances)
        stats = {
            "min_distance": float(np.min(dist_array)),
            "max_distance": float(np.max(dist_array)),
            "mean_distance": float(np.mean(dist_array)),
            "std_distance": float(np.std(dist_array)),
            "total_transects": len(dist_array),
            "retreats": int(np.sum(dist_array < 0)),
            "advances": int(np.sum(dist_array > 0))
        }

        base_name = f"transects_{os.path.splitext(file1)[0]}_vs_{os.path.splitext(file2)[0]}"
        output_geojson = f"{base_name}.geojson"
        output_geojson_path = os.path.join(app.config['RESULTS_FOLDER'], output_geojson)
        diff_gdf.to_file(output_geojson_path, driver="GeoJSON")

        # Save as shapefile
        shapefile_dir = os.path.join(app.config['RESULTS_FOLDER'], f"{base_name}_shp")
        os.makedirs(shapefile_dir, exist_ok=True)
        shapefile_path = os.path.join(shapefile_dir, "difference.shp")
        diff_gdf.to_file(shapefile_path)

        # Save CSV in projected CRS with geometry as WKT (in meters)
        csv_path = os.path.join(shapefile_dir, "difference.csv")
        diff_gdf_projected.to_csv(csv_path, index=False)

        # Create ZIP
        zip_path = os.path.join(app.config['RESULTS_FOLDER'], f"{base_name}_shp.zip")
        shutil.make_archive(zip_path.replace(".zip", ""), 'zip', shapefile_dir)
        shutil.rmtree(shapefile_dir)

        # Track processed file
        if "processed_files" not in session:
            session["processed_files"] = []
        session["processed_files"].append(output_geojson_path)
        session.modified = True

        return jsonify({
            "message": "Comparison successful with perpendicular transects!",
            "geojson": os.path.basename(output_geojson_path),
            "geojson_url": url_for('static', filename=f"results/{os.path.basename(output_geojson_path)}"),
            "download_link": f"/download/{os.path.basename(zip_path)}",
            "stats": stats
        })

    except Exception as e:
        return jsonify({
            "error": "Comparison failed.",
            "details": str(e),
            "trace": traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
