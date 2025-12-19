import numpy as np
import cv2
import svgwrite
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box
from sklearn.cluster import KMeans
import os
import sys

# ===============================
# Voronoi finite polygons helper
# ===============================
def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if all(v >= 0 for v in region):
            new_regions.append(region)
            continue
        if p1 not in all_ridges:
            continue
        ridges = all_ridges[p1]
        new_region = [v for v in region if v >= 0]
        for p2, v1, v2 in ridges:
            if v1 >= 0 and v2 >= 0:
                continue
            v = v1 if v1 >= 0 else v2
            tangent = vor.points[p2] - vor.points[p1]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[v] + direction * radius
            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = [v for _, v in sorted(zip(angles, new_region))]
        new_regions.append(new_region)
    return new_regions, np.asarray(new_vertices)

# ===============================
# Color helpers
# ===============================
def average_color(img, poly):
    mask = np.zeros(img.shape[:2], np.uint8)
    pts = np.array(poly.exterior.coords)
    cv2.fillPoly(mask, [pts.astype(int)], 255)
    mean = cv2.mean(img, mask=mask)
    return np.array(mean[:3], dtype=int)

def reduce_palette(colors, n):
    if n <= 0 or n >= len(colors):
        return colors
    km = KMeans(n_clusters=n, n_init=10)
    labels = km.fit_predict(colors)
    return km.cluster_centers_[labels].astype(int)

def rgb_to_hex(col):
    return '#{:02X}{:02X}{:02X}'.format(int(col[0]), int(col[1]), int(col[2]))

# ===============================
# CLI progress
# ===============================
def print_progress(current, total, prefix=''):
    bar_len = 40
    filled_len = int(round(bar_len * current / float(total)))
    percents = round(100.0 * current / float(total), 1)
    bar = '#' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write(f'\r{prefix} [{bar}] {percents}% ({current}/{total})')
    sys.stdout.flush()

# ===============================
# MAIN
# ===============================
print("\n=== VORONOI SVG GENERATOR ===\n")

img_path = input("Полный путь к INPUT изображению (png/jpg): ").strip('"')
if not os.path.exists(img_path):
    raise FileNotFoundError("Файл не найден")

svg_name = input("Имя выходного SVG (например result.svg): ").strip()
if not svg_name.lower().endswith(".svg"):
    svg_name += ".svg"

edge_n = int(input("Точек по контуру: "))
detail_n = int(input("Точек детализации: "))
bg_n = int(input("Точек фона: "))
low = int(input("LOW для Canny (30–100): "))
high = int(input("HIGH для Canny (120–300): "))
palette_n = int(input("Количество цветов в палитре (0 = без ограничения): "))
group_by_color = input("Группировать объекты по цветам? (y/n): ").lower() == "y"
export_each_color = input("Экспортировать каждый цвет в отдельный SVG? (y/n): ").lower() == "y"

# ===============================
# Load image
# ===============================
img_bgr = cv2.imread(img_path)
if img_bgr is None:
    raise ValueError("Не удалось прочитать изображение")

img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

# ===============================
# Generate points
# ===============================
pts = []

# фоновые точки						   
for _ in range(bg_n):
    pts.append([np.random.randint(0, w), np.random.randint(0, h)])

# точки по контуру через Canny
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray, low, high)
coords = np.column_stack(np.where(edges > 0))
for _ in range(edge_n):
    y, x = coords[np.random.randint(0, len(coords))]
    pts.append([x, y])

# случайные детализирующие точки															
for _ in range(detail_n):
    pts.append([np.random.randint(0, w), np.random.randint(0, h)])

pts = np.array(pts)

# ===============================
# Voronoi
# ===============================
vor = Voronoi(pts)
regions, vertices = voronoi_finite_polygons_2d(vor)
bbox = box(0, 0, w, h)

polygons = []
colors = []

for region in regions:
    poly = Polygon(vertices[region]).intersection(bbox)
    if poly.is_empty or not poly.is_valid:
        continue
    polygons.append(poly)
    colors.append(average_color(img, poly))

colors = np.array(colors)
colors = reduce_palette(colors, palette_n)

# ===============================
# SVG export
# ===============================
def save_svg(dwg_name, polygons, colors, group_colors):
    # Все SVG имеют одинаковый размер и viewBox
    dwg = svgwrite.Drawing(dwg_name, size=(w, h), viewBox=f"0 0 {w} {h}")
    if group_colors:
        color_groups = {}
        for poly, col in zip(polygons, colors):
            key = rgb_to_hex(col)
            color_groups.setdefault(key, []).append(poly)

        total = sum(len(polys) for polys in color_groups.values())
        count = 0

        for hex_col, polys in sorted(color_groups.items(), key=lambda x: int(x[0][1:], 16)):
            group = dwg.g(fill=hex_col, stroke="none", id=hex_col)
            for poly in polys:
                group.add(dwg.polygon(points=list(poly.exterior.coords)))
                count += 1
                print_progress(count, total, prefix=f'SVG прогресс {dwg_name}')
            dwg.add(group)
    else:
        total = len(polygons)
        for i, (poly, col) in enumerate(zip(polygons, colors), 1):
            dwg.add(
                dwg.polygon(points=list(poly.exterior.coords),
                            fill=rgb_to_hex(col),
                            stroke="none")
            )
            print_progress(i, total, prefix=f'SVG прогресс {dwg_name}')

    dwg.save()
    print(f"\nСохранён SVG: {dwg_name}")

# Главный SVG
save_svg(svg_name, polygons, colors, group_by_color)

# Экспорт каждого цвета в отдельный SVG в папке
if export_each_color and group_by_color:
    folder_name = os.path.splitext(svg_name)[0]
    os.makedirs(folder_name, exist_ok=True)

    color_groups = {}
    for poly, col in zip(polygons, colors):
        key = rgb_to_hex(col)
        color_groups.setdefault(key, []).append(poly)

    total_files = len(color_groups)
    file_count = 0

    for hex_col, polys in color_groups.items():
        file_count += 1
        file_name = os.path.join(folder_name, f"{hex_col}.svg")
        save_svg(file_name, polys, [np.array([int(hex_col[1:3],16),
                                              int(hex_col[3:5],16),
                                              int(hex_col[5:7],16)])]*len(polys),
                 group_colors=False)
        print_progress(file_count, total_files, prefix=f'Сохранение файлов по цвету')

print("\nГОТОВО ✅ Все SVG сохранены и готовы!")
