from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

PROFILE = bool(os.environ.get("PLATE_PROFILE"))
RESET_EVERY = int(os.environ.get("PLATE_RESET_EVERY", "24"))
STAGES: dict[str, float] = {}


def tick(name: str, start: float) -> float:
    now = time.perf_counter()
    if PROFILE:
        STAGES[name] = STAGES.get(name, 0.0) + (now - start)
    return now


LIGHT_STRENGTH = {
    "sun": (0.22, 2.3, 0.010), "overcast": (0.55, 0.8, 0.10), "shade": (0.36, 0.45, 0.09),
    "dusk": (0.055, 0.45, 0.05), "night": (0.010, 16.0, 0.30),
    "ir850": (0.002, 9.0, 0.08), "ir940": (0.002, 6.5, 0.08),
}


def surface_planes(path: str) -> dict[str, np.ndarray]:
    planes = np.load(path)
    return {
        "albedo": np.ascontiguousarray(planes[0:3].transpose(1, 2, 0)),
        "height_mm": planes[3], "roughness": planes[4], "metallic": planes[5],
        "alpha": planes[6], "nir": planes[7],
    }


def rotation(angles: list[float]) -> np.ndarray:
    yaw, pitch, roll = np.radians(angles)
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    return np.diag([1.0, -1.0, -1.0]) @ rz @ ry @ rx


def image_node(nodes, name: str, values: np.ndarray, colour: bool = False):
    height, width = values.shape[:2]
    rgba = np.ones((height, width, 4), dtype=np.float32)
    if values.ndim == 2:
        rgba[..., :3] = values[..., None]
    else:
        rgba[..., :values.shape[2]] = values
    image = bpy.data.images.new(name, width=width, height=height, alpha=True, float_buffer=True)
    image.colorspace_settings.name = "sRGB" if colour else "Non-Color"
    image.pixels.foreach_set(np.ascontiguousarray(rgba[::-1]).ravel())
    image.update()
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = image
    texture.extension = "EXTEND"
    texture.interpolation = "Cubic"
    return texture


def simple_material(name: str, colour, metallic=0.0, roughness=0.5, coat=0.0):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*colour, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Coat Weight"].default_value = coat
    return material


def plate_material(data, job: dict) -> bpy.types.Material:
    parameters = job["parameters"]
    infrared = job["lighting"].startswith("ir")
    material = bpy.data.materials.new("PlateSurface")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes["Principled BSDF"]
    colour = (np.repeat(data["nir"][..., None], 3, axis=2) if infrared else data["albedo"])
    rgba = np.concatenate((colour, data["alpha"][..., None]), axis=2)
    albedo = image_node(nodes, "PlateAlbedo", rgba)
    links.new(albedo.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(albedo.outputs["Alpha"], bsdf.inputs["Alpha"])
    roughness = image_node(nodes, "PlateRoughness", data["roughness"])
    metallic = image_node(nodes, "PlateMetallic", data["metallic"])
    links.new(roughness.outputs["Color"], bsdf.inputs["Roughness"])
    links.new(metallic.outputs["Color"], bsdf.inputs["Metallic"])
    bsdf.inputs["Coat Weight"].default_value = min(1.0, parameters["coat"]
                                                   + parameters["wetness"] * 0.5)
    bsdf.inputs["Coat Roughness"].default_value = 0.10 if parameters["wetness"] > 0.2 else 0.28
    if parameters["retroreflective"]:
        yaw, pitch, _ = np.radians(job["angles"])
        response = max(0.0, math.cos(yaw) * math.cos(pitch)) ** 2
        strength = response * (1.15 if infrared else 0.35)
        if infrared:
            strength *= 0.7 if job["lighting"] == "ir940" else 1.0
        links.new(albedo.outputs["Color"], bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = strength * job["retro_gain"]
    return material


def build_plate(job: dict, data) -> bpy.types.Object:
    width_mm, height_mm = job["size_mm"]
    heights = data["height_mm"]
    rows, cols = heights.shape
    target = max(72, int(job["width"] * 0.9))
    nx = int(min(cols, max(64, target)))
    ny = int(min(rows, max(48, target * height_mm / width_mm)))
    u, v = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    ix = np.clip(np.rint(u * (cols - 1)).astype(int), 0, cols - 1)
    iy = np.clip(np.rint(v * (rows - 1)).astype(int), 0, rows - 1)
    coords = np.stack([(u - 0.5) * width_mm, (v - 0.5) * height_mm,
                       -heights[iy, ix]], -1) / 1000.0
    grid = np.arange(nx * ny).reshape(ny, nx)
    faces = np.stack([grid[:-1, :-1], grid[1:, :-1], grid[1:, 1:], grid[:-1, 1:]], -1).reshape(-1, 4)
    mesh = bpy.data.meshes.new("PlateMesh")
    flat = coords.reshape(-1, 3).astype(np.float32)
    mesh.vertices.add(len(flat))
    mesh.vertices.foreach_set("co", flat.ravel())
    mesh.loops.add(faces.size)
    mesh.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
    mesh.polygons.add(len(faces))
    mesh.polygons.foreach_set("loop_start", np.arange(len(faces), dtype=np.int32) * 4)
    mesh.polygons.foreach_set("loop_total", np.full(len(faces), 4, np.int32))
    mesh.polygons.foreach_set("use_smooth", np.ones(len(faces), dtype=bool))
    uv = np.stack([u, 1 - v], -1).reshape(-1, 2)
    layer = mesh.uv_layers.new(name="PlateUV")
    layer.data.foreach_set("uv", uv[faces.ravel()].astype(np.float32).ravel())
    mesh.update()
    obj = bpy.data.objects.new("Plate", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(plate_material(data, job))
    obj.data.materials.append(simple_material("PlateBack", (0.52, 0.54, 0.56), 1.0, 0.42))
    thickness = obj.modifiers.new("Thickness", "SOLIDIFY")
    thickness.thickness = job["parameters"]["thickness_mm"] / 1000.0
    thickness.offset = -1.0
    thickness.material_offset_rim = 1
    thickness.material_offset = 1
    return obj


CUBE_VERTS = np.array([[-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
                       [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5]],
                      dtype=np.float32)
CUBE_FACES = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
                       [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]], dtype=np.int32)


def boxes_mesh(name: str, placements, material) -> bpy.types.Object:
    blocks = list(placements)
    if not blocks:
        return None
    vertices = np.concatenate([CUBE_VERTS * np.asarray(size, np.float32)
                               + np.asarray(location, np.float32)
                               for size, location in blocks])
    faces = np.concatenate([CUBE_FACES + index * 8 for index in range(len(blocks))])
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(vertices))
    mesh.vertices.foreach_set("co", vertices.ravel())
    mesh.loops.add(faces.size)
    mesh.loops.foreach_set("vertex_index", faces.ravel())
    mesh.polygons.add(len(faces))
    mesh.polygons.foreach_set("loop_start", np.arange(len(faces), dtype=np.int32) * 4)
    mesh.polygons.foreach_set("loop_total", np.full(len(faces), 4, np.int32))
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(material)
    bpy.context.collection.objects.link(obj)
    return obj


def add_box(name: str, dimensions, location, material, bevel: float = 0.0):
    return boxes_mesh(name, [(dimensions, location)], material)


def build_context(job: dict, rng) -> None:
    width, height = np.asarray(job["size_mm"]) / 1000.0
    infrared = job["lighting"].startswith("ir")
    tint = np.array(job["background_color"], dtype=float)
    if infrared:
        tint = np.full(3, float(tint.mean()))
    if job["is_vehicle"]:
        body = simple_material("Body", tuple(np.clip(tint * rng.uniform(0.6, 2.4), 0.005, 0.85)),
                               0.35, 0.24, 0.75)
        add_box("BodyPanel", (width * 3.2, height * 4.2, 0.10), (0, 0, 0.058), body)
        dark = simple_material("Grille", (0.014, 0.015, 0.016), 0.2, 0.42)
        add_box("Grille", (width * 2.2, height * 1.1, 0.05), (0, height * 1.35, 0.032), dark)
        rib = simple_material("Rib", (0.03, 0.031, 0.033), 0.4, 0.3)
        boxes_mesh("Ribs", [((0.006, height * 1.05, 0.02), (float(x), height * 1.35, 0.016))
                            for x in np.linspace(-width, width, 14)], rib)
    else:
        board = simple_material("Board", tuple(np.clip(tint * rng.uniform(0.8, 3.0), 0.01, 0.9)),
                                0.0, 0.75)
        add_box("Board", (width * 3.0, height * 3.6, 0.06), (0, 0, 0.042), board)
    if job["holder"]:
        holder = simple_material("Holder", (0.02, 0.02, 0.022), 0.1, 0.45)
        boxes_mesh("Holder", [
            ((width * 1.06, 0.012, 0.011), (0.0, -height / 2 - 0.006, -0.003)),
            ((width * 1.06, 0.012, 0.011), (0.0, height / 2 + 0.006, -0.003)),
            ((0.011, height * 1.02, 0.011), (-width / 2 - 0.006, 0.0, -0.003)),
            ((0.011, height * 1.02, 0.011), (width / 2 + 0.006, 0.0, -0.003)),
        ], holder)


def add_light(name: str, location, target, energy: float, size: float, colour,
              kind: str = "AREA"):
    data = bpy.data.lights.new(name, kind)
    data.color = colour
    if kind == "SUN":
        data.energy = energy
        data.angle = size
    else:
        distance = max(0.4, float(np.linalg.norm(np.asarray(location) - np.asarray(target))))
        data.energy = energy * distance ** 2
        data.size = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()
    return obj


def prepare_engine(render: dict) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.05
    scene.cycles.use_denoising = render["denoise"]
    scene.cycles.max_bounces = 3
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    scene.cycles.transmission_bounces = 2
    scene.cycles.transparent_max_bounces = 2
    scene.cycles.use_light_tree = False
    scene.cycles.blur_glossy = 1.5
    scene.render.use_persistent_data = True
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "16"
    scene.render.image_settings.color_mode = "RGB"
    if render["device"] != "CPU":
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = render["device"]
        preferences.get_devices()
        available = [device for device in preferences.devices if device.type == render["device"]]
        if not available:
            raise RuntimeError(f"No Cycles {render['device']} device available")
        for device in preferences.devices:
            device.use = device.type == render["device"]
        scene.cycles.device = "GPU"
    else:
        scene.cycles.device = "CPU"
        scene.render.threads_mode = "FIXED"
        scene.render.threads = max(1, render.get("threads", 4))
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.sensor_width = 36.0
    camera_data.clip_start = 0.02
    camera_data.clip_end = 400.0
    camera_data.shift_x = 0.0
    camera_data.shift_y = 0.0
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    scene.world = world


def clear_frame() -> None:
    camera = bpy.context.scene.camera
    collection = bpy.context.collection
    for obj in list(collection.objects):
        if obj is not camera:
            collection.objects.unlink(obj)


def configure(job: dict, render: dict) -> None:
    scene = bpy.context.scene
    scene.cycles.samples = render["samples"]
    scene.cycles.seed = job["seed"] % (2 ** 31 - 1)
    scale = render["supersampling"]
    scene.render.resolution_x = job["width"] * scale
    scene.render.resolution_y = job["height"] * scale
    camera = scene.camera
    camera_data = camera.data
    camera_data.lens = job["focal_px"] / (job["width"]) * 36.0
    centre_x, centre_y = job["centre"]
    camera_data.shift_x = -(centre_x - job["width"] / 2.0) / job["width"]
    camera_data.shift_y = (centre_y - job["height"] / 2.0) / job["width"]
    camera.location = (0.0, 0.0, 0.0)
    camera.rotation_euler = (0.0, 0.0, 0.0)

    lighting = job["lighting"]
    world_strength, key_energy, key_size = LIGHT_STRENGTH[lighting]
    world = scene.world
    background = world.node_tree.nodes["Background"]
    infrared = lighting.startswith("ir")
    background.inputs["Color"].default_value = ((0.35, 0.35, 0.35, 1.0) if infrared
                                                else (0.5, 0.62, 0.8, 1.0))
    background.inputs["Strength"].default_value = world_strength

    depth = job["depth"]
    target = (0.0, 0.0, -depth)
    if infrared:
        add_light("IRLamp", (0.03, 0.02, -0.02), target, key_energy, 0.06, (1.0, 1.0, 1.0))
    elif lighting == "night":
        add_light("StreetLamp", (depth * 0.3, depth * 0.45, -depth + 0.8), target,
                  key_energy, 0.4, (1.0, 0.72, 0.42))
        add_light("Headlamp", (-depth * 0.35, 0.15, -depth * 0.25), target,
                  key_energy * 0.5, 0.25, (0.8, 0.88, 1.0))
    else:
        direction = np.asarray(job["light_direction"], dtype=float)
        position = np.array([0.0, 0.0, -depth]) + direction * max(1.5, depth * 0.6)
        add_light("Key", tuple(position), target, key_energy, key_size,
                  (1.0, 0.95, 0.88), kind="SUN")
        add_light("Fill", (-depth * 0.5, depth * 0.35, -depth * 0.4), target,
                  key_energy * 0.25, max(0.05, key_size * 3), (0.75, 0.85, 1.0), kind="SUN")


def render_job(job: dict, render: dict, output: Path) -> None:
    mark = time.perf_counter()
    configure(job, render)
    mark = tick("configure", mark)
    rng = np.random.default_rng(job["seed"])
    data = surface_planes(job["surface"])
    mark = tick("load", mark)
    build_context(job, rng)
    mark = tick("context", mark)
    plate = build_plate(job, data)
    mark = tick("plate", mark)
    matrix = rotation(job["angles"])
    plate.location = Vector((0.0, 0.0, -job["depth"]))
    plate.rotation_euler = Matrix(matrix.tolist()).to_euler()
    for obj in bpy.context.collection.objects:
        if obj.type == "MESH" and obj is not plate:
            obj.parent = plate
            obj.matrix_parent_inverse = plate.matrix_world.inverted()
    bpy.context.view_layer.update()
    mark = tick("update", mark)
    if job.get("debug"):
        from bpy_extras.object_utils import world_to_camera_view
        scene = bpy.context.scene
        corners = [Vector(plate.matrix_world @ Vector(v.co)) for v in plate.data.vertices[:1]]
        box = [plate.matrix_world @ Vector(corner) for corner in plate.bound_box]
        for point in box[:8]:
            screen = world_to_camera_view(scene, scene.camera, point)
            print("DEBUG_CORNER", round(screen.x * job["width"], 1),
                  round((1 - screen.y) * job["height"], 1), round(screen.z, 3), flush=True)
        for obj in bpy.context.collection.objects:
            print("DEBUG_OBJ", obj.name, [round(v, 3) for v in obj.matrix_world.translation],
                  flush=True)
    scratch = Path("/dev/shm") if Path("/dev/shm").is_dir() else output.parent
    exr = scratch / f"{output.stem}.exr"
    exr_path = exr.as_posix()
    bpy.context.scene.render.filepath = exr_path
    bpy.ops.render.render(write_still=True)
    mark = tick("render", mark)
    image = bpy.data.images.load(exr_path)
    width, height = image.size
    buffer = np.empty(width * height * image.channels, dtype=np.float32)
    image.pixels.foreach_get(buffer)
    frame = buffer.reshape(height, width, image.channels)[::-1, :, :3].copy()
    np.save(output, frame)
    bpy.data.images.remove(image)
    exr.unlink(missing_ok=True)
    tick("readback", mark)


def main() -> None:
    arguments = sys.argv[sys.argv.index("--") + 1:]
    payload = json.loads(Path(arguments[0]).read_text(encoding="utf-8"))
    render = payload["render"]
    since = RESET_EVERY
    for job in payload["jobs"]:
        mark = time.perf_counter()
        if since >= RESET_EVERY:
            prepare_engine(render)
            since = 0
            tick("engine", mark)
        else:
            clear_frame()
            tick("clear", mark)
        since += 1
        render_job(job, render, Path(job["output"]))
        print(f"PLATE_FRAME_DONE {job['index']}", flush=True)
    if PROFILE:
        total = sum(STAGES.values())
        count = max(1, len(payload["jobs"]))
        report = "  ".join(f"{name}={value / count * 1000:.0f}ms"
                           for name, value in sorted(STAGES.items(), key=lambda kv: -kv[1]))
        print(f"PLATE_PROFILE frames={count} total={total / count * 1000:.0f}ms  {report}",
              flush=True)


if __name__ == "__main__":
    main()
