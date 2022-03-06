"""
Run as: blender -b --python blender-render-google.py -- regexp

Renders all models s.t.  model_name matches regexp
"""
import json
import os
import os.path as osp
import re
import sys
import traceback
from shutil import copyfile

import bpy
import numpy as np

# Add this folder to path
sys.path.append(osp.dirname(osp.abspath(__file__)))
import utils
from blender_misc import (eprint, hdrihaven_fetch, import_obj,
                          initialize_blender_cuda)

GOOGLE_DATASET_DIR =     'gso_data/'  # demo
GOOGLE_DATASET_VER =     '1'
RESULTS_DIR =            'renders_env/'

RESUME =                True
VIEWS =                 20
RESOLUTION =            2048
RENDER_DEPTH =          True
RENDER_NORMALS =        True
COLOR_DEPTH =           16
DEPTH_FORMAT =          'OPEN_EXR'
COLOR_FORMAT =          'PNG'
NORMAL_FORMAT =         'PNG'
CAMERA_FOV_RANGE =      [20, 50]
LIGHT_NUM =             6
LIGHT_ENERGY =          8
RANDOM_SEED =           0xaaaa_aaaa_aaaa_aaaa

# Environment lighting things
HDRI_NAME =             'photo_studio_01'
USE_ENV_LIGHTING =      True
ENV_LIGHTING_PATH =     hdrihaven_fetch(HDRI_NAME, res='16k')

default_rng = np.random.default_rng(RANDOM_SEED)

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting
    bpy.context.scene.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    return b_empty

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

def setup_nodegraph(scene):
    # Render Optimizations
    scene.render.use_persistent_data = True

    # Set up rendering of depth map.
    scene.use_nodes = True
    tree = scene.node_tree
    links = tree.links

    # Add passes for additionally dumping albedo and normals.
    scene.view_layers["View Layer"].use_pass_normal = True
    # scene.view_layers["View Layer"].use_pass_color = True

    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')

    if RENDER_DEPTH:
        depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        depth_file_output.label = 'Depth Output'
        links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
        depth_file_output.format.file_format = str(DEPTH_FORMAT)
        depth_file_output.base_path = ''
    else:
        depth_file_output = None

    if RENDER_NORMALS:
        normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        normal_file_output.label = 'Normal Output'
        links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])
        normal_file_output.format.file_format = str(NORMAL_FORMAT)
        normal_file_output.base_path = ''
    else:
        normal_file_output = None

    # albedo_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    # albedo_file_output.label = 'Albedo Output'
    # links.new(render_layers.outputs['Color'], albedo_file_output.inputs[0])
    return depth_file_output, normal_file_output

def add_environment_lighting(scene, rotate_angles=(0,0,0)):
    world = scene.world
    world.use_nodes = True
    node_tree = world.node_tree
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    enode = nodes.new('ShaderNodeTexEnvironment')
    enode.image = bpy.data.images.load(ENV_LIGHTING_PATH)
    links.new(enode.outputs['Color'], node_tree.nodes['Background'].inputs['Color'])

    # Rotate environment hdri
    # Follows https://blender.stackexchange.com/questions/47833/how-do-i-rotate-an-environment-texture
    mapping = nodes.new('ShaderNodeMapping')
    mapping.inputs['Rotation'].default_value = rotate_angles
    links.new(mapping.outputs['Vector'], enode.inputs['Vector'])

    tex_coord = nodes.new('ShaderNodeTexCoord')
    links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])


def create_random_point_lights(number, radius, energy=10, rng=default_rng):
    lights = []

    for i in range(number):
        # create light datablock, set attributes
        light_data = bpy.data.lights.new(name=f'ptlight{i}', type='POINT')
        light_data.energy = energy

        # create new object with our light datablock
        light_object = bpy.data.objects.new(name=f'ptlight{i}', object_data=light_data)

        #change location
        light_object.location = rng.uniform(-1., 1., size=3)
        light_object.location *= radius / np.linalg.norm(light_object.location)

        lights.append(light_object)

    for light in lights:
        # link light object
        bpy.context.collection.objects.link(light)

    return lights

def render_multiple(obj_path, output_dir, views, resolution,
        depth=True, normals=True, color_depth=16, rng = default_rng):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if RESUME:
        try:
            with open(output_dir + '/' + 'transforms.json', 'r') as out_file:
                data = json.load(out_file)
                if len(data['frames'])>=VIEWS:
                    print('#' * 30)
                    print('#' * 30)
                    print('#' * 30)
                    print('#' * 30)
                    print(f'Returning because enough frames ({len(data["frames"])}) are already present in transforms.json')
                    print('#' * 30)
                    print('#' * 30)
                    print('#' * 30)
                    print('#' * 30)
                    return
        except FileNotFoundError:
            pass

    # Clear scene
    utils.clean_objects()

    # Import obj
    obj_object = import_obj(obj_path)
    print('Imported name: ', obj_object.name, flush=True)
    verts = np.array([tuple(obj_object.matrix_world @ v.co) for v in obj_object.data.vertices])
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    vcen = (vmin+vmax)/2
    obj_size = np.abs(verts - vcen).max()

    scene = bpy.context.scene

    # Setup Node graph for rendering rgbs,depth,normals
    (depth_file_output, normal_file_output) = setup_nodegraph(scene)

    # Add random lighting
    if USE_ENV_LIGHTING:
        add_environment_lighting(scene, rotate_angles=(-np.pi/2,0,0))
        light_objects = []
    else:
        light_objects = create_random_point_lights(LIGHT_NUM, 3*obj_size, energy=LIGHT_ENERGY)

    # Create collection for objects not to render with background
    objs = [ob for ob in scene.objects if ob.type in ('EMPTY') and 'Empty' in ob.name]
    bpy.ops.object.delete({"selected_objects": objs})

    # Setup camera, constraint to empty object
    cam = utils.create_camera(location=(0, 0, 1))
    cam.data.sensor_fit = 'HORIZONTAL'
    cam.data.sensor_width = 36.0
    cam.data.sensor_height = 36.0
    b_empty = parent_obj_to_camera(cam)
    utils.add_track_to_constraint(cam, b_empty)

    # Move everything to be centered at vcen
    b_empty.location = vcen
    for light in light_objects:
        light.location += b_empty.location

    # Image settings
    scene.camera = cam
    scene.render.engine = 'CYCLES'
    scene.render.image_settings.file_format = str(COLOR_FORMAT)
    scene.render.image_settings.color_depth = str(COLOR_DEPTH)
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.dither_intensity = 0.0
    scene.render.film_transparent = True
    scene.view_layers[0].cycles.use_denoising = True
    scene.cycles.samples = 128

    out_data = {
        'obj_path':remove_prefix(obj_path, GOOGLE_DATASET_DIR),
    }
    out_data['frames'] = []

    for i in range(0, VIEWS):
        scene.render.filepath = output_dir + '/r_' + str(i)
        b_empty.rotation_euler = rng.uniform(0, 2*np.pi, size=3)

        # Update camera location and angle
        bpy.context.view_layer.update()
        # cam = scene.camera
        cam.data.angle = rng.uniform(CAMERA_FOV_RANGE[0],CAMERA_FOV_RANGE[1]) * np.pi/180
        cam.location =  (0, 0, 1.8 * obj_size/np.tan(cam.data.angle/2))
        # cam.data.angle = 0.691111147403717
        # cam.location = (0,0,4/9)
        bpy.context.view_layer.update()

        if RENDER_DEPTH:
            depth_file_output.file_slots[0].path = scene.render.filepath + "_depth_"
        if RENDER_NORMALS:
            normal_file_output.file_slots[0].path = scene.render.filepath + "_normal_"

        bpy.ops.render.render(write_still=True)  # render still

        bpy.context.view_layer.update()
        frame_data = {
            'file_path': remove_prefix(scene.render.filepath, GOOGLE_DATASET_DIR),
            'transform_matrix': listify_matrix(cam.matrix_world),

            # Independent components that make up transformation matrix
            'camera':{
                'angle_x': cam.data.angle_x,
                'angle_y': cam.data.angle_y,
                'shift_x': cam.data.shift_x,
                'shift_y': cam.data.shift_y,
                'sensor_height': cam.data.sensor_height,
                'sensor_width': cam.data.sensor_width,
                'sensor_fit': cam.data.sensor_fit,
                # 'location': list(cam.location),
                # 'scale': list(cam.scale),
                # 'rotation_quaternion': list(cam.rotation_quaternion),
                # 'be_location': list(b_empty.location),
                # 'be_scale': list(b_empty.scale),
                # 'be_rotation_euler': list(b_empty.rotation_euler),
                # 'be_rotation_matrix': listify_matrix(b_empty.matrix_world),
            }
        }
        out_data['frames'].append(frame_data)

    with open(output_dir + '/' + 'transforms.json', 'w') as out_file:
        json.dump(out_data, out_file, indent=4)


if __name__ == "__main__":

    # Initialize cuda
    initialize_blender_cuda()

    # Check if only part of data has to be rerun
    try:
        regexp = str(sys.argv[sys.argv.index('--') + 1])
    except IndexError:
        regexp = ".*"
    pattern = re.compile(regexp)

    for _mi, model_name in enumerate(os.listdir(GOOGLE_DATASET_DIR)):
        if not pattern.match(model_name):
            continue
        if not osp.isdir(f'{GOOGLE_DATASET_DIR}/{model_name}'):
            continue
        print(f'{_mi:04d}: {model_name}')
        eprint(f'{_mi:04d}: {model_name}')
        OBJ_PATH = f'{GOOGLE_DATASET_DIR}/{model_name}/{GOOGLE_DATASET_VER}/meshes/model.obj'
        MTL_PATH = f'{GOOGLE_DATASET_DIR}/{model_name}/{GOOGLE_DATASET_VER}/meshes/model.mtl'
        TEX_PATH_OLD = f'{GOOGLE_DATASET_DIR}/{model_name}/{GOOGLE_DATASET_VER}/materials/textures/texture.png'
        TEX_PATH_NEW = f'{GOOGLE_DATASET_DIR}/{model_name}/{GOOGLE_DATASET_VER}/meshes/texture.png'

        try:

            # Reset RANDOM_SEED for each instance
            rng = np.random.default_rng(RANDOM_SEED)

            # Copy texture from TEX_PATH_OLD -> TEX_PATH_NEW
            print(f'Copying texture from {TEX_PATH_OLD} to {TEX_PATH_NEW}')
            copyfile(TEX_PATH_OLD, TEX_PATH_NEW)

            OUTPUT_DIR = f'{GOOGLE_DATASET_DIR}/{model_name}/{GOOGLE_DATASET_VER}/{RESULTS_DIR}'
            render_multiple(
                OBJ_PATH,
                OUTPUT_DIR,
                VIEWS,
                RESOLUTION,
                depth=RENDER_DEPTH,
                normals=RENDER_NORMALS,
                rng = rng,
            )
        except:
            eprint("*** failed", model_name)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            eprint("*** print_tb:")
            traceback.print_tb(exc_traceback, limit=1, file=sys.stderr)
            eprint("*** print_exception:")
            # exc_type below is ignored on 3.5 and later
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                    limit=2, file=sys.stderr)
