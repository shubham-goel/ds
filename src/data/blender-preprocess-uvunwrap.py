"""
Run as: blender -b --python blender-preprocess-uvunwrap.py -- <infile> <outfile> <simplify>

1. Loads a mesh,
2. Applies modifiers:
        Laplacian Smooth
        Remesh
        Triangulate
3. Compute Smart UV unwrap
4. Exports to disk
"""
import bpy
import sys

in_file = sys.argv[sys.argv.index('--')+1]      # '../dummy_data/blender_test/B002JAYMEE.obj'
out_file = sys.argv[sys.argv.index('--')+2]     # '../dummy_data/blender_test/B002JAYMEE_uv.obj'
simplify = sys.argv[sys.argv.index('--')+3]     # True
print('in_file', in_file)
print('out_file', out_file)
print('simplify', simplify)
simplify = str(simplify).lower()=='true'
print('simplify', simplify)

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=True)

imported_object = bpy.ops.import_scene.obj(filepath=in_file)
obj_object = bpy.context.selected_objects[0] ####<--Fix
print('Imported name: ', obj_object.name)

# Set active object
vers = bpy.app.version_string
print(vers)
if vers.startswith('2.7'):
        bpy.context.scene.objects.active = obj_object         # blender 2.79
elif vers.startswith('2.8'):
        bpy.context.view_layer.objects.active = obj_object      # blender 2.8

if simplify:
        print('Laplacian Smooth ...')
        bpy.ops.object.modifier_add(type='LAPLACIANSMOOTH')
        obj_object.modifiers["Laplacian Smooth"].iterations = 5
        obj_object.modifiers["Laplacian Smooth"].lambda_factor = 0.5
        bpy.ops.object.modifier_apply(modifier='Laplacian Smooth')

        print('Remesh ...')
        bpy.ops.object.modifier_add(type='REMESH')
        obj_object.modifiers["Remesh"].mode = 'SMOOTH'
        obj_object.modifiers["Remesh"].octree_depth = 5
        bpy.ops.object.modifier_apply(modifier='Remesh')

print('Triangulate ...')
bpy.ops.object.modifier_add(type='TRIANGULATE')
bpy.ops.object.modifier_apply(modifier='Triangulate')

print('UV smart_project ...')
bpy.ops.object.editmode_toggle() #entering edit mode
bpy.ops.mesh.select_all(action='SELECT') #select all objects elements
bpy.ops.uv.smart_project()

print('Exporting', bpy.context.selected_objects)
bpy.ops.export_scene.obj(filepath=out_file, use_selection=True, use_uvs=True, use_materials=False, use_triangles=True)
