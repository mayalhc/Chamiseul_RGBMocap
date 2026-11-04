"""RGB MoCap Bridge: rig mapping, flat-ground snapping and FBX/GLB export.

The extension manifest (blender_manifest.toml) carries the package metadata;
Blender 4.2+ extensions do not use bl_info.
"""

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper

__version__ = "1.0.0"

NAMES = ['Hips', 'LeftUpLeg', 'RightUpLeg', 'Spine', 'LeftLeg', 'RightLeg', 'Spine1',
         'LeftFoot', 'RightFoot', 'Spine2', 'LeftToeBase', 'RightToeBase', 'Neck',
         'LeftShoulder', 'RightShoulder', 'Head', 'LeftArm', 'RightArm', 'LeftForeArm',
         'RightForeArm', 'LeftHand', 'RightHand']

from . import exporter, ground, live, mapping  # noqa: E402  (NAMES must exist first)


def active_armature(context):
    obj = context.active_object
    if obj and obj.type == 'ARMATURE':
        return obj
    selected = [item for item in context.selected_objects if item.type == 'ARMATURE']
    return selected[0] if selected else None


def describe(armature):
    found, missing, conflicts = mapping.detect(armature)
    lines = ['%s: %d bones, %d/%d body bones matched'
             % (armature.name, len(armature.data.bones), len(found), len(NAMES))]
    for canonical in NAMES:
        if canonical in found:
            lines.append('  %-12s <- %s' % (canonical, found[canonical]))
    if missing:
        lines.append('Missing: ' + ', '.join(missing))
    if conflicts:
        lines.append('Conflicts: ' + ', '.join('%s/%s' % pair for pair in conflicts))
    return found, missing, lines


class RGBMOCAP_OT_scan(bpy.types.Operator):
    bl_idname = 'rgbmocap.scan'
    bl_label = 'Scan Humanoid Rig'
    bl_description = 'Match this rig against the canonical Mixamo body bones'
    bl_options = {'REGISTER'}

    def execute(self, context):
        armature = active_armature(context)
        if armature is None:
            self.report({'ERROR'}, 'Select an armature')
            return {'CANCELLED'}
        found, missing, lines = describe(armature)
        context.scene.rgbmocap.report = '\n'.join(lines)
        self.report({'INFO'}, '%d/%d body bones matched' % (len(found), len(NAMES)))
        return {'FINISHED'}


class RGBMOCAP_OT_rename(bpy.types.Operator):
    bl_idname = 'rgbmocap.rename'
    bl_label = 'Use Canonical Names'
    bl_description = 'Rename mapped bones and skin groups to the canonical body names'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        armature = active_armature(context)
        if armature is None:
            self.report({'ERROR'}, 'Select an armature')
            return {'CANCELLED'}
        found, missing, _lines = describe(armature)
        if missing:
            self.report({'ERROR'}, 'Not a complete humanoid rig: ' + ', '.join(missing))
            return {'CANCELLED'}
        result = mapping.rename(armature, found)
        renamed = result.get('bones', {})
        lines = ['Renamed %d bones (%d skin groups)' % (len(renamed), result.get('vertex_groups', 0))]
        lines += ['  %s -> %s' % pair for pair in renamed.items()]
        context.scene.rgbmocap.report = '\n'.join(lines)
        self.report({'INFO'}, 'Renamed %d bones' % len(renamed))
        return {'FINISHED'}


class RGBMOCAP_OT_ground(bpy.types.Operator):
    bl_idname = 'rgbmocap.ground'
    bl_label = 'Snap To Floor'
    bl_description = 'Offset the root bone so the skinned feet sit on the floor plane'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        armature = active_armature(context)
        if armature is None:
            self.report({'ERROR'}, 'Select an armature')
            return {'CANCELLED'}
        settings = context.scene.rgbmocap
        try:
            result = ground.snap(armature, floor=settings.floor_height, mode=settings.ground_mode,
                                 smooth=settings.ground_smooth, foot_only=settings.foot_only)
        except (ValueError, RuntimeError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        settings.stats = ('%s: %d frames, %s\nbefore %.4f..%.4f m -> after %.4f..%.4f m\n'
                          'residual %.6f..%.6f m' % (
                              result['mode'], result['frames'], result['target'],
                              result['before_min'], result['before_max'],
                              result['after_min'], result['after_max'],
                              result['residual_low'], result['residual_high']))
        self.report({'INFO'}, 'Lowest point now %.4f m above the floor' % result['residual_low'])
        return {'FINISHED'}


class RGBMOCAP_OT_export_fbx(bpy.types.Operator, ExportHelper):
    bl_idname = 'rgbmocap.export_fbx'
    bl_label = 'Export FBX'
    bl_description = 'Export the character and baked bone animation as FBX'
    filename_ext = '.fbx'
    filter_glob = StringProperty(default='*.fbx', options={'HIDDEN'})

    def execute(self, context):
        armature = active_armature(context)
        if armature is None:
            self.report({'ERROR'}, 'Select an armature')
            return {'CANCELLED'}
        include_mesh = context.scene.rgbmocap.include_mesh
        exporter.export_fbx(self.filepath, armature, include_mesh)
        self.report({'INFO'}, 'Wrote ' + self.filepath)
        return {'FINISHED'}


class RGBMOCAP_OT_export_glb(bpy.types.Operator, ExportHelper):
    bl_idname = 'rgbmocap.export_glb'
    bl_label = 'Export GLB'
    bl_description = 'Export the character and animation as binary glTF'
    filename_ext = '.glb'
    filter_glob = StringProperty(default='*.glb', options={'HIDDEN'})

    def execute(self, context):
        armature = active_armature(context)
        if armature is None:
            self.report({'ERROR'}, 'Select an armature')
            return {'CANCELLED'}
        include_mesh = context.scene.rgbmocap.include_mesh
        exporter.export_glb(self.filepath, armature, include_mesh)
        self.report({'INFO'}, 'Wrote ' + self.filepath)
        return {'FINISHED'}


class RGBMOCAP_OT_live_start(bpy.types.Operator):
    bl_idname = 'rgbmocap.live_start'
    bl_label = 'Start Live Link'
    bl_description = 'Play the motion RGB MoCap displays on the selected armature, live'
    bl_options = {'REGISTER'}

    def execute(self, context):
        settings = context.scene.rgbmocap
        if live.start('127.0.0.1', settings.live_port):
            # The app streams its own character, so it appears here on the first
            # packet: nothing is built before it describes what it is showing.
            settings.live_status = 'Live: connected, waiting for the character'
            self.report({'INFO'}, 'Live link connected on port %d' % settings.live_port)
            return {'FINISHED'}
        error = live.status()['error']
        settings.live_status = 'Live: ' + error
        self.report({'ERROR'}, error)
        return {'CANCELLED'}


class RGBMOCAP_OT_live_stop(bpy.types.Operator):
    bl_idname = 'rgbmocap.live_stop'
    bl_label = 'Stop Live Link'
    bl_description = 'Disconnect from RGB MoCap and stop applying its pose'
    bl_options = {'REGISTER'}

    def execute(self, context):
        live.stop()
        context.scene.rgbmocap.live_status = 'Live: stopped'
        return {'FINISHED'}


class RGBMOCAP_OT_live_record(bpy.types.Operator):
    bl_idname = 'rgbmocap.live_record'
    bl_label = 'Record Live to Action'
    bl_description = 'Key every incoming live pose into a new action, starting at the current frame'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        armature = active_armature(context) or live.current_armature()
        if armature is None:
            self.report({'ERROR'}, 'Select a humanoid armature')
            return {'CANCELLED'}
        if not live.running():
            self.report({'ERROR'}, 'Start Live Link first')
            return {'CANCELLED'}
        state = live.record_start(armature, context.scene.frame_current)
        context.scene.rgbmocap.live_status = 'Live: recording from frame %d' % state['start']
        self.report({'INFO'}, 'Recording live poses from frame %d' % state['start'])
        return {'FINISHED'}


class RGBMOCAP_OT_live_record_stop(bpy.types.Operator):
    bl_idname = 'rgbmocap.live_record_stop'
    bl_label = 'Stop Recording'
    bl_description = 'Finish the recorded action and keep its keys'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        state = live.record_stop()
        if not state['keys']:
            context.scene.rgbmocap.live_status = 'Live: nothing recorded'
            self.report({'WARNING'}, 'No live pose arrived while recording')
            return {'CANCELLED'}
        context.scene.rgbmocap.live_status = ('Recorded %d keys on %s (frames %d-%d)'
                                              % (state['keys'], state['action'],
                                                 state['start'], state['last']))
        self.report({'INFO'}, 'Recorded %d keys to %s' % (state['keys'], state['action']))
        return {'FINISHED'}


class RGBMOCAP_OT_load_base(bpy.types.Operator):
    bl_idname = 'rgbmocap.load_base'
    bl_label = 'Load Base Character'
    bl_description = ('Bring in the character RGB MoCap is showing - its own base body '
                      'or the imported character - exactly as it is there')
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.rgbmocap
        armature = live.load_base('127.0.0.1', settings.live_port)
        if armature is None:
            message = live.status()['error'] or 'Waiting for RGB MoCap to send its character.'
            settings.live_status = 'Live: ' + message
            self.report({'ERROR'}, message)
            return {'CANCELLED'}
        bpy.ops.object.select_all(action='DESELECT')
        armature.select_set(True)
        context.view_layer.objects.active = armature
        found, missing, lines = describe(armature)
        settings.report = '\n'.join(lines)
        settings.live_status = 'Live: ' + armature.name + ' loaded from RGB MoCap'
        self.report({'INFO'}, '%s loaded from RGB MoCap, %d body bones' % (armature.name, len(found)))
        return {'FINISHED'}


class RGBMOCAP_OT_import(bpy.types.Operator, ImportHelper):
    bl_idname = 'rgbmocap.import_character'
    bl_label = 'Import Character'
    bl_description = 'Import a humanoid character and report its body bone mapping'
    filename_ext = ''
    filter_glob = StringProperty(default='*.fbx;*.glb;*.gltf', options={'HIDDEN'})

    def execute(self, context):
        path = self.filepath
        before = set(bpy.context.scene.objects)
        suffix = path.lower().rsplit('.', 1)[-1]
        if suffix == 'fbx':
            bpy.ops.import_scene.fbx(filepath=path, automatic_bone_orientation=False,
                                     use_prepost_rot=True)
        elif suffix in ('glb', 'gltf'):
            bpy.ops.import_scene.gltf(filepath=path)
        else:
            self.report({'ERROR'}, 'Import FBX, GLB or glTF')
            return {'CANCELLED'}
        imported = [obj for obj in bpy.context.scene.objects if obj not in before]
        armatures = [obj for obj in imported if obj.type == 'ARMATURE']
        if not armatures:
            self.report({'ERROR'}, 'No armature in this file')
            return {'CANCELLED'}
        bpy.ops.object.select_all(action='DESELECT')
        armatures[0].select_set(True)
        bpy.context.view_layer.objects.active = armatures[0]
        found, _missing, lines = describe(armatures[0])
        context.scene.rgbmocap.report = '\n'.join(lines)
        self.report({'INFO'}, '%d imported objects, %d/%d body bones matched'
                    % (len(imported), len(found), len(NAMES)))
        return {'FINISHED'}


class RGBMOCAP_PT_panel(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'RGB MoCap'
    bl_label = 'RGB MoCap'

    def draw(self, context):
        layout = self.layout
        settings = context.scene.rgbmocap

        box = layout.box()
        box.label(text='Live link', icon='LINKED')
        box.prop(settings, 'live_port')
        row = box.row(align=True)
        row.operator('rgbmocap.live_start', icon='PLAY')
        row.operator('rgbmocap.live_stop', icon='PAUSE')
        row = box.row(align=True)
        row.operator('rgbmocap.live_record', icon='REC')
        row.operator('rgbmocap.live_record_stop', icon='RECORD_OFF')
        box.operator('rgbmocap.load_base', icon='OUTLINER_OB_ARMATURE')
        if settings.live_status:
            box.label(text=settings.live_status)

        box = layout.box()
        box.label(text='1. Humanoid rig', icon='ARMATURE_DATA')
        row = box.row(align=True)
        row.operator('rgbmocap.scan', icon='VIEWZOOM')
        row.operator('rgbmocap.rename', icon='SORTALPHA')

        box = layout.box()
        box.label(text='2. Flat ground', icon='SNAP_FACE')
        box.prop(settings, 'ground_mode')
        box.prop(settings, 'floor_height')
        row = box.row()
        row.prop(settings, 'foot_only')
        row.prop(settings, 'ground_smooth')
        box.operator('rgbmocap.ground', icon='IMPORT')

        box = layout.box()
        box.label(text='3. Export', icon='EXPORT')
        box.prop(settings, 'include_mesh')
        row = box.row(align=True)
        row.operator('rgbmocap.export_fbx', text='FBX')
        row.operator('rgbmocap.export_glb', text='GLB')
        box.operator('rgbmocap.import_character', icon='IMPORT')

        if settings.stats:
            box = layout.box()
            box.label(text='Last snap')
            for line in settings.stats.splitlines():
                box.label(text=line)
        if settings.report:
            box = layout.box()
            box.label(text='Rig report')
            column = box.column(align=True)
            for line in settings.report.splitlines()[:40]:
                column.label(text=line)


class RGBMOCAP_Properties(bpy.types.PropertyGroup):
    ground_mode: EnumProperty(
        name='Mode',
        items=[('PLANT', 'Plant feet', 'Correct every frame so a contacted sole touches the floor'),
               ('AUTO', 'Auto drop', 'Drop to a low percentile, keeping airborne frames above the floor'),
               ('DROP', 'Drop', 'One offset from the lowest frame of the clip')],
        default='PLANT')
    floor_height: FloatProperty(name='Floor', default=0.0, precision=4, unit='LENGTH',
                                description='World height of the ground plane in metres')
    foot_only: BoolProperty(name='Feet only', default=True,
                            description='Measure foot and toe vertices only')
    ground_smooth: IntProperty(name='Smooth', default=0, min=0, max=30,
                               description='Frames of offset smoothing; 0 keeps exact contact')
    include_mesh: BoolProperty(name='Mesh', default=True,
                               description='Include the skinned mesh in exports')
    live_port: IntProperty(name='Port', default=9187, min=1, max=65535,
                           description='Port RGB MoCap listens on (File > Live Link to Blender)')
    live_status: StringProperty(default='')
    report: StringProperty(default='')
    stats: StringProperty(default='')


CLASSES = (RGBMOCAP_Properties, RGBMOCAP_OT_scan, RGBMOCAP_OT_rename, RGBMOCAP_OT_ground,
           RGBMOCAP_OT_export_fbx, RGBMOCAP_OT_export_glb, RGBMOCAP_OT_import,
           RGBMOCAP_OT_live_start, RGBMOCAP_OT_live_stop, RGBMOCAP_OT_live_record,
           RGBMOCAP_OT_live_record_stop, RGBMOCAP_OT_load_base, RGBMOCAP_PT_panel)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.rgbmocap = bpy.props.PointerProperty(type=RGBMOCAP_Properties)


def unregister():
    live.stop()
    del bpy.types.Scene.rgbmocap
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == '__main__':
    register()
