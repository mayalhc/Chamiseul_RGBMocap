"""Known humanoid import and compact, bone-driven viewport data."""
import json
import os
import shutil
import sys
import struct
import tempfile
import uuid
from pathlib import Path
import numpy as np
import bpy
import addon_utils
from .mapping import A, PARENTS, body_mapping, body_trajectory_root
from .file_paths import io_path, save_blend


def report(message):
    print('CHARACTER: '+message, flush=True)


def import_character(filename, out):
    Path(io_path(out)).mkdir(parents=True,exist_ok=True)
    kind={'.fbx':'mixamo','.glb':'glb','.gltf':'glb','.vrm':'vrm','.pmx':'pmx'}.get(filename.suffix.lower())
    if kind is None:raise ValueError('Unsupported character format. Import FBX, GLB/glTF, VRM or PMX.')
    bpy.ops.wm.read_factory_settings(use_empty=True)
    addons=Path(__file__).resolve().parents[2]/'third_party/character_import'
    sys.path.insert(0,str(addons))
    semantic={}
    report('Importing '+filename.name)
    if kind=='mixamo':
        bpy.ops.import_scene.fbx(filepath=str(filename),automatic_bone_orientation=False,use_prepost_rot=True)
    elif kind=='glb':
        # Blender's bundled glTF importer keeps joint names, so the humanoid
        # mapping can match them without a manual bone map.
        bpy.ops.import_scene.gltf(filepath=str(filename))
    elif kind=='vrm':
        addon_utils.enable('io_scene_vrm',default_set=True,persistent=False)
        bpy.ops.import_scene.vrm(filepath=str(filename))
        with open(io_path(filename),'rb') as f:
            if f.read(4)!=b'glTF':raise ValueError('Invalid VRM container')
            f.read(8);length,chunk=struct.unpack('<II',f.read(8))
            if length>32*1024*1024:raise ValueError('VRM metadata is too large')
            gltf=json.loads(f.read(length))
        ext=gltf['extensions']
        if 'VRM' in ext:
            semantic={b['bone']:gltf['nodes'][b['node']]['name'] for b in ext['VRM']['humanoid']['humanBones'] if 'node' in b}
        else:
            semantic={k:gltf['nodes'][b['node']]['name'] for k,b in ext['VRMC_vrm']['humanoid']['humanBones'].items()}
            # Normalize VRM 1's thumb labels to the three physical phalanges
            # used by the common finger mapper (VRM 0 terminology).
            for side in ('left','right'):
                thumb=[semantic.get(side+'Thumb'+part) for part in ('Metacarpal','Proximal','Distal')]
                for part,name in zip(('Proximal','Intermediate','Distal'),thumb):
                    if name:semantic[side+'Thumb'+part]=name
    else:
        addon_utils.enable('mmd_tools',default_set=False,persistent=False)
        bpy.ops.mmd_tools.import_model(filepath=str(filename),types={'MESH','ARMATURE'},scale=.08,log_level='ERROR')
    arms=[o for o in bpy.context.scene.objects if o.type=='ARMATURE']
    if len(arms)!=1:raise ValueError('Import one humanoid armature per character file.')
    arm=arms[0]
    mapping=body_mapping(kind,arm.data.bones.keys(),semantic)
    trajectory_root=body_trajectory_root({b.name:b.parent.name if b.parent else None for b in arm.data.bones},mapping)
    for o in bpy.context.scene.objects:
        o.animation_data_clear()
        if o.type=='MESH':
            for mod in o.modifiers:
                if mod.type=='NODES':mod.show_viewport=False;mod.show_render=False
    for pb in arm.pose.bones:pb.matrix_basis.identity()
    bpy.context.view_layer.update()
    if not any(o.type=='MESH' for o in bpy.context.scene.objects):
        raise ValueError('This file has a skeleton but no mesh. Download the character from Mixamo with Skin (FBX, With Skin) and import that file.')
    # Place the native rest shape at the scene origin, without scaling its bones.
    from mathutils import Vector
    hip=arm.matrix_world@arm.data.bones[mapping[0]].head_local
    meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
    bottom=min((o.matrix_world@Vector(c)).z for o in meshes for c in o.bound_box)
    shift=Vector((-hip.x,-hip.y,-bottom))
    for obj in bpy.context.scene.objects:
        if obj.parent is None:obj.location+=shift
    bpy.context.view_layer.update()
    rest=np.array([arm.matrix_world@arm.data.bones[n].head_local for n in mapping])@A
    limb=np.array([1,2,4,5,7,8,10,11,16,17,18,19,20,21])
    if np.any(np.linalg.norm(rest[limb]-rest[PARENTS[limb]],axis=1)<1e-5):
        raise ValueError('The humanoid rig has zero-length body segments.')
    meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
    count=sum(len(o.data.vertices) for o in meshes)
    if not 0<count<=1_000_000:raise ValueError('Character preview supports 1 to 1,000,000 vertices.')
    rig=dict(kind=kind,armature=arm.name,semantic=semantic,trajectory_root=trajectory_root,bones=[dict(name=b.name,parent=b.parent.name if b.parent else None) for b in arm.data.bones],meshes=[dict(name=o.name,vertices=len(o.data.vertices)) for o in meshes])
    Path(io_path(out/f'{kind}-rig.json')).write_text(json.dumps(rig,ensure_ascii=False),encoding='utf8')
    np.savez(io_path(out/f'{kind}-canonical.npz'),rest=rest,names=mapping,parents=PARENTS)
    from .blender_geometry import register
    report('Registering soles, palms and humanoid proportions')
    register(out,kind,rig,arm,mapping,rest)
    # A file imported from another machine may reference textures that no
    # longer exist; those materials fall back to their base color.
    try:bpy.ops.file.pack_all()
    except RuntimeError:report('Some original textures are missing; their materials use base colors')
    save_blend(out/f'{kind}-rest.blend')
    preview(out,arm,None,1)
    # The live view poses this character directly; a failure here only means
    # the rig is prepared on first live use instead.
    try:write_live_rig(out,kind,arm,mapping)
    except Exception as error:report('Live rig skipped: '+str(error)[:120])
    metadata=dict(kind=kind,name=filename.stem,vertices=count,bones=len(arm.data.bones))
    if kind=='pmx':metadata['mmd_import_scale']=.08
    Path(io_path(out/'import.json')).write_text(json.dumps(metadata),encoding='utf8')
    from .blender_textures import extract
    report('Preparing original-resolution textures')
    extract(out)


PMX_DEFORM = (('足D', 1), ('ひざD', 4), ('足首D', 7), ('足先EX', 10))


def write_live_rig(out, kind, arm, mapping):
    """Rest bone data for the live retarget: one file, no animation, no keys.

    The live view poses the character from numpy every frame, so it only needs
    where each palette bone sits at rest and which of the 22 body joints drives
    it. Palette order is the rest preview's (sorted bone names), so a pose built
    from this file skins ``preview.npz`` directly.
    """
    for pb in arm.pose.bones:
        pb.matrix_basis.identity()
    bpy.context.view_layer.update()
    palette = sorted(b.name for b in arm.data.bones)
    index = {name: i for i, name in enumerate(palette)}
    joint_of = {name: j for j, name in enumerate(mapping)}
    if kind == 'pmx':
        for side, suffix in enumerate(('.L', '.R')):
            for stem, joint in PMX_DEFORM:
                joint_of[stem + suffix] = joint + side
    head = np.array([arm.matrix_world @ arm.data.bones[n].head_local for n in palette]) @ A
    tail = np.array([arm.matrix_world @ arm.data.bones[n].tail_local for n in palette]) @ A
    parents = np.array([index[arm.data.bones[n].parent.name] if arm.data.bones[n].parent else -1 for n in palette], np.int32)
    joints = np.array([joint_of.get(n, -1) for n in palette], np.int32)
    canonical = np.load(io_path(out / (kind + '-canonical.npz')))
    np.savez(io_path(out / 'live-rig.npz'), names=np.asarray(palette), parents=parents, head=head,
             joints=joints, canonical_rest=canonical['rest'], tail=tail, version=np.int32(2))


def live_rig(out):
    """Add live-rig.npz to a character imported before the live retarget existed."""
    out = Path(out)
    kind = json.loads(Path(io_path(out / 'import.json')).read_text(encoding='utf8'))['kind']
    bpy.ops.wm.open_mainfile(filepath=io_path(out / (kind + '-rest.blend')), load_ui=False)
    rig = json.loads(Path(io_path(out / (kind + '-rig.json'))).read_text(encoding='utf8'))
    arm = bpy.data.objects[rig['armature']]
    arm.animation_data_clear()
    mapping = np.load(io_path(out / (kind + '-canonical.npz')))['names'].tolist()
    write_live_rig(out, kind, arm, mapping)
    report('Live rig ready')


def vertex_colors(obj):
    """Sample packed base-color textures once. Runtime uses a bounded color buffer."""
    mesh=obj.data
    result=np.zeros((len(mesh.vertices),4),np.float32);hits=np.zeros(len(mesh.vertices),np.int32)
    materials=[]
    for mat in mesh.materials:
        col=np.array(mat.diffuse_color if mat else (.65,.65,.65,1),float)
        tex=None
        if mat and mat.use_nodes:
            nodes=mat.node_tree.nodes
            bs=next((n for n in nodes if n.type=='BSDF_PRINCIPLED'),None)
            if bs:
                inp=bs.inputs.get('Base Color')
                if inp:
                    col=np.array(inp.default_value)
                    if inp.is_linked:
                        node=inp.links[0].from_node
                        if node.type=='TEX_IMAGE' and node.image:tex=node.image
            if tex is None:
                node=next((n for n in nodes if n.type=='TEX_IMAGE' and n.image and ('shade' not in n.name.lower())),None)
                if node:tex=node.image
        pixels=None
        # A texture whose file is missing has zero size; use the material color.
        if tex and tex.size[0] and tex.size[1] and tex.size[0]*tex.size[1]<=16_777_216:
            pixels=np.empty(tex.size[0]*tex.size[1]*4,np.float32)
            tex.pixels.foreach_get(pixels);pixels=pixels.reshape(tex.size[1],tex.size[0],4)
        materials.append((col,pixels))
    uv=mesh.uv_layers.active
    for polygon in mesh.polygons:
        col,pixels=materials[min(polygon.material_index,len(materials)-1)] if materials else (np.array([.65,.65,.65,1]),None)
        for li in polygon.loop_indices:
            vi=mesh.loops[li].vertex_index;c=col
            if pixels is not None and uv:
                u,v=uv.data[li].uv;c=pixels[int((v%1)*pixels.shape[0])%pixels.shape[0],int((u%1)*pixels.shape[1])%pixels.shape[1]]
            result[vi]+=c;hits[vi]+=1
    result/=np.maximum(hits[:,None],1)
    return result[:,:3]


def preview(out,arm,keyed,n):
    from .selective_fbx import body_dependencies
    metadata=Path(io_path(out/'import.json'))
    kind=json.loads(metadata.read_text(encoding='utf-8'))['kind'] if metadata.exists() else next(p.stem.split('-')[0] for p in Path(io_path(out)).glob('*-rig.json'))
    dynamic=set(keyed or [])|body_dependencies(kind,arm)
    if not keyed:dynamic={b.name for b in arm.data.bones}
    dynamic={name for name in dynamic if name in arm.data.bones}
    palette=sorted(dynamic);index={name:i for i,name in enumerate(palette)}
    # Unkeyed descendants inherit the nearest moving bone's deformation.
    owner={}
    for bone in arm.data.bones:
        b=bone
        while b is not None and b.name not in index:b=b.parent
        owner[bone.name]=index[b.name] if b else len(palette)
    vertices=[];colors=[];faces=[];rows=[];influences=[];offset=0
    for obj in bpy.context.scene.objects:
        if obj.type!='MESH':continue
        mesh=obj.data;mesh.calc_loop_triangles();M=np.array(obj.matrix_world)
        v=np.empty(len(mesh.vertices)*3);mesh.vertices.foreach_get('co',v);v=v.reshape(-1,3)
        vertices.append((v@M[:3,:3].T+M[:3,3])@A);colors.append(vertex_colors(obj))
        faces.extend([[offset+i for i in t.vertices] for t in mesh.loop_triangles]);offset+=len(v)
        for vertex in mesh.vertices:
            weights={}
            for g in vertex.groups:
                name=obj.vertex_groups[g.group].name
                if name in owner:weights[owner[name]]=weights.get(owner[name],0)+g.weight
            total=sum(weights.values())
            if total<=1e-8:weights={len(palette):1.};total=1.
            rows.append((len(influences),len(weights)))
            influences.extend((bone,w/total) for bone,w in weights.items())
    from mocap.mesh_data import adjacency
    vertices=np.concatenate(vertices).astype('float32');faces=np.asarray(faces,np.uint32)
    offsets,neighbors=adjacency(faces,len(vertices))
    np.savez(io_path(out/'preview.npz'),vertices=vertices,colors=np.concatenate(colors),faces=faces,weights=np.asarray(influences,np.float32),rows=np.asarray(rows,np.uint32),offsets=offsets,neighbors=neighbors)
    frames=np.lib.format.open_memmap(io_path(out/'preview-frames.npy'),mode='w+',dtype='float32',shape=(n,len(palette)+1,4,4))
    C=np.eye(4);C[:3,:3]=A.T
    rest_inv={name:(arm.matrix_world@arm.data.bones[name].matrix_local).inverted() for name in palette}
    for i in range(n):
        bpy.context.scene.frame_set(i+1);bpy.context.view_layer.update()
        for j,name in enumerate(palette):
            frames[i,j]=C@np.array(arm.matrix_world@arm.pose.bones[name].matrix@rest_inv[name])@C.T
        frames[i,-1]=np.eye(4)
        if i%30==0:report(f'Preparing animation preview {i+1}/{n}')
    frames.flush();del frames


def export_glb(target):
    """Write the selected armature and meshes as one glTF binary file.

    glTF keeps the armature's rest joints as node transforms and stores the baked
    motion as sampled animation channels, with no armature-object keys, mirroring
    the FBX path. All bones are exported as joints, so terminal finger bones do
    not need the FBX leaf-bone workaround.

    Blender's glTF exporter derives its output folder from the file path and
    rejects the Windows extended prefix, so the file is written to a short
    temporary folder and published to the (possibly long) destination after.
    """
    with tempfile.TemporaryDirectory(prefix='rgbmocap-glb-') as stage:
        staged=Path(stage)/'character.glb'
        bpy.ops.export_scene.gltf(
            filepath=str(staged),export_format='GLB',use_selection=True,export_apply=True,
            export_animations=True,export_animation_mode='ACTIVE_ACTIONS',export_anim_single_armature=True,
            export_force_sampling=True,export_frame_range=True,export_anim_slide_to_zero=True,
            export_optimize_animation_size=False,export_reset_pose_bones=True,
            export_rest_position_armature=True,export_leaf_bone=False,export_def_bones=False,
            export_skins=True,export_yup=True,export_materials='EXPORT',
            export_cameras=False,export_lights=False,export_extras=False)
        pending=Path(io_path(str(target)+'.'+uuid.uuid4().hex+'.partial'))
        try:
            shutil.copyfile(staged,pending)
            os.replace(pending,io_path(target))
        finally:
            pending.unlink(missing_ok=True)


def main(args):
    mode,folder,*rest=args;out=Path(folder)
    if mode=='import':import_character(Path(rest[0]),out)
    elif mode=='liverig':live_rig(out)
    elif mode=='textures':
        kind=json.loads(Path(io_path(out/'import.json')).read_text(encoding='utf8'))['kind']
        if kind not in {'mixamo','glb','vrm','pmx'}:raise ValueError('Unsupported character kind')
        bpy.ops.wm.open_mainfile(filepath=io_path(out/f'{kind}-rest.blend'),load_ui=False)
        from .blender_textures import extract
        extract(out)
    elif mode=='bake':
        from . import blender_bake
        blender_bake.OUT=out
        arm,keyed=blender_bake.bake(rest[0],'motion','stance')
        n=bpy.context.scene.frame_end
        preview(out,arm,keyed,n)
        Path(io_path(out/'bake.json')).write_text(json.dumps(dict(keyed_bones=keyed,frames=n,object_animation_channels=0)),encoding='utf8')
    elif mode=='export':
        from .selective_fbx import body_only_sampler,body_dependencies
        from mathutils import Matrix
        import uuid
        folder=Path(rest[0]);target=Path(rest[1]);transform=json.loads(rest[2])
        options=json.loads(rest[4]) if len(rest)>4 else {}
        fmt=options.get('format','fbx')
        if fmt not in {'fbx','glb','vmd'}:raise ValueError('Unsupported character export format')
        kind=json.loads(Path(io_path(folder/'import.json')).read_text(encoding='utf8'))['kind']
        if fmt=='vmd':
            if kind!='pmx':raise ValueError('VMD export requires a PMX character')
            from .vmd_export import export_vmd
            export_vmd(folder,target,transform,rest[3])
            return
        bpy.ops.wm.open_mainfile(filepath=io_path(folder/'animation.blend'),load_ui=False)
        arm=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
        arm.name=rest[3]
        info=json.loads(Path(io_path(folder/'bake.json')).read_text(encoding='utf8'))
        kind=json.loads(Path(io_path(folder/'import.json')).read_text(encoding='utf8'))['kind']
        if transform is not None:
            C=np.eye(4);C[:3,:3]=A
            M=Matrix((C@np.asarray(transform)@C.T).tolist())
            for obj in bpy.context.scene.objects:
                if obj.parent is None:obj.matrix_world=M@obj.matrix_world
        bpy.ops.object.select_all(action='DESELECT')
        types={'ARMATURE','MESH'} if options.get('include_mesh',True) else {'ARMATURE'}
        for obj in bpy.context.scene.objects:
            if obj.type in types:obj.select_set(True)
        bpy.context.view_layer.objects.active=arm
        if fmt=='glb':
            export_glb(target)
            return
        temp=target.with_name('.'+target.stem+'-'+uuid.uuid4().hex+'.partial.fbx')
        try:
            with body_only_sampler(arm,set(info['keyed_bones'])|body_dependencies(kind,arm)):
                # Explicit terminal markers keep Blender's Ignore Leaf Bones
                # option from removing real terminal finger bones.
                bpy.ops.export_scene.fbx(filepath=io_path(temp),use_selection=True,object_types=types,add_leaf_bones=True,
                    bake_anim=True,bake_anim_use_all_bones=False,bake_anim_use_all_actions=False,bake_anim_use_nla_strips=False,
                    bake_anim_simplify_factor=0.,use_mesh_modifiers=True,path_mode='COPY',embed_textures=True)
            Path(io_path(temp)).replace(io_path(target))
        finally:Path(io_path(temp)).unlink(missing_ok=True)
    else:raise ValueError('Unknown character operation')
