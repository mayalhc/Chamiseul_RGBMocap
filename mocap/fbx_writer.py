"""Small original binary FBX 7.4 skeleton/animation writer.

No SDK, Blender runtime, third-party implementation or registered body-model
data is embedded. Node defaults carry the registered rest pose; PreRotation
carries its axes. Only Hips has animated translation. End markers preserve
terminal lengths. The optional skinned mesh is either the application's own
rigid display geometry, one bone per vertex, or the displayed body surface with
multi-influence cluster weights and explicit bind matrices.
"""
import struct
import warnings
import zlib
import numpy as np
from scipy.spatial.transform import Rotation
from .source_bvh import continuous_angles

TICKS = 46186158000


def node(name, *props, children=()):
    return name, props, children


def prop(name, kind, *values, flags=''):
    subtype={'int':'Integer','double':'Number'}.get(kind,'')
    if kind in ('int','enum','bool'):values=tuple(np.int32(v) for v in values)
    return node('P', name, kind, subtype, flags, *values)


def properties(*items):
    return node('Properties70', children=items)


def _property(value):
    if isinstance(value, str):
        raw=value.encode('utf-8'); return b'S'+struct.pack('<I',len(raw))+raw
    if isinstance(value, (bool, np.bool_)):
        return b'C'+struct.pack('<?',value)
    if isinstance(value,np.int32):
        return b'I'+struct.pack('<i',value)
    if isinstance(value, (int, np.integer)):
        return b'L'+struct.pack('<q',value)
    if isinstance(value, (float, np.floating)):
        return b'D'+struct.pack('<d',value)
    value=np.ascontiguousarray(value)
    codes={np.dtype('int64'):b'l',np.dtype('int32'):b'i',np.dtype('float64'):b'd',np.dtype('float32'):b'f'}
    if value.dtype not in codes:raise ValueError('Unsupported FBX array')
    raw=zlib.compress(value.tobytes(),1)
    return codes[value.dtype]+struct.pack('<III',value.size,1,len(raw))+raw


def _element(out, item):
    name, props, children=item; name=name.encode('ascii')
    start=out.tell();out.write(b'\0'*13+name)
    prop_start=out.tell()
    for value in props:out.write(_property(value))
    prop_size=out.tell()-prop_start
    for child in children:_element(out,child)
    if children:out.write(b'\0'*13)
    end=out.tell()
    if end>=2**32:raise ValueError('FBX exceeds the 4 GiB format limit; split the clip')
    out.seek(start);out.write(struct.pack('<IIIB',end,len(props),prop_size,len(name)));out.seek(end)


def _euler(matrix):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Rotation.from_matrix(matrix).as_euler('xyz',degrees=True)


def skinned_geometry(skin, rest, rest_frames, display):
    """Place rigid display vertices in the FBX bind pose.

    The viewer draws a display vertex as M(t) @ v, where M is the committed
    animated bone matrix. FBX skinning computes
    NodeWorld(t) @ TransformLink^-1 @ v', and this writer keeps
    NodeWorld(t) = M(t) @ display. With TransformLink set to the bone's rest
    world matrix and v' = W_rest @ display^T @ v both expressions are identical,
    so the exported skin reproduces the displayed geometry exactly.
    """
    vertices=np.asarray(skin['vertices'],float)
    normals=np.asarray(skin['normals'],float)
    faces=np.asarray(skin['faces'],np.int32)
    bones=np.asarray(skin['bones']).astype(int)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or not len(vertices):
        raise ValueError('Invalid FBX skin vertices')
    if normals.shape!=vertices.shape or bones.shape!=(len(vertices),):
        raise ValueError('Invalid FBX skin attributes')
    if faces.ndim!=2 or faces.shape[1]!=3 or not len(faces) or faces.min()<0 or faces.max()>=len(vertices):
        raise ValueError('Invalid FBX skin topology')
    if bones.min()<0 or bones.max()>=len(rest):raise ValueError('Invalid FBX skin bone index')
    if not np.isfinite(vertices).all() or not np.isfinite(normals).all():
        raise ValueError('Invalid FBX skin data')
    rotation=rest_frames[bones]@display[bones].swapaxes(-1,-2)
    placed=np.einsum('vij,vj->vi',rotation,vertices)+rest[bones]
    normals=np.einsum('vij,vj->vi',rotation,normals)
    length=np.linalg.norm(normals,axis=1)
    if (length<1e-9).any():raise ValueError('Invalid FBX skin normal')
    return placed,normals/length[:,None],faces,bones


def bind_matrix(rotation, position):
    """A bone's cluster TransformLink in the mesh's global frame."""
    matrix=np.eye(4);matrix[:3,:3]=np.asarray(rotation,float);matrix[:3,3]=np.asarray(position,float)
    return matrix


def rigid_clusters(skin, rest, rest_frames, display):
    """One full weight per vertex, bound to the skeleton's registered rest."""
    placed,normals,faces,bones=skinned_geometry(skin,rest,rest_frames,display)
    clusters=[]
    for bone in np.unique(bones):
        ids=np.flatnonzero(bones==bone).astype(np.int32)
        clusters.append((int(bone),ids,np.ones(len(ids)),bind_matrix(rest_frames[bone],rest[bone])))
    return placed,normals,faces,clusters


def weighted_clusters(skin):
    """Multi-influence skin whose stored vertices are already in the bind pose.

    Every cluster carries its own bind matrix, so the exported surface follows
    the displayed body exactly even though those bind matrices are not the
    skeleton's own rest frames.
    """
    vertices=np.asarray(skin['vertices'],float)
    normals=np.asarray(skin['normals'],float)
    faces=np.asarray(skin['faces'],np.int32)
    indices=np.asarray(skin['bones']).reshape(-1)
    weights=np.asarray(skin['weights'],float).reshape(-1)
    offsets=np.asarray(skin['offsets']).reshape(-1)
    bind=np.asarray(skin['bind'],float)
    count=len(vertices)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or not count or not np.isfinite(vertices).all():
        raise ValueError('Invalid FBX skin vertices')
    if normals.shape!=vertices.shape or not np.isfinite(normals).all() or np.min(np.linalg.norm(normals,axis=1))<1e-9:
        raise ValueError('Invalid FBX skin normals')
    if faces.ndim!=2 or faces.shape[1]!=3 or not len(faces) or faces.min()<0 or faces.max()>=count:
        raise ValueError('Invalid FBX skin topology')
    if offsets.shape!=(count+1,) or offsets[0]!=0 or offsets[-1]!=len(indices) or np.any(np.diff(offsets)<1):
        raise ValueError('Invalid FBX skin influence offsets')
    if weights.shape!=indices.shape or len(indices)!=offsets[-1]:
        raise ValueError('Invalid FBX skin influence arrays')
    if len(indices) and (indices.min()<0 or indices.max()>=len(bind)):
        raise ValueError('Invalid FBX skin bone index')
    if not np.isfinite(weights).all() or (weights<=0).any() or (weights>1).any():
        raise ValueError('Invalid FBX skin weight')
    if np.max(abs(np.add.reduceat(weights,offsets[:-1])-1))>1e-5:
        raise ValueError('FBX skin weights must sum to one per vertex')
    if bind.ndim!=3 or bind.shape[1:]!=(4,4) or not np.isfinite(bind).all():
        raise ValueError('Invalid FBX skin cluster bind matrices')
    for matrix in bind:
        rotation=matrix[:3,:3]
        if np.max(abs(rotation.T@rotation-np.eye(3)))>1e-5 or np.linalg.det(rotation)<0.99999:
            raise ValueError('Invalid FBX skin cluster bind rotation')
    owner=np.repeat(np.arange(count),np.diff(offsets))
    clusters=[]
    for bone in np.unique(indices):
        selected=np.flatnonzero(indices==bone)
        clusters.append((int(bone),owner[selected].astype(np.int32),weights[selected],bind[bone]))
    return vertices,normals,faces,clusters


def mesh_elements(rig_name, names, skin, rest, rest_frames, display):
    """Geometry, Mesh model, Skin deformer and one weighted cluster per used bone."""
    if skin.get('weights') is None:
        placed,normals,faces,clusters=rigid_clusters(skin,rest,rest_frames,display)
    else:
        placed,normals,faces,clusters=weighted_clusters(skin)
    polygons=faces.copy();polygons[:,2]=~polygons[:,2]
    # No NodeAttribute: importers attach the connected Geometry as the mesh data,
    # and a Mesh attribute would replace it with an empty object.
    geometry_id,mesh_id,skin_id=7000,7100,7300
    objects=[node('Geometry',geometry_id,rig_name+'Mesh\0\1Geometry','Mesh',children=[
        properties(),node('GeometryVersion',124),
        node('LayerElementNormal',0,children=[node('Version',101),node('Name',''),
            node('MappingInformationType','ByPolygonVertex'),node('ReferenceInformationType','Direct'),
            node('Normals',np.ascontiguousarray(normals[faces].reshape(-1)))]),
        node('Layer',0,children=[node('Version',100),
            node('LayerElement',children=[node('Type','LayerElementNormal'),node('TypedIndex',0)])]),
        node('Vertices',np.ascontiguousarray(placed.reshape(-1))),
        node('PolygonVertexIndex',np.ascontiguousarray(polygons.reshape(-1)))]),
        node('Model',mesh_id,rig_name+'Mesh\0\1Model','Mesh',children=[node('Version',232),properties(
            prop('Lcl Translation','Lcl Translation',0.,0.,0.,flags='A'),
            prop('Lcl Rotation','Lcl Rotation',0.,0.,0.,flags='A'),
            prop('Lcl Scaling','Lcl Scaling',1.,1.,1.,flags='A'),
            prop('InheritType','enum',1))]),
        node('Deformer',skin_id,'skin\0\1SubDeformer','Skin',children=[node('Version',101),node('Link_DeformAcuracy',50.)])]
    connections=[node('C','OO',geometry_id,mesh_id),
        node('C','OO',skin_id,geometry_id),node('C','OO',mesh_id,900)]
    used=0;influences=0
    for j,ids,weights,bind in clusters:
        used+=1;influences+=len(ids)
        cluster=7400+j
        objects.append(node('Deformer',cluster,names[j]+'\0\1SubDeformer','Cluster',children=[
            node('Version',100),node('UserData','',''),
            node('Indexes',np.ascontiguousarray(ids.astype(np.int32))),
            node('Weights',np.ascontiguousarray(weights.astype(np.float64))),
            node('Transform',np.ascontiguousarray(np.linalg.inv(bind).T.reshape(-1))),
            node('TransformLink',np.ascontiguousarray(bind.T.reshape(-1)))]))
        connections.append(node('C','OO',cluster,skin_id))
        connections.append(node('C','OO',1000+j,cluster))
    return objects,connections,dict(vertices=len(placed),polygons=len(faces),clusters=used,influences=influences)


def write(path, names, parents, rest, rest_frames, display, tip_local, frames, fps,
          transform=None, progress=lambda *args:None, rig_name='SourceAnimation', skin=None, prepend_tpose=False,symmetric_bind=None,
          expected_bones=52):
    """Write one committed mapped take; working rotation arrays are one bone wide."""
    from .person_names import validate_name
    rig_name=validate_name(rig_name)
    n=len(frames);parents=np.asarray(parents);rest=np.asarray(rest,float)
    if n<1 or frames.shape!=(n,expected_bones,4,4) or not np.isfinite(fps) or fps<=0:
        raise ValueError('Invalid FBX frame dimensions/timing')
    if len(names)!=expected_bones or len(set(names))!=expected_bones or names[0]!='Hips' or parents.shape!=(expected_bones,) or parents[0]!=-1:
        raise ValueError(f'Expected {expected_bones}-bone Hips hierarchy')
    if any(not name.isascii() or any(c in name for c in ':\0\1') for name in names):
        raise ValueError('Invalid FBX bone name')
    if rest.shape!=(expected_bones,3) or np.shape(rest_frames)!=(expected_bones,3,3) or np.shape(display)!=(expected_bones,3,3) or np.shape(tip_local)!=(expected_bones,3):
        raise ValueError('Invalid FBX rest dimensions')
    if any(not 0<=parents[j]<j for j in range(1,expected_bones)):
        raise ValueError('Invalid FBX parents')
    for a in (rest,rest_frames,display,tip_local):
        if not np.isfinite(a).all():raise ValueError('Invalid FBX rest data')
    for a in (rest_frames,display):
        if np.max(abs(a.swapaxes(-1,-2)@a-np.eye(3)))>1e-5 or np.linalg.det(a).min()<.99999:
            raise ValueError('Invalid FBX rest rotation')
    source_display=display
    source_offsets=[rest[j] if p<0 else rest_frames[p].T@(rest[j]-rest[p]) for j,p in enumerate(parents)]
    if symmetric_bind is not None:
        from .export_options import symmetric_rest
        from .bone_overlay import reference_frame
        rest,ends=symmetric_rest(names,parents,symmetric_bind['rest'],symmetric_bind['ends'])
        display=[]
        for j in range(52):
            children=np.flatnonzero(parents==j)
            delta=rest[children].mean(0)-rest[j] if len(children) else ends[j]
            if np.linalg.norm(delta)<.001:delta=delta+[0.,.1,0.]
            display.append(reference_frame(delta))
        display=np.asarray(display);rest_frames=display
        tip_local=np.einsum('nji,nj->ni',display,ends)
    from .scene_camera import validate_transform
    transform=validate_transform(np.eye(4) if transform is None else transform)
    ids=np.arange(1000,1000+expected_bones);objects=[];connections=[]
    link=lambda a,b,p=None:connections.append(node('C','OO' if p is None else 'OP',int(a),int(b),*(() if p is None else (p,))))
    # Empty parent is an FBX armature container, not an extra animated bone.
    objects.append(node('Model',900,rig_name+'\0\1Model','Null',children=[node('Version',232),properties()]))
    link(900,0)
    # The exported node world is the displayed basis of each bone, so the rest used
    # for PreRotation and for the animation deltas carries the same display basis.
    display=np.asarray(display,float)
    rest_display=display if symmetric_bind is not None else np.asarray(rest_frames,float)@display
    local_rest=[];offsets=[]
    for j,p in enumerate(parents):
        R=rest_display[j] if p<0 else rest_display[p].T@rest_display[j]
        t=rest[j] if p<0 else rest_frames[p].T@(rest[j]-rest[p])
        local_rest.append(R);offsets.append(t)
        children=[node('Version',232),properties(
            prop('Lcl Translation','Lcl Translation',*map(float,t),flags='A'),
            prop('Lcl Rotation','Lcl Rotation',0.,0.,0.,flags='A'),
            prop('Lcl Scaling','Lcl Scaling',1.,1.,1.,flags='A'),
            prop('PreRotation','Vector3D',*map(float,_euler(R))),
            prop('RotationActive','bool',True),prop('RotationOrder','enum',0),
            prop('InheritType','enum',1))]
        objects.append(node('Model',int(ids[j]),names[j]+'\0\1Model','LimbNode',children=children))
        objects.append(node('NodeAttribute',2000+j,names[j]+'\0\1NodeAttribute','LimbNode',children=[node('TypeFlags','Skeleton')]))
        link(2000+j,ids[j]);link(ids[j],900 if p<0 else ids[p])
    for j in range(expected_bones):
        if j in parents:continue
        marker=3000+j
        objects.append(node('Model',marker,names[j]+'_end\0\1Model','LimbNode',children=[
            node('Version',232),properties(prop('Lcl Translation','Lcl Translation',*map(float,tip_local[j]),flags='A'))]))
        objects.append(node('NodeAttribute',4000+j,names[j]+'_end\0\1NodeAttribute','LimbNode',children=[node('TypeFlags','Skeleton')]))
        link(4000+j,marker);link(marker,ids[j])
    if skin is not None:
        mesh,mesh_links,counts=mesh_elements(rig_name,names,skin,rest,rest_frames,display)
        objects.extend(mesh);connections.extend(mesh_links)
    # Store an explicit bind-pose table in addition to the node defaults.
    pose_nodes=[node('PoseNode',children=[node('Node',900),node('Matrix',np.eye(4).T.reshape(-1))])]
    for j in range(expected_bones):
        M=np.eye(4);M[:3,:3]=rest_display[j];M[:3,3]=rest[j]
        pose_nodes.append(node('PoseNode',children=[node('Node',int(ids[j])),node('Matrix',M.T.reshape(-1))]))
    objects.append(node('Pose',5000,'BindPose\0\1Pose','BindPose',children=[node('Type','BindPose'),node('Version',100),node('NbPoseNodes',len(pose_nodes)),*pose_nodes]))
    # Zero-based file times import at frame 1 with Blender's default offset 1.
    key_count=n+int(bool(prepend_tpose))
    times=np.rint(np.arange(key_count)*TICKS/fps).astype(np.int64)
    stop=int(times[-1]);stack=6000;layer=6001
    objects.append(node('AnimationStack',stack,'Motion\0\1AnimStack','',children=[properties(
        prop('LocalStart','KTime',0),prop('LocalStop','KTime',stop),prop('ReferenceStart','KTime',0),prop('ReferenceStop','KTime',stop))]))
    objects.append(node('AnimationLayer',layer,'BaseLayer\0\1AnimLayer','',children=[properties()]))
    link(layer,stack)
    settings=properties(*(prop(k,'int',v) for k,v in [('UpAxis',1),('UpAxisSign',1),('FrontAxis',2),('FrontAxisSign',1),('CoordAxis',0),('CoordAxisSign',1)]),
        prop('UnitScaleFactor','double',100.),prop('OriginalUnitScaleFactor','double',100.),
        prop('TimeMode','enum',14),prop('CustomFrameRate','double',float(fps)),
        prop('TimeSpanStart','KTime',0),prop('TimeSpanStop','KTime',stop))
    # Stream nodes/curves individually so serialized animation never accumulates.
    with open(path,'xb') as out:
        out.write(b'Kaydara FBX Binary  \0\x1a\0'+struct.pack('<I',7400))
        _element(out,node('FBXHeaderExtension',children=[node('FBXHeaderVersion',1003),node('FBXVersion',7400),node('Creator','RGB MoCap')]))
        _element(out,node('GlobalSettings',children=[node('Version',1000),settings]))
        _element(out,node('Documents',children=[node('Count',1),node('Document',800,'Scene','Scene',children=[properties(),node('RootNode',0)])]))
        _element(out,node('References'))
        _element(out,node('Definitions',children=[node('Version',100),node('Count',len(objects))]))
        start=out.tell();out.write(b'\0'*13+b'Objects')
        for item in objects:_element(out,item)
        nextid=10000
        def curves(j,kind,values):
            nonlocal nextid
            cid=nextid;nextid+=1
            _element(out,node('AnimationCurveNode',cid,('T' if kind.endswith('Translation') else 'R')+'\0\1AnimCurveNode','',children=[properties(*(prop('d|'+a,'Number',0.) for a in 'XYZ'))]))
            link(cid,layer);link(cid,ids[j],kind)
            for axis,a in enumerate('XYZ'):
                uid=nextid;nextid+=1
                _element(out,node('AnimationCurve',uid,'\0\1AnimCurve','',children=[node('Default',float(values[0,axis])),node('KeyVer',4008),
                    node('KeyTime',times),node('KeyValueFloat',values[:,axis].astype(np.float32)),
                    node('KeyAttrFlags',np.array([4],np.int32)),node('KeyAttrDataFloat',np.zeros(4,np.float32)),node('KeyAttrRefCount',np.array([key_count],np.int32))]))
                link(uid,cid,'d|'+a)
        for j,p in enumerate(parents):
            world=np.asarray(frames[:,j,:3,:3],float)@display[j]
            if p<0:
                local=transform[:3,:3]@world
                positions=np.asarray(frames[:,0,:3,3],float)@transform[:3,:3].T+transform[:3,3]
                if not np.isfinite(positions).all():raise ValueError('Non-finite FBX root')
                if prepend_tpose:positions=np.concatenate((positions[:1],positions))
                curves(j,'Lcl Translation',positions)
            else:
                parent=np.asarray(frames[:,p,:3,:3],float)@display[p]
                local=parent.swapaxes(-1,-2)@world
                source_parent=np.asarray(frames[:,p,:3,:3],float)@source_display[p]
                actual=np.einsum('tji,tj->ti',source_parent,np.asarray(frames[:,j,:3,3],float)-frames[:,p,:3,3])
                if not np.isfinite(actual).all() or np.max(abs(actual-source_offsets[j]))>2e-5:
                    raise ValueError('Non-rigid bone offset in committed source animation')
            if not np.isfinite(local).all():raise ValueError('Non-finite FBX animation')
            if prepend_tpose:
                # Identity BVH world frames describe the straight-arm export bind.
                # Display axes transport that T-pose into the FBX bone basis.
                pose=transform[:3,:3]@display[j] if p<0 else display[p].T@display[j]
                local=np.concatenate((pose[None],local))
            angles=np.degrees(continuous_angles(np.radians(_euler(np.asarray(local_rest[j]).T@local))))
            curves(j,'Lcl Rotation',angles);progress(j+1,expected_bones)
        out.write(b'\0'*13);end=out.tell();out.seek(start);out.write(struct.pack('<IIIB',end,0,0,7));out.seek(end)
        _element(out,node('Connections',children=connections))
        _element(out,node('Takes',children=[node('Current','Motion')]))
        out.write(b'\0'*13)
        out.write(bytes.fromhex('fabcab09d0c8d466b176fb831cf7267e'))
        out.write(b'\0'*((-out.tell())%16));out.write(struct.pack('<I',7400));out.write(b'\0'*120)
        out.write(bytes.fromhex('f85a8c6adef5d97eece90ce3758f290b'))

