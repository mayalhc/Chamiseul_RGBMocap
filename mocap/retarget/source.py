"""Read one committed source selection without inference or source writes."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from .mapping import PARENTS


def stamp(paths):
    return {str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in paths}


def support_heights(project,kind,weights,n):
    """Lowest sole height per frame, sampled from the registered surface in chunks."""
    from mocap.project_storage import resolve_reference
    project=Path(project);mi=json.loads((project/(kind+'_mesh.json')).read_text(encoding='utf8'))
    with np.load(resolve_reference(project,mi['asset'])) as asset:
        count=int(asset['vertex_count']);packed=asset['packed']
    ids=[np.flatnonzero(weights[:,[7+s,10+s]].sum(1)>.7) for s in (0,1)]
    use=np.unique(np.concatenate(ids));pick=[np.searchsorted(use,ids[s]) for s in (0,1)]
    template=packed[:count*3].reshape(count,3)[use].astype(float)
    shapes=packed[count*3:count*63].reshape(20,count,3)[:,use].astype(float)
    poses=packed[count*63:count*1521].reshape(486,count,3)[:,use].astype(float)
    skinning=packed[count*1521:].reshape(count,55)[use].astype(float)
    frames=np.load(project/mi['frames_file'],mmap_mode='r',allow_pickle=False)
    heights=np.empty((n,2))
    try:
        for a in range(0,n,128):
            f=np.asarray(frames[a:min(n,a+128)],float)
            posed=template+np.einsum('tb,bvc->tvc',f[:,1366:1386],shapes)+np.einsum('tp,pvc->tvc',f[:,880:1366],poses)
            transform=np.einsum('vj,tjk->tvk',skinning,f[:,:880].reshape(-1,55,16)).reshape(-1,len(use),4,4)
            vertices=np.einsum('tvij,tvj->tvi',transform[:,:,:3,:3],posed)+transform[:,:,:3,3]+f[:,-3:,None].transpose(0,2,1)
            for s in (0,1):heights[a:a+len(f),s]=vertices[:,pick[s],1].min(1)
    finally:
        frames._mmap.close()
    return heights


def prepare(project,kind,out):
    import torch
    from mocap.paths import MODELS
    from mocap.motion_tools import RESULTS
    from mocap.project_storage import resolve_reference
    from mocap.source_surface import native_endpoints,smplx_endpoints
    from . import surface
    from mocap.surface_roll import confidence_solve
    from mocap.bone_overlay import reference_frame
    torch.set_num_threads(2)
    project=Path(project);metadata=project/(kind+'_mixamo.json')
    info=json.loads(metadata.read_text(encoding='utf8'))
    fps=float(info.get('fps',30.))
    if not 1<=fps<=240:raise ValueError('Unsupported source frame rate.')
    for key in ('frames_file','asset_file'):
        if Path(info[key]).name!=info[key]:raise ValueError('Invalid source cache path')
    model=json.loads((project/'project.json').read_text(encoding='utf8'))['pipeline']['model']
    filename=RESULTS[kind][0]
    if model=='flowhmr' and kind=='initial':filename='flowhmr_results.pt'
    # FlowHMR result names are recorded in the source rig receipt on old caches.
    if not (project/filename).exists() and model=='flowhmr' and kind=='initial':filename='hmr4d_results.pt'
    paths=[metadata,project/info['frames_file'],project/info['asset_file'],project/filename]
    contacts=project/(kind+'_contacts.npz');correction=project/(kind+'_correction.json')
    if contacts.exists():paths.append(contacts)
    if correction.exists():paths.append(correction)
    if model=='flowhmr':paths.append(project/info['raw_file'])
    else:
        mesh_metadata=project/(kind+'_mesh.json')
        mesh_info=json.loads(mesh_metadata.read_text(encoding='utf8'))
        paths += [mesh_metadata,resolve_reference(project,mesh_info['asset']),project/mesh_info['frames_file']]
    topology='smplh' if model=='flowhmr' else 'smplx'
    if model=='mamma':
        from mocap.mamma_body import body_path
        bodypath=body_path(project)
    else:bodypath=MODELS/('body_models/smplh/neutral/model.npz' if topology=='smplh' else 'body_models/smplx/SMPLX_NEUTRAL.npz')
    paths.append(bodypath)
    before=stamp(paths)
    frames=np.load(paths[1],mmap_mode='r');asset=np.load(paths[2]);n=len(frames)
    if n<2:raise ValueError('Retargeting requires at least two source frames.')
    if n>54000:raise ValueError('Select a shorter take (up to 30 minutes at 30 fps).')
    points=np.array(frames[:,:22,:3,3],float)
    world=R.from_matrix(frames[:,:22,:3,:3].reshape(-1,3,3)).as_matrix().reshape(n,22,3,3)
    rest=np.array(asset['rest'][:22],float)
    params=torch.load(paths[3],map_location='cpu',weights_only=False)
    body=np.load(bodypath,allow_pickle=True)
    if topology=='smplh':
        beta=params['shapes'].numpy().reshape(-1)
        x=params['rot6d'].numpy().reshape(n,52,3,2);u=x[...,0];u=u/np.maximum(np.linalg.norm(u,axis=-1,keepdims=True),1e-12)
        v=x[...,1]-(u*x[...,1]).sum(-1,keepdims=True)*u;v/=np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12)
        native=np.stack([u,v,np.cross(u,v)],-1)[:,:22]
    else:
        d=params['smpl_params_global'];beta=d['betas'].numpy().reshape(-1,d['betas'].shape[-1]).mean(0)
        arrays=[d[k].numpy() for k in ('global_orient','body_pose')]
        if arrays[1].shape[-2:]==(3,3):native=np.concatenate([arrays[0].reshape(n,1,3,3),arrays[1].reshape(n,21,3,3)],1)
        else:native=R.from_rotvec(np.concatenate([arrays[0].reshape(n,1,3),arrays[1].reshape(n,21,3)],1).reshape(-1,3)).as_matrix().reshape(n,22,3,3)
    for j,p in enumerate(PARENTS):
        if p>=0:native[:,j]=native[:,p]@native[:,j]
    native=R.from_matrix(native.reshape(-1,3,3)).as_matrix().reshape(n,22,3,3)
    vertices=body['v_template']+np.einsum('vck,k->vc',body['shapedirs'][:,:,:len(beta)],beta)
    neutral=body['J_regressor']@vertices;weights=body['weights']
    active=np.zeros((n,4),bool)
    if contacts.exists():
        with np.load(contacts) as c:
            for side,a,b,*_ in c['segments']:active[max(0,int(a)):min(n,int(b)),int(side)]=True
    settings=json.loads(correction.read_text(encoding='utf8')) if correction.exists() else {}
    floor=float(settings.get('floor',0.));origin=points[0,0].copy();origin[1]=floor
    # Registered surface evidence is independent of world translation.
    # Surface support heights are only consumed by multi-level correction.
    level=settings.get('mode',settings.get('correction_mode'))=='multi_level'
    heights=None
    if topology=='smplh':
        raw=np.load(project/info['raw_file'],mmap_mode='r')
        ep=native_endpoints(raw)
        if level:
            heights=np.empty((n,2))
            for s in [0,1]:
                ids=np.flatnonzero(weights[:,[7+s,10+s]].sum(1)>.7)
                for a in range(0,n,128):heights[a:a+128,s]=raw[a:a+128,ids,1].min(1)
        raw._mmap.close()
    else:
        ep,_=smplx_endpoints(project,kind,n)
        if level:heights=support_heights(project,kind,weights,n)
    points-=origin
    fitted=confidence_solve(points,native,neutral[:22],ep,fps)[0]
    ep0=surface.endpoints(vertices,topology)[None]
    fitted0=confidence_solve(neutral[None,:22],np.eye(3)[None,None].repeat(22,1),neutral[:22],ep0,30)[0][0]
    F=[];F0=[]
    for h,k in [(1,4),(2,5)]:
        basis=reference_frame(neutral[k]-neutral[h]);F.append(fitted[:,h]@basis);F0.append(fitted0[h]@basis)
    from .solver import local_from_world
    extra={}
    if settings.get('mode',settings.get('correction_mode'))=='multi_level':extra['support_height']=heights-floor
    np.savez(out/'motion-source.npz',names=info['names'][:22],parents=PARENTS,rest=rest,points=points,world=world,local=local_from_world(world),origin=origin,fps=fps,**extra)
    np.savez(out/'motion-source-contacts.npz',active=active,fps=fps)
    np.savez(out/'motion-source-neutral.npz',vertices=vertices,rest=neutral,faces=body['f'],weights=weights,native_world=native)
    if 'fbx_rest_frames' not in asset:raise ValueError('Prepare Preview again to create the current FBX neutral reference.')
    N=asset['fbx_rest_frames'][:22]@asset['fbx_display_frames'][:22].swapaxes(-1,-2)
    np.savez(out/'motion-neutral-calibration.npz',neutral=N)
    np.savez(out/'motion-evidence.npz',frames=np.stack(F,1),neutral_frames=np.stack(F0))
    (out/'sources.json').write_text(json.dumps([dict(label='motion',folder=str(project),info=info)]),encoding='utf8')
    frames._mmap.close();asset.close();body.close()
    if stamp(paths)!=before:raise ValueError('Source motion changed while preparing retargeting. Retry after processing finishes.')
    return dict(source_stats=before,frames=n,fps=fps,origin=origin.tolist(),result=kind,model=model)
