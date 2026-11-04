"""One original-motion SMPL-H scene floor, shared by every actor/take."""
import numpy as np
from .scene_floor import inputs,sample_windows,read_config,signature,VERSION
from .people_store import safe_child
from .storage import atomic_json


def resolve(root,options):
    from .flow_correction import motion_geometry
    from .correction import prepare_contacts
    from .people_store import read_people_json
    context=inputs(root);height=options['floor'];samples=0;warning='';board=None
    method='One common SMPL-H plane; equal-actor median of original sole estimates'
    if options.get('floor_board'):
        from .floor_board import scene_floor as measure
        measured,board=measure(root,context)
        if measured is None:warning='No readable floor reference board in the scene video; the floor was inferred from foot contacts.'
        else:
            height=float(measured)
            method='One common plane measured from the floor reference board in the scene video'
    if options['auto_floor'] and board is None:
        rows=context['actors'];budget=max(2,min(512,8192//max(1,len(rows))));estimates={}
        for row in rows:
            project=safe_child(root,row['project']);params,geometry,roots,quats,confidence=motion_geometry(project)
            fps=float(read_people_json(project/'project.json')['fps']);values=[]
            for first,end in sample_windows(len(roots),budget):
                probes=geometry.sample(roots[first:end],quats[first:end])[0]
                active=prepare_contacts(probes,confidence[first:end],fps,preset='normal')[0]
                heights=probes[...,1].reshape(-1,2,2).min(2)
                if active.any():values.extend(heights[active].tolist())
                samples+=end-first
            if values:estimates.setdefault(row['person'],[]).append(float(np.quantile(values,.05)))
        height=float(np.median([np.median(values) for values in estimates.values()])) if estimates else 0.
        if not estimates:warning='No reliable ground samples. Shared reference floor 0 m is used; adjust it manually if needed.'
    if inputs(root)!=context or read_config(root)!=options:raise ValueError('Shared floor changed; Apply Correction again.')
    record=dict(version=VERSION,options=options,inputs=context,floor=float(height),samples=samples,warning=warning,
        signature=signature(options,context,height),method=method)
    if board is not None:record['floor_board']=board
    atomic_json(root/'scene_floor_resolved.json',record)
    return record
