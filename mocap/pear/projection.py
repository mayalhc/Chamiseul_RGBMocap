"""Crop-camera -> fixed assumed full-image camera, without changing body scale.

The root's reprojection is exact for the official crop projection. The inferred
translation uses weak-perspective depth lifting and an assumed focal length;
it is NOT calibrated world trajectory or a metric camera measurement.
"""
import numpy as np

def lock_position(pose,root):
    """Translate a finished preview uniformly; never change rotations or inference."""
    root=np.asarray(root,dtype=np.float32)
    if root.shape!=(3,) or not np.isfinite(root).all():raise ValueError('Invalid fixed pelvis position')
    result=dict(pose);delta=root-pose['joints'][0]
    for name in ('joints','bounds','vertices','surface'):
        if name in pose:result[name]=np.ascontiguousarray(pose[name]+delta,dtype=np.float32)
    result['joints'][0]=root
    return result
def processed_bbox(raw,width,height):
    x,y,x2,y2=np.asarray(raw,dtype=np.float64);w=x2-x;h=y2-y
    x1=max(0,x);y1=max(0,y);x2=min(width-1,x1+max(0,w-1));y2=min(height-1,y1+max(0,h-1))
    if not (w*h>0 and x2>x1 and y2>y1):raise ValueError('Invalid person bbox')
    w=x2-x1;h=y2-y1;cx=x1+w/2;cy=y1+h/2;side=max(w,h)*1.25
    return np.array([cx-side/2,cy-side/2,side,side],dtype=np.float32)
def full_camera(root,camera,crop,image_size,focal_factor=1.2):
    root=np.asarray(root,dtype=np.float64);camera=np.asarray(camera,dtype=np.float64)
    width,height=np.asarray(image_size,dtype=np.float64);x,y,w,h=np.asarray(crop,dtype=np.float64)
    if abs(w-h)>1e-3:raise ValueError('Expected the official square PEAR crop')
    q=camera[:3,:3]@root+camera[:3,3]
    if not np.isfinite(q).all() or q[2]<=.05:raise ValueError('Invalid crop camera depth')
    # PEAR GS_Camera: crop pixel = 128 * (1 - 24 * camera_xy / camera_z).
    uv=np.array([x+w/2,y+h/2])-12*w*q[:2]/q[2]
    focal=float(max(width,height)*focal_factor);depth=q[2]*focal/(12*w)
    placed=np.r_[(np.array([width,height])/2-uv)/focal*depth,depth]
    translation=placed-camera[:3,:3]@root
    return translation,placed,uv,focal
class Placement:
    def __init__(self):self.reset()
    def reset(self):self.origin=None;self.reset_filter()
    def reset_filter(self):self.root_filter=None;self.last_time=None;self.last_smoothing=None
    def points(self,vertices,joints,camera,crop,image_size,smoothing=False,timestamp=None,position_offset=None,strength=50.):
        t,root,uv,f=full_camera(joints[0],camera,crop,image_size)
        def basis(a):return np.ascontiguousarray(a[:,[0,2,1]]*np.array([1.,-1.,1.]),dtype=np.float32)
        v=basis(vertices@camera[:3,:3].T+t);j=basis(joints@camera[:3,:3].T+t)
        if position_offset is not None:
            v+=position_offset;j+=position_offset
        elif smoothing:
            from .temporal_filter import AdaptiveFilter,strength_scale
            if self.root_filter is None or not self.last_smoothing:self.root_filter=AdaptiveFilter(1.2,4.,vector=True)
            scale=strength_scale(strength)
            self.root_filter.min_cutoff=1.2/scale;self.root_filter.beta=4./scale;self.root_filter.max_cutoff=8./scale
            dt=float(timestamp)-self.last_time if timestamp is not None and self.last_time is not None else 0.
            filtered=self.root_filter.value.astype(np.float32) if dt==0 and self.root_filter.value is not None else self.root_filter.update(j[0],dt)
            delta=filtered-j[0];v+=delta;j+=delta
        self.last_smoothing=smoothing;self.last_time=float(timestamp) if timestamp is not None else None
        if self.origin is None:
            # One constant origin/initial ground reference. No per-frame root subtraction.
            self.origin=j[0].copy();self.origin[2]=float(v[:,2].min())
        return v-self.origin,j-self.origin
