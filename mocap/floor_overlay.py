"""Bounded, display-only correction plane in viewport world coordinates."""
import numpy as np

COLOR = '#e8b66b'
_ticks = np.arange(-1., 1.001, .25)
LINES = np.asarray([segment for t in _ticks for segment in
                    (([t,0,-1],[t,0,1]),([-1,0,t],[1,0,t]))],float)
CORNERS = np.asarray([[-1,0,-1],[1,0,-1],[1,0,1],[-1,0,1]],float)


def geometry(center,height,grid_height,half=(1.,1.)):
    """half: half the plate's size along X and Z, so it can cover a whole clip."""
    offset=np.array([center[0],height,center[2]],float)
    scale=np.array([float(half[0]),1.,float(half[1])])
    corners=CORNERS*scale+offset
    # Keep the ruler beside the actor rather than at a corner that is often
    # outside a tightly framed viewport.
    a=offset+[.55,0,.1];b=a.copy();b[1]=grid_height
    ruler=np.asarray([[a,b],[a+[-.06,0,0],a+[.06,0,0]],
                      [b+[-.06,0,0],b+[.06,0,0]]])
    return corners,LINES*scale+offset,ruler


def clip_polygon(points,eye,basis,near=.08):
    """Clip before projection, including cameras below/inside the plane."""
    view=(np.asarray(points)-eye)@basis.T
    clipped=[]
    for a,b in zip(view,np.roll(view,-1,axis=0)):
        inside_a,inside_b=a[2]>=near,b[2]>=near
        if inside_a:clipped.append(a)
        if inside_a!=inside_b:clipped.append(a+(b-a)*((near-a[2])/(b[2]-a[2])))
    return np.asarray(clipped).reshape(-1,3)


def label(height,grid,automatic,pending,enabled):
    delta=height-grid
    relation='at grid' if abs(delta)<.0005 else f'{abs(delta)*100:.1f} cm '+('above grid' if delta>0 else 'below grid')
    title=f'Correction floor  Y {height:+.3f} m | {relation} | '+('Auto' if automatic else 'Manual')
    status=('Apply Correction to update motion' if pending else 'Applied correction floor') if enabled else 'Correction off | Reference plane only'
    return title,status
