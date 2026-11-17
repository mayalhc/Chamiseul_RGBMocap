"""Native live mesh/grid pass, independent of offline Qt overlay state."""
import numpy as np
from PySide6.QtGui import QMatrix4x4
from PySide6.QtOpenGL import QOpenGLBuffer,QOpenGLVertexArrayObject,QOpenGLShader
from ..rig_gl import RigRenderer
from ..mesh_gl import MeshRenderer
from ..bone_gl import BoneRenderer
from .preview_bones import PreviewBones

VERTEX='''#version 430 core
layout(location=0) in vec3 position;
layout(location=1) in vec3 tint;
uniform mat4 mvp;
out vec3 lineColor;
void main(){gl_Position=mvp*vec4(position,1);lineColor=tint;}'''
FRAGMENT='''#version 430 core
in vec3 lineColor;out vec4 color;
void main(){color=vec4(lineColor,1);}'''

class LiveRenderer:
    def __init__(self,view):
        self.mesh=None;self.bones=BoneRenderer();self.bone_preview=None;self.bone_key=None;self.bone_revision=None;self.gl=self.bones.gl
        self.program=MeshRenderer.program(((QOpenGLShader.Vertex,VERTEX),(QOpenGLShader.Fragment,FRAGMENT)))
        batches=[]
        for lines,color in ((view.grid_minor,(.27,.27,.27)),(view.grid_major,(.34,.34,.34)),
                            (np.array([[[-10,.001,0],[10,.001,0]]]),(.60,.32,.36)),
                            (np.array([[[0,.001,-10],[0,.001,10]]]),(.30,.46,.61))):
            v=np.asarray(lines,dtype='float32').reshape(-1,3);batches.append(np.column_stack([v,np.tile(color,(len(v),1))]))
        data=np.ascontiguousarray(np.concatenate(batches),dtype='float32');self.count=len(data)
        self.vao=QOpenGLVertexArrayObject();self.vao.create();self.vao.bind()
        self.buffer=QOpenGLBuffer();self.buffer.create();self.buffer.bind();self.buffer.allocate(data.tobytes(),data.nbytes)
        self.program.bind()
        for loc,offset in ((0,0),(1,12)):
            self.program.enableAttributeArray(loc);self.program.setAttributeBuffer(loc,0x1406,offset,3,24)
        self.vao.release();self.program.release()

    def paint(self,view):
        gl=self.gl;gl.glDisable(0x0C11);gl.glDisable(0x0B90);gl.glDisable(0x0BE2)
        gl.glColorMask(True,True,True,True);gl.glDepthMask(True);gl.glStencilMask(0xffffffff)
        gl.glClearColor(.2196,.2196,.2196,1.);gl.glClear(0x4000|0x100|0x400)
        if view.show_grid:
            eye,basis=view._paint_camera;matrix=np.eye(4);matrix[:3,:3]=basis*np.array([[1],[1],[-1]]);matrix[:3,3]=-matrix[:3,:3]@eye
            projection=QMatrix4x4();projection.perspective(40,view.width()/max(1,view.height()),.03,2000)
            self.program.bind();self.program.setUniformValue('mvp',projection*QMatrix4x4(*matrix.reshape(-1).tolist()))
            gl.glDisable(0x0B71);self.vao.bind();gl.glDrawArrays(0x1,0,self.count);self.vao.release();self.program.release()
        if view.live_data is None or view.live_asset is None:return
        if view.live_asset.get('render_mode')!='bones':
            if self.mesh is None:self.mesh=RigRenderer()
            self.mesh.load(view.live_asset);self.mesh.deform(view.live_data['vertices'],view.live_key)
            self.mesh.draw(*view._paint_camera,view.width(),view.height(),(.64,.79,.86))
        surface=getattr(view,'live_surface',None)
        if surface is not None and 'surface' in view.live_data:
            if self.mesh is None:self.mesh=RigRenderer()
            self.mesh.load(surface);self.mesh.deform(view.live_data['surface'],view.live_key)
            self.mesh.draw(*view._paint_camera,view.width(),view.height(),(.64,.79,.86))
        if view.show_bones:
            j=view.live_data['joints'];parents=view.live_parents
            revision=view.live_data.get('rig_revision',0)
            if self.bone_preview is None or revision!=self.bone_revision:
                self.bone_preview=PreviewBones(j,parents,('pear-live',revision));self.bone_key=None;self.bone_revision=revision
            if self.bone_key!=view.live_key:
                self.bone_frames=self.bone_preview.frames(j,view.live_data['bone_facing']);self.bone_key=view.live_key
            self.bones.load(self.bone_preview.asset);self.bones.deform(self.bone_frames,view.live_key)
            self.bones.draw(*view._paint_camera,view.width(),view.height())

    def close(self):
        self.bones.close()
        if self.mesh is not None:self.mesh.close()
        self.buffer.destroy();self.vao.destroy();self.program.removeAllShaders()
