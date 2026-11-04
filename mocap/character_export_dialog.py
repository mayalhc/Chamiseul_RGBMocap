"""Export choices for an already baked character."""
from PySide6.QtWidgets import QDialog,QVBoxLayout,QFormLayout,QComboBox,QCheckBox,QLabel,QDialogButtonBox


class CharacterExportDialog(QDialog):
    def __init__(self,kind,parent=None,default_format='fbx'):
        super().__init__(parent)
        self.setWindowTitle('Export Character')
        self.setMinimumWidth(max(440,self.fontMetrics().horizontalAdvance('M')*32))
        layout=QVBoxLayout(self);form=QFormLayout();layout.addLayout(form)
        self.format=QComboBox();self.format.addItem('FBX (.fbx)','fbx')
        self.format.addItem('GLB (.glb)','glb')
        if kind=='pmx':self.format.addItem('VMD (.vmd)','vmd')
        # File menu entries name a format; preselect it but keep every choice.
        index=self.format.findData(default_format)
        if index>=0:self.format.setCurrentIndex(index)
        form.addRow('Format',self.format)
        self.mesh=QCheckBox('Include character mesh');self.mesh.setChecked(True)
        self.mesh.setToolTip('Checked: save the character mesh and skeleton animation. Unchecked: save only the skeleton and animation.')
        layout.addWidget(self.mesh)
        self.note=QLabel();self.note.setWordWrap(True);layout.addWidget(self.note)
        buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
        self.format.currentIndexChanged.connect(self.update_format);self.update_format()

    def update_format(self):
        style=self.format.currentData();vmd=style=='vmd';self.mesh.setVisible(not vmd)
        notes={'vmd':'Motion only, sampled at 30 fps for the same PMX model. Body and finger animation is saved; facial animation and physics are not generated.',
               'glb':'glTF binary: mesh, skeleton and animation in one file for Blender, Unity, Godot and web viewers. Each take is saved as a separate file.'}
        self.note.setText(notes.get(style,'Save the retargeted motion. Each take is saved as a separate file.'))
        self.layout().activate()
        self.fit_note()
        self.adjustSize()

    def fit_note(self):
        self.note.setMinimumHeight(self.note.heightForWidth(max(1,self.note.width())))

    def resizeEvent(self,event):
        super().resizeEvent(event)
        self.fit_note()

    def options(self):
        style=self.format.currentData()
        return dict(format=style,include_mesh=style in {'fbx','glb'} and self.mesh.isChecked())
