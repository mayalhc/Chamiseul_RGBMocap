"""Imported targets are distinct from source people and extraction selections."""
import json,os,sys,time,uuid,shutil,codecs
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt,QProcess,QTimer,QProcessEnvironment
from PySide6.QtWidgets import QFileDialog,QListWidgetItem,QMenu,QMessageBox
from .paths import ROOT,HOME
from .storage import atomic_json
from .character_data import CharacterLoader

class CharacterWorkspace:
    def init_characters(self):
        self.character_root=None;self.characters=[];self.character_assets={};self.character_process=None
        self.character_refresh_queue=[];self.character_refresh_total=0
        self.character_session=HOME/'character_scenes'/uuid.uuid4().hex
        self.character_loader=CharacterLoader(self)
        self.character_loader.ready.connect(self.character_ready)
        self.character_loader.evicted.connect(lambda key:self.character_assets.pop(key,None))
        self.character_loader.changed.connect(self.motion_view.update)
        self.character_loader.failed.connect(lambda key,msg:self.status.setText('Character preview: '+msg))
        self.character_loader.preparing.connect(lambda msg:self.status.setText(msg) if self.character_process is None else None)
        self.character_loader.start()
        self.character_timer=QTimer(self);self.character_timer.setInterval(500)
        self.character_timer.timeout.connect(self.character_progress)
        self.people_list.characterMenuRequested.connect(self.character_menu)
        self.motion_view.character_workspace=self

    def sync_characters(self):
        root=Path(self.workspace_root or self.project or self.character_session)
        if root!=self.character_root:
            pending=list(self.characters) if self.character_root==self.character_session else []
            blocked=self.people_list.blockSignals(True)
            for i in range(self.people_list.count()-1,-1,-1):
                if isinstance(self.people_list.item(i).data(Qt.UserRole),str):self.people_list.takeItem(i)
            self.people_list.blockSignals(blocked)
            self.character_loader.reset();self.character_assets.clear();self.motion_view.clear_characters()
            self.character_root=root;self.characters=[]
            p=root/'characters/index.json'
            if p.is_file():
                try:
                    data=json.loads(p.read_text(encoding='utf8'))
                    if len(data)>8:raise ValueError('Too many imported characters (maximum 8).')
                    for c in data:
                        self.character_path(c['asset'])
                        if c.get('animation'):
                            self.character_path(c['animation'])
                            for take in c.get('takes',[]):self.character_path(c['animation']+'/'+take['folder'])
                    self.characters=data
                except (ValueError,OSError,KeyError) as exc:self.status.setText('Cannot load characters: '+str(exc))
            for c in self.characters:self.request_character(c)
            if pending and len(pending)+len(self.characters)>8:
                self.status.setText('Imported session characters were kept separately: the project would exceed 8 characters.')
                pending=[]
            if pending:
                folder='import-'+uuid.uuid4().hex
                QTimer.singleShot(0,lambda:self.start_character_job(dict(mode='adopt',
                    output=str(self.character_path(folder)),prefix=folder,characters=pending,
                    source=str(self.character_session/'characters'))))
        self.append_character_rows()

    def character_path(self,relative):
        base=(self.character_root/'characters').resolve();p=(base/relative).resolve()
        if not p.is_relative_to(base):raise ValueError('Invalid character project path')
        return p

    def save_characters(self):atomic_json(self.character_root/'characters/index.json',self.characters)

    def selected_character(self):
        item=self.people_list.currentItem()
        key=item.data(Qt.UserRole) if item else None
        return next((c for c in getattr(self,'characters',[]) if key=='character:'+c['id']),None)

    def export_character_from_menu(self,default_format='fbx'):
        """File > Export Character: the selected row, or the only retargeted one."""
        if self.busy:return
        c=self.selected_character()
        if c is None or not c.get('takes'):
            exportable=[row for row in getattr(self,'characters',[]) if row.get('takes')]
            c=exportable[0] if len(exportable)==1 else None
        if c is None:
            self.status.setText('Select a retargeted character in the People list, then export it again.')
            return
        self.export_character(c,default_format)

    def export_character(self,c,default_format='fbx'):
        if not c.get('takes') or self.busy:return
        if c.get('needs_retarget'):
            self.status.setText('Character motion is out of date. Apply Correction or retarget it again.');return
        from .person_names import export_stem
        from .character_export_dialog import CharacterExportDialog
        dialog=CharacterExportDialog(c['kind'],self,default_format)
        accepted=dialog.exec();options=dialog.options();dialog.deleteLater()
        if not accepted:return
        extension=options['format'];label=extension.upper()
        stem=export_stem(c['name']);takes=c['takes']
        if len(takes)==1:
            name,_=QFileDialog.getSaveFileName(self,'Export Character',stem+'.'+extension,f'{label} (*.{extension})')
            if not name:return
            if Path(name).suffix.lower()!='.'+extension:
                name+='.'+extension
                if Path(name).exists() and QMessageBox.question(self,'Export Character','Replace the existing file?')!=QMessageBox.Yes:return
            files=[name]
        else:
            name=QFileDialog.getExistingDirectory(self,'Export Character Takes')
            if not name:return
            files=[str(Path(name)/(stem+f'_take_{i+1}.{extension}')) for i in range(len(takes))]
            if any(Path(p).exists() for p in files) and QMessageBox.question(self,'Export Character','Replace existing files with these names?')!=QMessageBox.Yes:return
        self.start_character_job(dict(mode='export',output=str(self.character_path('export-'+uuid.uuid4().hex)),
            animation=str(self.character_path(c['animation'])),takes=takes,files=files,source_name=c['name'],**options))

    def append_character_rows(self):
        if not hasattr(self,'characters'):return
        blocked=self.people_list.blockSignals(True)
        try:
            existing={self.people_list.item(i).data(Qt.UserRole):self.people_list.item(i) for i in range(self.people_list.count())}
            for c in self.characters:
                key='character:'+c['id'];item=existing.get(key)
                if item is None:
                    item=QListWidgetItem();item.setData(Qt.UserRole,key);item.setData(Qt.UserRole+1,'#69c9b8')
                    item.setFlags(item.flags()|Qt.ItemIsUserCheckable);self.people_list.addItem(item)
                item.setText(c['name']+(' · Needs retarget' if c.get('needs_retarget') else ''));item.setCheckState(Qt.Checked if c.get('visible',True) else Qt.Unchecked)
                item.setToolTip('Character · '+c['kind'].upper()+'\nRight-click > Retarget to choose a source person. Check to show; uncheck to hide.\n'+
                    (('Baked from '+c.get('source_name','source')+' ('+c.get('result','')+'). Apply Correction also updates linked characters. Retarget again after replacing source motion.') if c.get('takes') else 'Rest pose. No motion assigned.'))
        finally:self.people_list.blockSignals(blocked)

    def character_checked(self,item):
        key=item.data(Qt.UserRole)
        if not isinstance(key,str) or not key.startswith('character:'):return False
        for c in self.characters:
            if c['id']==key.split(':',1)[1]:c['visible']=item.checkState()==Qt.Checked
        self.save_characters();self.motion_view.update();return True

    def choose_character(self):
        if self.busy:return
        self.sync_characters()
        if len(self.characters)>=8:
            QMessageBox.information(self,'Import Character','Up to 8 characters are supported per project.');return
        name,_=QFileDialog.getOpenFileName(self,'Import Character','','Humanoid characters (*.fbx *.glb *.gltf *.vrm *.pmx)')
        if not name:return
        ident=uuid.uuid4().hex
        self.start_character_job(dict(mode='import',file=name,output=str(self.character_path(ident+'/asset')),id=ident))

    def character_sources(self):
        from .motion_tools import result_is_current
        from .people_store import safe_child
        if self.people_state:
            people=self.people_state['people']
        elif self.project:people=[dict(id=1,name=(self._single_actor or {}).get('name','Person 1'),takes=[dict(project=str(self.project),start=self.source_offset)])]
        else:people=[]
        result=[]
        for p in people:
            entries=p.get('takes',[])
            paths=[safe_child(self.people_root,t['project']) if self.people_state else Path(t['project']) for t in entries]
            # A selected variant without a committed preview must not hide the
            # source motion a character can be retargeted to.
            kind=self.displayable_result(paths[0]) if paths else self.displayable_result()
            takes=[]
            for path in paths:
                metadata=path/(kind+'_mixamo.json')
                if not metadata.is_file() or not result_is_current(path,kind):continue
                manifest=json.loads((path/'project.json').read_text(encoding='utf8'))
                start=float(manifest.get('start',0.))
                # Preview scene's stored camera transform is the authoritative shared world.
                transform=None
                if self.scene_data:
                    st=self.scene_data.by_project.get(path.resolve())
                    if st is not None:transform=st.transform.tolist()
                takes.append(dict(project=str(path),start=start,transform=transform))
            if takes:result.append(dict(name=p['name'],kind=kind,takes=sorted(takes,key=lambda t:t['start'])))
        return result
    def character_menu(self,ident,position):
        if self.busy:return
        c=next((c for c in self.characters if 'character:'+c['id']==ident),None)
        if c is None:return
        menu=QMenu(self);retarget=menu.addMenu('Retarget')
        sources=self.character_sources()
        if not sources:retarget.addAction('No prepared source motion').setEnabled(False)
        for source in sources:
            action=retarget.addAction(source['name'])
            action.triggered.connect(lambda checked=False,s=source:self.retarget_character(c,s))
        export=menu.addAction('Export Character...');export.setEnabled(bool(c.get('takes')) and not c.get('needs_retarget'))
        export.triggered.connect(lambda:self.export_character(c))
        menu.addSeparator()
        remove=menu.addAction('Remove Character')
        try:
            if menu.exec(position)==remove:self.remove_character(c['id'])
        finally:menu.deleteLater()

    def remove_character(self,ident):
        if self.busy:return
        remaining=[c for c in self.characters if c['id']!=ident]
        if len(remaining)==len(self.characters):return
        # Commit before changing the UI; a failed save must leave the row usable.
        try:atomic_json(self.character_root/'characters/index.json',remaining)
        except OSError as exc:
            self.status.setText('Cannot remove character: '+str(exc));return
        self.characters=remaining
        blocked=self.people_list.blockSignals(True)
        try:
            key='character:'+ident
            current=self.people_list.currentItem()
            selected=current is not None and current.data(Qt.UserRole)==key
            # takeItem changes the current row synchronously. Without blocking,
            # person_selected can reload/clear this same list inside Qt's removal,
            # deleting the item while Qt still uses it (native access violation).
            for row in range(self.people_list.count()-1,-1,-1):
                if self.people_list.item(row).data(Qt.UserRole)==key:
                    self.people_list.takeItem(row)
            if selected:
                active=getattr(self,'active_person_id',None)
                items=[self.people_list.item(i) for i in range(self.people_list.count())]
                fallback=next((item for item in items if item.data(Qt.UserRole)==active),None)
                if fallback is None:
                    fallback=next((item for item in items if isinstance(item.data(Qt.UserRole),str)),None)
                self.people_list.setCurrentItem(fallback)
        finally:self.people_list.blockSignals(blocked)
        # Selection restoration is display-only: keep the source, frame and camera.
        self.update_readiness()
        self.motion_view.update()

    def retarget_character(self,c,source):
        if self.busy or self.contact_saving:return
        takes=source.get('takes') or []
        if not takes:return
        kind=source.get('kind') or self.displayable_result()
        if kind.startswith('corrected_') and self.current_export(kind,Path(takes[0]['project'])) is None:
            self.status.setText('Apply contact changes before retargeting.');return
        folder=c['id']+'/animation-'+uuid.uuid4().hex
        job=dict(mode='retarget',output=str(self.character_path(folder)),asset=str(self.character_path(c['asset'])),id=c['id'],
                 animation=folder,result=kind,source_name=source['name'],takes=takes)
        self.start_character_job(job)
    def prepare_character_refresh(self,jobs):
        from .retarget.refresh import correction_refreshes
        self.character_refresh_queue=correction_refreshes(self.characters,jobs,self.project)
        self.character_refresh_total=len(self.character_refresh_queue)
        ids={p['id'] for p in self.character_refresh_queue}
        for c in self.characters:
            if c['id'] in ids:c['needs_retarget']=True
        if ids:self.save_characters();self.append_character_rows()

    def next_character_refresh(self):
        if not self.character_refresh_queue:return
        plan=self.character_refresh_queue.pop(0)
        c=next(c for c in self.characters if c['id']==plan['id'])
        folder=c['id']+'/animation-'+uuid.uuid4().hex
        job=dict(plan,mode='retarget',output=str(self.character_path(folder)),
                 asset=str(self.character_path(c['asset'])),animation=folder,
                 refresh_label=f"Updating {c['name']} · {self.character_refresh_total-len(self.character_refresh_queue)}/{self.character_refresh_total}")
        self.start_character_job(job)

    def start_character_job(self,job):
        from .retarget.runtime import blender
        from .runtime_registry import command
        try:
            if job['mode']!='adopt':blender()
            path=self.character_root/'characters'/('job-'+uuid.uuid4().hex+'.json');atomic_json(path,job)
            args=['--request',str(path)]
            if job['mode'] in {'retarget','solve'}:args+=['--project',job['takes'][0]['project']]
            cmd=command(ROOT,'character',args,frozen=getattr(sys,'frozen',False))
        except Exception as exc:
            self.character_refresh_queue=[]
            self.status.setText(str(exc));return
        self.stop_playback();self.character_job=job;self.character_started=time.monotonic();self.character_line='Starting character processing...';self.character_buffer='';self.character_cancelled=False
        self._character_decoder=codecs.getincrementaldecoder("utf-8")("replace");self._character_pending=""
        p=QProcess(self);self.character_process=p
        env=QProcessEnvironment.systemEnvironment();env.insert('PYTHONUTF8','1');env.insert('PYTHONIOENCODING','utf-8');p.setProcessEnvironment(env)
        p.setWorkingDirectory(str(ROOT));p.setProcessChannelMode(QProcess.MergedChannels)
        p.readyReadStandardOutput.connect(self.character_output)
        p.finished.connect(self.character_finished)
        p.errorOccurred.connect(lambda error:self.character_finished(-1) if error==QProcess.FailedToStart else None)
        self.busy=True;self.progress.setRange(0,0);self.character_timer.start();self.character_progress();self.update_readiness()
        p.start(cmd[0],cmd[1:])

    def character_output(self, final=False):
        if not self.character_process:return
        text=self._character_decoder.decode(bytes(self.character_process.readAllStandardOutput()),final=final)
        self.character_buffer=(self.character_buffer+text)[-16000:]
        self._character_pending=(self._character_pending+text)[-16000:]
        lines=self._character_pending.split("\n");self._character_pending=lines.pop()
        if final and self._character_pending:lines.append(self._character_pending);self._character_pending=""
        for line in lines:
            if line.startswith('CHARACTER:'):
                self.character_line=line.partition(':')[2].strip()
                self.log.status_terminal(line.strip())
        self.character_progress()

    def character_progress(self):
        label=self.character_job.get('refresh_label')
        self.status.setText((label+' · ' if label else '')+self.character_line)
        self.task_info.setText(f'Character · {time.monotonic()-self.character_started:.0f}s elapsed')

    def character_finished(self,code,*args):
        if self.character_process is None:return
        self.character_output(final=True);p=self.character_process;self.character_process=None;p.deleteLater();self.character_timer.stop();self.busy=False
        job=self.character_job;complete=Path(job['output'])/'complete.json'
        if code==0 and complete.is_file() and not self.character_cancelled:
            result=json.loads(complete.read_text(encoding='utf8'))
            if job['mode']=='adopt':
                self.characters.extend(result['characters'])
                self.save_characters()
                for c in result['characters']:self.request_character(c)
                self.append_character_rows()
            elif job['mode']=='import':
                c=dict(id=job['id'],name=Path(job['file']).stem,kind=result['kind'],asset=job['id']+'/asset',visible=True)
                self.characters.append(c)
            elif job['mode']=='retarget':
                c=next(c for c in self.characters if c['id']==job['id'])
                c.pop('_starts',None)
                c.pop('needs_retarget',None)
                c.update(animation=job['animation'],takes=result['takes'],source_name=result['source_name'],result=result['result'])
            elif job['mode']=='liverig':
                shutil.rmtree(job['output'],ignore_errors=True)
                self.__dict__.setdefault('_live_characters',{}).pop(job['id'],None)
                self.__dict__.setdefault('_rig_ready',{}).pop(job['id'],None)
            elif job['mode']=='solve':
                c=next((c for c in self.characters if c['id']==job['id']),None)
                if c is not None:
                    old=c.get('follow')
                    c['follow']=dict(animation=job['animation'],takes=result['takes'],source_name=result['source_name'],result=result['result'])
                    if old and old.get('animation')!=job['animation']:
                        shutil.rmtree(self.character_path(old['animation']),ignore_errors=True)
                    self.save_characters()
            if job['mode'] in {'import','retarget'}:
                self.save_characters();self.request_character(c);self.append_character_rows()
            self.status.setText(f'Character ready · {result["seconds"]:.1f}s');self.progress.setRange(0,100);self.progress.setValue(100)
        else:
            self.character_refresh_queue=[]
            self.status.setText('Character job cancelled.' if self.character_cancelled else 'Character processing failed. See Console.')
            self.log.append(self.character_buffer);self.progress.setRange(0,100);self.progress.setValue(0)
        self.task_info.setText('');self.update_readiness();self.motion_view.update()
        if job.get('refresh_label'):
            self.next_character_refresh()
        if job['mode'] in {'import','liverig'} and getattr(self,'live_channel',None) is not None:
            self.prepare_live_characters()

    def cancel_character(self):
        p=self.character_process
        if not p:return
        self.character_cancelled=True;self.character_line='Cancelling character job...'
        helper=QProcess(self);self._character_killer=helper
        helper.start('taskkill',['/PID',str(p.processId()),'/T','/F'])
        helper.finished.connect(helper.deleteLater)

    def request_character(self,c):
        path=self.character_path(c['asset']);self.character_loader.request(path)

    def character_ready(self,key,data):
        if self.character_loader.stop:return
        self.character_assets[key]=data
        if self.status.text()=='Loading original character textures...':self.status.setText('Character preview ready.')
        if self.motion_view.joints is None and not self.motion_view._camera_initialized:
            v=data['asset']['vertices'];lo=v.min(0);hi=v.max(0)
            self.motion_view.scene_center=(lo+hi)*.5
            self.motion_view.distance=max(2.,float(np.linalg.norm(hi-lo))*1.5)
            self.motion_view._camera_initialized=True
        self.motion_view.update()

    def character_live_take(self):
        """The character take on screen, for the live link (path, frame, matrix)."""
        from bisect import bisect_right
        time_ = self.motion_view.scene_time
        for char in self.characters:
            if not char.get('visible', True):
                continue
            path = self.character_path(char['asset'])
            frame = 0
            transform = None
            if char.get('takes'):
                starts = char.setdefault('_starts', [t['start'] for t in char['takes']])
                index = bisect_right(starts, time_) - 1
                take = char['takes'][index] if index >= 0 else None
                if take is not None and time_ >= take['start'] + take['frames'] / take['fps']:
                    take = None
                if take is None:
                    continue
                path = self.character_path(char['animation'] + '/' + take['folder'])
                frame = int((time_ - take['start']) * take['fps'] + 1e-6)
                transform = take.get('transform')
            data = self.character_assets.get(str(path))
            if not data:
                self.character_loader.request(path, frame, self.character_path(char['asset']))
                continue
            matrix = self.character_loader.frame(path, frame)
            if matrix is None:
                continue
            return char, path, frame, transform, matrix
        return None

    def character_live_info(self, path):
        """Rest file and bone palette behind a character take, read once."""
        cache = getattr(self, '_live_character_cache', None)
        if cache is None:
            cache = self._live_character_cache = {}
        key = str(path)
        if key in cache:
            return cache[key]
        import json
        info = None
        try:
            bake = json.loads((path / 'bake.json').read_text(encoding='utf-8'))
            kind = json.loads((path / 'import.json').read_text(encoding='utf-8'))['kind']
            rig_path = path / f'{kind}-rig.json'
            rig = json.loads(rig_path.read_text(encoding='utf-8')) if rig_path.is_file() else {}
            import numpy as np
            with np.load(path / 'preview.npz') as preview:
                vertices = preview['vertices']
            # The app draws this character in metres; the rest file may keep its
            # importer's units, so the live link sends the height to scale by.
            height = float(vertices[:, 1].max() - vertices[:, 1].min())
            info = dict(blend=path / f'{kind}-rest.blend', palette=sorted(bake['keyed_bones']),
                        armature=rig.get('armature', 'Armature'), kind=kind, height=height)
            if not info['blend'].is_file():
                raise FileNotFoundError(str(info['blend']))
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.log.append('Live link: cannot describe character: ' + str(error))
            info = None
        cache[key] = info
        return info

    # -- live puppeteering ---------------------------------------------------
    # An imported character is the live view's model: each solved frame poses
    # it directly (live_character.py). There are no keys and nothing is baked.
    def live_character(self,c,source=None):
        """The numpy rig that poses character ``c`` live, or None until it exists."""
        if source is None:source=getattr(self,'_live_source_rest',None)
        if source is None:return None
        cache=self.__dict__.setdefault('_live_characters',{})
        if c['id'] in cache and cache[c['id']] is not None and getattr(cache[c['id']],'source_key',None)!=hash(np.asarray(source).tobytes()):cache.pop(c['id'])
        if c['id'] in cache:return cache[c['id']]
        from .live_character import LiveCharacter,rig_file
        asset=self.character_path(c['asset'])
        if not rig_file(asset).is_file():return None
        try:
            cache[c['id']]=LiveCharacter(asset,source);cache[c['id']].source_key=hash(np.asarray(source).tobytes())
        except (OSError,ValueError,KeyError) as error:
            self.log.append('Live character '+c['name']+': '+str(error));cache[c['id']]=None
        return cache[c['id']]

    def prepare_live_characters(self):
        """Write the live rig of a visible character that lacks one (a one-off
        read of its bones, not an animation); characters imported now have it."""
        if self.busy or self.character_process is not None:return
        tried=self.__dict__.setdefault('_live_rig_tried',set())
        from .live_character import rig_current
        for c in getattr(self,'characters',[]):
            if not c.get('visible',True) or c['id'] in tried:continue
            asset=self.character_path(c['asset'])
            if rig_current(asset):continue
            tried.add(c['id'])
            self.start_character_job(dict(mode='liverig',output=str(self.character_path('liverig-'+uuid.uuid4().hex)),
                asset=str(asset),id=c['id'],name=c['name']))
            return

    def update_live_characters(self,row):
        """Pose every visible character from one live (52, 4, 4) rig row."""
        if not getattr(self,'follow_source',False):
            self.live_character_poses={};return {}
        poses={}
        serial=getattr(self,'_live_pose_serial',0)+1;self._live_pose_serial=serial
        for c in getattr(self,'characters',[]):
            if not c.get('visible',True):continue
            live=self.live_character(c)
            if live is not None:poses[c['id']]=(serial,live.pose(row))
        self.live_character_poses=poses
        return poses

    def clear_live_characters(self):
        self._direct_poses={};self.live_character_poses={};self._live_characters={};self._live_source_rest=None

    def rest_bone_row(self,c,transform):
        """Skeleton of a character that is not driven by a person: its rest bones."""
        cache=self.__dict__.setdefault('_rest_bones',{})
        if c['id'] not in cache:
            from .live_character import rig_file
            file=rig_file(self.character_path(c['asset']))
            try:
                with np.load(file,allow_pickle=False) as data:cache[c['id']]=(data['head'].astype(float),data['parents'].astype(int))
            except (OSError,ValueError,KeyError):cache[c['id']]=None
        if cache[c['id']] is not None:
            self.__dict__.setdefault('character_bone_rows',{})[c['id']]=(cache[c['id']][0],cache[c['id']][1],transform)

    def source_pose(self):
        """(rig row, rest, stamp, transform) of the person the view shows now.

        Characters are posed straight from it at the timeline's frame, the same
        row the bone overlay draws: moving the playhead moves the character,
        and nothing is keyed or baked.
        """
        view=self.motion_view
        try:
            if view.people_mode and view.people_scene is not None:
                actors=view.scene_actors()
                if not actors:return None
                take,frame=next((a for a in actors if a[0].ident==getattr(self,'active_person_id',None)),actors[0])
                asset=take.target_asset;rig=take.arrays.get('rig')
                if not asset or 'bones' not in asset or 'rest' not in asset['bones'] or rig is None:return None
                row=np.asarray(rig[frame]);stamp=(take.ident,frame)
                return row,asset['bones']['rest'],(stamp,hash(row.tobytes())),take.transform,str(take.project)
            if view.joints is not None and view.target_asset and 'bones' in view.target_asset                     and 'rest' in view.target_asset['bones'] and view.target_frames is not None:
                row=np.asarray(view.target_frames[view.index])
                return row,view.target_asset['bones']['rest'],(('single',view.index),hash(row.tobytes())),None,str(self.project) if self.project else None
        except (AttributeError,KeyError,IndexError,TypeError):return None
        return None

    def rig_ready(self,c):
        cache=self.__dict__.setdefault('_rig_ready',{})
        if c['id'] not in cache:
            from .live_character import rig_current
            cache[c['id']]=rig_current(self.character_path(c['asset']))
        return cache[c['id']]

    @staticmethod
    def same_path(a,b):
        return bool(a) and bool(b) and os.path.normcase(os.path.realpath(a))==os.path.normcase(os.path.realpath(b))

    def follow_take(self,c,project):
        """The solved retarget of ``c`` for this source, while its source is unchanged."""
        follow=c.get('follow')
        take=next((t for t in follow['takes'] if self.same_path(t.get('source'),project)),None) if follow and project else None
        if take is None:return None
        checks=self.__dict__.setdefault('_follow_checks',{});key=(c['id'],follow['animation'],take['folder'])
        seen=checks.get(key);now=time.monotonic()
        if seen and now-seen[0]<1.0:return take if seen[1] else None
        ok=True
        try:
            receipt=json.loads((self.character_path(follow['animation']+'/'+take['folder'])/'receipt.json').read_text(encoding='utf8'))
            for path,(size,stamp) in receipt['source_stats'].items():
                info=os.stat(path)
                if info.st_size!=size or info.st_mtime_ns!=stamp:ok=False;break
        except (OSError,ValueError,KeyError,TypeError):ok=False
        checks[key]=(now,ok)
        return take if ok else None

    def follow_arrays(self,c,take):
        cache=self.__dict__.setdefault('_follow_arrays',{})
        path=self.character_path(c['follow']['animation']+'/'+take['folder']+'/solve.npz')
        if str(path) not in cache:
            with np.load(path,allow_pickle=False) as data:cache[str(path)]={k:data[k] for k in data.files}
        return cache[str(path)]

    def ensure_follow_solves(self):
        """Solve the retarget for the shown source (Blender-free, no bake) when
        a character has none yet; the viewport copies the person meanwhile."""
        project=getattr(self,'_follow_project',None)
        if not project or self.busy or self.character_process is not None:return
        tried=self.__dict__.setdefault('_follow_tried',set())
        for c in self.characters:
            if not c.get('visible',True) or not self.rig_ready(c) or self.follow_take(c,project) is not None:continue
            source=next((s for s in self.character_sources() if any(self.same_path(t['project'],project) for t in s['takes'])),None)
            if source is None:continue
            kind=source.get('kind') or self.displayable_result()
            if kind.startswith('corrected_') and self.current_export(kind,Path(project)) is None:continue
            try:stamp=(Path(project)/(kind+'_mixamo.json')).stat().st_mtime_ns
            except OSError:continue
            key=(c['id'],project,stamp)
            if key in tried:continue
            tried.add(key)
            folder=c['id']+'/follow-'+uuid.uuid4().hex
            self.start_character_job(dict(mode='solve',output=str(self.character_path(folder)),asset=str(self.character_path(c['asset'])),
                id=c['id'],animation=folder,result=kind,source_name=source['name'],takes=source['takes']))
            return

    def direct_draws(self,source):
        row,rest,stamp,transform,project=source
        cache=self.__dict__.setdefault('_direct_poses',{})
        self._follow_project=project
        time_=self.motion_view.scene_time
        for c in self.characters:
            if not c.get('visible',True):continue
            live=self.live_character(c,rest)
            if live is None:
                QTimer.singleShot(0,self.prepare_live_characters);continue
            path=self.character_path(c['asset']);data=self.character_assets.get(str(path))
            if not data:self.character_loader.request(path,0,path);continue
            take=self.follow_take(c,project) if self.rig_ready(c) else None
            if take is not None:
                frame=int((time_-take['start'])*take['fps']+1e-6)
                if not 0<=frame<take['frames']:continue
                key=(('solved',c['follow']['animation'],take['folder'],frame),)
                held=cache.get(c['id'])
                if held is None or held[0]!=key:
                    arrays=self.follow_arrays(c,take)
                    extra={int(b):arrays['finger_world'][frame,k] for k,b in enumerate(arrays['finger_bones'])}
                    self._live_pose_serial=getattr(self,'_live_pose_serial',0)+1
                    held=cache[c['id']]=(key,live.pose_solved(arrays['world'][frame],arrays['root'][frame],extra),self._live_pose_serial)
                self.__dict__.setdefault('character_bone_rows',{})[c['id']]=(live.positions,live.parents,transform)
                yield c,data,held[1],held[2],transform
                continue
            if self.rig_ready(c):QTimer.singleShot(0,self.ensure_follow_solves)
            else:QTimer.singleShot(0,self.prepare_live_characters)
            held=cache.get(c['id'])
            if held is None or held[0]!=stamp:
                self._live_pose_serial=getattr(self,'_live_pose_serial',0)+1
                held=cache[c['id']]=(stamp,live.pose(row),self._live_pose_serial)
            self.__dict__.setdefault('character_bone_rows',{})[c['id']]=(live.positions,live.parents,transform)
            yield c,data,held[1],held[2],transform

    def character_draws(self):
        from bisect import bisect_right
        time_=self.motion_view.scene_time
        follow=getattr(self,'follow_source',False)
        if follow and getattr(self,'live_channel',None) is None:
            source=self.source_pose()
            if source is not None:
                drawn=list(self.direct_draws(source))
                if drawn:
                    yield from drawn
                    return
        live=getattr(self,'live_character_poses',None) if follow and getattr(self,'live_channel',None) is not None else None
        for c in self.characters:
            if not c.get('visible',True):continue
            if live and c['id'] in live:
                serial,matrix=live[c['id']];path=self.character_path(c['asset']);data=self.character_assets.get(str(path))
                if not data:self.character_loader.request(path,0,path);continue
                lc=self.live_character(c)
                if lc is not None:self.__dict__.setdefault('character_bone_rows',{})[c['id']]=(lc.positions,lc.parents,None)
                yield c,data,matrix,serial,None;continue
            path=self.character_path(c['asset']);frame=0;transform=None
            if c.get('takes'):
                starts=c.setdefault('_starts',[t['start'] for t in c['takes']])
                i=bisect_right(starts,time_)-1
                take=c['takes'][i] if i>=0 else None
                if take is not None and time_>=take['start']+take['frames']/take['fps']:take=None
                if take is None:continue
                path=self.character_path(c['animation']+'/'+take['folder']);frame=int((time_-take['start'])*take['fps']+1e-6);transform=take.get('transform')
            data=self.character_assets.get(str(path))
            if not data:
                self.character_loader.request(path,frame,self.character_path(c['asset']));continue
            matrix=self.character_loader.frame(path,frame)
            if matrix is not None:
                self.rest_bone_row(c,transform);yield c,data,matrix,frame,transform
