"""Verify the automatic post-reconstruction grounding decision without a GUI.

Calls the real PeopleUi method with a stub window so the flat-floor switch and
the job list can be inspected directly.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mocap.runtime_registry import activate  # noqa: E402
activate(ROOT, 'gui')
from mocap import people_ui  # noqa: E402
from mocap import gui as gui_module  # noqa: E402
from mocap.people_store import load_people  # noqa: E402
from mocap.scene_floor import ensure_config, read_config, save_config  # noqa: E402
from mocap.motion_tools import result_is_current  # noqa: E402

PROJECT = Path(sys.argv[1] if len(sys.argv) > 1 else
               ROOT / 'projects' / 'Download_35b14c2294807c_project')
KIND = 'corrected_initial'


def find(cls_name, method):
    for value in vars(people_ui).values():
        if isinstance(value, type) and hasattr(value, method):
            return value
    raise SystemExit('owner of ' + method + ' not found')


class Status:
    text = ''

    def setText(self, value):
        self.text = value


class Toggle:
    def __init__(self, value=True):
        self.value = value

    def isChecked(self):
        return self.value


class Stub:
    busy = False

    def __init__(self, root):
        self.people_root = root
        self.people_state = load_people(root)
        self.motion_correction = Toggle()
        self.active_person_id = self.people_state['people'][0]['id']
        self.project = None
        self.status = Status()
        self.jobs = []
        self.started = []

    def start_jobs(self, jobs, refresh_characters=False):
        self.started.append((list(jobs), refresh_characters))

    def sync_shared_floor_controls(self):
        self.floor_controls_synced = True


def actors(root):
    state = load_people(root)
    return [root / take['project'] for person in state['people'] if person.get('checked')
            for take in person.get('takes', [])]


class LoneStub(Stub):
    people_root = None

    def __init__(self, take):
        self.people_state = None
        self.motion_correction = Toggle()
        self.active_person_id = 1
        self.project = take
        self.status = Status()
        self.jobs = []
        self.started = []


def take_project(root):
    state = load_people(root)
    person = next(person for person in state['people'] if person.get('takes'))
    return root / person['takes'][0]['project']


def check_lone(root, method, report):
    take = take_project(root)
    stub = LoneStub(take)
    stub._single_result_kind = KIND
    method(stub)
    report['lone_project'] = take.name
    report['lone_current_jobs'] = len(stub.started)
    victim = take / (KIND + '.pt')
    backup = victim.with_name(victim.name + '.bak')
    try:
        victim.rename(backup)
        stub = LoneStub(take)
        stub._single_result_kind = KIND
        method(stub)
        report['lone_missing_jobs'] = [[stage for stage, _ in jobs] for jobs, _ in stub.started]
        report['lone_missing_status'] = stub.status.text
        report['lone_missing_refresh'] = stub.started[0][1] if stub.started else None
    finally:
        backup.rename(victim)
    stub = LoneStub(take)
    stub.motion_correction = Toggle(False)
    method(stub)
    report['lone_opt_out_jobs'] = len(stub.started)
    report['lone_restored'] = result_is_current(take, KIND)


def main():
    root = PROJECT.resolve()
    owner = find(None, 'ground_scene_after_reconstruction')
    method = getattr(owner, 'ground_scene_after_reconstruction')
    report = {'project': str(root)}

    # 1. Already grounded: flat plane and current results -> nothing to queue.
    save_config(root, {**ensure_config(root), 'mode': 'flat'})
    stub = Stub(root)
    method(stub)
    report['flat_current_jobs'] = len(stub.jobs)
    report['flat_current_ok'] = all(result_is_current(project, KIND) for project in actors(root))

    # 2. A saved Multi Level plane leaves every actor ungrounced -> the hook must
    #    flip back to flat and queue one correction per actor.
    save_config(root, {**ensure_config(root), 'mode': 'multi_level'})
    report['multi_level_stale'] = [result_is_current(project, KIND) for project in actors(root)]
    stub = Stub(root)
    method(stub)
    report['multi_level_jobs'] = sorted({stage for jobs, _ in stub.started for stage, _ in jobs})
    report['multi_level_actor_jobs'] = sum(len(jobs) for jobs, _ in stub.started)
    report['mode_after_hook'] = read_config(root).get('mode', 'flat')
    report['status_after_hook'] = stub.status.text

    # 3. A reconstruction that never produced a corrected result still gets
    #    grounded without a manual Apply Correction.
    victim = actors(root)[0] / (KIND + '.pt')
    backup = victim.with_name(victim.name + '.bak')
    try:
        victim.rename(backup)
        stub = Stub(root)
        method(stub)
        report['missing_result_jobs'] = [[stage for stage, _ in jobs] for jobs, _ in stub.started]
        report['missing_result_refresh'] = stub.started[0][1] if stub.started else None
        report['missing_result_status'] = stub.status.text
    finally:
        backup.rename(victim)

    # 4. Explicit opt-out of Motion Correction is respected.
    stub = Stub(root)
    stub.motion_correction = Toggle(False)
    method(stub)
    report['opt_out_jobs'] = len(stub.started)

    # Restore the grounded configuration used by the app.
    save_config(root, {**ensure_config(root), 'mode': 'flat'})
    report['restored_current'] = all(result_is_current(project, KIND) for project in actors(root))

    lone_owner = next(value for value in vars(gui_module).values()
                      if isinstance(value, type)
                      and hasattr(value, 'ground_lone_actor_after_reconstruction'))
    check_lone(root, getattr(lone_owner, 'ground_lone_actor_after_reconstruction'), report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
