"""Create the Unreal side of a take: a playable character and a place to stand.

Run inside Unreal's Python. It builds (or refreshes) a character Blueprint whose
animation class follows the RGBMocap stream, and puts one in the level per
person, named so each one follows its own subject. This is the file the app
hands to Unreal when a take is sent, so nothing has to be arranged by hand.

Actors already standing in the level (a Mannequin put there by hand) and selected
there are switched over in place: their animation class becomes the RGBMocap one
and they take the next free RGBMoCapLive_PersonN name, which is how the stream
finds them.

Skeletal meshes selected in the Content Browser get their own Blueprint
(BP_RGBMoCapLive_<mesh>) in the same way, so a character of the user's own can
follow the stream without building an animation Blueprint for it.
"""
import unreal

SUBJECTS = 4
PATH = "/Game/RGBMocap"
NAME = "BP_RGBMoCapLive"
MESH = "/Game/RGBMocap/base_body"
FALLBACK = "/Game/Characters/Mannequins/Meshes/SKM_Quinn_Simple"
SPACING = 180.0

log = []


def say(*parts):
    log.append(" ".join(str(part) for part in parts))


def build_blueprint(name=NAME, mesh_path=None):
    full = PATH + "/" + name
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    blueprint = unreal.EditorAssetLibrary.load_asset(full)
    if blueprint is None:
        factory = unreal.BlueprintFactory()
        factory.set_editor_property("parent_class", unreal.SkeletalMeshActor)
        blueprint = tools.create_asset(name, PATH, unreal.Blueprint, factory)
        say("blueprint created")
    if blueprint is None:
        return None

    generated = unreal.load_object(None, full + "." + name + "_C")
    component = unreal.get_default_object(generated).get_editor_property("skeletal_mesh_component")
    wanted = mesh_path or (MESH if unreal.EditorAssetLibrary.does_asset_exist(MESH) else FALLBACK)
    component.set_editor_property("skeletal_mesh_asset", unreal.EditorAssetLibrary.load_asset(wanted))
    component.set_editor_property("animation_mode", unreal.AnimationMode.ANIMATION_BLUEPRINT)
    # The anim class this project ships: it follows the RGBMocap stream, reads the
    # subject off the actor's name and brings its own retarget. The MotionForge
    # class of the same kind composes a different convention and bends the
    # body about the wrong axes.
    anim_class = getattr(unreal, "RGBMocapLiveInstance", None) or unreal.MotionForgeLiveLinkInstance
    component.set_editor_property("anim_class", anim_class)
    unreal.BlueprintEditorLibrary.compile_blueprint(blueprint)
    unreal.EditorAssetLibrary.save_asset(full)
    say("blueprint ready, mesh", wanted.split("/")[-1])
    return blueprint


def free_person(taken):
    """A subject the app really streams (Person 1..SUBJECTS) for the next actor.

    The plugin serves one source per subject, so an actor named past the last
    one would wait for a stream that never comes. Slots nobody stands on come
    first; with every slot taken the actor shares Person 1, which is allowed.
    """
    labels = {a.get_actor_label() for a in unreal.EditorLevelLibrary.get_all_level_actors()}
    for person in range(1, SUBJECTS + 1):
        if "RGBMoCapLive_Person%d" % person not in labels and person not in taken:
            return person
    return 1


def place_own(blueprint, mesh_path, taken):
    """One actor of the user's own character, on the next free subject."""
    person = free_person(taken)
    taken.add(person)
    actor = unreal.EditorLevelLibrary.spawn_actor_from_object(
        blueprint, unreal.Vector((person - 1) * SPACING, 0.0, 0.0))
    actor.set_actor_label("RGBMoCapLive_Person%d" % person)
    component = actor.get_component_by_class(unreal.SkeletalMeshComponent)
    if component.get_skeletal_mesh_asset() is None:
        component.set_skeletal_mesh_asset(unreal.EditorAssetLibrary.load_asset(mesh_path))
    say("placed", mesh_path.split("/")[-1], "as Person", person)


def selected_actors():
    """Level actors that wear a skeletal mesh and are selected in the viewport."""
    forced = globals().get('SELECTED_ACTORS')
    if forced is not None:
        return list(forced)
    found = []
    for actor in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_selected_level_actors():
        if actor.get_components_by_class(unreal.SkeletalMeshComponent):
            found.append(actor)
    return found


def adopt_actor(actor, taken):
    """Make an actor the level already holds follow the stream."""
    anim_class = getattr(unreal, "RGBMocapLiveInstance", None) or unreal.MotionForgeLiveLinkInstance
    label = actor.get_actor_label()
    person = 0
    if label.startswith("RGBMoCapLive_Person"):
        tail = label.rsplit("Person", 1)[1]
        person = int(tail) if tail.isdigit() and 1 <= int(tail) <= SUBJECTS else 0
    person = person or free_person(taken)
    taken.add(person)
    actor.set_actor_label("RGBMoCapLive_Person%d" % person)
    for component in actor.get_components_by_class(unreal.SkeletalMeshComponent):
        component.set_editor_property("animation_mode", unreal.AnimationMode.ANIMATION_BLUEPRINT)
        # Setting the class again starts a fresh instance, which reads the new name.
        component.set_editor_property("anim_class", None)
        component.set_editor_property("anim_class", anim_class)
        try:
            component.set_editor_property("update_animation_in_editor", True)
        except Exception:                                  # noqa: BLE001 - older engines lack it
            pass
        mesh = component.get_skeletal_mesh_asset()
        say("switched", label, "(", mesh.get_name() if mesh else "no mesh", ") to Person", person)
    actor.modify()


def selected_meshes():
    forced = globals().get('SELECTED')   # a caller that already knows which meshes
    if forced is not None:
        return list(forced)
    chosen = []
    for asset in unreal.EditorUtilityLibrary.get_selected_assets():
        if isinstance(asset, unreal.SkeletalMesh):
            chosen.append(asset.get_path_name().split(".")[0])
    return chosen


def place(blueprint, subjects):
    level = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    mesh = unreal.EditorAssetLibrary.load_asset(
        MESH if unreal.EditorAssetLibrary.does_asset_exist(MESH) else FALLBACK)
    for person in range(1, subjects + 1):
        label = "RGBMoCapLive_Person%d" % person
        found = [a for a in unreal.EditorLevelLibrary.get_all_level_actors()
                 if a.get_actor_label() == label]
        if found:
            actor = found[0]
        else:
            actor = unreal.EditorLevelLibrary.spawn_actor_from_object(
                blueprint, unreal.Vector((person - 1) * SPACING, 0.0, 0.0))
            actor.set_actor_label(label)
            say("placed", label)
        component = actor.get_component_by_class(unreal.SkeletalMeshComponent)
        if component.get_skeletal_mesh_asset() is None:
            component.set_skeletal_mesh_asset(mesh)
    level.save_current_level()
    say("level saved")


def main():
    chosen = selected_actors()
    if chosen:
        taken = set()
        for actor in chosen:
            adopt_actor(actor, taken)
        unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).save_current_level()
        unreal.log("RGBMOCAP_SETUP " + " | ".join(log))
        return
    own = selected_meshes()
    if own:
        taken = set()
        for mesh_path in own:
            stem = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in mesh_path.split("/")[-1])
            blueprint = build_blueprint(NAME + "_" + stem, mesh_path)
            if blueprint is not None:
                place_own(blueprint, mesh_path, taken)
        unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).save_current_level()
        unreal.log("RGBMOCAP_SETUP " + " | ".join(log))
        return
    blueprint = build_blueprint()
    if blueprint is not None:
        place(blueprint, SUBJECTS)
    unreal.log("RGBMOCAP_SETUP " + " | ".join(log))


main()
