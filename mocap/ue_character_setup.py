"""Create the Unreal side of a take: a playable character and a place to stand.

Run inside Unreal's Python. It builds (or refreshes) a character Blueprint whose
animation class follows the MotionForge stream, and puts one in the level per
person, named so each one follows its own subject. This is the file the app
hands to Unreal when a take is sent, so nothing has to be arranged by hand.
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


def build_blueprint():
    full = PATH + "/" + NAME
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    blueprint = unreal.EditorAssetLibrary.load_asset(full)
    if blueprint is None:
        factory = unreal.BlueprintFactory()
        factory.set_editor_property("parent_class", unreal.SkeletalMeshActor)
        blueprint = tools.create_asset(NAME, PATH, unreal.Blueprint, factory)
        say("blueprint created")
    if blueprint is None:
        return None

    generated = unreal.load_object(None, full + "." + NAME + "_C")
    component = unreal.get_default_object(generated).get_editor_property("skeletal_mesh_component")
    wanted = MESH if unreal.EditorAssetLibrary.does_asset_exist(MESH) else FALLBACK
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
    blueprint = build_blueprint()
    if blueprint is not None:
        place(blueprint, SUBJECTS)
    unreal.log("RGBMOCAP_SETUP " + " | ".join(log))


main()
