"""Read-only raw SOMA animation overlay, backed by embedded response data."""
import json
import math
import bpy
from mathutils import Vector, Matrix
from .core import Calibration, PAIRS, interval, require
from .animation import Motion, write_action

_CACHE={}


def store(s,request,responses,anchors,scene_fps,model_fps):
    payload={'request':request,'responses':responses,'anchors':list(anchors),
             'scene_fps':scene_fps,'model_fps':model_fps}
    text=bpy.data.texts.new('R15 Kimodo — Raw Motion.json')
    text.write(json.dumps(payload,ensure_ascii=False,separators=(',',':')))
    text.use_fake_user=True
    s.motion_data=text
    if s.result_action is not None: s.result_action.r15_kimodo_data=text
    clear_objects(s)
    _CACHE.clear()
    if s.raw_view!='OFF':refresh(s)
    return text


def data(s):
    require(s.motion_data is not None, '没有原始响应，请生成动画或导入接口数据')
    key=(s.motion_data.as_pointer(),s.target.as_pointer())
    if key not in _CACHE:
        payload=json.loads(s.motion_data.as_string())
        cal=Calibration(s.target,payload['request']['skeleton'])
        motions=[Motion(g,cal) for g in payload['responses']]
        require(len(motions)==len(payload['anchors'])-1,'原始数据的段数与关键姿势不匹配')
        _CACHE[key]=(payload,cal,motions)
    return _CACHE[key]


def raw_joints(s,frame,side_offset=True):
    payload,cal,motions=data(s)
    anchors=payload['anchors']
    if frame<anchors[0] or frame>anchors[-1]: return None
    a,b,t=interval(anchors,frame)
    i=anchors.index(a)
    indices=[c['frame'] for c in payload['request']['constraints']]
    time=t*(indices[i+1]-indices[i])/payload['model_fps']
    rotations,root=motions[i].sample(time)
    positions={};globals_={}
    for joint in cal.joints:
        name=joint['name'];parent=joint['parent']
        if parent is None:
            positions[name]=root
            globals_[name]=rotations[name]
        else:
            positions[name]=positions[parent]+globals_[parent] @ Vector(joint['rest_translation'])
            globals_[name]=globals_[parent] @ rotations[name]
    offset=cal.axes.col[0]*s.raw_spacing if side_offset and s.raw_view=='SIDE' else Vector((0,0,0))
    matrix=s.target.matrix_world
    points={n:matrix @ (cal.pelvis+cal.axes @ (p-cal.positions['Hips'])*cal.scale+offset)
            for n,p in positions.items()}
    return points,cal.parent


def _build(s):
    payload,cal,motions=data(s)
    previous=bpy.context.view_layer.objects.active
    selected=list(bpy.context.selected_objects)
    mode=previous.mode if previous is not None else 'OBJECT'
    if mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    arm_data=bpy.data.armatures.new('Kimodo Raw SOMA30')
    raw=bpy.data.objects.new('Kimodo · Original SOMA Motion',arm_data)
    bpy.context.scene.collection.objects.link(raw)
    raw.parent=s.target;raw.matrix_parent_inverse=Matrix.Identity(4)
    raw.show_in_front=True;raw.hide_render=True;raw['r15_kimodo_raw']=True
    arm_data.display_type='STICK'
    for obj in selected:obj.select_set(False)
    raw.select_set(True);bpy.context.view_layer.objects.active=raw
    try:
        bpy.ops.object.mode_set(mode='EDIT')
        for name in cal.order:
            bone=arm_data.edit_bones.new(name)
            bone.head=cal.pelvis+cal.axes @ (cal.positions[name]-cal.positions['Hips'])*cal.scale
            children=[n for n,p in cal.parent.items() if p==name]
            bone.tail=(cal.pelvis+cal.axes @ (cal.positions[children[0]]-cal.positions['Hips'])*cal.scale
                       if children else bone.head+cal.axes.col[1]*0.08)
            parent=cal.parent[name]
            if parent is not None:bone.parent=arm_data.edit_bones[parent]
        bpy.ops.object.mode_set(mode='OBJECT')
        for bone in arm_data.bones:
            bone.color.palette='CUSTOM'
            bone.color.custom.normal=(0.10,0.55,1.0) if bone.name.startswith('Left') else (1.0,0.25,0.04) if bone.name.startswith('Right') else (1.0,0.75,0.05)
            bone.color.custom.select=(0.4,1.0,0.7);bone.color.custom.active=(0.6,1.0,0.8)
            raw.pose.bones[bone.name].rotation_mode='QUATERNION'
        baked={};positions_by_frame={}
        anchors=payload['anchors'];indices=[c['frame'] for c in payload['request']['constraints']]
        for i,(a,b) in enumerate(zip(anchors,anchors[1:])):
            count=indices[i+1]-indices[i]
            for index in range(count+1):
                time=index/payload['model_fps']
                frame=a+(b-a)*index/count
                local,root=motions[i].sample(time)
                positions={};rotations={};matrices={};bases={}
                for joint in cal.joints:
                    n=joint['name'];parent=joint['parent']
                    if parent is None:
                        positions[n]=root;rotations[n]=local[n]
                    else:
                        positions[n]=positions[parent]+rotations[parent] @ Vector(joint['rest_translation'])
                        rotations[n]=rotations[parent] @ local[n]
                    rest=arm_data.bones[n].matrix_local
                    desired=(cal.axes @ rotations[n].to_matrix() @ cal.axes.transposed() @ rest.to_3x3()).to_4x4()
                    desired.translation=cal.pelvis+cal.axes @ (positions[n]-cal.positions['Hips'])*cal.scale
                    bases[n]=arm_data.bones[n].convert_local_to_pose(desired,rest,
                            parent_matrix=matrices[parent] if parent is not None else Matrix.Identity(4),
                            parent_matrix_local=arm_data.bones[parent].matrix_local if parent is not None else Matrix.Identity(4),invert=True)
                    matrices[n]=desired
                baked[frame]=bases
                positions_by_frame[frame]={n:m.translation.copy() for n,m in matrices.items()}
        template=bpy.data.actions.new('Kimodo Raw Template')
        slot=template.slots.new(id_type='OBJECT',name=raw.name)
        template.layers.new('Raw').strips.new(type='KEYFRAME').channelbag(slot,ensure=True)
        action,out_slot=write_action(raw,template,slot,baked,[],{n:'QUATERNION' for n in cal.order},
                                    'Kimodo · Unprocessed Response',bone_names=cal.order)
        bpy.data.actions.remove(template)
        action['r15_kimodo_raw']=True
        raw.animation_data_create();raw.animation_data.action=action;raw.animation_data.action_slot=out_slot
        s.raw_object=raw
        _build_geometry(raw,cal,positions_by_frame)
    except Exception:
        if raw.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
        bpy.data.objects.remove(raw,do_unlink=True)
        if arm_data.users==0:bpy.data.armatures.remove(arm_data)
        raise
    finally:
        for obj in bpy.context.selected_objects:obj.select_set(False)
        for obj in selected:obj.select_set(True)
        bpy.context.view_layer.objects.active=previous
        if previous is not None and mode!='OBJECT':bpy.ops.object.mode_set(mode=mode)
    return raw


def _build_geometry(raw,cal,positions):
    groups={'Left':(0.08,0.55,1.0,1),'Right':(1.0,0.25,0.035,1),'Body':(1.0,0.75,0.05,1)}
    frames=list(positions)
    for group,color in groups.items():
        names=[n for n in cal.order if cal.parent[n] is not None and
               ('Left' if n.startswith('Left') else 'Right' if n.startswith('Right') else 'Body')==group]
        curve=bpy.data.curves.new('Raw SOMA '+group,'CURVE')
        curve.dimensions='3D';curve.bevel_depth=cal.scale*0.006
        curve.bevel_resolution=1;curve.use_fill_caps=True
        for name in names:
            spline=curve.splines.new('POLY');spline.points.add(1)
            spline.points[0].co=(*positions[frames[0]][cal.parent[name]],1)
            spline.points[1].co=(*positions[frames[0]][name],1)
        material=bpy.data.materials.get('Kimodo Raw '+group)
        if material is None:material=bpy.data.materials.new('Kimodo Raw '+group)
        material.diffuse_color=color;curve.materials.append(material)
        obj=bpy.data.objects.new('Kimodo Raw · '+group,curve)
        bpy.context.scene.collection.objects.link(obj)
        obj.parent=raw;obj.matrix_parent_inverse=Matrix.Identity(4)
        obj.hide_render=True;obj.hide_select=True;obj.color=color;obj.show_in_front=True
        action=bpy.data.actions.new('Raw SOMA '+group+' Lines')
        slot=action.slots.new(id_type='CURVE',name=curve.name)
        bag=action.layers.new('Raw').strips.new(type='KEYFRAME').channelbag(slot,ensure=True)
        for i,name in enumerate(names):
            for j,joint in enumerate((cal.parent[name],name)):
                for axis in range(3):
                    fc=bag.fcurves.new(data_path=f'splines[{i}].points[{j}].co',index=axis)
                    fc.keyframe_points.add(len(frames))
                    fc.keyframe_points.foreach_set('co',[v for f in frames for v in (f,positions[f][joint][axis])])
                    for kp in fc.keyframe_points:kp.interpolation='LINEAR';kp.type='GENERATED'
                    fc.update()
        curve.animation_data_create();curve.animation_data.action=action;curve.animation_data.action_slot=slot
    return


def _trail(s):
    if s.raw_trail is not None:
        obj=s.raw_trail;curve=obj.data
        bpy.data.objects.remove(obj,do_unlink=True);bpy.data.curves.remove(curve)
    payload,cal,_=data(s)
    name=PAIRS[s.selected_bone] if s.selected_bone in PAIRS else 'Hips'
    inverse=s.target.matrix_world.inverted()
    frames=range(payload['anchors'][0],payload['anchors'][-1]+1)
    points=[inverse @ raw_joints(s,f,side_offset=False)[0][name] for f in frames]
    curve=bpy.data.curves.new('Kimodo Raw Joint Trajectory','CURVE')
    curve.dimensions='3D';curve.bevel_depth=0.008;curve.bevel_resolution=1
    spline=curve.splines.new('POLY');spline.points.add(len(points)-1)
    for point,co in zip(spline.points,points):point.co=(*co,1)
    obj=bpy.data.objects.new('Kimodo · Raw '+name+' Path',curve)
    bpy.context.scene.collection.objects.link(obj)
    obj.parent=s.target;obj.matrix_parent_inverse=Matrix.Identity(4);obj.hide_render=True
    obj.color=(0.1,1.0,0.4,1)
    material=bpy.data.materials.get('Kimodo Raw Trajectory')
    if material is None:material=bpy.data.materials.new('Kimodo Raw Trajectory')
    material.diffuse_color=(0.05,1.0,0.25,1.0);curve.materials.append(material)
    obj.hide_select=True
    s.raw_trail=obj
    return


def refresh(s):
    if s.motion_data is None:
        require(s.raw_view=='OFF','缺少原始动画数据')
        return
    if s.raw_view=='OFF':
        if s.raw_object is not None:
            s.raw_object.hide_set(True)
            for obj in s.raw_object.children:obj.hide_set(True)
        if s.raw_trail is not None:s.raw_trail.hide_set(True)
        return
    if s.raw_object is None:_build(s)
    _,cal,_=data(s)
    offset=cal.axes.col[0]*s.raw_spacing if s.raw_view=='SIDE' else Vector((0,0,0))
    s.raw_object.location=offset;s.raw_object.hide_set(True)
    for obj in s.raw_object.children:obj.hide_set(False)
    if s.raw_trajectory:
        _trail(s)
        s.raw_trail.location=offset;s.raw_trail.hide_set(False)
    elif s.raw_trail is not None:s.raw_trail.hide_set(True)
    bpy.context.view_layer.update()
    return


def changed(self,context):
    try:refresh(self)
    except Exception as exc:
        self.error='原始数据可视化失败: '+str(exc)
        print('[R15 Kimodo] '+self.error)
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type=='VIEW_3D':area.tag_redraw()
    return


def clear_objects(s):
    if s.raw_object is not None:
        raw=s.raw_object;data=raw.data;action=raw.animation_data.action
        for obj in list(raw.children):
            curve=obj.data;curve_action=curve.animation_data.action
            bpy.data.objects.remove(obj,do_unlink=True)
            bpy.data.curves.remove(curve);bpy.data.actions.remove(curve_action)
        bpy.data.objects.remove(raw,do_unlink=True)
        bpy.data.armatures.remove(data);bpy.data.actions.remove(action)
    if s.raw_trail is not None:
        obj=s.raw_trail;data=obj.data
        bpy.data.objects.remove(obj,do_unlink=True);bpy.data.curves.remove(data)
    return


def register():
    return


def clear():
    _CACHE.clear()
    return


def unregister():
    clear()
    return
