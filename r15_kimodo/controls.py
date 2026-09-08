"""Per-effector Reach controls and non-stretching CCD IK for R15."""
import time
import traceback
import bpy
from mathutils import Matrix, Quaternion, Vector
from .core import ALL_BODY, BODY, ROOT, PARENTS, require, qblend, interval
from . import animation

_BUSY=False
_PENDING=None
_CHANGED_AT=0.0
_CACHE={}
_OFFSETS={}
IK_ITERATIONS=16
IK_TOLERANCE=0.0001


def _descendants(joint):
    result=[]
    for n in ALL_BODY:
        current=n
        while current!=joint and current!=ROOT:current=PARENTS[current]
        if current==joint:result.append(n)
    return result


DESCENDANTS={n:_descendants(n) for n in ALL_BODY}


def _rotate(matrices,name,rotation):
    pivot=matrices[name].translation
    transform=Matrix.Translation(pivot) @ rotation.to_matrix().to_4x4() @ Matrix.Translation(-pivot)
    for child in DESCENDANTS[name]:matrices[child]=transform @ matrices[child]
    return


def solve(rest,reference,generated,items,offsets):
    """Blend FK motion, then solve weighted effector head goals in armature space."""
    matrices={}
    for n in ALL_BODY:
        item=items[n]
        a=reference['bases'][n];b=generated['bases'][n]
        loc=a.translation.lerp(b.translation,item.translation_strength)
        q=qblend(a.to_quaternion(),b.to_quaternion(),item.rotation_strength)
        basis=Matrix.LocRotScale(loc,q,Vector((1,1,1)))
        matrices[n]=(matrices[PARENTS[n]] @ rest[PARENTS[n]].inverted() @ rest[n] @ basis
                     if n!=ROOT else rest[n] @ basis)
    # Effector rotation offsets operate in that bone's animated local axes.
    for n in ALL_BODY:
        q=offsets[n].to_quaternion()
        if abs(q.w)<1-1e-9:
            current=matrices[n].to_quaternion()
            desired=current @ qblend(Quaternion(),q,items[n].rotation_strength)
            _rotate(matrices,n,desired @ current.inverted())
    orientations={n:matrices[n].to_quaternion() for n in ALL_BODY
                  if items[n].ik_strength>0 and items[n].rotation_strength>0}
    goals={}
    for n in ALL_BODY:
        item=items[n]
        influence=item.ik_strength*item.translation_strength
        if influence>0:
            target=(generated['matrices'][n] @ offsets[n]).translation
            goals[n]=matrices[n].translation.lerp(target,influence)
    for _ in range(IK_ITERATIONS):
        largest=0.0
        for n,target in goals.items():
            error=(target-matrices[n].translation).length
            largest=max(largest,error)
            if error<IK_TOLERANCE:continue
            if n in (ROOT,'LowerTorso'):
                shift=target-matrices[n].translation
                for child in DESCENDANTS[n]:matrices[child].translation+=shift
                continue
            parent=PARENTS[n]
            for depth in range(items[n].chain_length):
                if parent==ROOT:break
                pivot=matrices[parent].translation
                current=matrices[n].translation-pivot
                goal=target-pivot
                # A target at a joint pivot is a valid singular IK pose: this joint
                # has no angular leverage, but the next ancestor may still move it.
                if current.length>1e-8 and goal.length>1e-8:
                    _rotate(matrices,parent,current.rotation_difference(goal))
                parent=PARENTS[parent]
        for n,desired in orientations.items():
            current=matrices[n].to_quaternion()
            reached=qblend(current,desired,items[n].rotation_strength)
            _rotate(matrices,n,reached @ current.inverted())
        if largest<IK_TOLERANCE:break
    bases={}
    for n in ALL_BODY:
        parent=PARENTS[n] if n!=ROOT else None
        bases[n]=(rest[n].inverted() @ rest[parent] @ matrices[parent].inverted() @ matrices[n]
                  if parent is not None else rest[n].inverted() @ matrices[n])
    residual={n:(matrices[n].translation-target).length for n,target in goals.items()}
    return bases,residual


def _cache(s):
    arm=s.target;baseline=s.baseline_action;source=s.source_action
    require(arm is not None and baseline is not None and source is not None,'请先生成动画并创建骨骼调节器')
    key=(arm.as_pointer(),baseline.as_pointer(),source.as_pointer(),s.source_slot)
    if key not in _CACHE:
        anchors=list(baseline['r15_kimodo_anchors'])
        frames=list(range(anchors[0],anchors[-1]+1))
        reference=animation.snapshot(arm,source,source.slots[s.source_slot],frames)
        generated=animation.snapshot(arm,baseline,baseline.slots[s.source_slot],frames)
        rest={n:arm.data.bones[n].matrix_local.copy() for n in ALL_BODY}
        _CACHE[key]=(anchors,frames,reference,generated,rest)
    return _CACHE[key]


def setup(s):
    global _BUSY
    from . import ensure_idle
    ensure_idle()
    require(s.target is not None and s.result_action is not None,'请先生成 R15 动画')
    require(not s.controls_enabled,'调节器已存在，可直接调节或点击刷新')
    _BUSY=True
    try:
        arm=s.target
        s.baseline_action=s.result_action
        s.baseline_action.use_fake_user=True
        anchors,frames,reference,generated,rest=_cache(s)
        collection=bpy.data.collections.new('R15 Kimodo · '+arm.name)
        bpy.context.scene.collection.children.link(collection)
        s.controls_collection=collection
        proxy=arm.copy();proxy.data=arm.data.copy();proxy.name=arm.name+' · Kimodo FK Source'
        collection.objects.link(proxy)
        proxy.animation_data_clear();proxy.animation_data_create()
        proxy.animation_data.action=s.baseline_action
        proxy.animation_data.action_slot=s.baseline_action.slots[s.source_slot]
        proxy.parent=arm;proxy.matrix_parent_inverse=Matrix.Identity(4);proxy.matrix_basis=Matrix.Identity(4)
        proxy.hide_render=True;proxy.hide_select=True;proxy.hide_set(True)
        proxy['r15_kimodo_helper']=True
        s.control_proxy=proxy
        s.bone_controls.clear()
        for n in ALL_BODY:
            item=s.bone_controls.add();item.name=n
            item.chain_length=2 if n.endswith(('Hand','Foot')) or n=='Head' else 1
            driver=bpy.data.objects.new(n+' · IK Follow',None)
            collection.objects.link(driver)
            driver.hide_render=True;driver.hide_select=True
            driver.empty_display_size=0.01
            driver['r15_kimodo_helper']=True
            constraint=driver.constraints.new('COPY_TRANSFORMS');constraint.target=proxy;constraint.subtarget=n
            driver.hide_set(True)
            control=bpy.data.objects.new(n+' · IK Target',None)
            collection.objects.link(control)
            control.parent=driver
            control.empty_display_type='SPHERE';control.empty_display_size=0.10
            control.show_in_front=True;control.hide_render=True
            control.color=(0.65,0.3,1.0,1.0)
            control['r15_kimodo_effector']=n
            item.effector=control
            _OFFSETS[control.as_pointer()]=tuple(v for row in control.matrix_basis for v in row)
        s.controls_enabled=True
        s.controls_status='调节器已创建；拖动目标或修改强度后自动更新'
        show_effectors(s)
    finally:_BUSY=False
    return


def show_effectors(s):
    for item in s.bone_controls:
        if item.effector is not None:
            item.effector.hide_set(not s.show_effectors or item.name!=s.selected_bone)
    return


def apply(s):
    global _BUSY
    from . import ensure_idle
    ensure_idle()
    require(s.controls_enabled,'请先创建骨骼调节器')
    require(s.target.animation_data.action is not s.source_action,'当前在编辑关键姿势，请切回生成结果再调节')
    _BUSY=True
    try:
        anchors,frames,reference,generated,rest=_cache(s)
        items={item.name:item for item in s.bone_controls}
        require(set(items)==set(ALL_BODY),'骨骼调节器不完整')
        require(all(items[n].effector is not None for n in ALL_BODY),'IK 目标被删除，请重新创建调节器')
        require(all(items[n].effector.animation_data is None or items[n].effector.animation_data.action is None for n in ALL_BODY),
                'IK 目标用于整段偏移；请在原始 R15 骨架上 K 关键姿势')
        offsets={n:items[n].effector.matrix_basis.copy() for n in ALL_BODY}
        require(all(max(abs(v-1) for v in offset.to_scale())<1e-5 for offset in offsets.values()),'IK 目标只支持位移和旋转，请勿缩放')
        baked={};residual={n:0.0 for n in ALL_BODY}
        for f in frames:
            if s.lock_anchors and f in anchors:
                baked[f]={n:generated[f]['bases'][n].copy() for n in ALL_BODY}
            else:
                baked[f],errors=solve(rest,reference[f],generated[f],items,offsets)
                for n,error in errors.items():residual[n]=max(residual[n],error)
                if s.lock_anchors and s.lock_fade>0:
                    a,b,_=interval(anchors,f)
                    fade_frames=s.lock_fade*bpy.context.scene.render.fps/bpy.context.scene.render.fps_base
                    t=min(1.0,min(f-a,b-f)/fade_frames)
                    blend=t*t*(3-2*t)
                    for n in ALL_BODY:
                        base=generated[f]['bases'][n];solved=baked[f][n]
                        baked[f][n]=Matrix.LocRotScale(base.translation.lerp(solved.translation,blend),
                                    qblend(base.to_quaternion(),solved.to_quaternion(),blend),Vector((1,1,1)))
        modes={n:s.target.pose.bones[n].rotation_mode for n in ALL_BODY}
        action,slot=animation.write_action(s.target,s.baseline_action,s.baseline_action.slots[s.source_slot],
                   baked,anchors,modes,s.baseline_action.name+' · IK',bone_names=ALL_BODY)
        action['r15_kimodo_controlled']=True
        action['r15_kimodo_controls']=json_parameters(s)
        action.r15_kimodo_data=s.motion_data
        # Verify authored anchors only when their lock is enabled.
        if s.lock_anchors:
            checked=animation.snapshot(s.target,action,slot,anchors)
            error=max(max(abs(a-b) for ra,rb in zip(checked[f]['matrices'][n],generated[f]['matrices'][n]) for a,b in zip(ra,rb))
                      for f in anchors for n in ALL_BODY)
            if error>0.001:
                bpy.data.actions.remove(action)
                raise ValueError(f'IK 调节未能保持锁定姿势: {error}')
        old=s.result_action
        s.result_action=action
        s.target.animation_data.action=action;s.target.animation_data.action_slot=slot
        if old is not s.baseline_action and old.get('r15_kimodo_controlled',False) and old.users-int(old.use_fake_user)==0:
            bpy.data.actions.remove(old)
        for n,error in residual.items():items[n].reach_error=error
        bpy.context.scene.frame_set(bpy.context.scene.frame_current)
        s.controls_status=f'已更新 {len(frames)} 帧；可直接导出当前 Action'
        s.error=''
        print('[R15 Kimodo] IK preview baked:',len(frames),'frames')
    finally:_BUSY=False
    return


def json_parameters(s):
    import json
    return json.dumps({'lock_anchors':s.lock_anchors,'lock_fade':s.lock_fade,'bones':{p.name:{'ik':p.ik_strength,
                      'rotation':p.rotation_strength,'translation':p.translation_strength,
                      'chain':p.chain_length,'offset_location':list(p.offset_location),
                      'offset_rotation':list(p.offset_rotation)} for p in s.bone_controls}})


def schedule(s):
    global _PENDING,_CHANGED_AT
    if _BUSY or not s.controls_enabled:return
    from . import _JOB
    if _JOB is not None:return
    if s.target.animation_data.action is s.source_action:return
    _PENDING=s.id_data
    _CHANGED_AT=time.monotonic()
    if not bpy.app.timers.is_registered(_timer):bpy.app.timers.register(_timer,first_interval=0.15)
    return


def _timer():
    global _PENDING
    if _PENDING is None:return None
    if time.monotonic()-_CHANGED_AT<0.12:return 0.1
    scene=_PENDING;_PENDING=None
    try:
        require(scene is bpy.context.scene,'请切回调节器所属场景再更新')
        apply(scene.r15_kimodo)
    except Exception as exc:
        scene.r15_kimodo.error='骨骼调节失败: '+str(exc)
        scene.r15_kimodo.controls_status='调节失败，保留上一次成功结果'
        print('[R15 Kimodo] '+traceback.format_exc())
    return None


def setting_changed(self,context):
    if _BUSY:return
    s=self.id_data.r15_kimodo
    schedule(s)
    return


def offset_changed(self,context):
    if _BUSY or self.effector is None:return
    self.effector.location=self.offset_location
    self.effector.rotation_mode='XYZ'
    self.effector.rotation_euler=self.offset_rotation
    schedule(self.id_data.r15_kimodo)
    return


@bpy.app.handlers.persistent
def watch_targets(scene,depsgraph):
    global _BUSY
    if _BUSY or not hasattr(scene,'r15_kimodo'):return
    s=scene.r15_kimodo
    if not s.controls_enabled:return
    changed=False
    _BUSY=True
    try:
        for item in s.bone_controls:
            obj=item.effector
            if obj is None:
                message='IK 目标已被删除: '+item.name
                if s.error!=message:print('[R15 Kimodo] '+message)
                s.error=message
                return
            key=obj.as_pointer()
            values=tuple(v for row in obj.matrix_basis for v in row)
            if key not in _OFFSETS:
                _OFFSETS[key]=values
            elif values!=_OFFSETS[key]:
                _OFFSETS[key]=values
                item.offset_location=obj.location
                item.offset_rotation=obj.rotation_euler
                changed=True
    finally:_BUSY=False
    if changed:schedule(s)
    return


def selected_changed(self,context):
    show_effectors(self)
    from . import visualization
    if self.motion_data is not None and self.raw_view!='OFF':visualization.changed(self,context)
    return


def clear_setup(s,restore=True):
    global _BUSY
    reset_runtime()
    _BUSY=True
    try:
        if restore and s.baseline_action is not None:
            s.result_action=s.baseline_action
            s.target.animation_data.action=s.baseline_action
            s.target.animation_data.action_slot=s.baseline_action.slots[s.source_slot]
        if s.controls_collection is not None:
            collection=s.controls_collection
            data=s.control_proxy.data if s.control_proxy is not None else None
            for obj in list(collection.objects):bpy.data.objects.remove(obj,do_unlink=True)
            bpy.data.collections.remove(collection)
            if data is not None and data.users==0:bpy.data.armatures.remove(data)
        s.controls_enabled=False;s.control_proxy=None;s.controls_collection=None
        s.bone_controls.clear();s.baseline_action=None
    finally:_BUSY=False
    return


def reset_runtime():
    global _PENDING
    _PENDING=None;_CACHE.clear();_OFFSETS.clear()
    if bpy.app.timers.is_registered(_timer):bpy.app.timers.unregister(_timer)
    return


@bpy.app.handlers.persistent
def seed_offsets(_):
    for scene in bpy.data.scenes:
        if not hasattr(scene,'r15_kimodo'):continue
        for item in scene.r15_kimodo.bone_controls:
            if item.effector is not None:
                obj=item.effector
                _OFFSETS[obj.as_pointer()]=tuple(v for row in obj.matrix_basis for v in row)
    return


def register():
    bpy.app.handlers.depsgraph_update_post.append(watch_targets)
    bpy.app.handlers.load_post.append(seed_offsets)
    seed_offsets(None)
    return


def unregister():
    reset_runtime()
    if seed_offsets in bpy.app.handlers.load_post:bpy.app.handlers.load_post.remove(seed_offsets)
    if watch_targets in bpy.app.handlers.depsgraph_update_post:bpy.app.handlers.depsgraph_update_post.remove(watch_targets)
    return
