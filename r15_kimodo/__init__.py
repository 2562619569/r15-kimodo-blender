bl_info = {
    "name": "R15 Kimodo — Key Poses + Text",
    "author": "RobloxPJ",
    "version": (1, 1, 0),
    "blender": (5, 0, 0),
    "location": "3D View > Sidebar > R15 Kimodo",
    "description": "Generate R15 transitions from authored key poses and text using Proscenium's Kimodo service",
    "category": "Animation",
}

import json
import copy
import queue
import threading
import time
import traceback
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import bpy
from bpy.props import PointerProperty, CollectionProperty, StringProperty, IntProperty, FloatProperty, FloatVectorProperty, BoolProperty, EnumProperty
from .core import ALL_BODY, require
from . import animation, visualization, controls

_JOB = None
LAST_RUN = None


def settings():
    return bpy.context.scene.r15_kimodo


def ensure_idle():
    require(_JOB is None, '生成正在进行，请等待完成或取消')


def redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()


def target_changed(self, context):
    if self.controls_enabled:controls.clear_setup(self,restore=False)
    visualization.clear_objects(self)
    self.motion_data=None
    self.raw_view='OFF'
    self.source_action=None
    self.result_action=None
    self.source_slot=''
    self.prompts.clear()
    self.status='读取目标骨架的关键帧，或记录当前姿势'


class R15KM_Transition(bpy.types.PropertyGroup):
    frame_start: IntProperty(name='起点')
    frame_end: IntProperty(name='终点')
    prompt: StringProperty(name='过渡描述',default='Move naturally and smoothly between the specified poses.')


BONE_LABELS={
    'HumanoidRootPart':'角色根', 'LowerTorso':'下躯干', 'UpperTorso':'上躯干', 'Head':'头部',
    'LeftUpperArm':'左上臂','LeftLowerArm':'左前臂','LeftHand':'左手',
    'RightUpperArm':'右上臂','RightLowerArm':'右前臂','RightHand':'右手',
    'LeftUpperLeg':'左大腿','LeftLowerLeg':'左小腿','LeftFoot':'左脚',
    'RightUpperLeg':'右大腿','RightLowerLeg':'右小腿','RightFoot':'右脚',
}


class R15KM_BoneControl(bpy.types.PropertyGroup):
    ik_strength: FloatProperty(name='IK 强度',min=0,max=1,default=0,subtype='FACTOR',options=set(),
                               update=controls.setting_changed,description='骨骼链追随 IK 目标的程度，0 关闭')
    rotation_strength: FloatProperty(name='旋转强度',min=0,max=1,default=1,subtype='FACTOR',options=set(),
                              update=controls.setting_changed,description='0 保留原始关键姿势插值，1 完全采用生成旋转及目标旋转偏移')
    translation_strength: FloatProperty(name='位移强度',min=0,max=1,default=1,subtype='FACTOR',options=set(),
                              update=controls.setting_changed,description='生成位移及 IK 位置跟随的强度；四肢需配合 IK 使用')
    chain_length: IntProperty(name='联动祖先数',min=1,max=4,default=2,options=set(),update=controls.setting_changed)
    offset_location: FloatVectorProperty(name='目标位移',subtype='TRANSLATION',options=set(),update=controls.offset_changed)
    offset_rotation: FloatVectorProperty(name='目标旋转',subtype='EULER',options=set(),update=controls.offset_changed)
    effector: PointerProperty(type=bpy.types.Object)
    reach_error: FloatProperty(name='最大未达距离',options=set())


class R15KM_Settings(bpy.types.PropertyGroup):
    target: PointerProperty(name='R15 角色',type=bpy.types.Object,
                           poll=lambda self,obj: obj.type=='ARMATURE',update=target_changed)
    source_action: PointerProperty(type=bpy.types.Action)
    result_action: PointerProperty(type=bpy.types.Action)
    source_slot: StringProperty()
    motion_data: PointerProperty(type=bpy.types.Text)
    raw_object: PointerProperty(type=bpy.types.Object)
    raw_trail: PointerProperty(type=bpy.types.Object)
    raw_view: EnumProperty(name='原始数据',items=[('OFF','关闭',''),('OVERLAY','叠加','与角色重叠显示'),
                          ('SIDE','并排','在角色旁边显示原始 SOMA 骨架')],default='OFF',update=visualization.changed)
    raw_spacing: FloatProperty(name='并排距离',default=3.0,min=0,max=20,update=visualization.changed)
    raw_trajectory: BoolProperty(name='显示所选骨骼轨迹',default=False,update=visualization.changed)
    baseline_action: PointerProperty(type=bpy.types.Action)
    controls_collection: PointerProperty(type=bpy.types.Collection)
    control_proxy: PointerProperty(type=bpy.types.Object)
    controls_enabled: BoolProperty(default=False)
    bone_controls: CollectionProperty(type=R15KM_BoneControl)
    selected_bone: EnumProperty(name='骨骼',items=[(n,BONE_LABELS[n],n) for n in ALL_BODY],
                               default='RightHand',update=controls.selected_changed)
    show_effectors: BoolProperty(name='显示所选 IK 目标',default=True,update=controls.selected_changed)
    lock_anchors: BoolProperty(name='锁定已 K 姿势',default=True,options=set(),update=controls.setting_changed)
    lock_fade: FloatProperty(name='锁帧缓入缓出（秒）',default=0.2,min=0,max=2,options=set(),update=controls.setting_changed)
    controls_status: StringProperty()
    prompts: CollectionProperty(type=R15KM_Transition)
    seed: IntProperty(name='随机种子',default=42,min=1,max=2147483647)
    smoothing: FloatProperty(name='过渡平滑（秒）',default=0.08,min=0,max=0.5,precision=2,
                             description='降低生成动作的高频抖动，关键姿势仍精确保留；0 关闭')
    steps: IntProperty(name='生成步数',default=50,min=10,max=200)
    status: StringProperty(default='选择 R15 角色，然后读取关键帧')
    error: StringProperty()


def read_keys(s):
    ensure_idle()
    controls.reset_runtime()
    arm=s.target
    require(arm is not None, '请选择 R15 角色')
    require(arm.animation_data is not None and arm.animation_data.action is not None, '角色还没有 Action，请先 K 姿势')
    action=arm.animation_data.action
    slot=arm.animation_data.action_slot
    require(not action.get('r15_kimodo_generated',False), '请先返回关键姿势再读取，避免把生成帧当作输入')
    frames=animation.key_frames(arm,action,slot)
    old={(p.frame_start,p.frame_end):p.prompt for p in s.prompts}
    s.prompts.clear()
    for a,b in zip(frames,frames[1:]):
        p=s.prompts.add();p.frame_start=a;p.frame_end=b
        if (a,b) in old: p.prompt=old[(a,b)]
    s.source_action=action
    s.source_slot=slot.identifier
    s.status=f'{len(frames)} 个关键姿势 · {frames[0]}–{frames[-1]} 帧'
    s.error=''
    return frames


def _http(url, headers, body=None, timeout=600):
    data=json.dumps(body).encode('utf-8') if body is not None else None
    request=Request(url,data=data,headers=headers)
    try:
        with urlopen(request,timeout=timeout) as response:
            require(response.status==200, f'接口返回 HTTP {response.status}；当前插件需要同步生成服务')
            return json.loads(response.read().decode('utf-8'))
    except HTTPError as exc:
        detail=exc.read().decode('utf-8')
        raise RuntimeError(f'Kimodo HTTP {exc.code}: {detail}') from exc


def _worker(outbox,url,headers,request,cancelled):
    try:
        caps=_http(url+'/capabilities',headers,timeout=20)
        models=[m for m in caps['models'] if m['id']==request['model']]
        require(len(models)==1, '接口模型不存在或重复')
        require(models[0]['canonical_skeleton']==request['skeleton'], '服务骨架已变化，请在 Proscenium 重新连接后生成')
        responses=[]
        # Each transition gets both boundary constraints in its own generation window.
        # Avoid the model's multi-prompt overlap changing the timing of authored seams.
        for i,segment in enumerate(request['segments']):
            if cancelled.is_set(): return
            a,b=request['constraints'][i:i+2]
            part=copy.deepcopy(request)
            part['segments']=[dict(segment,duration_frames=b['frame']-a['frame']+1)]
            part['constraints']=[dict(a,frame=0),dict(b,frame=b['frame']-a['frame'])]
            part['options']['transition_frames']=0
            outbox.put(('progress',i+1))
            responses.append(_http(url+'/generate',headers,body=part))
        outbox.put(('success',responses))
    except Exception as exc:
        outbox.put(('error',(str(exc),traceback.format_exc())))
    return


def _tick():
    global _JOB,LAST_RUN
    if _JOB is None: return None
    job=_JOB
    s=job['scene'].r15_kimodo
    if job['queue'].empty():
        s.status=f"Kimodo 正在生成第 {job['part']}/{len(job['request']['segments'])} 段… {int(time.monotonic()-job['started'])} 秒"
        redraw()
        return 0.5
    state,value=job['queue'].get_nowait()
    if state=='progress':
        job['part']=value
        return 0.1
    try:
        if state=='error':
            message,trace=value
            print('[R15 Kimodo] ERROR\n'+trace)
            raise RuntimeError(message)
        require(bpy.context.scene is job['scene'], '生成期间切换了场景；请切回后重新生成')
        require(s.target is job['data']['arm'], '生成期间更换了目标角色')
        s.status='正在烘焙 R15 动画…'
        result,slot=animation.bake(value,job['data'])
        arm=job['data']['arm']
        require(s.target is arm, '生成期间更换了目标角色')
        if s.controls_enabled:controls.clear_setup(s,restore=False)
        s.result_action=result
        visualization.store(s,job['request'],value,job['data']['anchors'],job['data']['scene_fps'],job['data']['model_fps'])
        arm.animation_data.action=result
        arm.animation_data.action_slot=slot
        bpy.context.scene.frame_set(job['data']['anchors'][0])
        s.status=f"已完成 {len(job['data']['samples'])} 帧；关键姿势已验证"
        s.error=''
        LAST_RUN={'state':'success','request':job['request'],'response':value,'action':result.name,
                  'anchors':job['data']['anchors'],'seconds':time.monotonic()-job['started']}
    except Exception as exc:
        s.status='生成失败'
        s.error=str(exc)
        print('[R15 Kimodo] ERROR\n'+traceback.format_exc())
        LAST_RUN={'state':'error','error':str(exc),'request':job['request']}
    finally:
        _JOB=None
        redraw()
    return None


def start_generation(s):
    global _JOB,LAST_RUN
    ensure_idle()
    controls.reset_runtime()
    from proscenium_blender import mmcp_client
    require(s.target is not None and s.source_action is not None, '请先读取关键帧')
    model_id=bpy.context.scene.proscenium.model_id
    model=mmcp_client.cached_model(model_id)
    require(model is not None, '请在 Proscenium 面板连接 Kimodo 服务并选择模型')
    require('pose_keyframe' in model['supported_constraints'], '模型不支持关键姿势约束')
    require(s.source_slot in s.source_action.slots, '原始 Action 槽位已变化，请重新读取关键帧')
    source_slot=s.source_action.slots[s.source_slot]
    scene=bpy.context.scene
    request,data=animation.build_request(s.target,s.source_action,source_slot,model,s.prompts,s,scene)
    headers={'Content-Type':'application/json; charset=utf-8','Accept':'application/json, model/gltf+json'}
    token=mmcp_client.get_access_token()
    if token: headers['Authorization']='Bearer '+token
    outbox=queue.Queue()
    _JOB={'scene':scene,'request':request,'data':data,'queue':outbox,'started':time.monotonic(),'part':1,'cancelled':threading.Event()}
    LAST_RUN={'state':'running','anchors':data['anchors']}
    s.status='正在向 Kimodo 发送关键姿势和文本…';s.error=''
    print(f"[R15 Kimodo] Request: {len(data['anchors'])} poses, {len(request['segments'])} text segments, {data['scene_fps']} scene FPS / {data['model_fps']} model FPS")
    threading.Thread(target=_worker,args=(outbox,mmcp_client.get_mmcp_url(),headers,request,_JOB['cancelled']),daemon=True).start()
    bpy.app.timers.register(_tick,first_interval=0.5)
    redraw()
    return


class R15KM_OT_read(bpy.types.Operator):
    bl_idname='r15_kimodo.read_keys'
    bl_label='读取关键帧'
    bl_description='从角色当前 Action 读取手工关键姿势，为每个间隔创建文本描述'
    def execute(self,context):
        try: read_keys(context.scene.r15_kimodo)
        except Exception as exc:
            self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}


class R15KM_OT_record(bpy.types.Operator):
    bl_idname='r15_kimodo.record_pose'
    bl_label='记录当前全身姿势'
    bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        try:
            ensure_idle();s=context.scene.r15_kimodo;arm=s.target
            require(arm is not None and set(ALL_BODY).issubset(arm.pose.bones.keys()),'请选择完整的 R15 骨架')
            require(arm.animation_data is None or arm.animation_data.action is None or
                    not arm.animation_data.action.get('r15_kimodo_generated',False),'请先返回关键姿势再记录')
            frame=context.scene.frame_current
            for n in ALL_BODY:
                pb=arm.pose.bones[n]
                prop='rotation_quaternion' if pb.rotation_mode=='QUATERNION' else 'rotation_axis_angle' if pb.rotation_mode=='AXIS_ANGLE' else 'rotation_euler'
                pb.keyframe_insert(data_path='location',frame=frame,group=n)
                pb.keyframe_insert(data_path=prop,frame=frame,group=n)
            s.status=f'已记录第 {frame} 帧全身姿势；摆好另一个姿势后再记录'
        except Exception as exc:
            self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}


class R15KM_OT_generate(bpy.types.Operator):
    bl_idname='r15_kimodo.generate'
    bl_label='生成文本过渡'
    bl_description='将 R15 关键姿势发送到 Kimodo，并将生成结果烘焙为独立 Action'
    def execute(self,context):
        try: start_generation(context.scene.r15_kimodo)
        except Exception as exc:
            context.scene.r15_kimodo.error=str(exc)
            self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}


class R15KM_OT_switch(bpy.types.Operator):
    bl_idname='r15_kimodo.switch_action'
    bl_label='切换动作'
    result: bpy.props.BoolProperty(default=False)
    def execute(self,context):
        try:
            ensure_idle();s=context.scene.r15_kimodo
            action=s.result_action if self.result else s.source_action
            require(s.target is not None and action is not None,'没有可切换的动作')
            require(s.source_slot in action.slots,'动作槽位已变化')
            s.target.animation_data.action=action
            s.target.animation_data.action_slot=action.slots[s.source_slot]
            context.scene.frame_set(context.scene.frame_current)
            s.status='预览生成动作' if self.result else '已返回原始关键姿势；修改后重新读取关键帧'
        except Exception as exc:
            self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}


class R15KM_OT_cancel(bpy.types.Operator):
    bl_idname='r15_kimodo.cancel'
    bl_label='取消本次写回'
    bl_description='停止接收本次结果；已经提交的服务端计算可能继续运行'
    def execute(self,context):
        global _JOB,LAST_RUN
        require(_JOB is not None,'没有进行中的生成')
        _JOB['cancelled'].set()
        _JOB=None
        if bpy.app.timers.is_registered(_tick): bpy.app.timers.unregister(_tick)
        LAST_RUN={'state':'cancelled'}
        context.scene.r15_kimodo.status='已取消写回；原始 Action 保留'
        redraw()
        return {'FINISHED'}


class R15KM_PT_panel(bpy.types.Panel):
    bl_label='R15 关键姿势 + 文本过渡'
    bl_idname='R15KM_PT_panel'
    bl_space_type='VIEW_3D'
    bl_region_type='UI'
    bl_category='R15 Kimodo'
    def draw(self,context):
        s=context.scene.r15_kimodo;layout=self.layout
        col=layout.column();col.enabled=_JOB is None
        col.prop(s,'target')
        col.label(text='在 R15 上 K 姿势，再为间隔填写动作描述')
        col.operator('r15_kimodo.record_pose',icon='KEY_HLT')
        col.operator('r15_kimodo.read_keys',icon='ACTION')
        if s.source_action: col.label(text='关键姿势: '+s.source_action.name)
        for p in s.prompts:
            box=col.box();box.label(text=f'{p.frame_start} → {p.frame_end} 帧')
            box.prop(p,'prompt',text='文本')
        row=col.row(align=True);row.prop(s,'seed');row.prop(s,'steps')
        col.prop(s,'smoothing')
        col.operator('r15_kimodo.generate',icon='PLAY')
        row=col.row(align=True)
        row.operator('r15_kimodo.switch_action',text='返回关键姿势').result=False
        row.operator('r15_kimodo.switch_action',text='预览生成结果').result=True
        if _JOB is not None: layout.operator('r15_kimodo.cancel',icon='CANCEL')
        layout.label(text=s.status)
        if s.error:
            box=layout.box();box.alert=True;box.label(text='错误：'+s.error,icon='ERROR')
        layout.label(text='服务与模型沿用 Proscenium 设置')


class R15KM_OT_select_bone(bpy.types.Operator):
    bl_idname='r15_kimodo.select_bone'
    bl_label='选择调节骨骼'
    bone: StringProperty()
    def execute(self,context):
        context.scene.r15_kimodo.selected_bone=self.bone
        return {'FINISHED'}


class R15KM_OT_setup_controls(bpy.types.Operator):
    bl_idname='r15_kimodo.setup_controls'
    bl_label='创建骨骼调节器'
    def execute(self,context):
        try:controls.setup(context.scene.r15_kimodo)
        except Exception as exc:
            self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}


class R15KM_OT_apply_controls(bpy.types.Operator):
    bl_idname='r15_kimodo.apply_controls'
    bl_label='刷新并烘焙调节'
    def execute(self,context):
        try:
            controls.reset_runtime()
            controls.apply(context.scene.r15_kimodo)
        except Exception as exc:
            self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}


class R15KM_OT_select_effector(bpy.types.Operator):
    bl_idname='r15_kimodo.select_effector'
    bl_label='启用并选择 IK 目标'
    bl_description='将所选骨骼 IK 强度设为 1，选择目标球后可使用 G / R 调整'
    def execute(self,context):
        try:
            ensure_idle();s=context.scene.r15_kimodo
            require(s.controls_enabled,'请先创建骨骼调节器')
            item=s.bone_controls[s.selected_bone]
            require(item.effector is not None,'所选 IK 目标已被删除')
            if context.object is not None and context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
            s.show_effectors=True
            for obj in context.selected_objects:obj.select_set(False)
            item.effector.hide_set(False);item.effector.select_set(True)
            context.view_layer.objects.active=item.effector
            item.ik_strength=1
            for area in context.screen.areas:
                if area.type=='VIEW_3D':area.spaces.active.overlay.show_overlays=True
        except Exception as exc:
            self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}


class R15KM_OT_reset_bone(bpy.types.Operator):
    bl_idname='r15_kimodo.reset_bone'
    bl_label='重置所选骨骼'
    def execute(self,context):
        s=context.scene.r15_kimodo;p=s.bone_controls[s.selected_bone]
        p.ik_strength=0;p.rotation_strength=1;p.translation_strength=1
        p.offset_location=(0,0,0);p.offset_rotation=(0,0,0)
        p.chain_length=2 if p.name.endswith(('Hand','Foot')) or p.name=='Head' else 1
        return {'FINISHED'}


class R15KM_OT_clear_controls(bpy.types.Operator):
    bl_idname='r15_kimodo.clear_controls'
    bl_label='移除调节器并恢复生成动作'
    def execute(self,context):
        try:controls.clear_setup(context.scene.r15_kimodo)
        except Exception as exc:
            self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}


from bpy_extras.io_utils import ImportHelper
class R15KM_OT_import_motion(bpy.types.Operator,ImportHelper):
    bl_idname='r15_kimodo.import_raw_motion'
    bl_label='导入已有接口数据'
    filename_ext='.json'
    filter_glob: StringProperty(default='*.json',options={'HIDDEN'})
    def execute(self,context):
        try:
            ensure_idle();s=context.scene.r15_kimodo
            require(s.target is not None and s.result_action is not None,'先选择已生成动画的 R15 角色')
            from pathlib import Path
            from proscenium_blender.gltf_to_blender import read_extension_metadata
            payload=json.loads(Path(self.filepath).read_text(encoding='utf-8'))
            fps=float(read_extension_metadata(payload['responses'][0])['fps'])
            anchors=list(s.result_action['r15_kimodo_anchors'])
            visualization.store(s,payload['request'],payload['responses'],anchors,
                                s.result_action['r15_kimodo_scene_fps'],fps)
            visualization.data(s)
            s.raw_view='SIDE';s.error=''
        except Exception as exc:
            self.report({'ERROR'},str(exc));return {'CANCELLED'}
        return {'FINISHED'}


class R15KM_PT_raw(bpy.types.Panel):
    bl_label='原始动画数据可视化'
    bl_idname='R15KM_PT_raw'
    bl_parent_id='R15KM_PT_panel'
    bl_space_type='VIEW_3D';bl_region_type='UI';bl_category='R15 Kimodo'
    def draw(self,context):
        s=context.scene.r15_kimodo;layout=self.layout
        if s.motion_data is None:
            layout.label(text='生成后自动保存原始响应到 blend')
            layout.operator('r15_kimodo.import_raw_motion',icon='IMPORT')
            return
        layout.prop(s,'raw_view',expand=True)
        if s.raw_view=='SIDE':layout.prop(s,'raw_spacing')
        layout.prop(s,'raw_trajectory')
        if s.raw_trajectory:layout.prop(s,'selected_bone',text='轨迹骨骼')
        layout.label(text='蓝：左侧 · 橙：右侧 · 黄：躯干')
        layout.label(text='显示服务器原始骨架，未经平滑或锁帧')


class R15KM_PT_ik(bpy.types.Panel):
    bl_label='骨骼调节 · IK / Reach'
    bl_idname='R15KM_PT_ik'
    bl_parent_id='R15KM_PT_panel'
    bl_space_type='VIEW_3D';bl_region_type='UI';bl_category='R15 Kimodo'
    def draw(self,context):
        s=context.scene.r15_kimodo;layout=self.layout
        layout.enabled=_JOB is None
        if not s.controls_enabled:
            layout.operator('r15_kimodo.setup_controls',icon='CON_KINEMATIC')
            layout.label(text='对每个 R15 骨骼分别调节，原始 Action 保留')
            return
        rows=[(None,'Head',None),('LeftUpperArm','UpperTorso','RightUpperArm'),
              ('LeftLowerArm','LowerTorso','RightLowerArm'),('LeftHand','HumanoidRootPart','RightHand'),
              ('LeftUpperLeg',None,'RightUpperLeg'),('LeftLowerLeg',None,'RightLowerLeg'),('LeftFoot',None,'RightFoot')]
        for names in rows:
            row=layout.row(align=True)
            for n in names:
                col=row.column()
                if n is None:col.label(text='')
                else:col.operator('r15_kimodo.select_bone',text=BONE_LABELS[n],depress=s.selected_bone==n).bone=n
        layout.prop(s,'selected_bone')
        p=s.bone_controls[s.selected_bone]
        box=layout.box()
        box.prop(p,'ik_strength',slider=True)
        box.prop(p,'rotation_strength',slider=True)
        box.prop(p,'translation_strength',slider=True)
        box.prop(p,'chain_length')
        box.operator('r15_kimodo.select_effector',icon='EMPTY_AXIS')
        box.prop(p,'offset_location')
        box.prop(p,'offset_rotation')
        box.prop(s,'show_effectors')
        box.operator('r15_kimodo.reset_bone',icon='LOOP_BACK')
        layout.prop(s,'lock_anchors')
        if s.lock_anchors:layout.prop(s,'lock_fade')
        if s.lock_anchors and context.scene.frame_current in s.baseline_action['r15_kimodo_anchors']:
            layout.label(text='当前帧是锁定姿势；在中间帧预览调节',icon='LOCKED')
        layout.label(text=f'所选目标最大未达距离：{p.reach_error:.4f}')
        layout.operator('r15_kimodo.apply_controls',icon='ACTION')
        layout.label(text=s.controls_status)
        layout.operator('r15_kimodo.clear_controls',icon='X')


CLASSES=(R15KM_Transition,R15KM_BoneControl,R15KM_Settings,R15KM_OT_read,R15KM_OT_record,R15KM_OT_generate,
         R15KM_OT_switch,R15KM_OT_cancel,R15KM_PT_panel,R15KM_OT_select_bone,R15KM_OT_setup_controls,
         R15KM_OT_apply_controls,R15KM_OT_select_effector,R15KM_OT_reset_bone,R15KM_OT_clear_controls,
         R15KM_OT_import_motion,R15KM_PT_raw,R15KM_PT_ik)


@bpy.app.handlers.persistent
def _before_load(_):
    global _JOB,LAST_RUN
    if _JOB is not None: _JOB['cancelled'].set()
    _JOB=None
    LAST_RUN=None
    controls.reset_runtime();visualization.clear()
    if bpy.app.timers.is_registered(_tick): bpy.app.timers.unregister(_tick)
    return


def register():
    for cls in CLASSES: bpy.utils.register_class(cls)
    bpy.types.Scene.r15_kimodo=PointerProperty(type=R15KM_Settings)
    bpy.app.handlers.load_pre.append(_before_load)
    bpy.types.Action.r15_kimodo_data=PointerProperty(type=bpy.types.Text)
    visualization.register();controls.register()
    return


def unregister():
    global _JOB
    if _JOB is not None: _JOB['cancelled'].set()
    _JOB=None
    if bpy.app.timers.is_registered(_tick): bpy.app.timers.unregister(_tick)
    if _before_load in bpy.app.handlers.load_pre: bpy.app.handlers.load_pre.remove(_before_load)
    visualization.unregister();controls.unregister()
    del bpy.types.Action.r15_kimodo_data
    del bpy.types.Scene.r15_kimodo
    for cls in reversed(CLASSES): bpy.utils.unregister_class(cls)
    return
