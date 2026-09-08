"""Action sampling, MMCP response decoding, and R15 baking."""
import bisect
import json
import math
import bpy
from mathutils import Matrix, Quaternion, Vector
from .core import ALL_BODY, BODY, ROOT, PARENTS, PAIRS, Calibration, require, qblend, interval, angular_error


def channelbag(action, slot):
    for layer in action.layers:
        for strip in layer.strips:
            if strip.type == 'KEYFRAME':
                bag = strip.channelbag(slot)
                if bag is not None:
                    return bag
    raise ValueError('Action 没有可读取的关键帧通道')


def curves(action, slot):
    return channelbag(action, slot).fcurves


def key_frames(arm, action, slot):
    prefixes = tuple(arm.pose.bones[n].path_from_id() + '.' for n in ALL_BODY)
    frames = set()
    for fc in curves(action, slot):
        if fc.data_path.startswith(prefixes):
            for kp in fc.keyframe_points:
                if kp.type != 'GENERATED':
                    f = float(kp.co.x)
                    require(abs(f-round(f)) < 1e-5, '关键姿势请放在整数帧')
                    frames.add(round(f))
    require(len(frames) >= 2, '至少需要两个 R15 关键姿势；摆姿势后按 I 插入关键帧')
    return sorted(frames)


def snapshot(arm, action, slot, frames):
    scene = bpy.context.scene
    ad = arm.animation_data
    saved_action, saved_slot = ad.action, ad.action_slot
    saved_frame, saved_subframe = scene.frame_current, scene.frame_subframe
    result = {}
    try:
        ad.action = action
        ad.action_slot = slot
        for frame in frames:
            scene.frame_set(int(frame), subframe=frame-int(frame))
            bpy.context.view_layer.update()
            matrices, bases = {}, {}
            for n in ALL_BODY:
                pb = arm.pose.bones[n]
                require(max(abs(x-1) for x in pb.matrix_basis.to_scale()) < 1e-4,
                        f'不支持缩放关键帧: {n} / {frame}')
                matrices[n] = pb.matrix.copy()
                bases[n] = pb.matrix_basis.copy()
            result[frame] = {'matrices':matrices, 'bases':bases}
    finally:
        ad.action = saved_action
        ad.action_slot = saved_slot
        scene.frame_set(saved_frame, subframe=saved_subframe)
        bpy.context.view_layer.update()
    return result


class Motion:
    def __init__(self, gltf, calibration):
        from proscenium_blender.gltf_to_blender import _read_floats
        require(len(gltf['animations']) == 1, '服务必须返回一段动画')
        animation = gltf['animations'][0]
        self.tracks = {}
        self.root = None
        nodes = gltf['nodes']
        for channel in animation['channels']:
            target = channel['target']
            name, path = nodes[target['node']]['name'], target['path']
            require(name in calibration.order, '响应中存在未知关节: '+name)
            sampler = animation['samplers'][channel['sampler']]
            require(sampler.get('interpolation', 'LINEAR') == 'LINEAR', '只支持 LINEAR 动画采样')
            times = _read_floats(gltf, sampler['input'], 'SCALAR')
            require(len(times)>=2 and all(a<b for a,b in zip(times,times[1:])), '响应时间轴无效')
            if path == 'rotation':
                raw = _read_floats(gltf,sampler['output'],'VEC4')
                require(len(raw)==len(times)*4 and name not in self.tracks, '旋转轨道数据无效: '+name)
                values=[]
                for i in range(0,len(raw),4):
                    q=Quaternion((raw[i+3],raw[i],raw[i+1],raw[i+2]))
                    require(all(math.isfinite(v) for v in q) and q.magnitude > 1e-6, '响应四元数无效')
                    values.append(q.normalized())
                self.tracks[name]=(times,values)
            elif path == 'translation':
                require(name=='Hips' and self.root is None, '响应包含非根平移或重复根轨道')
                raw=_read_floats(gltf,sampler['output'],'VEC3')
                require(len(raw)==len(times)*3 and all(math.isfinite(v) for v in raw),'根轨道无效')
                self.root=(times,[Vector(raw[i:i+3]) for i in range(0,len(raw),3)])
            else:
                raise ValueError('不支持的响应通道: '+path)
        require(set(self.tracks)==set(calibration.order) and self.root is not None, '响应缺少源骨架旋转或根平移')
        self.duration=min(track[0][-1] for track in list(self.tracks.values())+[self.root])

    @staticmethod
    def sample_track(track, time, rotation):
        times,values=track
        require(times[0]-1e-5 <= time <= times[-1]+1e-5, '响应未覆盖请求的时间范围')
        i=max(0,min(len(times)-2,bisect.bisect_right(times,time)-1))
        t=max(0.0,min(1.0,(time-times[i])/(times[i+1]-times[i])))
        return qblend(values[i],values[i+1],t) if rotation else values[i].lerp(values[i+1],t)

    def sample(self,time):
        return {n:self.sample_track(t,time,True) for n,t in self.tracks.items()},self.sample_track(self.root,time,False)


def build_request(arm, action, slot, model, prompts, settings, scene):
    require(not action.get('r15_kimodo_generated', False), '请选择原始关键姿势 Action，或点击返回关键姿势')
    require(not arm.animation_data.nla_tracks or not arm.animation_data.use_nla,
            '请先关闭该骨架的 NLA 求值，再生成过渡')
    calibration=Calibration(arm,model['canonical_skeleton'])
    anchors=key_frames(arm,action,slot)
    scene_fps=scene.render.fps/scene.render.fps_base
    model_fps=float(model['fps'])
    indices=[round((f-anchors[0])*model_fps/scene_fps) for f in anchors]
    require(all(a<b for a,b in zip(indices,indices[1:])), '相邻关键姿势太近，低于模型的一帧')
    count=indices[-1]+1
    require(count/model_fps <= model['limits']['max_duration_seconds'], '关键姿势跨度超过服务时长限制')
    require(len(anchors)<=model['limits']['max_constraints_per_request'], '关键姿势数量超过服务约束上限')
    require(len(prompts)==len(anchors)-1, '关键帧已变化，请重新读取关键帧')
    segments=[]
    for i,p in enumerate(prompts):
        require((p.frame_start,p.frame_end)==(anchors[i],anchors[i+1]), '关键帧已变化，请重新读取关键帧')
        text=p.prompt.strip()
        require(bool(text), f'请填写 {p.frame_start}–{p.frame_end} 的过渡描述')
        require(len(text)<=model['limits']['max_prompt_length'], '文本超过服务长度限制')
        segments.append({'type':'text','prompt':text,'duration_frames':indices[i+1]-indices[i]+(i==len(prompts)-1)})
    samples=snapshot(arm,action,slot,range(anchors[0],anchors[-1]+1))
    constraints=[]
    for f,index in zip(anchors,indices):
        rotations,root=calibration.to_source(samples[f]['matrices'])
        constraints.append({'type':'pose_keyframe','frame':index,'joint_rotations':rotations,
                            'root_position':root,'fill_mode':'rest'})
    request={'protocol_version':'1.0','model':model['id'],'skeleton':model['canonical_skeleton'],
             'segments':segments,'constraints':constraints,
             'options':{'diffusion_steps':settings.steps,'num_samples':1,'seed':settings.seed,
                        'post_processing':True,'transition_frames':min(10,min(b-a for a,b in zip(indices,indices[1:]))),
                        'guidance':{'type':'separated','weight':[2.0,2.0]}}}
    require(len(json.dumps(request).encode('utf-8'))<=model['limits']['max_request_bytes'],'请求超过服务大小限制')
    return request, {'arm':arm,'action':action,'slot':slot,'anchors':anchors,'indices':indices,
                     'scene_fps':scene_fps,'model_fps':model_fps,'calibration':calibration,'samples':samples,
                     'smoothing':settings.smoothing,
                     'rotation_modes':{n:arm.pose.bones[n].rotation_mode for n in ALL_BODY}}


def write_action(arm, template, slot, baked, anchors, rotation_modes, name, bone_names=BODY):
    result=template.copy()
    result.name=name
    result.use_fake_user=True
    result_slot=result.slots[slot.identifier]
    bag=channelbag(result,result_slot)
    paths={arm.pose.bones[n].path_from_id():n for n in bone_names}
    for fc in list(bag.fcurves):
        bone_path,sep,prop=fc.data_path.rpartition('.')
        if bone_path in paths and prop in {'location','rotation_euler','rotation_quaternion','rotation_axis_angle'}:
            bag.fcurves.remove(fc)
    times=list(baked)
    for n in bone_names:
        mode=rotation_modes[n]
        rotations=[]
        previous=None
        for f in times:
            q=baked[f][n].to_quaternion()
            if mode=='QUATERNION':
                if previous is not None: q.make_compatible(previous)
                value=q
            elif mode=='AXIS_ANGLE':
                axis,angle=q.to_axis_angle();value=(angle,*axis)
            else:
                value=q.to_euler(mode,previous) if previous is not None else q.to_euler(mode)
            previous=value
            rotations.append(tuple(value))
        prop='rotation_quaternion' if mode=='QUATERNION' else 'rotation_axis_angle' if mode=='AXIS_ANGLE' else 'rotation_euler'
        for property_name,values in [('location',[tuple(baked[f][n].translation) for f in times]),(prop,rotations)]:
            for index in range(len(values[0])):
                fc=bag.fcurves.new(data_path=arm.pose.bones[n].path_from_id()+'.'+property_name,index=index)
                fc.keyframe_points.add(len(times))
                fc.keyframe_points.foreach_set('co',[v for f,row in zip(times,values) for v in (float(f),row[index])])
                for kp,f in zip(fc.keyframe_points,times):
                    kp.interpolation='LINEAR'
                    kp.type='KEYFRAME' if f in anchors else 'GENERATED'
                fc.update()
    return result,result_slot


def bake(gltf, job):
    arm, original, slot=job['arm'],job['action'],job['slot']
    require(bpy.data.objects.get(arm.name) is arm, '生成期间目标骨架已删除')
    require(all(arm.pose.bones[n].rotation_mode==m for n,m in job['rotation_modes'].items()),
            '生成期间旋转模式发生变化，请恢复后重新生成')
    cal,anchors,indices=job['calibration'],job['anchors'],job['indices']
    require(len(gltf)==len(anchors)-1, '服务返回的过渡段数不正确')
    motions=[Motion(part,cal) for part in gltf]
    for i,motion in enumerate(motions):
        require(motion.duration+1e-4>=(indices[i+1]-indices[i])/job['model_fps'],'服务返回动画过短')
    corrected_samples={}
    for i,(a,b) in enumerate(zip(anchors,anchors[1:])):
        raw={}
        for f in range(a,b+1):
            time=(f-a)/(b-a)*(indices[i+1]-indices[i])/job['model_fps']
            raw[f]=cal.to_target(*motions[i].sample(time))
        sigma=job['smoothing']*job['scene_fps']
        filtered={}
        for f in raw:
            if sigma==0:
                filtered[f]=raw[f]
                continue
            radius=math.ceil(2*sigma)
            neighbors=list(range(max(a,f-radius),min(b,f+radius)+1))
            weights=[math.exp(-0.5*((k-f)/sigma)**2) for k in neighbors]
            total=sum(weights)
            rotations={}
            for n in BODY:
                center=raw[f][0][n]
                accum=Vector((0,0,0,0))
                for k,w in zip(neighbors,weights):
                    q=raw[k][0][n].copy();q.make_compatible(center)
                    accum+=Vector(q)*w
                rotations[n]=Quaternion(accum).normalized()
            position=sum((raw[k][1]*w for k,w in zip(neighbors,weights)),Vector((0,0,0)))/total
            filtered[f]=(rotations,position)
        corrections={}
        for f in (a,b):
            rotations,position=filtered[f]
            ref=job['samples'][f]['matrices']
            corrections[f]=({n:ref[n].to_quaternion() @ rotations[n].inverted() for n in BODY},
                            ref['LowerTorso'].translation-position)
        qa,pa=corrections[a];qb,pb=corrections[b]
        for f,(rotations,position) in filtered.items():
            t=(f-a)/(b-a);smooth=t*t*(3-2*t)
            corrected={n:qblend(qa[n],qb[n],smooth) @ rotations[n] for n in BODY}
            corrected_samples[f]=(corrected,position+pa.lerp(pb,smooth))
    baked={}
    for f,(corrected,root) in corrected_samples.items():
        ref=job['samples'][f]
        matrices={ROOT:ref['matrices'][ROOT]}
        bases={}
        for n in BODY:
            parent=PARENTS[n]
            parent_matrix=matrices[parent]
            local_rest=cal.rest[parent].inverted() @ cal.rest[n]
            position_n=root if n=='LowerTorso' else (
                parent_matrix @ local_rest @ Matrix.Translation(ref['bases'][n].translation)).translation
            desired=corrected[n].to_matrix().to_4x4()
            desired.translation=position_n
            basis=arm.data.bones[n].convert_local_to_pose(desired,cal.rest[n],
                  parent_matrix=parent_matrix,parent_matrix_local=cal.rest[parent],invert=True)
            bases[n]=basis
            matrices[n]=desired
        baked[f]=bases
    result,result_slot=write_action(arm,original,slot,baked,anchors,job['rotation_modes'],original.name+' · Kimodo')
    result['r15_kimodo_generated']=True
    result['r15_kimodo_source']=original.name
    result['r15_kimodo_anchors']=anchors
    result['r15_kimodo_scene_fps']=job['scene_fps']
    result['r15_kimodo_smoothing']=job['smoothing']
    original.use_fake_user=True
    # Evaluate against the original rig, including its Roblox joint metadata.
    try:
        checks=snapshot(arm,result,result_slot,anchors)
    except Exception:
        bpy.data.actions.remove(result)
        raise
    max_rotation=max(angular_error(checks[f]['matrices'][n].to_quaternion(),job['samples'][f]['matrices'][n].to_quaternion())
                     for f in anchors for n in ALL_BODY)
    max_position=max((checks[f]['matrices'][n].translation-job['samples'][f]['matrices'][n].translation).length
                     for f in anchors for n in ALL_BODY)
    if max_rotation>=0.003 or max_position>=0.001:
        bpy.data.actions.remove(result)
        raise ValueError(f'关键姿势写回验证失败: angle={max_rotation}, distance={max_position}')
    result['r15_kimodo_max_anchor_angle']=max_rotation
    result['r15_kimodo_max_anchor_position']=max_position
    print(f'[R15 Kimodo] Baked {len(baked)} frames; {len(anchors)} anchors; max angle {max_rotation:.6f} rad; position {max_position:.8f}')
    return result,result_slot
