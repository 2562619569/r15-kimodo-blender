"""Bidirectional SOMA30/R15 pose conversion in armature space."""
import bisect
import math
from mathutils import Matrix, Quaternion, Vector

ROOT = "HumanoidRootPart"
PAIRS = {
    "LowerTorso": "Hips", "UpperTorso": "Chest", "Head": "Head",
    "LeftUpperArm": "LeftArm", "LeftLowerArm": "LeftForeArm", "LeftHand": "LeftHand",
    "RightUpperArm": "RightArm", "RightLowerArm": "RightForeArm", "RightHand": "RightHand",
    "LeftUpperLeg": "LeftLeg", "LeftLowerLeg": "LeftShin", "LeftFoot": "LeftFoot",
    "RightUpperLeg": "RightLeg", "RightLowerLeg": "RightShin", "RightFoot": "RightFoot",
}
BODY = tuple(PAIRS)
ALL_BODY = (ROOT,) + BODY
PARENTS = {"LowerTorso": ROOT, "UpperTorso": "LowerTorso", "Head": "UpperTorso"}
for side in ("Left", "Right"):
    PARENTS.update({side+"UpperArm": "UpperTorso", side+"LowerArm": side+"UpperArm",
                    side+"Hand": side+"LowerArm", side+"UpperLeg": "LowerTorso",
                    side+"LowerLeg": side+"UpperLeg", side+"Foot": side+"LowerLeg"})
ENDPOINTS = {
    "UpperTorso": ("Head", "Neck1"),
    "LeftUpperArm": ("LeftLowerArm", "LeftForeArm"),
    "LeftLowerArm": ("LeftHand", "LeftHand"),
    "LeftHand": (None, "LeftHandMiddleEnd"),
    "RightUpperArm": ("RightLowerArm", "RightForeArm"),
    "RightLowerArm": ("RightHand", "RightHand"),
    "RightHand": (None, "RightHandMiddleEnd"),
    "LeftUpperLeg": ("LeftLowerLeg", "LeftShin"),
    "LeftLowerLeg": ("LeftFoot", "LeftFoot"),
    "LeftFoot": (None, "LeftToeBase"),
    "RightUpperLeg": ("RightLowerLeg", "RightShin"),
    "RightLowerLeg": ("RightFoot", "RightFoot"),
    "RightFoot": (None, "RightToeBase"),
}
EPSILON = 1e-6


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unit(vector):
    require(vector.length > EPSILON, "骨架校准方向长度为零")
    return vector.normalized()


def segment_frame(direction, secondary):
    y = unit(direction)
    x = unit(y.cross(secondary))
    z = unit(x.cross(y))
    return Matrix((x, y, z)).transposed()


def qblend(a, b, factor):
    b = b.copy()
    b.make_compatible(a)
    return a.slerp(b, factor).normalized()


def interval(frames, frame):
    i = max(0, min(len(frames)-2, bisect.bisect_right(frames, frame)-1))
    a, b = frames[i:i+2]
    return a, b, (frame-a)/(b-a)


class Calibration:
    def __init__(self, arm, skeleton):
        require(arm is not None and arm.type == 'ARMATURE', "请选择 R15 骨架")
        require(skeleton['coordinate_system'] == 'right_handed_y_up' and skeleton['units'] == 'meters',
                "服务骨架必须使用 Y-up / meters")
        bones = arm.data.bones
        require(set(ALL_BODY).issubset(bones.keys()), "骨架缺少标准 R15 身体骨骼")
        require(bones[ROOT].parent is None, "HumanoidRootPart 必须是根骨骼")
        for name in BODY:
            require(bones[name].parent is not None and bones[name].parent.name == PARENTS[name],
                    "R15 父子关系错误: " + name)
        for name in ALL_BODY:
            pb = arm.pose.bones[name]
            require(not any(not c.mute and c.influence > 0 for c in pb.constraints),
                    "请先烘焙或关闭身体骨骼上的约束: " + name)
            require(pb.bone.inherit_scale == 'FULL', "不支持特殊继承缩放: " + name)
        self.joints = skeleton['joints']
        self.order = [j['name'] for j in self.joints]
        needed = set(PAIRS.values()) | {'Spine1', 'Spine2', 'Neck1', 'Neck2',
                 'LeftShoulder', 'RightShoulder', 'LeftHandMiddleEnd', 'RightHandMiddleEnd',
                 'LeftToeBase', 'RightToeBase'}
        require(needed.issubset(self.order), "服务骨架不符合 SOMA30 命名")
        self.parent = {}
        self.positions = {}
        for j in self.joints:
            name, parent = j['name'], j['parent']
            require(parent is None or parent in self.positions, "源骨架必须按父子顺序排列")
            require(abs(j['rest_rotation'][3]) > 1-EPSILON, "源骨架初始旋转必须为单位旋转")
            self.parent[name] = parent
            self.positions[name] = Vector(j['rest_translation']) + (
                self.positions[parent] if parent is not None else Vector((0,0,0)))
        require(self.parent['Hips'] is None, "SOMA 根关节必须为 Hips")
        left = unit(bones['LeftUpperLeg'].head_local-bones['RightUpperLeg'].head_local)
        up_raw = bones['Head'].head_local-bones['LowerTorso'].head_local
        up = unit(up_raw-left*up_raw.dot(left))
        forward = unit(left.cross(up))
        self.axes = Matrix((left, up, forward)).transposed()
        self.rest = {n: bones[n].matrix_local.copy() for n in ALL_BODY}
        self.pelvis = bones['LowerTorso'].head_local.copy()
        self.scale = sum((bones[s+'LowerLeg'].head_local-bones[s+'UpperLeg'].head_local).length +
                         (bones[s+'Foot'].head_local-bones[s+'LowerLeg'].head_local).length
                         for s in ('Left','Right')) / sum(
                         (self.positions[s+'Shin']-self.positions[s+'Leg']).length +
                         (self.positions[s+'Foot']-self.positions[s+'Shin']).length
                         for s in ('Left','Right'))
        require(self.scale > EPSILON, "腿长比例无效")
        self.offsets = {}
        for target, source in PAIRS.items():
            if target == 'LowerTorso':
                ft, fs = self.axes, Matrix.Identity(3)
            elif target == 'Head':
                ft = segment_frame(bones[target].tail_local-bones[target].head_local, forward)
                fs = segment_frame(self.positions['Head']-self.positions['Neck2'], Vector((0,0,1)))
            else:
                te, se = ENDPOINTS[target]
                tip = bones[te].head_local if te is not None else bones[target].tail_local
                secondary_t = up if target.endswith('Foot') else forward
                secondary_s = Vector((0,1,0)) if target.endswith('Foot') else Vector((0,0,1))
                ft = segment_frame(tip-bones[target].head_local, secondary_t)
                fs = segment_frame(self.positions[se]-self.positions[source], secondary_s)
            # Source absolute orientation = axes^-1 * target_pose * offset.
            self.offsets[target] = self.rest[target].to_quaternion().to_matrix().transposed() @ ft @ fs.transposed()

    def to_source(self, matrices):
        desired = {src: (self.axes.transposed() @ matrices[tgt].to_quaternion().to_matrix() @
                   self.offsets[tgt]).to_quaternion() for tgt,src in PAIRS.items()}
        globals_ = {}
        locals_ = {}
        for name in self.order:
            parent = self.parent[name]
            if name in ('Spine1','Spine2'):
                q = qblend(desired['Hips'], desired['Chest'], {'Spine1':1/3,'Spine2':2/3}[name])
            elif name in ('Neck1','Neck2'):
                q = qblend(desired['Chest'], desired['Head'], {'Neck1':1/3,'Neck2':2/3}[name])
            elif name in desired:
                q = desired[name]
            else:
                require(parent is not None, "未映射的源根骨骼: " + name)
                q = globals_[parent].copy()
            globals_[name] = q
            local = globals_[parent].inverted() @ q if parent is not None else q
            locals_[name] = [local.x,local.y,local.z,local.w]
        position = self.positions['Hips'] + self.axes.transposed() @ (
            matrices['LowerTorso'].translation-self.pelvis)/self.scale
        return locals_, list(position)

    def to_target(self, local_rotations, root_position):
        globals_ = {}
        for name in self.order:
            parent = self.parent[name]
            q = local_rotations[name]
            globals_[name] = globals_[parent] @ q if parent is not None else q
        rotations = {tgt: (self.axes @ globals_[src].to_matrix() @
                     self.offsets[tgt].transposed()).to_quaternion()
                     for tgt,src in PAIRS.items()}
        position = self.pelvis + self.axes @ (root_position-self.positions['Hips'])*self.scale
        return rotations, position


def angular_error(a, b):
    dot=sum(x*y for x,y in zip(a,b))
    norm=math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))
    return 2*math.acos(min(1.0, abs(dot/norm)))
