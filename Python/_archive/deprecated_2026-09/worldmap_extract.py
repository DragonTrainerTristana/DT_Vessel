"""WorldMap.unity → 항해 가능 영역 지오메트리 추출.

Unity 씬 YAML 을 파싱해 육지(섬·해안)·벽을 **월드 좌표 프리미티브**로 꺼낸다.
부모 체인을 타고 로컬→월드 변환을 직접 계산한다(Unity 없이).

출력: worldmap_geometry.json
  { unit_per_?, bounds, landmarks{부산,대만,...},
    boxes[{cx,cz,hx,hz,yaw}], capsules[{cx,cz,r}], meshes[...] }
"""
import json
import math
import os
import re
import sys

SCENE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'Scenes', 'WorldMap.unity')


def unesc(s):
    s = s.strip().strip('"')
    try:
        return s.encode().decode('unicode_escape')
    except Exception:
        return s


def load(path=SCENE):
    txt = open(path, encoding='utf-8', errors='replace').read()
    parts = re.split(r'\n--- !u!(\d+) &(\d+)(?: stripped)?\n', txt)
    docs = {}
    for i in range(1, len(parts), 3):
        docs[parts[i + 1]] = (parts[i], parts[i + 2])
    return docs


F3 = r'\{x: ([-\d.eE+]+), y: ([-\d.eE+]+), z: ([-\d.eE+]+)\}'
F4 = r'\{x: ([-\d.eE+]+), y: ([-\d.eE+]+), z: ([-\d.eE+]+), w: ([-\d.eE+]+)\}'


def parse(docs):
    """GameObject/Transform/Collider 를 표로 정리."""
    go = {}          # fid -> name
    tf = {}          # fid -> dict(go, pos, rot(quat), scale, father)
    cols = []        # (kind, go_fid, params)
    for fid, (cid, body) in docs.items():
        if cid == '1':
            m = re.search(r'm_Name: (.+)', body)
            if m:
                go[fid] = unesc(m.group(1))
        elif cid == '4':
            g = re.search(r'm_GameObject: \{fileID: (\d+)\}', body)
            p = re.search(r'm_LocalPosition: ' + F3, body)
            q = re.search(r'm_LocalRotation: ' + F4, body)
            s = re.search(r'm_LocalScale: ' + F3, body)
            f = re.search(r'm_Father: \{fileID: (\d+)\}', body)
            if g and p:
                tf[fid] = dict(
                    go=g.group(1),
                    pos=[float(x) for x in p.groups()],
                    rot=[float(x) for x in q.groups()] if q else [0, 0, 0, 1],
                    scale=[float(x) for x in s.groups()] if s else [1, 1, 1],
                    father=f.group(1) if f else '0')
        elif cid in ('65', '136', '64'):     # Box, Capsule, Mesh collider
            g = re.search(r'm_GameObject: \{fileID: (\d+)\}', body)
            if not g:
                continue
            kind = {'65': 'box', '136': 'capsule', '64': 'mesh'}[cid]
            prm = {}
            c = re.search(r'm_Center: ' + F3, body)
            if c:
                prm['center'] = [float(x) for x in c.groups()]
            sz = re.search(r'm_Size: ' + F3, body)
            if sz:
                prm['size'] = [float(x) for x in sz.groups()]
            for key, rx in (('radius', r'm_Radius: ([-\d.eE+]+)'),
                            ('height', r'm_Height: ([-\d.eE+]+)'),
                            ('direction', r'm_Direction: (\d+)')):
                m = re.search(rx, body)
                if m:
                    prm[key] = float(m.group(1))
            cols.append((kind, g.group(1), prm))
    # GameObject fid -> Transform fid
    go2tf = {v['go']: k for k, v in tf.items()}
    return go, tf, cols, go2tf


def quat_yaw(q):
    x, y, z, w = q
    return math.atan2(2 * (w * y + x * z), 1 - 2 * (y * y + x * x))


def world(tf_fid, tf):
    """부모 체인을 타고 월드 (x, z, yaw, sx, sz) 계산 (Y축 회전만 취급)."""
    px, pz, yaw, sx, sz = 0.0, 0.0, 0.0, 1.0, 1.0
    chain = []
    f = tf_fid
    seen = set()
    while f in tf and f not in seen:
        seen.add(f)
        chain.append(f)
        f = tf[f]['father']
    for f in reversed(chain):                       # 루트→리프
        t = tf[f]
        lx, _, lz = t['pos']
        lsx, _, lsz = t['scale']
        ly = quat_yaw(t['rot'])
        c, s = math.cos(yaw), math.sin(yaw)
        # 부모 스케일·회전 적용 후 누적
        wx = px + (lx * sx) * c + (lz * sz) * s
        wz = pz - (lx * sx) * s + (lz * sz) * c
        px, pz = wx, wz
        sx, sz = sx * lsx, sz * lsz
        yaw = yaw + ly
    return px, pz, yaw, sx, sz


def main():
    docs = load()
    go, tf, cols, go2tf = parse(docs)
    print(f'GameObject {len(go)} / Transform {len(tf)} / Collider {len(cols)}')

    out = dict(boxes=[], capsules=[], meshes=[], landmarks={}, named=[])
    for kind, gfid, prm in cols:
        tfid = go2tf.get(gfid)
        if tfid is None:
            continue
        x, z, yaw, sx, sz = world(tfid, tf)
        nm = go.get(gfid, '')
        cx, cz = prm.get('center', [0, 0, 0])[0], prm.get('center', [0, 0, 0])[2]
        c, s = math.cos(yaw), math.sin(yaw)
        x += (cx * sx) * c + (cz * sz) * s
        z += -(cx * sx) * s + (cz * sz) * c
        if kind == 'box':
            w_, _, d_ = prm.get('size', [1, 1, 1])
            out['boxes'].append(dict(name=nm, cx=x, cz=z, hx=abs(w_ * sx) / 2,
                                     hz=abs(d_ * sz) / 2, yaw=math.degrees(yaw)))
        elif kind == 'capsule':
            r = prm.get('radius', 0.5) * max(abs(sx), abs(sz))
            out['capsules'].append(dict(name=nm, cx=x, cz=z, r=r))
        else:
            out['meshes'].append(dict(name=nm, cx=x, cz=z, sx=sx, sz=sz))

    for gfid, nm in go.items():
        tfid = go2tf.get(gfid)
        if tfid is None:
            continue
        x, z, yaw, sx, sz = world(tfid, tf)
        out['named'].append(dict(name=nm, x=x, z=z, sx=sx, sz=sz))
        if nm in ('부산', '대만', '대한해협', '해안', 'WorldGrid'):
            out['landmarks'][nm] = dict(x=x, z=z, sx=sx, sz=sz)

    allx = [b['cx'] for b in out['boxes']] + [c['cx'] for c in out['capsules']] + \
           [m['cx'] for m in out['meshes']]
    allz = [b['cz'] for b in out['boxes']] + [c['cz'] for c in out['capsules']] + \
           [m['cz'] for m in out['meshes']]
    if allx:
        out['bounds'] = dict(xmin=min(allx), xmax=max(allx), zmin=min(allz), zmax=max(allz))

    print(f"  box {len(out['boxes'])}  capsule {len(out['capsules'])}  mesh {len(out['meshes'])}")
    print('  landmarks:')
    for k, v in out['landmarks'].items():
        print(f"    {k:10s} ({v['x']:10.1f}, {v['z']:10.1f})  scale ({v['sx']:.2f},{v['sz']:.2f})")
    if 'bounds' in out:
        b = out['bounds']
        print(f"  콜라이더 범위 x {b['xmin']:.0f}~{b['xmax']:.0f}  z {b['zmin']:.0f}~{b['zmax']:.0f}")
    lm = out['landmarks']
    if '부산' in lm and '대만' in lm:
        d = math.dist((lm['부산']['x'], lm['부산']['z']), (lm['대만']['x'], lm['대만']['z']))
        print(f"  부산↔대만 월드 직선 {d:,.1f} 유닛")
        out['busan_taiwan_units'] = d

    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'worldmap_geometry.json')
    if len(sys.argv) > 1:
        dst = sys.argv[1]
    json.dump(out, open(dst, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('저장:', dst)


if __name__ == '__main__':
    main()
