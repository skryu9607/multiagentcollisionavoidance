import numpy as np
import numpy.linalg as LA

class Util():
    def __init__(self) -> None:
        pass

    def chk_inside(self, p1, obs):
        chk = 0
        flag = 0
        for i in range(len(obs) - 1):
            if np.cross([obs[i + 1][0] - obs[i][0], obs[i + 1][1] - obs[i][1], 0],
                        [p1[0] - obs[i][0], p1[1] - obs[i][1], 0])[2] < 0:
                chk = chk + 1
                # print(obs[i+1],obs[i],chk)
        if chk == (len(obs) - 1):
            flag = 1
        return flag

    def chk_cross(self, a,b,p,q):
        aa = np.array([a[0], a[1], 0])
        bb = np.array([b[0], b[1], 0])
        pp = np.array([p[0], p[1], 0])
        qq = np.array([q[0], q[1], 0])
        # print(aa,bb,pp,qq)
        c1 = np.cross(bb - aa, pp - aa)
        c2 = np.cross(bb - aa, qq - aa)
        c3 = np.cross(pp - qq, aa - qq)
        c4 = np.cross(pp - qq, bb - qq)
        # print(c1,c2,c3,c4)
        if (np.dot(c1,c2) < 0) & (np.dot(c3,c4) < 0):
            out = 1
        else:
            out = 0

        return out

    def get_crosspt(self, P11, P12, P21, P22):
        x11 = P11[0]
        y11 = P11[1]
        x12 = P12[0]
        y12 = P12[1]
        x21 = P21[0]
        y21 = P21[1]
        x22 = P22[0]
        y22 = P22[1]

        if x12==x11 or x21==x22:

            if x12 == x11:
                cx = x11
                m2 = (y22 - y21) / (x22 - x21)
                cy = m2 * cx - m2 * x21 + y21
            if x21 == x22:
                cx = x21
                m1 = (y12 - y11) / (x12 - x11)
                cy = m1 * cx - m1 * x11 + y11

            return [cx, cy]
        m1 = (y12 - y11) / (x12 - x11)
        m2 = (y22 - y21) / (x22 - x21)

        if m1==m2:
            print('parallel')
            return None
        # print(x11,y11, x12, y12, x21, y21, x22, y22, m1, m2)
        cx = (x11 * m1 - y11 - x21 * m2 + y21) / (m1 - m2)
        cy = m1 * (cx - x11) + y11

        return [cx, cy]

    def dist_pointnline(self, line, point):
        x0 = point[0]
        y0 = point[1]
        x1 = line[0][0]
        y1 = line[0][1]
        x2 = line[1][0]
        y2 = line[1][1]

        return np.abs((x2 - x1)*(y1 - y0) - (x1 - x0)*(y2 - y1)) / np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def gen_buffer(self, polygon, d):
        polygon = np.asarray(polygon)
        N = len(polygon) # N by 2
        mat = list(range(N))
        p1 = mat[0]
        p2 = mat[1]
        flg = True
        new_polygon = []
        for i in range(N):
            vec1 = polygon[p1%N] - polygon[(p1+1)%N]
            vec2 = polygon[(p2+1)%N] - polygon[p2%N]
            vec1 = vec1 / LA.norm(vec1)
            vec2 = vec2 / LA.norm(vec2)
            new_vec = vec1 + vec2
            new_vec = new_vec / LA.norm(new_vec)
            theta = np.arccos(np.dot(vec1,new_vec))
            length = d * 1 / np.sin(theta)
            new_point = polygon[(p1+1)%N] + length * new_vec
            new_point[0] = np.round(new_point[0], 3)
            new_point[1] = np.round(new_point[1], 3)
            p1 = mat[(mat.index(p1) + 1)%len(mat)]
            p2 = mat[(mat.index(p2) + 1)%len(mat)]
            new_polygon.append(list(new_point))

        return np.asarray(new_polygon)