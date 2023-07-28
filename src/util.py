import numpy as np

class Util():
    def __init__(self) -> None:
        pass

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