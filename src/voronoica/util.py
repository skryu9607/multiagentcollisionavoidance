import numpy as np
import numpy.linalg as LA

class Util():
    def __init__(self) -> None:
        pass

    def cal_perimeter(self, polygon):
        ## polygon = [[x1, y1], [x2, y2], ..., [xn, yn]]
        ll = 0
        for i in range(len(polygon)):
            xd = polygon[i][0] - polygon[(i+1)%len(polygon)][0]
            yd = polygon[i][1] - polygon[(i+1)%len(polygon)][1]
            ll += LA.norm([xd, yd])
        return ll

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

        flg = True
        while flg:
            if len(new_polygon) > 3:
                E = []
                for i in range(len(new_polygon)):
                    E.append([i, (i + 1) % len(new_polygon)])
                # print(E)
                Comb = []
                for i in range(len(E) - 2):
                    for j in range(i + 2, len(E) - (i == 0)):
                        Comb.append([i, j])
                # print(Comb)
                for i in range(len(Comb)):
                    # print(E[Comb[i][0]])
                    p11 = new_polygon[E[Comb[i][0]][0]]
                    p12 = new_polygon[E[Comb[i][0]][1]]
                    p21 = new_polygon[E[Comb[i][1]][0]]
                    p22 = new_polygon[E[Comb[i][1]][1]]
                    if self.chk_cross(p11, p12, p21, p22):
                        new_pt = self.get_crosspt(p11, p12, p21, p22)
                        poly1 = []
                        poly2 = []
                        for j in range(len(E)):
                            if j < E[Comb[i][0]][1] or j > E[Comb[i][1]][0]:
                                poly2.append(new_polygon[j])
                            else:
                                poly1.append(new_polygon[j])
                        poly1.append(new_pt)
                        poly2.insert(E[Comb[i][0]][0]+1, new_pt)
                        l_poly1 = self.cal_perimeter(poly1)
                        l_poly2 = self.cal_perimeter(poly2)
                        if l_poly1 > l_poly2:
                            new_polygon = poly1
                            # print(polygon)
                            break
                        else:
                            new_polygon = poly2
                            break
                    if i == len(Comb) - 1:
                        flg = False
            else:
                flg = False

        return np.asarray(new_polygon)