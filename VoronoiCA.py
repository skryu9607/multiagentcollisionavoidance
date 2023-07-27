import random

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi

def chk_cross(a,b,p,q):
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

def get_crosspt(P11, P12, P21, P22):
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

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

# make up data points
np.random.seed(10)


N_agent = 20
N_obs = 10
N = N_obs + N_agent
# cur = np.random.rand(N, 2)
# goal = np.random.rand(N, 2)
cur = np.zeros((N, 2))
goal = np.zeros((N, 2))
for i in range(N_agent):
    cur[i, 0] = 0.5 + 0.4 * np.cos(i / (N_agent) * 2 * np.pi) + random.random()*0.0001
    cur[i, 1] = 0.5 + 0.4 * np.sin(i / (N_agent) * 2 * np.pi) + random.random()*0.0001
    goal[i, 0] = 0.5 + 0.4 * np.cos(i / (N_agent) * 2 * np.pi + np.pi) + random.random()*0.0001
    goal[i, 1] = 0.5 + 0.4 * np.sin(i / (N_agent) * 2 * np.pi + np.pi) + random.random()*0.0001


obs_x = np.ones(N_obs)*0.5
obs_y = np.arange(0,N_obs)*0.05 + 0.25
print(obs_y)
for i in range(N_obs):
    cur[i + N_agent, 0] = obs_x[i]
    cur[i + N_agent, 1] = obs_y[i]
    goal[i + N_agent, 0] = cur[i + N_agent, 0]
    goal[i + N_agent, 1] = cur[i + N_agent, 1]
print(cur)
# cur[N-1, 0] = 0.5
# cur[N-1, 1] = 0.5
# goal[N-1, 0] = 0.5
# goal[N-1, 1] = 0.5



Coeff = 0.025 # repulsive 반영 정도

for i in range(200):

    ## 1) Preferred Velocity _ to Goal
    d_vec = goal - cur
    vv = np.zeros((N, 1))
    v_x = np.zeros((N, 1))
    v_y = np.zeros((N, 1))
    for j in range(N):
        vv[j] = np.sqrt(d_vec[j, 0]**2 + d_vec[j, 1]**2)
        v_x[j] = d_vec[j, 0] / vv[j] * 0.02
        v_y[j] = d_vec[j, 1] / vv[j] * 0.02

    ## 2) Voronoi Cell
    # compute Voronoi tesselation
    # print(cur)
    vor = Voronoi(cur)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    ## 3) 모든 벽과의 cross_pt 계산
    cnt = 0
    for region in regions:
        if vv[cnt] < 0.01:
            cur[cnt] = goal[cnt]
        else:
            polygon = vertices[region]
            polygon = np.asarray(polygon)
            pt = []
            d_from_wall = []
            # print(region, polygon)
            for j in range(len(polygon) - 1):
                if chk_cross(polygon[j], polygon[j+1], cur[cnt], goal[cnt]):
                    pt.append(get_crosspt(polygon[j], polygon[j+1], cur[cnt], goal[cnt]))
                else:
                    pt.append([-1000, -1000])
            if chk_cross(polygon[-1], polygon[0], cur[cnt], goal[cnt]):
                pt.append(get_crosspt(polygon[-1], polygon[0], cur[cnt], goal[cnt]))
            else:
                pt.append([-1000, -1000])
            pt = np.asarray(pt)
            for j in range(len(pt)):
                d_from_wall.append(LA.norm(pt[j] - cur[cnt]))

            rep_d = []
            for j in range(len(pt)):
                if d_from_wall[j] < Coeff*5:
                    rep_d.append(cur[cnt] - pt[j])
                else:
                    rep_d.append([0, 0])

            rep_v = []

            for j in range(len(pt)):
                if LA.norm(rep_d[j]) < 1e-6:
                    rep_v.append([0, 0])
                else:
                    rep_v.append(rep_d[j] / LA.norm(rep_d[j]))

            sum_rep_v = [0, 0]
            for j in range(len(pt)):
                sum_rep_v[0] += rep_v[j][0]
                sum_rep_v[1] += rep_v[j][1]
            norm_sum_rep_v = LA.norm(sum_rep_v)

            # tp_x = v_x[cnt]
            # tp_y = v_y[cnt]
            if norm_sum_rep_v < 1e-6:
                tp_x = v_x[cnt][0]
                tp_y = v_y[cnt][0]
                print(tp_x, tp_y)
                # cur[cnt] = cur[cnt] + [tp_x[0], tp_y[0]]
            else:
                rx = Coeff * sum_rep_v[0] / norm_sum_rep_v / 2
                ry = Coeff * sum_rep_v[1] / norm_sum_rep_v / 2
                tp_x = v_x[cnt][0] + rx
                tp_y = v_y[cnt][0] + ry
                #
                # # if np.dot([v_x[cnt][0], v_y[cnt][0]], sum_rep_v) < 0:
                tp_cross = np.cross([0, 0, 1], [rx, ry, 0])
                tp_x += tp_cross[0] * 0.5 - rx * 0.5
                tp_y += tp_cross[1] * 0.5 - ry * 0.5
                # cur[cnt] = cur[cnt] + [tp_x[0], tp_y[0]] + [tp_cross[0]*0.1, tp_cross[1]*0.1] - [rx*0.9, ry*0.9]
                # # else:
                # cur[cnt] = cur[cnt] + [tp_x[0], tp_y[0]]
            # print(tp_x, tp_y)
            pt = []
            tp_cur = cur[cnt]+[tp_x, tp_y]
            print("tp_cur", tp_cur)
            for j in range(len(polygon) - 1):
                if chk_cross(polygon[j], polygon[j+1], cur[cnt], tp_cur):
                    pt.append(get_crosspt(polygon[j], polygon[j+1], cur[cnt], tp_cur))
            if chk_cross(polygon[-1], polygon[0], cur[cnt], tp_cur):
                pt.append(get_crosspt(polygon[-1], polygon[0], cur[cnt], tp_cur))

            if not len(pt) == 0:
                cur[cnt][0] = cur[cnt][0] / 2 + pt[0][0] / 2
                cur[cnt][1] = cur[cnt][1] / 2 + pt[0][1] / 2
            else:
                # print(tp_x, tp_y)
                cur[cnt] = cur[cnt]+[tp_x, tp_y]

        cnt += 1

    ## colorize
    if i == 0:
        pol = []
        for region in regions:
            polygon = vertices[region]
            polygon = np.asarray(polygon)
            # print(polygon)
            aa, = plt.fill(polygon[:, 0], polygon[:, 1], alpha=0.4)
            pol.append(aa)
        a, = plt.plot(cur[:, 0], cur[:, 1], 'ko')
        bb, = plt.plot(goal[:, 0], goal[:, 1], 'bo')
        ttt = []
        for j in range(len(cur)):
            tt = plt.text(cur[j, 0], cur[j, 1], str(j))
            ttt.append(tt)
    else:
        cnt = 0
        for region in regions:
            polygon = vertices[region]
            polygon = np.asarray(polygon)
            pol[cnt].set_xy(polygon)
            cnt += 1
            # pol[cnt].set_ydata()
        a.set_xdata(cur[:, 0])
        a.set_ydata(cur[:, 1])
        for j in range(len(cur)):
            ttt[j].set_x(cur[j, 0])
            ttt[j].set_y(cur[j, 1])


    plt.axis('equal')
    # plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
    # plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    # plt.savefig('data/' + str(i) + '.png') 
    plt.pause(0.01)
plt.show()
