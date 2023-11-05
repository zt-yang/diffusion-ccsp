import numpy as np
import fcl


def test_fcl():
    """ https://github.com/BerkeleyAutomation/python-fcl/ """

    def example_shapes():
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([2.0, 1.0, 3.0])
        v3 = np.array([3.0, 2.0, 1.0])
        x, y, z = 1, 2, 3
        rad, lz = 1.0, 3.0
        n = np.array([1.0, 0.0, 0.0])
        d = 5.0

        t = fcl.TriangleP(v1, v2, v3)  # Triangle defined by three points
        b = fcl.Box(x, y, z)  # Axis-aligned box with given side lengths
        s = fcl.Sphere(rad)  # Sphere with given radius
        e = fcl.Ellipsoid(x, y, z)  # Axis-aligned ellipsoid with given radii
        c = fcl.Capsule(rad, lz)  # Capsule with given radius and height along z-axis
        c = fcl.Cone(rad, lz)  # Cone with given radius and cylinder height along z-axis
        c = fcl.Cylinder(rad, lz)  # Cylinder with given radius and height along z-axis
        h = fcl.Halfspace(n, d)  # Half-space defined by {x : <n, x> < d}
        p = fcl.Plane(n, d)  # Plane defined by {x : <n, x> = d}

    def example_transforms():
        R = np.array([[0.0, -1.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0]])
        T = np.array([1.0, 2.0, 3.0])
        q = np.array([0.707, 0.0, 0.0, 0.707])

        tf = fcl.Transform()  # Default gives identity transform
        tf = fcl.Transform(q)  # Quaternion rotation, zero translation
        tf = fcl.Transform(R)  # Matrix rotation, zero translation
        tf = fcl.Transform(T)  # Translation, identity rotation
        tf = fcl.Transform(q, T)  # Quaternion rotation and translation
        tf = fcl.Transform(R, T)  # Matrix rotation and translation
        tf1 = fcl.Transform(tf)  # Can also initialize with another Transform
        return tf

    g1 = fcl.Box(1, 2, 3)
    t1 = fcl.Transform()
    o1 = fcl.CollisionObject(g1, t1)

    g2 = fcl.Cone(1, 3)
    t2 = fcl.Transform(np.array([3, 3, 4]))
    o2 = fcl.CollisionObject(g2, t2)

    request = fcl.CollisionRequest()
    result = fcl.CollisionResult()

    ret = fcl.collide(o1, o2, request, result)
    print(ret)


def check_collisions_in_scene(objects, rotations=None, **kwargs):
    # data = world.generate_json(**kwargs)
    bodies = {}
    for i, (name, obj) in enumerate(objects.items()):
        shape = obj['shape']
        if shape == 'box':
            geom = fcl.Box(*obj['extents'])
        elif shape == 'cylinder':
            geom = fcl.Cylinder(*obj['extents'])
        elif shape == 'arbitrary_triangle':
            verts = np.array(obj['vertices_centered'])
            tris = np.array(obj['faces'])
            num_faces = len(tris)

            ## ----------------- version 1
            faces = np.concatenate((3 * np.ones((len(tris), 1), dtype=np.int64), tris), axis=1).flatten()
            geom = fcl.Convex(verts, num_faces, faces)

            ## ----------------- version 2
            # faces = np.concatenate(
            #     [np.array([
            #         [3, 0, 1, 2],
            #         [3, 3, 4, 5]
            #     ]).flatten(),
            #     np.array([
            #         [4, 1, 0, 3, 4],
            #         [4, 2, 1, 4, 5],
            #         [4, 0, 2, 5, 3]
            #     ]).flatten()
            #     ], axis=0)
            # num_faces = 5
            # geom = fcl.Convex(verts, num_faces, faces)

            ## ----------------- version 3
            # m = fcl.BVHModel()
            # m.beginModel(len(verts), len(tris))
            # m.addSubModel(verts, tris)
            # m.endModel()

        # ## impossible, will cause error
        # elif shape == 'pointcloud':
        #     verts = np.array(obj['vertices_centered'])
        #     geom = fcl.BVHModel()
        #     geom.beginModel(len(verts), 1)
        #     geom.addSubModel(verts, [])
        #     geom.endModel()

        # else:
        #     print('Shape {} of type {} not supported'.format(obj['label'], obj['shape']))

        if rotations is not None and obj['label'] in rotations:
            from transformations import quaternion_about_axis
            R = quaternion_about_axis(rotations[obj['label']], (0, 0, 1))
            tf = fcl.Transform(R, obj['centroid'])
        else:
            tf = fcl.Transform(obj['centroid'])
        bodies[obj['label']] = fcl.CollisionObject(geom, tf)

        # print('gg', obj['label'], bodies[obj['label']].getRotation(), bodies[obj['label']].getTranslation())

    request = fcl.CollisionRequest()
    result = fcl.CollisionResult()
    collisions = []
    for i in bodies:
        geom1 = bodies[i]
        for j in bodies:
            if i == j:
                continue
            geom2 = bodies[j]
            ret = fcl.collide(geom1, geom2, request, result)
            if ret and (j, i) not in collisions:
                collisions.append((i, j))
    return collisions


def test_check_collisions_in_scene():
    from worlds import ShapeSettingWorld

    world = ShapeSettingWorld()
    world.sample_scene()
    collisions = world.check_collisions_in_scene()
    world.render(show=True)


if __name__ == '__main__':
    # test_fcl()
    test_check_collisions_in_scene()
