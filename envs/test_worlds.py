from os.path import dirname, join
import matplotlib.pyplot as plt

from worlds import *
from robot_worlds import *
from collections import defaultdict
from tqdm import tqdm

seed = None
if seed is None:
    seed = random.randint(0, 10**6-1)
random.seed(seed)
np.random.seed(seed % (2**32))
print('Seed:', seed)

RENDER_PATH = join(dirname(__file__), '..', 'renders')
if not isdir(RENDER_PATH):
    os.mkdir(RENDER_PATH)


def test_shape_setting_world():
    scene = ShapeSettingWorld()
    scene.add_shape('square', size=0.5, constraints=[('LeftIn',)])
    scene.add_shape('square', size=0.5, constraints=[('RightIn',)])
    scene.render(show=True)
    # for t in range(10):
    #     scene.randomize_scene()
    #     time.sleep(1.1)
    # scene.export_gif()


def randomize_shape_setting_world(debug=False):
    images = []
    name = None
    num_worlds = 10 if not debug else 1
    for t in range(num_worlds):
        scene = ShapeSettingWorld(h=0.5, orthographic=False)
        scene.sample_scene(case=0)
        img = scene.render(show=debug, array=True)
        images.append(img)
        name = scene.name
    export_gif(images, join(RENDER_PATH, f'{name}.gif'))
    files = [join(RENDER_PATH, f) for f in listdir(RENDER_PATH) if name in f and '.png' in f]
    for f in files:
        os.remove(f)


def test_qiqiao_world(case='logo'):
    if case == 'logo':
        world = QiQiaoWorld(w=12.5, h=1, orthographic=False)
    else:
        world = QiQiaoWorld(w=10)
    world.sample_scene(case=case)
    png_name = f'{world.name}_{case}.png'
    world.render(show=True, img_name=png_name)


def test_random_split_world():
    world = RandomSplitWorld(grid_size=0.5)
    world.sample_scene()
    world.generate_pt()
    world.render(show=True, show_grid=True, save=True)


def test_shake_worlds():
    world = RandomSplitWorld(grid_size=0.5)
    world.sample_scene()
    world.render(show=False, show_grid=True, save=True, img_name='RS.png')
    world.render(show=False, show_grid=True, save=True, array=True)
    funk = world.shake_scenes_gen(num=4, img_name_template='RS_{}.png', verbose=True, visualize=True)
    for new_world in funk:
        print()


def test_triangular_split_world():
    world = TriangularRandomSplitWorld(grid_size=0.5)
    world.sample_scene()
    # world.generate_pt()
    world.generate_meshes()
    world.render(show=True, show_grid=True, save=True)


def get_world_json(world, world_dir, input_mode, **kwargs):
    world.sample_scene(**kwargs)
    if world_dir is None:
        world_dir = join(RENDER_PATH, 'TableToBoxWorld')
    if not isdir(world_dir):
        os.mkdir(world_dir)
    json_name = join(world_dir, 'world.json')
    world.generate_json(input_mode=input_mode, json_name=json_name)
    return world_dir


def test_2d_box_split_world(min_num_objects=6, max_num_objects=10, render=True, world_dir=None):
    world = RandomSplitWorld(h=0.5, orthographic=False, grid_size=0.5)
    get_world_json(world, world_dir, input_mode='stability_flat',
                   min_num_objects=min_num_objects, max_num_objects=max_num_objects)
    if render:
        world.render(show=render, show_grid=False, save=True)


def test_3d_box_split_world(min_num_objects=6, max_num_objects=10, render=True, world_dir=None):
    world = RandomSplitWorld3D(grid_size=0.5)
    get_world_json(world, world_dir, input_mode='stability',
                   min_num_objects=min_num_objects, max_num_objects=max_num_objects)
    if render:
        world.render(show=render, show_grid=False, save=True)


def test_robot_world(min_num_objects=3, max_num_objects=4, render=True, world_dir=None):
    world = TableToBoxWorld(grid_size=0.5)
    world_dir = get_world_json(world, world_dir, input_mode='robot_box',
                               min_num_objects=min_num_objects, max_num_objects=max_num_objects)
    png_name = join(world_dir, 'world.png')
    if render:
        world.render(show=False, show_grid=True, save=True, img_name=png_name)


def test_qualitative_traj():
    train_dir_name = 'RandomSplitQualitativeWorld(30000)_qualitative_train_m=False_t=1000_qualitative_id=fo6ux6jg'
    prediction_name = 'denoised_t=1_n=2_i=1_k=2.json'
    prediction_file = join(RENDER_PATH, train_dir_name, prediction_name)
    prediction_data = json.load(open(prediction_file, 'r'))

    num_object, idx = prediction_name.split('_')[2:4]
    test_dir_name = f'RandomSplitQualitativeWorld(10)_qualitative_test_{num_object}_split'
    solution_path = join(RENDER_PATH, test_dir_name, f'idx={idx}.json')
    solution_data = json.load(open(solution_path, 'r'))

    print('Prediction:')


def test_qualitative_world():
    world = RandomSplitQualitativeWorld()
    world.sample_scene(min_num_objects=2, max_num_objects=6)
    world.generate_json(input_mode='robot_qualitative')


def visualize_qualitative_constraints(num_samples=10000, file_type='pdf'):
    """ t, bw, bl, x, y, yaw = nodes[i] """
    from denoise_fn import qualitative_constraints
    from data_monitor import RED, BLUE
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    VIZ_DIR = join('..', 'visualizations', 'failure_qualitative')
    w = 0.5
    nodes = [[0, 3, 2, 0, 0, 0], [1, w, w, 0, 0, 0]]

    satisfied = defaultdict(list)
    for k in tqdm(range(num_samples)):
        x = np.random.uniform(-1.5+w/2, 1.5-w/2)
        y = np.random.uniform(-1+w/2, 1-w/2)

        world = RandomSplitQualitativeWorld()
        world.construct_scene_from_graph_data(np.asarray(nodes+[[1, w, w, x, y, 0]]), constraints=[])
        current_constraints = world.get_current_constraints()
        for con in qualitative_constraints:
            cons = [k[0] for k in current_constraints if (k[1] == 2 or k[2] == 2)
                    and not (con in ['left-of', 'top-of'] and k[1] == 2)]
            if con in cons:
                if con == 'center-in' or not (abs(x) < w and abs(y) < w):
                    satisfied[con].append([x, y])

    # for con, points in satisfied.items():
    #     fig, ax = plt.subplots(1)
    #
    #     ## add box for object A
    #     pc = PatchCollection([Rectangle((-w/2, -w/2), w, w)], facecolor=BLUE, alpha=0.8)
    #     ax.add_collection(pc)
    #     ax.annotate('A', (-0.05, -0.04), fontsize=16)
    #
    #     points = np.asarray(points)
    #     plt.scatter(points[:, 0], points[:, 1], label=con[0], alpha=0.3, color=RED)
    #     plt.xlim([-1.5, 1.5])
    #     plt.ylim([-1, 1])
    #     plt.title(f"{con}(A, (x, y))", fontsize=16)
    #     plt.tight_layout()
    #     plt.savefig(join(VIZ_DIR, f'{con}.{file_type}'))
    #     # plt.show()
    #     plt.close()

    satisfied = dict(sorted(satisfied.items(), key=lambda x: len(x[1])))  ## , reverse=True
    satisfied = {k: len(v) for k, v in satisfied.items()}
    total_area = (3-w) * (2-w)
    total_area_cfree = total_area - w * w

    fig, ax = plt.subplots()
    areas = np.asarray(list(satisfied.values())) / num_samples * total_area_cfree
    areas[list(satisfied.keys()).index('center-in')] *= total_area / total_area_cfree
    ax.bar(range(len(satisfied)), areas, color=BLUE)
    plt.xticks(range(len(satisfied)), list(satisfied.keys()), rotation=45, fontsize=14, ha='right')
    plt.yticks(fontsize=14)
    plt.xlabel('Qualitative Constraints', fontsize=14)
    plt.ylabel('Estimated Area', fontsize=14)
    plt.title('Estimated Area Occupied by Qualitative Constraints', fontsize=16)
    plt.subplots_adjust(left=0.14, right=0.98, top=0.9, bottom=0.24)
    plt.tight_layout()
    plt.savefig(join(VIZ_DIR, f'area.{file_type}'))


if __name__ == '__main__':
    # test_shape_setting_world()
    # randomize_shape_setting_world()
    # test_qiqiao_world()
    # test_random_split_world()
    # test_shake_worlds()
    # test_triangular_split_world()
    test_3d_box_split_world()
    # test_robot_world()
    # test_qualitative_traj()
    # test_qualitative_world()
    # visualize_qualitative_constraints()
