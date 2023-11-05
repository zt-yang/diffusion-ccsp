from os.path import join, dirname, abspath, isdir, isfile

PROJECT_PATH = abspath(join(dirname(__file__), '..'))
MESH_PATH = abspath(join(PROJECT_PATH, 'models', 'meshes'))
RENDER_PATH = abspath(join(PROJECT_PATH, 'renders'))
DATASET_PATH = abspath(join(PROJECT_PATH, 'data'))
VISUALIZATION_PATH = abspath(join(PROJECT_PATH, 'visualizations'))
