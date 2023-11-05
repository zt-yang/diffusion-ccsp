from config import *
import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix as R


def export_gif(images, gif_file, pause=1, save_pngs=False):
    import imageio
    with imageio.get_writer(gif_file, mode='I') as writer:
        counter = 0
        for img in images:
            for i in range(pause):
                writer.append_data(img)
                if save_pngs:
                    from PIL import Image
                    image_file = join(RENDER_PATH, f'{counter}.png')
                    Image.fromarray(img).save(image_file)
                    print('Saved png to {}'.format(image_file))
                    counter += 1
    print('Saved gif to {}'.format(gif_file))


def adjust_camera_topdown(scene, resolution=(1600, 1200)):
    scene.camera.resolution = resolution
    scene.camera.fov = [60, 60]
    scene.camera.transform = R(angle=np.radians(0), direction=[1, 0, 0], point=[0, 0, 0])
    scene.camera.transform[0:3, 3] = [0, 0, 30]


def adjust_camera_tilted(scene, transform, resolution=(1600, 1200)):
    scene.camera.resolution = resolution
    scene.camera.fov = [6, 6]
    scene.camera.transform = transform


def show_and_save(scene, img_name='top_down.png', resolution=(1600, 1200),
                    show=False, save=True, array=False, topdown=True):
    if topdown:
        adjust_camera_topdown(scene, resolution)
    if show:
        scene.show()
    if save:
        from pyglet import gl
        import io
        from PIL import Image

        window_conf = gl.Config(double_buffer=True, depth_size=24)
        data = scene.save_image(resolution=resolution, window_conf=window_conf, visible=False)
        img = Image.open(io.BytesIO(data))
        if array:
            return np.array(img)
        image_file = join(RENDER_PATH, img_name)
        img.save(image_file)
        return img

        # window = windowed.SceneViewer(scene, start_loop=False, visible=False)
        # window.save_image()
