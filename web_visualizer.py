import websockets
import flatbuffers
import numpy as np
from typing import List
import webviewer

from queue import Queue
import asyncio
import threading
import webbrowser
from webviewer import Frame, Box, Vec3, Color, Image

from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import matplotlib.cm as cm


def pc_show(pc=None, boxes=None, images=None, ip=None):
    if WebViewer.main_display is None:
        WebViewer.main_display = WebViewer(ip=ip)
        # if WebViewer.main_display.ip == 'localhost':
            # webbrowser.get('firefox').open_new_tab('localhost:8000')

    WebViewer.main_display.display(pc, boxes, images)
    return WebViewer.main_display


class QueuedWebSocket(threading.Thread):
    """
        Create and
    """

    def __init__(self, ip, port, queue: Queue):
        super().__init__(name='webs_socket')
        self._queue = queue
        self.ip = ip
        self._port = port
        self._start_server = None
        self._loop = None

    def connect(self):
        self._start_server = websockets.serve(self._webserver_loop, self.ip, self._port)

    async def _webserver_loop(self, websocket, path):
        while True:
            # print(0)
            message = self._queue.get()
            # print(1)
            await websocket.send(message)
            # print(2)
            self._queue.task_done()

    def run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self.connect()
        asyncio.get_event_loop().run_until_complete(self._start_server)
        asyncio.get_event_loop().run_forever()


class WebViewer:
    main_display = None

    def __init__(self, ip=None):
        self.alive = False
        self._builder = None
        self._points = None
        self._reflectivities = None
        self._boxes = None
        self._images = None
        self._start_new_message()
        if ip is None:
            ip = "localhost"
        self.ip = ip
        self._port = 8080
        self._queue = Queue(1)
        self._queue = Queue(1)
        wvc = QueuedWebSocket(ip=self.ip, port=self._port, queue=self._queue)
        wvc.daemon = True
        wvc.start()

    def display(self, pc, boxes, images):
        self._start_new_message()
        if pc is not None:
            if pc.shape[1] == 4:
                pc_points = pc[:, :3]
                reflectivity = pc[:, 3].astype(np.uint8)
            elif pc.shape[1] == 3:
                pc_points = pc
                reflectivity = np.zeros((pc.shape[0], )).astype(np.uint8)
            self._build_pc_message(pc=pc_points, reflectivity=reflectivity)

        if boxes is not None:
            translations = [box[:3] for box in boxes]
            rotations = [box[6] for box in boxes]
            sizes = [box[3:6] for box in boxes]
            clss = [box[7] for box in boxes]
            colors = [object_type_to_color(box[7]) for box in boxes]
            self._build_box_message(translations=translations, rotations=rotations, sizes=sizes,
                                    colors=colors, clss=clss, names=clss)

        if images is not None:
            self._build_image_message(images)

        message = self._end_message()
        self._send_buffer(message)

    def _send_buffer(self, buffer):
        # print('trying_send_message')
        self._queue.put(buffer)
        # print('send_message')

    def _start_new_message(self, buffer_size=0):
        """
        starts new flatbuffer message
        :param buffer_size:
        """
        self._builder = flatbuffers.Builder(buffer_size)

    def _end_message(self):
        """
        ends flatbuf message and generate buffer
        :return: flatbuf buffer (that can be sent with socket)
        """
        # self._builder = flatbuffers.Builder(0)
        Frame.FrameStart(self._builder)
        if self._points:
            Frame.FrameAddPoints(self._builder, self._points)

        if self._reflectivities:
            Frame.FrameAddReflectivities(self._builder, self._reflectivities)

        if self._boxes:
            Frame.FrameAddBoxes(self._builder, self._boxes)

        if self._images:
            Frame.FrameAddImages(self._builder, self._images)

        frame = Frame.FrameEnd(self._builder)
        self._builder.Finish(frame)
        buf = self._builder.Output()
        self._points = None
        self._reflectivities = None
        self._boxes = None
        self._images = None
        return buf

    def _build_pc_message(self, pc: np.ndarray, reflectivity: np.ndarray):
        """
        builds frame point cloud flatbuffer
        :param pc: point cloud as float np array [N, 3] (x, y, z)
        :param reflectivity: uint8 nd array [N, 1]
        """
        assert len(pc) == len(reflectivity)
        if reflectivity.dtype != np.uint8:
            raise TypeError('reflectivity must be unit8')

        self._points = self._builder.CreateNumpyVector(pc.astype(np.float32).reshape([-1]))
        self._reflectivities = self._builder.CreateNumpyVector(reflectivity)

    def _build_box_message(self, translations, rotations, sizes, colors, clss, names):
        assert len(translations) == len(rotations) == len(sizes) == len(clss) == len(names)
        if len(translations) == 0:
            return
        # box_names = [self._builder.CreateString(name) for name in names]
        # box_types = [self._builder.CreateString(cls) for cls in clss]
        box_name = self._builder.CreateString("Sample-Box")
        box_type = self._builder.CreateString("Car")
        boxes = []
        for translation, rotation, size, color in zip(translations, rotations, sizes, colors):
            Box.BoxStart(self._builder)
            Box.BoxAddName(self._builder, box_name)
            Box.BoxAddType(self._builder, box_type)
            Box.BoxAddScore(self._builder, 99.0)
            box_color = Color.CreateColor(self._builder, color[0], color[1], color[2], color[3])
            Box.BoxAddColor(self._builder, box_color)

            box_rotation = Vec3.CreateVec3(self._builder, 0, 0, rotation + np.pi / 2)
            Box.BoxAddRotation(self._builder, box_rotation)
            box_pos = Vec3.CreateVec3(self._builder, translation[0], translation[1],
                                      translation[2])
            Box.BoxAddPos(self._builder, box_pos)
            box_size = Vec3.CreateVec3(self._builder, size[0], size[1], size[2])
            Box.BoxAddSize(self._builder, box_size)
            boxes.append(Box.BoxEnd(self._builder))

        Frame.FrameStartBoxesVector(self._builder, len(boxes))
        for box in boxes:
            self._builder.PrependSOffsetTRelative(box)
        self._boxes = self._builder.EndVector(len(boxes))

    def _build_image_message(self, images):
        import cv2
        ret_images = []
        for image in images:
            _, en_buff = cv2.imencode('.jpg', image)
            new_buff = np.array(en_buff)
            buff = self._builder.CreateNumpyVector(new_buff[:, 0])
            Image.ImageStart(self._builder)
            Image.ImageAddHeight(self._builder, image.shape[0])
            Image.ImageAddWidth(self._builder, image.shape[1])
            Image.ImageAddRawdata(self._builder, buff)
            ret_images.append(Image.ImageEnd(self._builder))

        Frame.FrameStartImagesVector(self._builder, len(ret_images))
        for ret_image in ret_images:
            self._builder.PrependSOffsetTRelative(ret_image)
        self._images = self._builder.EndVector(len(ret_images))


def object_type_to_color(object_type):
    """
    Constant mapping between object types and colors
    :param object_type:
    :return:
    """
    cmap = ListedColormap(
        ['w', 'magenta', 'orange', 'mediumspringgreen', 'deepskyblue', 'pink', 'y', 'g', 'r', 'purple',
         'lime', 'crimson', 'aqua'])
    coloring = np.mod(object_type, len(cmap.colors))
    c = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=-1, vmax=len(cmap.colors) - 1))
    color = c.to_rgba(coloring)
    return tuple(int(c * 255) for c in color)


if __name__ == '__main__':
    from object_detection_repo.analytics.display_samples_utils import INVZFramesViewer, CoordinateSystem
    from infra.visualizations.visualization_panda import display_pc
    from infra.data_layer.pc_reader import PCReaderKitti

    reader = PCReaderKitti()
    pc = reader.read('/home/odedf/repos/PointRCNN/data/KITTI/object/training/velodyne/007474.bin')

    for i in range(10000):
        pc_show(pc)

    print("Hi")
    # for frame in range(1000):
    #     sample = vv.read_sample(frame_id=frame, pc_path=data_path)
    #     pc = sample.pc_data
    #     # pc_show(pc)
    #     display_pc(point_cloud=pc)
    #     # pc.change_coordinate_system(CoordinateSystem.CAR, sample.calib_data)
    #     # try:
    #     #     pc_show(pc, sample.gt_data.gt_objects)
    #     #     a=1
    #     # except:
    #     #     pc_show(pc)
