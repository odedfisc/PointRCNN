# automatically generated by the FlatBuffers compiler, do not modify

# namespace: WebViewer

import flatbuffers

class Frame(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsFrame(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Frame()
        x.Init(buf, n + offset)
        return x

    # Frame
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Frame
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Frame
    def Id(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Frame
    def Points(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Float32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # Frame
    def PointsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Float32Flags, o)
        return 0

    # Frame
    def PointsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Frame
    def Reflectivities(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # Frame
    def ReflectivitiesAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # Frame
    def ReflectivitiesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Frame
    def Overridecolors(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            from .Color import Color
            obj = Color()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Frame
    def OverridecolorsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Frame
    def Boxes(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from .Box import Box
            obj = Box()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Frame
    def BoxesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Frame
    def Lines(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from .Line import Line
            obj = Line()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Frame
    def LinesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Frame
    def Images(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from .Image import Image
            obj = Image()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Frame
    def ImagesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

def FrameStart(builder): builder.StartObject(8)
def FrameAddName(builder, name): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)
def FrameAddId(builder, id): builder.PrependInt32Slot(1, id, 0)
def FrameAddPoints(builder, points): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(points), 0)
def FrameStartPointsVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def FrameAddReflectivities(builder, reflectivities): builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(reflectivities), 0)
def FrameStartReflectivitiesVector(builder, numElems): return builder.StartVector(1, numElems, 1)
def FrameAddOverridecolors(builder, overridecolors): builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(overridecolors), 0)
def FrameStartOverridecolorsVector(builder, numElems): return builder.StartVector(4, numElems, 1)
def FrameAddBoxes(builder, boxes): builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(boxes), 0)
def FrameStartBoxesVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def FrameAddLines(builder, lines): builder.PrependUOffsetTRelativeSlot(6, flatbuffers.number_types.UOffsetTFlags.py_type(lines), 0)
def FrameStartLinesVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def FrameAddImages(builder, images): builder.PrependUOffsetTRelativeSlot(7, flatbuffers.number_types.UOffsetTFlags.py_type(images), 0)
def FrameStartImagesVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def FrameEnd(builder): return builder.EndObject()