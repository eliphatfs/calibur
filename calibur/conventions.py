# Convention entry: Right, Up, Forward

class CameraConventions:
    GL = ("X", "Y", "-Z")
    Godot = Blender = OpenGL = GL
    CV = ("X", "-Y", "Z")
    OpenCV = CV
    ROS = ("-Y", "Z", "X")
    DirectXLH = ("X", "Y", "Z")
    Unity = DirectXLH
    UE = ("Y", "Z", "X")

class WorldConventions:
    Blender = ("X", "Z", "Y")
    GL = ("X", "Y", "Z")  # TODO: double check
    GLTF = Godot = Unity = DirectXLH = GL
    ROS = CameraConventions.ROS
    UE = ("-X", "Z", "Y")
