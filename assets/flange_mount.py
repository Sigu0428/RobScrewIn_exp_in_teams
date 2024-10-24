from build123d import *
from ocp_vscode import *
import numpy as np

CAM_WIDTH = 23
CAM_LENGTH = 70
CAM_HOLE_DIST = 45
CAM_HOLE_RADIUS = 3 / 2
MOUNT_THICKNESS = 6
FLANGE_RADIUS = 63 / 2
FLANGE_RADIUS2 = 50 / 2
FLANGE_HOLE_RADIUS = 6 / 2

PITCH_INWARD = 45

STICK_RADIUS = 3/2
STICK_CLEARANCE = 0.2
STICK_MOUNT_RADIUS = 10

with BuildPart() as mount:
    with BuildSketch():
        Circle(radius=CAM_LENGTH/2)
        Rectangle(width=CAM_LENGTH, height=CAM_LENGTH/2+CAM_LENGTH, align=(Align.CENTER, Align.MAX, Align.CENTER))
        with PolarLocations(radius=FLANGE_RADIUS2, count=8):
            Circle(radius=FLANGE_HOLE_RADIUS, mode=Mode.SUBTRACT)
    flat = extrude(amount=MOUNT_THICKNESS)

    OFFSET_DOWN = 20
    with Locations(Location((0, -(CAM_LENGTH/2 + CAM_WIDTH/2), MOUNT_THICKNESS/2 + OFFSET_DOWN), (-PITCH_INWARD, 0, 0))):
        box = Box(CAM_LENGTH, width=CAM_WIDTH, height=MOUNT_THICKNESS, align=(Align.CENTER, Align.CENTER, Align.CENTER))
        with GridLocations(x_spacing=CAM_HOLE_DIST, y_spacing=0, x_count=2, y_count=1):
            Cylinder(CAM_HOLE_RADIUS, MOUNT_THICKNESS, mode=Mode.SUBTRACT)
        box_faces = box.faces().sort_by(Axis.Z)[0]
    cam_block = extrude(box_faces, dir=(0, 0, -1), until=Until.LAST, target=flat)
    faces = cam_block.faces().filter_by(Axis.Y).sort_by(Axis.Y)[0:2]
    extrude(faces, amount=CAM_LENGTH, mode=Mode.SUBTRACT)
    
    Cylinder(radius=CENTER_STICK_RADIUS, height=110, align=(Align.CENTER, Align.CENTER, Align.MIN))

show_all()