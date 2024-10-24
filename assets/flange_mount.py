from build123d import *
from ocp_vscode import *
import numpy as np

CAM_WIDTH = 23
CAM_LENGTH = 100
CAM_HOLE_DIST = 45
CAM_HOLE_RADIUS = 3 / 2
MOUNT_THICKNESS = 6
FLANGE_RADIUS = 63 / 2
FLANGE_RADIUS2 = 50 / 2
FLANGE_HOLE_RADIUS = 6 / 2

PITCH_INWARD = 45
PITCH_SIDEWAYS = 0.00001

CENTER_STICK_RADIUS = 3/2

with BuildPart() as mount:
    with BuildSketch():
        Circle(radius=CAM_LENGTH/2)
        Rectangle(width=CAM_LENGTH, height=CAM_LENGTH/2+CAM_LENGTH, align=(Align.CENTER, Align.MAX, Align.CENTER))
        with PolarLocations(radius=FLANGE_RADIUS2, count=8):
            Circle(radius=FLANGE_HOLE_RADIUS, mode=Mode.SUBTRACT)
    flat = extrude(amount=MOUNT_THICKNESS)

    OFFSET_DOWN = 20
    with Locations(Location((0, -(CAM_LENGTH/2 + CAM_WIDTH/2), MOUNT_THICKNESS/2 + OFFSET_DOWN), (-PITCH_INWARD, PITCH_SIDEWAYS, 0))):
        box = Box(CAM_LENGTH, width=CAM_WIDTH, height=MOUNT_THICKNESS, align=(Align.CENTER, Align.CENTER, Align.CENTER))
        box_faces = box.faces().sort_by(Axis.Z)[0:3]
    cam_block = extrude(box_faces, dir=(0, 0, -1), until=Until.LAST, target=flat)
    # This could be replaced by a distance check from the edge of flat as well:
    faces = cam_block.faces().filter_by(Axis.Y, tolerance=PITCH_SIDEWAYS).sort_by(Axis.Y)[0]
    #faces = cam_block.faces().sort_by_distance((0, -CAM_LENGTH*2, 0))[0]
    with BuildSketch(faces):
        Rectangle(CAM_LENGTH*2, CAM_LENGTH*2)
    extrude(amount=CAM_LENGTH*2, mode=Mode.SUBTRACT)
    face = box.faces().sort_by()
    
    
    #Cylinder(radius=CENTER_STICK_RADIUS, height=110, align=(Align.CENTER, Align.CENTER, Align.MIN))

show_all()