from build123d import *
from ocp_vscode import *
import numpy as np

CAM_WIDTH = 23
CAM_LENGTH = 70*1.5
CAM_HOLE_RADIUS = 3 / 2 + 0.2
CAM_HOLE_DIST = CAM_LENGTH - 10 - CAM_HOLE_RADIUS #45
MOUNT_THICKNESS = 6
FLANGE_RADIUS = 63 / 2
FLANGE_RADIUS2 = 50 / 2
FLANGE_HOLE_RADIUS = 6 / 2

PITCH_INWARD = 20 + 10

STICK_RADIUS = 3/2 + 0.05
STICK_HEIGHT = 110
STICK_CLEARANCE = 0.15 - 0.05
STICK_MOUNT_RADIUS = 10
STICK_MOUNT_HEIGHT = 20

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
        Box(length=CAM_HOLE_DIST, width=CAM_HOLE_RADIUS*2, height=MOUNT_THICKNESS, mode=Mode.SUBTRACT)
        box_faces = box.faces().sort_by(Axis.Z)[0]
    cam_block = extrude(box_faces, dir=(0, 0, -1), until=Until.LAST, target=flat)
    faces = cam_block.faces().filter_by(Axis.Y).sort_by(Axis.Y)[0:2]
    extrude(faces, amount=CAM_LENGTH, mode=Mode.SUBTRACT)
    
    Cylinder(radius=STICK_MOUNT_RADIUS, height=STICK_MOUNT_HEIGHT+MOUNT_THICKNESS, align=(Align.CENTER, Align.CENTER, Align.MIN))
    Cylinder(radius=STICK_CLEARANCE+STICK_RADIUS, height=STICK_MOUNT_HEIGHT+MOUNT_THICKNESS, align=(Align.CENTER, Align.CENTER, Align.MIN), mode=Mode.SUBTRACT)

    with Locations(Location((0, 0, MOUNT_THICKNESS + STICK_MOUNT_HEIGHT/2), (0, 90, 0))):
        Cylinder(2, STICK_MOUNT_RADIUS, mode=Mode.SUBTRACT, align=(Align.CENTER, Align.CENTER, Align.MAX))
    
    new_extrude = extrude(mount.faces().filter_by(Axis.Y).sort_by(Axis.Y)[1], amount=-MOUNT_THICKNESS)
    extrude(new_extrude.faces().sort_by(Axis.Z)[0], amount=MOUNT_THICKNESS)

with BuildPart() as stick:
    Cylinder(radius=STICK_RADIUS, height=STICK_HEIGHT, align=(Align.CENTER, Align.CENTER, Align.MIN))
    with Locations(Location((0, 0, STICK_MOUNT_HEIGHT+MOUNT_THICKNESS), (0, 0, 0))):
        Cylinder(STICK_MOUNT_RADIUS, MOUNT_THICKNESS, align=(Align.CENTER, Align.CENTER, Align.MIN))
    chamfer(stick.faces().sort_by(Axis.Z)[-3].edges().sort_by(SortBy.LENGTH)[0], STICK_MOUNT_RADIUS/2)

with BuildPart() as calibration_stick:
    Cylinder(radius=STICK_RADIUS, height=STICK_HEIGHT, align=(Align.CENTER, Align.CENTER, Align.MIN))
    with Locations(Location((0, 0, STICK_MOUNT_HEIGHT+MOUNT_THICKNESS), (0, 0, 0))):
        Cylinder(STICK_MOUNT_RADIUS, MOUNT_THICKNESS, align=(Align.CENTER, Align.CENTER, Align.MIN))

    face = calibration_stick.faces().sort_by(Axis.Z)[-2]
    thicken(face, amount=STICK_RADIUS)
    
    chamfer(calibration_stick.faces().sort_by(Axis.Z)[-3].edges().sort_by(SortBy.LENGTH)[0], STICK_MOUNT_RADIUS/2)
    
    with Locations(Location((0, 0, STICK_HEIGHT), (0, 0, 0))):
        Box(length=50, width=50, height=MOUNT_THICKNESS, align=(Align.CENTER, Align.CENTER, Align.MIN))
    chamfer(calibration_stick.faces().sort_by(Axis.Z)[-1].edges(), MOUNT_THICKNESS*0.95)

show_all()

export_stl(mount.part, "mount.stl")
export_stl(stick.part, "stick.stl")
export_stl(calibration_stick, "calibration_stick.stl")