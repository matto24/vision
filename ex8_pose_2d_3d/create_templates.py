import pyrender
import numpy as np
import cv2
from pathlib import Path
import argparse

import trimesh

# Function to help parse input arguments
def parse_multi_arg(value):
    return [float(val) for val in value.split(',')]

# Function to auto crop the image by color
def autocrop(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return src

    x, y, w, h = cv2.boundingRect(contours[0])
    return src[y:y+h, x:x+w]


def generate_templates(cad_file_path, output_dir, fov, width, height, radii, lat_begin, lat_end, lon_begin, lon_end, rotations, rot_begin, rot_end, subdiv, north, background_color):
    # Print the input arguments
    print("Selecting camera positions within specified radius and lat-lon intervals and applying rotations...")
    print("\tRadii: ", radii)
    print("\tLatitude interval: [", lat_begin, ",", lat_end, "]")
    print("\tLongitude interval: [", lon_begin, ",", lon_end, "]")
    print("\tRotation(s) at each camera: ", rotations, " over [", rot_begin, ",", rot_end, "]")
    
    # Create a pyrender scene
    scene = pyrender.Scene()

    # Load the CAD model
    mesh_data = trimesh.load_mesh(cad_file_path)
    mesh = pyrender.Mesh.from_trimesh(mesh_data)

    # Calculate the 3D centroid of the CAD model
    centroid = np.mean(mesh_data.vertices, axis=0)

    # Print the centroid
    print(f"Centroid: {centroid}")

    # Create an icosahedron
    icosphere = trimesh.creation.icosphere(subdivisions=subdiv)

    # Get the vertices and faces
    vertices = icosphere.vertices
    faces = icosphere.faces

    # Print amount of vertices and faces
    print(f"The View (Ico)sphere has Vertices: {len(vertices)}")
    print(f"And Faces: {len(faces)}")

    # Create a camera
    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(fov), aspectRatio=width/height)

    t = False # Debug argument, is set to false at the end of the inner for loop, so it only prints once.

    # Generate camera poses
    camera_poses = []
    labels = []
    # Loop over all the radii to be applied to the sphere
    for radius in radii:
        # Loop over all the points on the icosphere
        for i in range(len(vertices)):
            # Get the point (Sampled camera position on the icosphere)
            point = vertices[i]

            # If i = 0, then print the point
            if t:
                print(f"Point from icosphere: {point}")

            # Normalize the point to be on the unit sphere
            point = point / np.linalg.norm(point)

            # Get the latitude and longitude
            lat = np.rad2deg(np.arccos(point[2]))
            lon = np.rad2deg(np.arctan2(point[1], point[0]))
            # If lon < 0, add 360 to make it positive
            if lon < 0:
                lon += 360

            # Check if the point is within the latitude and longitude range
            tol = 1e-5
            if lat_begin - tol <= lat <= lat_end + tol and lon_begin - tol <= lon <= lon_end + tol:
                # Create a new camera pose
                T_mat = trimesh.transformations.translation_matrix(point)

                if t:
                    print(f"Point (trimesh transform): {T_mat}")

                # Generate a rotation, with the aim of pointing the y-axis in the opposite direction as the input pole
                R_mat = np.eye(3)
                R_mat[:, 2] = -T_mat[:3, 3]  # Z-axis points towards focal point

                # If i = 0, print the rotation
                if t:
                    print(f"Rotation: {R_mat}")

                # North pole, unity
                vnorth = np.array([north[0], north[1], north[2]])
                vnorth = vnorth / np.linalg.norm(vnorth)

                # If z-axis is (anti-)parallel with the north pole, skip
                ndot = np.dot(R_mat[:, 2], vnorth)
                isNorth = np.isclose(abs(ndot), 1, rtol=1e-5)
                if isNorth:
                    print(f"Frame with index {i} (anti-) parallel with north pole, using arbitrary camera orientation...")

                # # Project to a sphere with specified radius - Already done by creating the icosphere with the radius
                T_mat[:3, 3] *= radius

                if t:
                    print(f"Point (scaled): {T_mat}")

                # Move the whole sphere so the focal point is at the object center
                T_mat[:3, 3] += centroid[:3]

                # If i = 0, print the translation
                if t:
                    print(f"Translation moved: {T_mat}")

                if isNorth:  # Use an arbitrary up-vector
                    R_mat[:, 1] = np.cross(R_mat[:, 2], np.array([0, 0, 1]))
                else:  # Use a well-defined up-vector (pointing north)
                    # North pole
                    vnorth *= radius
                    # Direction vector from camera to north pole
                    dirCamNorth = (vnorth - T_mat[:3, 3]) / np.linalg.norm(vnorth - T_mat[:3, 3])
                    # Projection of that vector to the image plane
                    dirCamNorthProj = dirCamNorth - np.dot(dirCamNorth, R_mat[:, 2]) * R_mat[:, 2]
                    # The projection negated makes the up vector (y-axis points down in the image)
                    R_mat[:, 1] = -dirCamNorthProj / np.linalg.norm(dirCamNorthProj)

                # Final image axis
                R_mat[:, 0] = np.cross(R_mat[:, 1], R_mat[:, 2])
                T_mat[:3, :3] = R_mat

                # If i = 0, print the camera pose with rotation
                if t:
                    print(f"Final camera pose: {T_mat}")

                # Generate all rotations around the view vector and append the resulting camera pose and a label
                for j in range(rotations):
                    # Generate a z-axis rotation matrix of the current angle (converted to radians)
                    angle_rad = (rot_begin + j * (rot_end - rot_begin) / rotations) * np.pi / 180.0
                    Rz = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                                [np.sin(angle_rad), np.cos(angle_rad), 0],
                                [0, 0, 1]])
                    # Apply rotation to the original camera pose
                    Tz = T_mat.copy()
                    Tz[:3, :3] = np.dot(T_mat[:3, :3], Rz)

                    camera_poses.append(Tz)
                    label = f"template{len(labels):04d}"
                    labels.append(label)

                    # If i = 0, print the final camera pose
                    if t:
                        print(f"Printed pose: {Tz}")
                        print(f"Rz: {Rz}")

                    t = False

    # Print the number of camera poses
    print(f"Generated {len(camera_poses)} camera poses.")
    print(f"Generated {len(labels)} labels.")

    # Loop over camera poses
    for i, camera_pose in enumerate(camera_poses):
        # Create a pyrender scene for rendering the templates
        template_scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4, 1.0], bg_color=np.array(list(background_color)) )
        template_scene.add(mesh)

        # Change the transformation frame' (camera_pose) coordinate frame - Pyrender uses X right, Y up, Z forward
        # Our frame is formatted as X forward, Y right, Z up
        # src - https://medium.com/check-visit-computer-vision/converting-camera-poses-from-opencv-to-opengl-can-be-easy-27ff6c413bdb
        R_mod = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        # # Apply the rotation to the camera pose rotation
        render_pose = camera_pose.copy()
        render_pose[:3, :3] = np.dot(render_pose[:3, :3], R_mod)

        # Set pose of camera_node and add it to scene
        camera_node = pyrender.Node(camera=camera, matrix=render_pose)
        template_scene.add_node(camera_node)

        # Render the scene
        r = pyrender.OffscreenRenderer(width, height)
        color, depth = r.render(template_scene)

        # Save templates (e.g., as PNG images)
        template_dir = output_dir
        template_dir.mkdir(parents=True, exist_ok=True)
        color_path = output_dir / f'template{i:04d}.png'
        cv2.imwrite(str(color_path), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

        # Save camera pose
        pose_dir = output_dir
        pose_dir.mkdir(parents=True, exist_ok=True)
        pose_path = output_dir / f'template{i:04d}_pose.txt'
        np.savetxt(str(pose_path), camera_pose)

        # Cleanup the renderer
        r.delete()

    print(f"Generated {len(camera_poses)} templates.")


def main():
    parser = argparse.ArgumentParser(description="Generate CAD templates using pyrender")
    
    # Positional argument - Means you dont have to call "--cad-file", this is the first expected argument after the script name
    parser.add_argument("cad-file", help="CAD file (must be in mesh format, e.g. PLY)")

    # Options
    parser.add_argument("--output-dir", default="./../templates", help="Output directory for templates and poses")
    parser.add_argument("--subdiv", '-s', type=int, default=1, help="subdivisions of an icosahedron to generate a view sphere - the number of points become 10*4^subdiv + 2")
    parser.add_argument("--radius", '-r', type=parse_multi_arg, nargs='+', default=[[650,700,750]], help="Radius of the tessellated sphere, i.e. distance of the camera from the object - you can use multiple values here to get multiple radii")
    parser.add_argument("--lat-begin", type=float, default=0, help="latitude start angle [deg] in [0,180] - 0 is at the pole (0,0,1)")
    parser.add_argument("--lat-end", type=float, default=180, help="latitude end angle [deg] in [0,180]")
    parser.add_argument("--lon-begin", type=float, default=0, help="longitude start angle [deg] in [0,360] - 0 is at the x-axis (1,0,0) and 180 is at (-1,0,0)")
    parser.add_argument("--lon-end", type=float, default=360, help="longitude end angle [deg] in [0,360]")
    parser.add_argument("--rotations", '-t', type=int, default=1, help="number of in-plane rotations of the optical camera axis at each position (> 0)")
    parser.add_argument("--rot-begin", type=float, default=0, help="rotation start angle [deg]")
    parser.add_argument("--rot-end", type=float, default=360, help="rotation end angle [deg]")
    parser.add_argument("--north", '-n', type=parse_multi_arg, nargs='+', default=[[0.0,0.0,1.0]], help="set a \"natural\" vertical axis for the model, defaults to the z-axis - the virtual camera's y-axis is aligned with the NEGATIVE of this axis")
    parser.add_argument("--width", type=int, default=640, help="horizontal resolution")
    parser.add_argument("--height", type=int, default=480, help="vertical resolution")
    parser.add_argument("--fov", type=float, default=49, help="vertical field of view [degree]")
    parser.add_argument("--bc", type=parse_multi_arg, nargs='+', default=[[1.0,0.0,0.0]], help="background color for the RGB template (grayscale value or RGB triplet in [0,1])")
    parser.add_argument('-v', '--visualize', action="store_true", help="show some visualizations")

    args = parser.parse_args()

    # Example usage:
    # python3 create_templates.py ./files/obj_09.ply --output-dir ./templates -r 650,700,750 --lat-begin 45 --lat-end 80 --lon-begin 0 --lon-end 360 --fov 45.5 --bc 1,0,0 -t 1 -s 4 -v
    
    # Load the arguments into variables
    cad_file_path = getattr(args, "cad-file") # Get the cad file
    output_dir = Path(args.output_dir)
    fov     = args.fov      # Camera field of view - default is 49
    width   = args.width    # Camera width - default is 640
    height  = args.height   # Camera height - default is 480
    radii   = args.radius[0]# Radius (plural) of the tessellated sphere - default is 1

    lat_begin = args.lat_begin
    lat_end = args.lat_end
    lon_begin = args.lon_begin
    lon_end = args.lon_end
    rotations = args.rotations
    rot_begin = args.rot_begin
    rot_end = args.rot_end

    subdiv = args.subdiv
    north = args.north[0]
    background_color = args.bc[0]   

    print(f"Generating templates for object: {cad_file_path} into folder: {output_dir}")
    
    generate_templates(cad_file_path, output_dir, fov, width, height, radii, lat_begin, lat_end, lon_begin, lon_end, rotations, rot_begin, rot_end, subdiv, north, background_color)

if __name__ == "__main__":
    main()