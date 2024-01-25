import cv2
import mediapipe as mp
import numpy as np

from scipy.spatial.transform import Rotation as R

import justpyplot as jplt

import screeninfo
#Most popular - Macbook Air "13, in centimenters
# px_sz = 0.01119
# monitor= screeninfo.get_monitors()[0]
# screen_width, screen_height = monitor.width, monitor.height
# screen_width_phys, screen_height_phys = monitor.width_mm, monitor.height_mm
# print("Screen Width: {}, Screen Height: {}\n\n".format(screen_width_phys, screen_height_phys))
# px_sz = screen_width_phys / screen_width
# print("Pixel Size on system: {}, cm".format(px_sz*.1))

def reject_outliers(data, max_len=10, m =2):
    if data.size < 3:
        return data

    tail, last_data = data[:-max_len],data[-max_len:]
    u = np.mean(data)
    s = np.std(data)
    filtered_data = last_data[abs(last_data - u) < s]
    data=np.concatenate([tail, filtered_data])
    return data

def main():
    # Initialize MediaPipe Objectron
    mp_objectron = mp.solutions.objectron
    objectron = mp_objectron.Objectron(static_image_mode=False,
                                        max_num_objects=1,
                                        min_detection_confidence=0.4,
                                        min_tracking_confidence=0.25,
                                        model_name='Cup')

    # Initialize MediaPipe Drawing Utils
    mp_drawing = mp.solutions.drawing_utils

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    angles_x = []
    angles_y = []
    angles_z = []
    distances = []
    volumes = []

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and retrieve the results
        results = objectron.process(image)

        if results.detected_objects:
            for detected_object in results.detected_objects:
                # Retrieve rotation, translation, and size/distance of the cup
                rotation = detected_object.rotation
                translation = detected_object.translation
                # Size/distance can be inferred from the translation
                distance = np.linalg.norm(translation)
                dimensions = detected_object.scale  # Calculate 3D distance
                volume_size = np.prod(dimensions)  # Store volume size in a NumPy array

                r = R.from_matrix(rotation)
                rotations = r.as_euler('zyx', degrees=True)
                # Print out the rotation, volume size, and 3D distance
                angles_z.append(rotations[0])
                angles_y.append(rotations[1])
                angles_x.append(rotations[2])
                volumes.append(volume_size)
                distances.append(distance)
                ang_zs = (np.array(angles_z))
                ang_ys = (np.array(angles_y))
                ang_xs = (np.array(angles_x))

                jplt.plot1_at(image, ang_zs,
                            title='Angle from Z axis', offset=(50,50), size=(270, 300),
                            point_color=(255,0,0),line_color=(255,0,0), label_color=(255,0,0), grid_color=(126,126,126))
                jplt.plot1_at(image, ang_ys,
                              title='Angle from Y axis', offset=(400,50), size=(270, 300),
                              point_color=(0,255,0), line_color=(0,255,0),label_color=(0,255,0), grid_color=(126,126,126),
                              scatter=False)
                jplt.plot1_at(image,ang_xs,
                              title='Angle from X axis', offset=(750,50), size=(270, 300),
                              point_color=(0,0,255), line_color=(0,0,255),label_color=(0,0,255), grid_color=(126,126,126),
                              scatter=False)
               


                # print("Rotation:\n", rotations)
                # print("Volume Size:", volume_size)
                # print("3D Distance:", distance)

                # Draw the 3D bounding box and axis
                mp_drawing.draw_landmarks(image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                mp_drawing.draw_axis(image, rotation, translation)

        # Display the annotated image
        cv2.imshow('MediaPipe Objectron', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        c = cv2.waitKey(5)
        if c == 27:
            break
        elif c == 32:
            angles_z.clear()
            angles_x.clear()
            angles_y.clear()

    # Release the webcam and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()