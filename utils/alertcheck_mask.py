import cv2
from playsound import playsound
import pandas as pd


# crossed=0

def drawboxtosafeline(image_np, box_list, Orientation):
    # global crossed
    box1_p1 = box_list[0]
    box1_p2 = box_list[1]
    box2_p1 = box_list[2]
    box2_p2 = box_list[3]
    if (Orientation == "bt"):
        #bounding_mid = (int((p1[0] + p2[0]) / 2), int(p1[1]))
        if box2_p2[0] < box1_p1[0]:
            bounding_mid_1 = (int(box1_p1[0]),int((box1_p1[1]+box1_p2[1])/2))
            bounding_mid_2 = (int(box2_p2[0]),int((box2_p1[1]+box2_p2[1])/2))
        else:
            bounding_mid_1 = (int(box1_p2[0]), int((box1_p1[1] + box1_p2[1]) / 2))
            bounding_mid_2 = (int(box2_p1[0]), int((box2_p1[1] + box2_p2[1]) / 2))

        if (bounding_mid_1):
            cv2.line(img=image_np, pt1=bounding_mid_1, pt2=bounding_mid_2, color=(255, 0, 0),
                     thickness=2, lineType=8, shift=0)
            distance_from_line = abs(bounding_mid_1[0] - bounding_mid_2[0])

            i = 1
            p = distance_from_line
            q = 0.0104166667
            i = (i * p) % 1000000007
            i = (i * q) % 1000000007
            cv2.putText(image_np, "Distance: " + str("{0:.2f} inch".format(i)), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,(22, 219, 78), 2)


    if (distance_from_line <= 150):

        # crossed+=1
        posii = int(image_np.shape[1] / 2)
        cv2.putText(image_np, "ALERT----> Keep Distance", (posii-50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        # sound = os.path.join()
        playsound(r"E:\Python_projects\facemask_ssdlite\utils\alert.wav")
        #cv2.rectangle(image_np, (posii - 20, 20), (posii + 85, 60), (255, 0, 0), thickness=3, lineType=8,
                     # shift=0)
        # to write into xl-sheet


        return 1,i

    else:
        return 0,i