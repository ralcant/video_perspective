import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import pi, sin, cos
from wand.color import Color 
from wand.image import Image as WandImage
from itertools import chain

def get_from_cv2(path): #returns RGB
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
def show_from_array(array):
    plt.imshow(array)
    plt.show()
def show_from_name(path):
    if type(path) == str:
        plt.imshow(get_from_cv2(path))
        plt.show()      
    elif type(path) == list:
        for singular_path in path:
            if type(singular_path) != str:
                raise ValueError("In this list, all elements should be strings")
            plt.imshow(get_from_cv2(singular_path))
            plt.show()
    else:
        raise ValueError(f"path should be a string or a list of strings, but got {type(path)}")

def combine_lists(source, coordinates):
    order = chain.from_iterable(zip(source, coordinates))
    arguments = list(chain.from_iterable(order)) 
    return arguments
'''
### Order:
     0.   1.

     2.   3.
'''
def get_distort_params(dimensions, face_coords): 
    w, h = dimensions
    source_points = (
        (0, 0),
        (w, 0),
        (0, h),
        (w, h)
    )
    return combine_lists(source_points, face_coords)

# raul = get_from_cv2("raul.png")
# fake = get_from_cv2("raul.png")
# other = get_from_cv2("raul.png")

#### Global variables ####
w = 900 #of the main frame
h = 600 #of the main frame
R = 100 #radius of the circular table
# all_users = [raul, fake, other] # position 0 is the current one, order is counterclockwise


frame_mask = np.zeros((h, w, 3), dtype=np.uint8) #need to have 3 channels! IMPORTANT
table = np.zeros((2*R, 2*R, 3), dtype=np.uint8)
# cv2.circle(table, (R, R), R, (255, 0, 0), thickness=-1)
# show_from_array(table)
theta = pi/180*80 #angle (in radians) between reality and perspective

cap = cv2.VideoCapture(0)

with WandImage.from_array(frame_mask) as mask_image:
    print(f"size of mask: {mask_image.size}")
    # table= cv2.cvtColor(table,  cv2.COLOR_BGR2RGB)
    ########### Table ##############################
    w, h = mask_image.size
    table_coordinates_in_perspective = np.array((
        (w//2-R+2*R*cos(theta),h-2*R*sin(theta)),  #top left
        (w//2+R-2*R*cos(theta),h-2*R*sin(theta)),  #top right
        (0,h),
        (w,h)
    ))
    ######### Person 1 #############3
    w_person_1, h_person_1 = mask_image.size
    person_1_coord = (
        (table_coordinates_in_perspective[0]-np.array([0,100])),
        (table_coordinates_in_perspective[1]-np.array([0,100])),
        table_coordinates_in_perspective[0],   #bottom left is same as top left of table
        table_coordinates_in_perspective[1]    #bottom right is same as top right of table
    )

    ########## Person 2 ##################3
    w_person_2, h_person_2 = mask_image.size
    person_2_coord = (
        (table_coordinates_in_perspective[3]+3*table_coordinates_in_perspective[1])/4 - np.array([0, 100]),
        (table_coordinates_in_perspective[1]+3*table_coordinates_in_perspective[3])/4 - np.array([0, 100]),
        (table_coordinates_in_perspective[3]+3*table_coordinates_in_perspective[1])/4,
        (table_coordinates_in_perspective[1]+3*table_coordinates_in_perspective[3])/4
    )

    ######### Person 3 ####################
    w_person_3, h_person_3 = mask_image.size
    person_3_coord = (
        (table_coordinates_in_perspective[0]+3*table_coordinates_in_perspective[2])/4 - np.array([0, 100]),
        (table_coordinates_in_perspective[2]+3*table_coordinates_in_perspective[0])/4 - np.array([0, 100]),
        (table_coordinates_in_perspective[0]+3*table_coordinates_in_perspective[2])/4,
        (table_coordinates_in_perspective[2]+3*table_coordinates_in_perspective[0])/4
    )
    while(True):
        ret, person_array = cap.read()
        # person_array_2 = person_array.copy()
        person_array = cv2.cvtColor(person_array,  cv2.COLOR_BGR2RGB)
        # show_from_array(person_array)
        with WandImage.from_array(table) as table_image, \
             WandImage.from_array(person_array) as person_1, \
             WandImage.from_array(person_array) as person_2, \
             WandImage.from_array(person_array) as person_3:
            
            table_image.virtual_pixel = 'background'
            table_image.background_color = Color("green")
            table_image.matte_color = Color("skyblue")
            person_1.virtual_pixel = 'transparent'
            person_2.virtual_pixel = 'transparent'
            person_3.virtual_pixel = 'transparent'

            table_image.resize(mask_image.size[0], mask_image.size[1])
            person_1.resize(mask_image.size[0], mask_image.size[1])
            person_2.resize(mask_image.size[0], mask_image.size[1])
            person_3.resize(mask_image.size[0], mask_image.size[1])


            table_image.distort('perspective', get_distort_params((w,h), table_coordinates_in_perspective))
            person_1.distort(   'perspective', get_distort_params((w_person_1,h_person_1), person_1_coord))
            person_2.distort(   'perspective', get_distort_params((w_person_2,h_person_2), person_2_coord))
            person_3.distort(   'perspective', get_distort_params((w_person_3,h_person_3), person_3_coord))
            # Overlay cover onto template 
            mask_image.composite(table_image,left=0,top=0)
            mask_image.composite(person_3, left=0, top=0)
            mask_image.composite(person_2, left=0, top=0)
            mask_image.composite(person_1, left=0, top=0)
            # mask_image.save(filename='result.png')
            final_pic = np.array(mask_image.export_pixels(channel_map="BGRA"), dtype=np.uint8).reshape((h, w, 4)) #RGBA
            # final_pic = cv2.cvtColor(final_pic, cv2.COLOR_RGBA2BGRA)
            # show_from_array(final_pic)
            cv2.imshow("Meeting", final_pic)
            # cv2.imshow("final pic", cv2.cvtColor(final_pic, cv2.COLOR_RGB2RGBA))
            if cv2.waitKey(1) & 0xFF == ord('q'):  #Press q to finish!
                break
cap.release()
cv2.destroyAllWindows()