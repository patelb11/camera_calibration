import cv2 
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from mpl_toolkits.mplot3d import Axes3D
import random as rand
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


#creates a homography given set of correponding points
def create_homography_estimation(x , x_prime):
    
    #if invalid corner point
    if x == None or x_prime == None:
        return None 
    
    #make the points into homogeanous points 
    x = np.array([np.append(point, 1) for point in x])
    y = np.array([np.append(point, 1) for point in x_prime])
    x_len = len(x)

    # b = A^-1 * c
    A = np.zeros((2 * x_len, 9))  
    # c = np.array(x_prime)
    # c = c.reshape((2 * x_len, 1))
    
    #make A 
    for i in range(x_len):
        A[2 * i] = [x[i][0], x[i][1], 1, 0, 0, 0, -y[i][0] * x[i][0],
                    -y[i][0] * x[i][1], -y[i][0]]
        A[2 * i + 1] = [0, 0, 0, -x[i][0], -x[i][1], -1, y[i][1] * x[i][0],
                        y[i][1] * x[i][1], y[i][1]]
    
    #using svd find the last eigenvector of A^TA
    _, _ ,v_t = np.linalg.svd(np.transpose(A) @ A)
    h = np.array(v_t[-1])
    
    homography = np.reshape(h,(3,3))
    homography = homography / homography[2][2]
    
    return homography

def find_extrinsic(homography_list, k):
    
    r_all = []
    t_all = []
    inv_k = np.linalg.inv(k)

    #loop through all the homographies and gets it extrinstic values
    for homography in homography_list:
        
        h_1 =  homography[:,0]
        h_2 =  homography[:,1]
        h_3 =  homography[:,2]
        
        scaling_factor = 1 / np.linalg.norm(np.matmul(inv_k,h_1))
        r_1 = scaling_factor*(np.matmul(inv_k,h_1))
        r_2 = scaling_factor*(np.matmul(inv_k,h_2))
        r_3 = np.cross(r_1,r_2)
        r = np.transpose(np.array([r_1,r_2,r_3]))
        t = scaling_factor*(np.matmul(inv_k,h_3))
        
        # you need to condition the rotation matrix because 
        # it is not orthonormal 
        u, d, v_t = np.linalg.svd(r)
        r_new = u @ v_t
    
        r_all.append(r)
        t_all.append(t)
        
    return r_all, t_all

def find_K(homography_list):
    
    #create the V 
    v = []
    for homography in homography_list:
        
        #get the values of homography rows 1 and 2
        #flip the row and column because of werid notation
        h_11, h_12, h_13 = homography[:,0]
        h_21, h_22, h_23 = homography[:,1]
        
        #calcaute the V_12 
        v.append([h_11*h_21, h_11*h_22+h_12*h_21, h_12*h_22, h_13*h_21+ h_11*h_23,
                  h_13*h_22+h_12*h_23, h_13*h_23])
        
        #calculate V_11 - V_22
        v_11 = np.array([h_11*h_11, h_11*h_12+h_12*h_11, h_12*h_12,
                         h_13*h_11+ h_11*h_13, h_13*h_12+h_12*h_13, h_13*h_13])
        v_22 = np.array([h_21*h_21, h_21*h_22+h_22*h_21, h_22*h_22,
                         h_23*h_21+ h_21*h_23, h_23*h_22+h_22*h_23, h_23*h_23])
        v.append(v_11 - v_22)
        
    # using linear least sqaures by svd solve for b in Vb = 0 
    v = np.array(v)
    _, _, v_t = np.linalg.svd(np.transpose(v) @ v)
    b = np.array(v_t[-1])
        
    #assign omega values
    omega = np.zeros((3,3))
    omega[0][0] = b[0]
    omega[0][1] = omega[1][0] = b[1]
    omega[1][1] = b[2]
    omega[0][2] = omega[2][0] = b[3]
    omega[1][2] = omega[2][1] = b[4]
    omega[2][2] = b[5]
        
    #normalize omega 
    omega /= omega[2][2]
        
    #now solve for K variables
    y_0 = (omega[0][1]*omega[0][2]-omega[0][0]*omega[1][2]) /\
        (omega[0][0]*omega[1][1]-omega[0][1]**2) 
    lam = omega[2][2] - (omega[0][2]**2+y_0*(omega[0][1]*\
        omega[0][2]-omega[0][0]*omega[1][2])) / omega[0][0]
    alpha_x = np.sqrt(lam/omega[0][0])
    alpha_y = np.sqrt(((lam*omega[0][0])/(omega[0][0]*omega[1][1\
        ]-omega[0][1]**2)))
    s =  -omega[0][1]*alpha_x**2*alpha_y / lam
    x_0 = ((s*y_0) / alpha_y) - ((omega[0][2]*alpha_x**2) / lam)
    
    #create the K matrix
    k = np.zeros((3,3))
    k[0][0] = alpha_x
    k[0][1] = s
    k[0][2] = x_0
    k[1][1] = alpha_y
    k[1][2] = y_0 
    k[2][2] = 1    
    
    return k

#finds intersection points from Hough Transform Lines 
def find_intersection(line1, line2):
    
    rho1,theta1 = line1
    rho2,theta2 = line2
    
    #get the intersection point
    a1 = np.cos(theta1)
    b1 = np.sin(theta1)
    a2 = np.cos(theta2)
    b2 = np.sin(theta2)
    
    #if determinant is 0 then lines are parallel and there is no 
    # intersection 
    det = a1*b2-a2*b1
    
    if abs(det) < 10e-6:
        return None 
    else:
        #find the intersection point
        x = (b2*rho1-b1*rho2) / det
        y = (a1*rho2-a2*rho1) / det
        return (int(x), int(y))
        

#this function elimates duplicate lines from Hough Transformation
def remove_dup_lines(lines):
    
    #number vertical and horizontal lines 
    vertical_count = 8
    horizontal_count = 10
    
    #the duplicate line ranges 
    vertical_rho_range = 10
    vertical_theta_range = 15 * (np.pi / 180)
    horizontal_rho_range = 10
    horizontal_theta_range = 15 * (np.pi / 180)
    
    #seperate all the lines into vertical and horizontal lines 
    vertical_lines = []
    horizontal_lines = []

    for line in lines:
        rho, theta = line[0]
        if abs(theta) < np.pi/4 or abs(theta-np.pi) < np.pi/4: 
            vertical_lines.append(line)
        else:  
            horizontal_lines.append(line)
      
    #keep incraese rho and theta for vertical lines until reach 8 lines 
    while True: 
        new_vertical_lines = filter_lines(vertical_lines, 
                                          vertical_rho_range, 
                                          vertical_theta_range)
        if len(new_vertical_lines) == vertical_count:
            break
        elif len(new_vertical_lines) > vertical_count:
            vertical_rho_range += 1
            vertical_theta_range += 2 * (np.pi / 180)
        else:
            print(f"Found {len(new_vertical_lines)} Vertical Lines.")
            break
        
    #keep incraese rho and theta for horizontal lines until reach 10 lines 
    while True: 
        new_horizontal_lines = filter_lines(horizontal_lines, 
                                            horizontal_rho_range, 
                                            horizontal_theta_range)
        if len(new_horizontal_lines) == horizontal_count:
            break
        elif len(new_horizontal_lines) > horizontal_count:
            horizontal_rho_range += 1
            horizontal_theta_range += 2 * (np.pi / 180)
        else:
            print(f"Found {len(new_horizontal_lines)} horizontal Lines.")
            break
        
    #sort the horizontal and vertical lines 
    new_vertical_lines.sort(key=lambda x: x[0][0])  
    new_horizontal_lines.sort(key=lambda x: x[0][0])
    
    return new_horizontal_lines,new_vertical_lines

#removes the duplicate lines based on rho and theta 
def filter_lines(line_list, rho_range, theta_range):
        filtered_lines = []
        
        for line in line_list:
            rho, theta = line[0]
            duplicate = False
            
            for f_line in filtered_lines:
                f_rho, f_theta = f_line[0]
                
                if (abs(rho-f_rho)<rho_range and abs(theta-f_theta)<\
                    theta_range)or(abs(rho+f_rho)<rho_range and \
                        abs(theta-(f_theta + np.pi))<theta_range):
                    duplicate = True
                    break
            
            if not duplicate:
                filtered_lines.append(line)
        
        return filtered_lines

def remove_incorrect_points(points,img_shape):
    
    x, y = img_shape
    new_points = []
    
    #loop through the points and make sure it is in image range 
    for point in points:
        point_x, point_y = point
        
        #check the range of the point
        if point_x > 0 and point_x < x and point_y > 0 and point_y < y:
            new_points.append(point)
        
    return new_points
    

#get the corner points for all calibration images
def get_corner_points(img_path, img_num, dataset):
    
    #set the threshold values for all preprocessing operations 
    canny_lower = 222
    canny_upper = 256
    hough_votes_for_line = 40
    
    #get the image 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    #use otsu to make the cannny detector more accurate 
    _, otsu_thresholded = cv2.threshold(img, 0, 255, 
                                        cv2.THRESH_BINARY + \
                                            cv2.THRESH_OTSU)
    
    #remove the noise with morphological operations 
    otsu_thresholded = cv2.bitwise_not(otsu_thresholded)
    otsu_thresholded = cv2.erode(otsu_thresholded,
                                 np.ones((5, 5), np.uint8), iterations=1)
    otsu_thresholded = cv2.dilate(otsu_thresholded, 
                                  np.ones((5, 5), np.uint8), iterations=1)
    cv2.imwrite(dataset+'_otsu_'+img_num+'.jpeg', otsu_thresholded) 
    
    #apply canny to get the edges 
    edges = cv2.Canny(otsu_thresholded, canny_lower, canny_upper)
    cv2.imwrite(dataset+'_canny_'+img_num+'.jpeg', edges) 
    
    #get lines using Hough Transformation 
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 
                           threshold=hough_votes_for_line)
    
    #elimate duplicate lines 
    hor_lines, vert_lines = remove_dup_lines(lines)
    
    #create image for Hough Lines 
    lines = hor_lines + vert_lines
    hough_img = np.copy(img) #create duplicate image 
    hough_img = cv2.cvtColor(hough_img, cv2.COLOR_GRAY2BGR)
    
    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x = a * rho
        y = b * rho
        pt1 = (int(x + 1000*(-b)), int(y + 1000*(a)))
        pt2 = (int(x - 1000*(-b)), int(y - 1000*(a)))
        cv2.line(hough_img, pt1, pt2, (0,0,255), 2)
    cv2.imwrite(dataset+'_hough_'+img_num+'.jpeg', hough_img) 
    
    #loop through all the lines and find intersection points
    points_img = np.copy(img) #create duplicate image 
    points_img = cv2.cvtColor(points_img, cv2.COLOR_GRAY2BGR)
    done = False
    
    while done == False:
        intersection_points = []
        for i in hor_lines:
            for j in vert_lines:
                intersection = find_intersection(i[0], j[0])
                if intersection is not None:
                    intersection_points.append(intersection)
        
        # check if point order is correct. then fix flip veritcal lines 
        # this case is all lines flipped
        if intersection_points[0][0] > intersection_points[7][0]:
            vert_lines = vert_lines[::-1]
        #only lines of 1 to 4 flipped
        elif intersection_points[0][0] > intersection_points[3][0]:
            vert_lines = vert_lines[0:4][::-1]+vert_lines[4:]
        #only lines of 1 to 2 flipped
        elif intersection_points[0][0] > intersection_points[1][0]:
            vert_lines = vert_lines[0:2][::-1]+vert_lines[2:]
        else:
            done = True
                    
    # eliminate the images which corner points are not in order and 
    # points are too close (error in corner detection)
    for i in range(7):
        if intersection_points[i][0] >= intersection_points[i+1][0]:
            return None  
        if abs(intersection_points[i][0] - intersection_points[i+1][0]) < 5:
            return None
    
    #display all the intersection points 
    for num,point in enumerate(intersection_points):
        cv2.circle(points_img, point, 3, (255,0,0), -1)
        point_num = f"{num+1}"
        cv2.putText(points_img, point_num, (point[0]+2,point[1]-2),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1,cv2.LINE_AA)
    cv2.imwrite(dataset+'_points_'+img_num+'.jpeg', points_img) 
                
    return intersection_points

def applyP(homography, world_coords):
    
    #convert to homogeneous coords 
    world_coords_homo = []
    for point in world_coords:
        world_coords_homo.append((point[0],point[1],0,1))
    world_coords_homo = np.array(world_coords_homo)
    
    #get projected points
    proj_points = []
    for point in world_coords_homo:
        temp = homography @ point
        temp = temp / temp[2]
        temp = (temp[0], temp[1])
        proj_points.append(temp)
    proj_points = np.array(proj_points)
    
    return proj_points

#this function takes the world coords to the image 
def calc_geometric_diff(points, projected_points):
    
    points = np.array(points)
    projected_points = np.array(projected_points)
    
    #calculate the distance between corresponding points
    error = (points - projected_points)**2
    error = np.sum(error, axis=1)
    error = np.sqrt(error)
    
    #find the mean and variance
    mean = np.mean(error)
    variance = np.var(error)
    
    return error, mean, variance
    
    
#this function takes the world coords to the image 
def reprojection(world_coords, points, K, R, t, img_num, dataset):

    #create a homography     
    P = np.zeros((3,4))
    P[:3,:3] = R
    P[:,3] = t
    P = K @ P 
   
    #get projected points 
    proj_points = applyP(P, world_coords)
    
    #get the eucilidan distance 
    if dataset is not None:
        error, mean, variance = calc_geometric_diff(points, proj_points)
    else:
        error = (points - proj_points)**2
        return error, -1, -1
    
    #plot the points 
    if dataset is not None:
        img_path = '/home/patelb/ece661/hw08/Dataset1/Pic_'+str(img_num)+'.jpg'
        points_img = cv2.imread(img_path)
        
        for num,(proj_point, point) in enumerate(zip(proj_points,points)):
            
            cv2.circle(points_img, point, 3, (255,0,0), -1)
            cv2.circle(points_img, (int(proj_point[0]),
                                    int(proj_point[1])), 3, (0,255,0), -1)
            
            point_num = f"{num+1}"
            cv2.putText(points_img,point_num,(point[0]+2,point[1]-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),1,cv2.LINE_AA)
           
            
        cv2.imwrite(dataset+'_reconstructed_'+str(img_num)+'.jpeg', points_img) 
      
    return error, mean, variance

#this is the cost function for the LM optimization 
def cost_func(parameters,world_coords, corner_points_all): 
    
    #first take the parameters and seperate r, t, k 
    #get k from the end of the parameters
    k = np.zeros((3,3))
    k[0][0] = parameters[-5]
    k[0][1] = parameters[-4]
    k[0][2] = parameters[-3]
    k[1][1] = parameters[-2]
    k[1][2] = parameters[-1]
    k[2][2] = 1
    
    #starting from parameters[5] get the r and t 
    r_all = []
    t_all = []
    
    for i in range(0,len(parameters)-5, 6):
        
        #get the rodriques representation and convert to r
        w = []
        w.append(parameters[i])
        w.append(parameters[i+1])
        w.append(parameters[i+2])
        w = np.array(w)
        r = cv2.Rodrigues(w)[0]
        r_all.append(r)
        
        #get the t
        t = []
        t.append(parameters[i+3])
        t.append(parameters[i+4])
        t.append(parameters[i+5])
        t_all.append(t)
    
    #now calculate the reprojection for every image 
    j = 0 
    error_all = np.array([])
    for i in range(len(corner_points_all)):
        if corner_points_all[i] != -1:
            error, _, _ = reprojection(world_coords, 
                                       corner_points_all[i], 
                                       k, r_all[j], t_all[j], 
                                       i+1, None)
            error_all = np.append(error_all,error)
            j+=1
            
    return error_all


def create_3d_rep(r_all, t_all, world_coodinates, square_size):
    
    #initlize the variables for each camera pose
    camera_center_x = []
    camera_center_y = []
    camera_center_z = []
    x_cam_world_all = []
    y_cam_world_all = []
    z_cam_world_all = []
    
    x_cam =np.array([1,0,0])
    y_cam =np.array([0,1,0])
    z_cam =np.array([0,0,1])   
    
    for i in range(len(r_all)):
        
        #get the camera center 
        camera_center = -1*np.transpose(r_all[i]) @ t_all[i]
        camera_center_x.append(camera_center[0])
        camera_center_y.append(camera_center[1])
        camera_center_z.append(camera_center[2])
        
        #get the x,y,z for the camera in world coords 
        x_cam_world = np.array((np.transpose(r_all[i]) @ x_cam))
        y_cam_world = np.array((np.transpose(r_all[i]) @ y_cam))
        z_cam_world = np.array((np.transpose(r_all[i]) @ z_cam))
        
        #normalize the vectors 
        mag_x = np.linalg.norm(x_cam_world)
        mag_y = np.linalg.norm(y_cam_world) 
        mag_z = np.linalg.norm(z_cam_world)
        x_cam_world_norm = x_cam_world / mag_x
        y_cam_world_norm = y_cam_world / mag_y
        z_cam_world_norm = z_cam_world / mag_z

        #append the normalized vectors
        x_cam_world_all.append(x_cam_world_norm*square_size)
        y_cam_world_all.append(y_cam_world_norm*square_size)
        z_cam_world_all.append(z_cam_world_norm*square_size)
        
    #convert to numpy 
    x_cam_world_all = np.array(x_cam_world_all)
    y_cam_world_all = np.array(y_cam_world_all)
    z_cam_world_all = np.array(z_cam_world_all)
        
    #create the 3d plot for all the camera poses
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    # Set the labels for the axes
    ax.set_xlabel('X World')
    ax.set_ylabel('Y World')
    ax.set_zlabel('Z World')
    
    #plot the camera centers
    ax.scatter(camera_center_x, camera_center_y, 
               camera_center_z, marker='*', color='yellow')
    
    #plot the vectors for x 
    x_x_vector = x_cam_world_all[:,0]
    y_x_vector = x_cam_world_all[:,1]
    z_x_vector = x_cam_world_all[:,2]
    ax.quiver(camera_center_x, camera_center_y, 
              camera_center_z, x_x_vector, 
              y_x_vector, z_x_vector,color='red')
    
    #plot the vectors for y
    x_y_vector = y_cam_world_all[:,0]
    y_y_vector = y_cam_world_all[:,1]
    z_y_vector = y_cam_world_all[:,2]
    ax.quiver(camera_center_x, camera_center_y, 
              camera_center_z, x_y_vector, 
              y_y_vector, z_y_vector,color='green')
        
    #plot the vectors for z
    x_z_vector = z_cam_world_all[:,0]
    y_z_vector = z_cam_world_all[:,1]
    z_z_vector = z_cam_world_all[:,2]
    ax.quiver(camera_center_x, camera_center_y, 
              camera_center_z, x_z_vector, 
              y_z_vector, z_z_vector,color='blue')
    
    
    for i in range(len(camera_center_x)):
        
        x_mesh, y_mesh = np.meshgrid(np.linspace(camera_center_x[i]-square_size,
                                                 camera_center_x[i]+square_size,30),
                                     np.linspace(camera_center_y[i]-square_size,\
                                         camera_center_y[i]+square_size,30))
        camera_center = np.array([camera_center_x[i], 
                                  camera_center_y[i], camera_center_z[i]])
        
        #use the normal vector 
        normal = z_cam_world_all[i]
        d = camera_center.dot(normal)
        z_mesh = (-normal[0] * x_mesh - normal[1] * y_mesh + d)/normal[2]
        
        random_color = (rand.randint(0,255),rand.randint(0,255),rand.randint(0,255))

        ax.plot_surface(x_mesh, y_mesh, z_mesh, color=np.random.rand(3,))
        
    #plot the checkerboard. Get the outermost points
    min_x = float('inf')
    max_x = float('-inf')
    min_y = float('inf')
    max_y = float('-inf')
    
    for x, y in world_coodinates:
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y
            
    calibration_points = np.array([[min_x,min_y,0],[max_x,min_y,0],
                                   [max_x,max_y,0],[min_x,max_y,0]])
    
    #plot the calibration pattern outline 
    ax.scatter(calibration_points[:,0],calibration_points[:,1],
               calibration_points[:,2], color='black')
    polygon = Poly3DCollection([calibration_points], color='black')
    ax.add_collection3d(polygon)
    plt.show()
    
    return

def main():
    
    
    #######################################
    dataset = 1
    num_img = 40 
    homography_all = []
    corner_points_all = []
    
    #first get the global coordinates 
    width = 8
    height = 10 
    square_size = 1
    world_coords = []
    
    #loop through and get world coords
    for i in range(height):  
        for j in range(width):  
            x = j * square_size 
            y = i * square_size 
            world_coords.append([x, y]) 
    #convert to numpy array
    np.array(world_coords)
        
    #get all the homographies for all the images 
    for img_num in range(num_img):
        img_num += 1
        img_path = '/home/patelb/ece661/hw08/Dataset1/Pic_'+str(img_num)+'.jpg'
        print(img_path)
        
        #get the corner points for the img 
        corner_points = get_corner_points(img_path,str(img_num),str(dataset))
        if corner_points is None:
            corner_points_all.append(-1)
        else:
            corner_points_all.append(corner_points)
            
        #calculate the homopgraphy
        homography = create_homography_estimation(world_coords, corner_points)
        if homography is not None:
            homography_all.append(homography)
            
    
    #use homogrpahy to find K (intrinstic params)
    k = find_K(homography_all)
    
    #find extrinsic params for all images 
    r_all, t_all = find_extrinsic(homography_all, k)
    
    #now we have all the intrinsic and extrinstic parameters 
    #lets get the projection before LM for all the images with the mean 
    # error and distance
    mean_all_before_lm = []
    variance_all_before_lm = []
    j = 0 
    for i in range(len(corner_points_all)):
        if corner_points_all[i] != -1:
            error, mean, variance = reprojection(world_coords, 
                                                 corner_points_all[i], 
                                                 k, r_all[j], t_all[j],
                                                 i+1, str(dataset))
            mean_all_before_lm.append(mean)
            variance_all_before_lm.append(variance)
            j+=1
            
    #########################################################
            
    # #now apply LM optimization
    # #first make the parameters for the LM function [K,w_1,t_1,w_2,t_2,....]
    parameters = []

    for r,t in zip(r_all, t_all):
        
        #convert the r to w (Rodriquez)
        w = cv2.Rodrigues(r)[0]
        w = w.tolist()
        
        #append w,r to the paramaters list 
        parameters.append(w[0][0])
        parameters.append(w[1][0])
        parameters.append(w[2][0])
        
        parameters.append(t[0])
        parameters.append(t[1])
        parameters.append(t[2])
        
    #add the k to the end
    parameters.append(k[0][0])
    parameters.append(k[0][1])
    parameters.append(k[0][2])
    parameters.append(k[1][1])
    parameters.append(k[1][2])
    
    parameters = [float(x) for x in parameters] 
    
    #call the least squres function 
    lm_out = least_squares(cost_func, parameters, method='lm', 
                           args=[world_coords, corner_points_all])
    
    print('lm done')
    
    #deconstruct the lm output 
    lm_out = lm_out.x
    
    #first take the parameters and seperate r, t, k 
    #get k from the end of the parameters
    k_lm = np.zeros((3,3))
    k_lm[0][0] = lm_out[-5]
    k_lm[0][1] = lm_out[-4]
    k_lm[0][2] = lm_out[-3]
    k_lm[1][1] = lm_out[-2]
    k_lm[1][2] = lm_out[-1]
    k_lm[2][2] = 1
    
    #starting from parameters[5] get the r and t 
    r_all_lm = []
    t_all_lm = []
    
    for i in range(0,len(lm_out)-5, 6):
        
        #get the rodriques representation and convert to r
        w = []
        w.append(lm_out[i])
        w.append(lm_out[i+1])
        w.append(lm_out[i+2])
        w = np.array(w)
        r = cv2.Rodrigues(w)[0]
        r_all_lm.append(r)
        
        #get the t
        t = []
        t.append(lm_out[i+3])
        t.append(lm_out[i+4])
        t.append(lm_out[i+5])
        t_all_lm.append(t)
        
        
    #get the reconstruction error for after the lm 
    mean_all_after_lm = []
    variance_all_after_lm = []
    j = 0 
    for i in range(len(corner_points_all)):
        if corner_points_all[i] != -1:
            error, mean, variance = reprojection(world_coords, 
                                                 corner_points_all[i],
                                                 k_lm, r_all_lm[j], 
                                                 t_all_lm[j], i+1, str(3))
            mean_all_after_lm.append(mean)
            variance_all_after_lm.append(variance)
            j+=1
            
    ###########################################################
    
    #now create a 3d represenation of everything 
    create_3d_rep(r_all, t_all, world_coords, square_size)
    
    return 


if __name__=="__main__":
    main()