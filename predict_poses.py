import numpy as np
import matplotlib.pyplot as plt
import cv2
#from utils import cam_pos
from glob import glob
import os
import json
import tqdm
import pandas as pd
import pickle

"""def show_matches(img1, img2):
    final_img = cv2.drawMatches(query_img,
                            queryKeypoints,
                            train_img,
                            trainKeypoints,
                            matches[:20],
                            None)
    final_img = cv2.resize(final_img, (1600,800))

    # Show the final image
    cv2.imshow("Feature Matches", final_img)
    cv2.waitKey(0)"""

def rotate_translate_points(i1, i2, translation, rotations):
    # Apply translation to i2
    i2_translated = i2 + translation

    # Apply rotations
    i2_transformed = i2_translated.copy()
    for rotation in rotations:
        rotation_matrix = euler_to_rotation_matrix(rotation)
        i2_transformed = np.dot(rotation_matrix, i2_transformed.T).T

    return i2_transformed

def euler_to_rotation_matrix(rotation):
    alpha, beta, gamma = rotation

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha), np.cos(alpha)]])

    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])

    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])

    rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))
    return rotation_matrix


def estimate(img1_path, img2_path, img3_path, n_features, focal_length, resize = False):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img3 = cv2.imread(img3_path)
    size = (img1.shape[0], img1.shape[0])
    #if resize:
    img1 = cv2.resize(img1, (1000, 1000))
    img2 = cv2.resize(img2, (1000, 1000))
    img3 = cv2.resize(img3, (1000, 1000))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

    principal_point_y = size[0]//2
    principal_point_x = size[1]//2
    
    # Step 3: Estimate the essential matrix
    camera_matrix = np.array([[focal_length, 0, principal_point_x],
                            [0, focal_length, principal_point_y],
                            [0, 0, 1]])

    sift = cv2.SIFT_create(nfeatures = n_features)

    img1_kp, img1_desc = sift.detectAndCompute(img1,None)
    img2_kp, img2_desc = sift.detectAndCompute(img2,None)
    img3_kp, img3_desc = sift.detectAndCompute(img3,None)

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # img 1 and 2
    matches_1_2 = matcher.match(img1_desc, img2_desc)
    matches_1_2 = sorted(matches_1_2, key=lambda x: x.distance)

    # img 2 and 3
    matches_2_3 = matcher.match(img2_desc, img3_desc)
    matches_2_3 = sorted(matches_2_3, key=lambda x: x.distance)

    # img 1 and 3
    matches_1_3 = matcher.match(img1_desc, img3_desc)
    matches_1_3 = sorted(matches_1_3, key=lambda x: x.distance)

    m1_2 = matches_1_2.shape[0]
    m2_3 = matches_1_2.shape[0]
    m1_3 = matches_1_3.shape[0]
    
    compare_list = [m1_2, m2_3, m1_3]
    trim = min(compare_list)
    matches_1_2 = matches_1_2[:trim]
    matches_2_3 = matches_2_3[:trim]
    matches_1_3 = matches_1_3[:trim]

    # Calculate the fundamental matrix
    fmat_1_2, _ = cv2.findFundamentalMat(matches_1_2, matches_1_2, cv2.FM_RANSAC)
    fmat_2_3, _ = cv2.findFundamentalMat(matches_2_3, matches_2_3, cv2.FM_RANSAC)
    fmat_1_3, _ = cv2.findFundamentalMat(matches_1_3, matches_1_3, cv2.FM_RANSAC)

    #emat_1_2, _ = cv2.findEssentialMat(img1_kp, img2_kp, camera_matrix, cv2.RANSAC, prob=0.999, threshold=1.0, maxIters=2048)
    #emat_2_3, _ = cv2.findEssentialMat(img2_kp, img3_kp, camera_matrix, cv2.RANSAC, prob=0.999, threshold=1.0, maxIters=2048)
    #emat_1_3, _ = cv2.findEssentialMat(img1_kp, img3_kp, camera_matrix, cv2.RANSAC, prob=0.999, threshold=1.0, maxIters=2048)

    emat_1_2 = np.dot(np.dot(camera_matrix.T, fmat_1_2), camera_matrix)
    emat_2_3 = np.dot(np.dot(camera_matrix.T, fmat_2_3), camera_matrix)
    emat_1_3 = np.dot(np.dot(camera_matrix.T, fmat_1_3), camera_matrix)
    
    # Step 4: Decompose the essential matrix
    _, r_1_2, t_1_2, mask_1_2 = cv2.recoverPose(emat_1_2, img1_kp, img2_kp, camera_matrix)
    _, r_2_3, t_2_3, mask_2_3 = cv2.recoverPose(emat_2_3, img2_kp, img3_kp, camera_matrix)
    _, r_1_3, t_1_3, mask_1_3 = cv2.recoverPose(emat_1_3, img1_kp, img3_kp, camera_matrix)
    
    pmat_1_2 = np.hstack((r_1_2, t_1_2))
    pmat_2_3 = np.hstack((r_2_3, t_2_3))
    pmat_1_3 = np.hstack((r_1_3, t_1_3))

    # Pack the data
    data_1_2 = [t_1_2, r_1_2, fmat_1_2, emat_1_2, pmat_1_2, matches_1_2]
    data_2_3 = [t_2_3, r_2_3, fmat_2_3, emat_2_3, pmat_2_3, matches_2_3]
    data_1_3 = [t_1_3, r_1_3, fmat_1_3, emat_1_3, pmat_1_3, matches_1_3]
    
    return data_1_2, data_2_3, data_1_3

def t_preprocessor(t_array, focal_length):
    translation = str(t_array).replace("\n", "")
    translation = translation.replace("  ", " ")
    translation = translation.replace("[", "")
    translation = translation.replace("]", "")
    translation = (translation).split(" ")
    translation = [np.float32(x) for x in translation if x != '']

    return translation

def r_preprocessor(r_array):
    rotation = str(r_array).replace("\n", "")
    rotation = rotation.replace("  ", " ")
    rotation = rotation.replace("[[ ", "[[")
    rotation = rotation.replace("] [", "], [")
    return rotation

def preprocess_pandas(filepath1, filepath2, t, r, fundamental_matrix, essential_matrix, projection_matrix):
    pass

def estimate_first(dir):
    image_paths = glob(dir + "*.jpg")
    image_paths = image_paths

    tr_dict = {}
    focal_length = 20.45
    pbar = tqdm.tqdm(total=len(image_paths)*len(image_paths), leave = True)
    cols = ["correlation", "translation", "rotation", "fundamental matrix", "essential matrix", "norm_dist", "calc_dist"]

    dump_dict = {}
    for i1, p1 in enumerate(image_paths):
        data = []
        temp_dict = {}
        temp_dict['path'] = p1.replace("\\", "/")
        relation_dict = {}
        for i2, p2 in enumerate(image_paths):
            if str(image_paths[i1]) != str(image_paths[i2]):
                d1, d2, d3 = estimate(image_paths[i1], image_paths[i2], 2048, focal_length, resize = False)
                t, r, f_matrix, e_matrix, p_matrix, matches = d1
                emulated_dist = np.linalg.norm(t)
                true_dist = emulated_dist * focal_length
                t = t_preprocessor(t, focal_length)
                r = r_preprocessor(r)
                relation_dict[i2] = {
                        'full path' : p2.replace("\\", "/"),
                        'translation' : str(t),
                        'rotation' : str(r).replace("\n", ""),
                        'virtual baseline' : str(emulated_dist),
                        'true baseline' : str(true_dist),
                        'projection matrix' : str(p_matrix)
                    }
                temp_dict['relations'] = relation_dict
                dump_dict[p1] = temp_dict
                matched_keypoints = [(match.queryIdx, match.trainIdx) for match in matches]
                
                #keypoints = map(bytes, keypoints)
                p1 = str(os.path.basename(p1)).replace(".jpg", "")
                p2 = str(os.path.basename(p2)).replace(".jpg", "")
                # Save the matched keypoints to a file using pickle
                with open(f"{dir}/{p1}-{p2}-matched.pickle", mode="wb") as f:
                    pickle.dump(matched_keypoints, f)
                pbar.update(1)

        data = pd.DataFrame(data, columns=cols)
        save_name = f"{dir}{os.path.basename(p1)}.csv"
        data.to_csv(save_name, index=False)
        json_obj = json.dumps(dump_dict, indent=4)
    
        with open("./dataset/room test.json", "w") as jfile:
            jfile.write(json_obj)
        

def estimate_dir(dir, n_features, resize_bool, trim_dataset = 0):
    if trim_dataset >=1:
        files_list = glob(dir)[:trim_dataset]
    else:
        files_list = glob(dir)
        
    tr_dict = {}
    focal_length_set = np.arange(4.0, 6.0, 0.5)
    focal_length = np.mean(focal_length_set)
    pbar = tqdm.tqdm(total=len(files_list)*len(files_list), leave = True)
    
    for root_idx, img1 in enumerate(files_list):
        #print(img1)
        temp_dict = {}
        temp_dict['path'] = img1.replace("\\", "/")
        relation_dict = {}
        for sub_idx, img2 in enumerate(files_list):
            if str(img1) != str(img2):
                t, r, focal_length, proj_mat = estimate(img1, img2, n_features, focal_length, resize = resize_bool)
                t = t_preprocessor(t, focal_length)
                r = r_preprocessor(r)
                emulated_dist = np.linalg.norm(t)
                true_dist = emulated_dist * focal_length
                relation_dict[sub_idx] = {
                        'full path' : img2.replace("\\", "/"),
                        'translation' : str(t),
                        'rotation' : str(r).replace("\n", ""),
                        'virtual baseline' : str(emulated_dist),
                        'true baseline' : str(true_dist),
                        'projection matrix' : str(proj_mat)
                    }

                temp_dict['relations'] = relation_dict
                pbar.update(1)
        
        pbar.update(1)
        tr_dict[root_idx] = temp_dict
    
    return tr_dict, focal_length

def sort_transdistance(json_data):
    if isinstance(json_data, str):
        if json_data[-5:] == ".json":
            # Load the JSON file as a dictionary
            with open(json_data) as file:
                data = json.load(file)
                for item in data.values():
                    relations = item['relations']
                    sorted_relations = sorted(relations.items(), key=lambda x: float(x[1]['trans_distance']))
                    item['relations'] = dict(sorted_relations)
        return data
    
    elif isinstance(json_data, dict):
        # Sort the 'relations' nested list by 'trans_distance' value for each 'path'
        for item in data.values():
            relations = item['relations']
            sorted_relations = sorted(relations.items(), key=lambda x: float(x[1]['trans_distance']))
            item['relations'] = dict(sorted_relations)
            
    return data

import json

def get_smallest_truebaseline(file_path):
    # Load the JSON file as a dictionary
    with open(file_path) as file:
        data = json.load(file)

    pair_data = []
    
    # Iterate over the first layer's data to find the smallest 'trans_distance' and its 'full_path'
    smallest_distance = float('inf')
    smallest_path = None
    for item in data.values():
        path = item['path']
        relations = item['relations']
        for relation in relations.values():
            trans_distance = float(relation['true baseline'])
            if trans_distance < smallest_distance:
                smallest_distance = trans_distance
                smallest_path = relation['full path']
        pair_data.append((path, smallest_path, smallest_distance))

    # Update the data dictionary with the smallest 'trans_distance' and its 'full_path'
    #data['smallest_trans_distance'] = smallest_distance
    #data['smallest_full_path'] = smallest_path

    # Return the updated data
    return pair_data


def create_depthmap(pair_list):
    depth_data = []
    for pair in pair_list:
        # Load the two closest images
        img1 = cv2.cvtColor(cv2.resize(cv2.imread(pair[1]), (800, 800)), cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(cv2.resize(cv2.imread(pair[0]), (800, 800)), cv2.COLOR_BGR2GRAY)

        # Convert the images to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Create a StereoBM object with specified parameters
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

        # Compute the disparity map
        disparity = stereo.compute(gray1, gray2)

        # Normalize the disparity map for visualization
        disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Display the disparity map
        cv2.imshow('Disparity Map', disparity_normalized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    return depth_data

if __name__ == "__main__":
    df = estimate_first("dataset/room test/")

    """relative_dict, focal_length = estimate_dir(
        "dataset/room test/*.jpg",
        n_features = 2048,
        resize_bool = True,
        trim_dataset = 0
    )
    df = estimate_dir("dataset/room test/")
    
    json_obj = json.dumps(relative_dict, indent=4)
    
    with open("./room test.json", "w") as jfile:
        jfile.write(json_obj)
    
    del relative_dict
    del json_obj
    focal_length = np.mean(np.arange(4.0, 6.0, 0.2))
    smallest_pairs = get_smallest_truebaseline("./BRICK.json")
    depthmaps = create_depthmap(smallest_pairs, focal_length)
    print(depthmaps[0].max())
    depthmaps[0] = depthmaps[0] * 255.0/depthmaps[0].max()
    plt.imshow(depthmaps[0], cmap="gray")
    plt.show()
    cv2.imshow("depth map", depthmaps[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """