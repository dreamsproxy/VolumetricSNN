import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
import os

def generate_tuples(strings):
    tuples_list = []
    n = len(strings)
    
    progbar = tqdm(total = (n ** 3)//3)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if i != j and j != k and k != i:
                    tuple_triple = (strings[i], strings[j], strings[k])
                    # Check if the tuple_triple is already present in tuples_list
                    if tuple_triple not in sorted(tuples_list):
                        tuples_list.append(tuple_triple)
                        progbar.update(1)
    
    return tuples_list

def run(data):
    img_path1, img_path2, img_path3 = data
    sift = cv2.SIFT_create(nfeatures=4096)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    focal_length = 20.45
    save_dir = os.path.dirname(img_path1)
    os.makedirs(f"{save_dir}/rmat/", exist_ok=True)
    os.makedirs(f"{save_dir}/tmat/", exist_ok=True)
    os.makedirs(f"{save_dir}/points/", exist_ok=True)
    img1 = cv2.imread(img_path1, 0)
    img2 = cv2.imread(img_path2, 0)
    img3 = cv2.imread(img_path3, 0)
    p_point = (img1.shape[0] // 2, img1.shape[1] // 2)

    # Extract SIFT features and compute matches for the img pairs
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    keypoints3, descriptors3 = sift.detectAndCompute(img3, None)

    matches_1_2 = bf.match(descriptors1, descriptors2)
    matches_1_2 = sorted(matches_1_2, key=lambda x: x.distance)

    matches_2_3 = bf.match(descriptors2, descriptors3)
    matches_2_3 = sorted(matches_2_3, key=lambda x: x.distance)

    matches_1_3 = bf.match(descriptors1, descriptors3)
    matches_1_3 = sorted(matches_1_3, key=lambda x: x.distance)

    # Get the matched keypoints for the img pairs
    matched_keypoints1_2 = np.float32([keypoints1[m.queryIdx].pt for m in matches_1_2]).reshape(-1, 1, 2)
    matched_keypoints2_2 = np.float32([keypoints2[m.trainIdx].pt for m in matches_1_2]).reshape(-1, 1, 2)

    matched_keypoints2_3 = np.float32([keypoints2[m.queryIdx].pt for m in matches_2_3]).reshape(-1, 1, 2)
    matched_keypoints3_3 = np.float32([keypoints3[m.trainIdx].pt for m in matches_2_3]).reshape(-1, 1, 2)

    matched_keypoints1_3 = np.float32([keypoints1[m.queryIdx].pt for m in matches_1_3]).reshape(-1, 1, 2)
    matched_keypoints3_3 = np.float32([keypoints3[m.trainIdx].pt for m in matches_1_3]).reshape(-1, 1, 2)

    m1_2 = matched_keypoints1_2.shape[0]
    m2_2 = matched_keypoints2_2.shape[0]
    m2_3 = matched_keypoints2_3.shape[0]
    m1_3 = matched_keypoints1_3.shape[0]
    m3_3 = matched_keypoints3_3.shape[0]
    
    compare_list = [m1_2, m2_2, m2_3, m1_3, m3_3]
    trim = min(compare_list)
    matched_keypoints1_2 = matched_keypoints1_2[:trim-1]
    matched_keypoints2_2 = matched_keypoints2_2[:trim-1]
    matched_keypoints2_3 = matched_keypoints2_3[:trim-1]
    matched_keypoints1_3 = matched_keypoints1_3[:trim-1]
    matched_keypoints3_3 = matched_keypoints3_3[:trim-1]

    fundamental_matrix_1_2, _ = cv2.findFundamentalMat(matched_keypoints1_2, matched_keypoints2_2, cv2.FM_RANSAC)
    fundamental_matrix_2_3, _ = cv2.findFundamentalMat(matched_keypoints2_3, matched_keypoints3_3, cv2.FM_RANSAC)
    fundamental_matrix_1_3, _ = cv2.findFundamentalMat(matched_keypoints1_3, matched_keypoints3_3, cv2.FM_RANSAC)

    K = np.array([[focal_length, 0, p_point[0]],
                  [0, focal_length, p_point[1]],
                  [0, 0, 1]])

    essential_matrix_1_2 = np.dot(np.dot(K.T, fundamental_matrix_1_2), K)
    essential_matrix_2_3 = np.dot(np.dot(K.T, fundamental_matrix_2_3), K)
    essential_matrix_1_3 = np.dot(np.dot(K.T, fundamental_matrix_1_3), K)

    _, rotation_1_2, translation_1_2, _ = cv2.recoverPose(
        essential_matrix_1_2, matched_keypoints1_2,
        matched_keypoints2_2, cameraMatrix=K)
    _, rotation_2_3, translation_2_3, _ = cv2.recoverPose(
        essential_matrix_2_3, matched_keypoints2_3,
        matched_keypoints3_3, cameraMatrix=K)
    _, rotation_1_3, translation_1_3, _ = cv2.recoverPose(
        essential_matrix_1_3, matched_keypoints1_3,
        matched_keypoints3_3, cameraMatrix=K)

    # Triangulate the 3D points using the camera matrices
    projection_matrix_1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
    projection_matrix_2 = np.dot(K, np.hstack((rotation_1_2, translation_1_2)))
    projection_matrix_3 = np.dot(K, np.hstack((rotation_1_3, translation_1_3)))

    # Reshape matched keypoints
    matched_keypoints1_2 = matched_keypoints1_2.reshape(-1, 2).astype(np.float32)
    matched_keypoints1_3 = matched_keypoints1_3.reshape(-1, 2).astype(np.float32)
    matched_keypoints2_2 = matched_keypoints2_2.reshape(-1, 2).astype(np.float32)
    matched_keypoints2_3 = matched_keypoints2_3.reshape(-1, 2).astype(np.float32)
    matched_keypoints3_3 = matched_keypoints3_3.reshape(-1, 2).astype(np.float32)
    
    points_4d_12 = cv2.triangulatePoints(projection_matrix_1, projection_matrix_2, matched_keypoints1_2.T, matched_keypoints2_2.T)
    points_4d_12 /= points_4d_12[3]
    points_3d_12 = points_4d_12[:3].T

    points_4d_23 = cv2.triangulatePoints(projection_matrix_2, projection_matrix_3, matched_keypoints2_3.T, matched_keypoints3_3.T)
    points_4d_23 /= points_4d_23[3]
    points_3d_23 = points_4d_23[:3].T

    points_4d_13 = cv2.triangulatePoints(projection_matrix_1, projection_matrix_3, matched_keypoints1_3.T, matched_keypoints3_3.T)
    points_4d_13 /= points_4d_13[3]
    points_3d_13 = points_4d_13[:3].T
    
    img1_name = os.path.basename(img_path1).replace(".jpg", "")
    img2_name = os.path.basename(img_path2).replace(".jpg", "")
    img3_name = os.path.basename(img_path3).replace(".jpg", "")

    np.savetxt(f"{save_dir}/rmat/{img1_name}-{img2_name}_rmat.txt", rotation_1_2)
    np.savetxt(f"{save_dir}/tmat/{img1_name}-{img2_name}_tmat.txt", translation_1_2)
    np.savetxt(f"{save_dir}/points/{img1_name}-{img2_name}_points.txt", points_3d_12)

    np.savetxt(f"{save_dir}/rmat/{img2_name}-{img3_name}_rmat.txt", rotation_2_3)
    np.savetxt(f"{save_dir}/tmat/{img2_name}-{img3_name}_tmat.txt", translation_2_3)
    np.savetxt(f"{save_dir}/points/{img2_name}-{img3_name}_points.txt", points_3d_23)

    np.savetxt(f"{save_dir}/rmat/{img1_name}-{img3_name}_rmat.txt", rotation_1_3)
    np.savetxt(f"{save_dir}/tmat/{img1_name}-{img3_name}_tmat.txt", translation_1_3)
    np.savetxt(f"{save_dir}/points/{img1_name}-{img3_name}_points.txt", points_3d_13)

if __name__ == "__main__":
    files_list = glob(f"./dataset/Big Tree/*.jpg")[:16]

    pairs = generate_tuples(files_list)
    for p in tqdm(pairs):
        run(p)
    #with Pool(2) as pool:
    #    for _ in pool.imap_unordered(run, pairs, chunksize=3):
    #        progbar.update(1)