import cv2
import numpy as np
import glob
import logging
import random


def read_images():
    logging.info('Reading images...')
    images = []
    for file_name in glob.glob('*.jpg'):
        img = cv2.imread(file_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(gray)

    return images


def draw_matches(img1, kp1, img2, kp2, matches, inliers=None):
    logging.info('Drawing matches...')
    # Create a new output image that concatenates the two images together
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')

    # Place the first image to the left
    out[:rows1, :cols1, :] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2, cols1:cols1 + cols2, :] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns, y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        inlier = False

        if inliers is not None:
            for i in inliers:
                if i.item(0) == x1 and i.item(1) == y1 and i.item(2) == x2 and i.item(3) == y2:
                    inlier = True

        # Draw a small circle at both coordinates
        cv2.circle(out, (int(x1), int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2) + cols1, int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points, draw inliers if we have them
        if inliers is not None and inlier:
            cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (0, 255, 0), 1)
        elif inliers is not None:
            cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (0, 0, 255), 1)

        if inliers is None:
            cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (255, 0, 0), 1)

    return out


def calculate_homography(correspondences):
    logging.info('Calculating homography...')
    a_list = []
    for correspondence in correspondences:
        # First, convert each points into homogeneous coordinates
        point1 = np.matrix([correspondence.item(0), correspondence.item(1), 1])
        point2 = np.matrix([correspondence.item(2), correspondence.item(3), 1])

        # For each correspondence xi <--> xi' compute the matrix Ai. Only the first two rows need to be used in general.
        a_row1 = [-point2.item(2) * point1.item(0), -point2.item(2) * point1.item(1), -point2.item(2) * point1.item(2),
                  0, 0, 0,
                  point2.item(0) * point1.item(0), point2.item(0) * point1.item(1), point2.item(0) * point1.item(2)]
        a_row2 = [0, 0, 0, -point2.item(2) * point1.item(0), -point2.item(2) * point1.item(1),
                  -point2.item(2) * point1.item(2),
                  point2.item(1) * point1.item(0), point2.item(1) * point1.item(1), point2.item(1) * point1.item(2)]

        # Assemble the n 2x9 matrices Ai into a single 2nx9 matrix A.
        a_list.append(a_row1)
        a_list.append(a_row2)

    A_matrix = np.matrix(a_list)

    # Obtain the SVD of A. The unit singular vector corresponding to the smallest singular value is the solution h.
    # Specifically, if A = UDV^T with D diagonal with positive diagonal entries, arranged in descending order down
    # the diagonal, then h is the last column of V.
    u, s, v = np.linalg.svd(A_matrix)

    # Reshape the minimum singular value into a 3 by 3 matrix.
    h = np.reshape(v[8], (3, 3))
    h = (1 / h.item(8)) * h

    return h


def residual_error(correspondence, h):
    logging.info('Calculating algebraic distance...')
    point1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
    estimated_point2 = np.dot(h, point1)
    estimated_point2 = (1 / estimated_point2.item(2)) * estimated_point2

    point2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
    error = point2 - estimated_point2
    return np.linalg.norm(error)


def ransac(corr, thresh):
    logging.info('Running RANSAC...')
    max_inliers = []
    final_h = None
    for i in range(1000):
        corr1 = corr[random.randrange(0, len(corr))]
        corr2 = corr[random.randrange(0, len(corr))]
        random_four = np.vstack((corr1, corr2))
        corr3 = corr[random.randrange(0, len(corr))]
        random_four = np.vstack((random_four, corr3))
        corr4 = corr[random.randrange(0, len(corr))]
        random_four = np.vstack((random_four, corr4))

        h = calculate_homography(random_four)
        print(h)
        inliers = []

        for i in range(len(corr)):
            d = residual_error(corr[i], h)
            if d < 5:
                inliers.append(corr[i])

        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            final_h = h
        print("Correspondence size: ", len(corr), " Number of inliers: ", len(inliers), "Maximum inliers: ", len(max_inliers))

        if len(max_inliers) > (len(corr)*thresh):
            break
    return final_h, max_inliers


def find_keypoints(image):
    logging.info('Finding keypoints...')
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    cv2.drawKeypoints(image, keypoints, image)
    cv2.imwrite('sift_keypoints.png', image)

    return keypoints, descriptors


def match_keypoints(kp1, kp2, desc1, desc2, image1, image2):
    logging.info('Matching keypoints...')
    matcher = cv2.BFMatcher(cv2.NORM_L2, True)
    matches = matcher.match(desc1, desc2)
    match_image = draw_matches(image1, kp1, image2, kp2, matches)
    cv2.imwrite('matches.png', match_image)

    return matches


def main():
    threshold = 0.60
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    images = read_images()
    correspondence_list = []
    kp1, desc1 = find_keypoints(images[0])
    kp2, desc2 = find_keypoints(images[1])
    keypoints = [kp1, kp2]
    matches = match_keypoints(kp1, kp2, desc1, desc2, images[0], images[1])
    for match in matches:
        (x1, y1) = keypoints[0][match.queryIdx].pt
        (x2, y2) = keypoints[1][match.trainIdx].pt
        correspondence_list.append([x1, y1, x2, y2])

    correspondences = np.matrix(correspondence_list)

    final_h, inliers = ransac(correspondences, threshold)
    print("Final homography: ", final_h)
    print("Final inliers count: ", len(inliers))

    match_img = draw_matches(images[0], kp1, images[1], kp2, matches, inliers)
    cv2.imwrite('Inlier_matches.png', match_img)


if __name__ == "__main__":
    main()
