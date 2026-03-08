import math
import numpy as np
import cv2


def main():
    vid = cv2.VideoCapture("./1.mp4")
    background_subtractor = cv2.createBackgroundSubtractorKNN()
    
    success, frame = vid.read()
    if not success:
        raise RuntimeError("Cannot read video")
    h,w = frame.shape[:2]
    #initialising accumulator space
    h,w = frame.shape[:2]
    #taking diagonal of the frame to be the maximum possible value of rho, so we can cover all possible lines in the frame
    diag = int(math.sqrt(h*h + w*w))
    #taking step of 1 in angles and rhos
    rhos = list(range(-diag, diag + 1))
    thetas = [math.radians(a) for a in range(-90, 90)] 
    #finding cos and sin from before so that we don't have to calcualte it during the frame process, that gives TLE
    cos_theta = np.array([math.cos(t) for t in thetas])
    sin_theta = np.array([math.sin(t) for t in thetas])
    theta_arr = np.array(thetas)
    #what np.array returns is vectorised version that follows numpy algebra
    count, success = 0, True
    while success:
        success, frame = vid.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not success:
            break
        background = background_subtractor.apply(frame)
        kernel = np.ones((3, 3), np.uint8)
        opened_bg = cv2.morphologyEx(background, cv2.MORPH_OPEN, kernel)
        edges = cv2.Canny(opened_bg, 100, 200)
        
        #Accumulator space for this frame:
        acc = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
        nonZeroY, nonZeroX = np.nonzero(edges) 
        #gives us the coordinates of the white pixels, faster than iteration
        if len(nonZeroY) > 0:
            #outer.. gives output as (all edges, 180 angles) matrix
            rho_mat = np.outer(nonZeroX, cos_theta) + np.outer(nonZeroY, sin_theta) #vectorised version of rho = x*cos(theta) + y*sin(theta) on two loops
            rho_matrix = np.round(rho_mat).astype(int) + diag #we round it to get the index of the rho in the accumulator space, and we add diag to make it positive
            rho_matrix = np.clip(rho_matrix, 0, 2*diag) #clip to make sure we don't go out of bounds
            theta_index = np.array(list(range(len(thetas))))
            # for i in range(len(nonZeroX)):
            #     acc[rho_matrix[i], theta_index] += 1 #we add 1 to the accumulator space for each edge pixel, we use the rho and theta index to find the cell in the accumulator space
            np.add.at(acc, (rho_matrix, np.arange(len(thetas))), 1)
        
        acc2 = cv2.normalize(acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) #we normalise the accumulator space to be between 0 and 255 for maxima method
        peaks = []
        dilated = cv2.dilate(acc2, np.ones((5, 5), np.uint8))
        rows,cols = np.where((acc2 == dilated) & (acc2 > 120)) #gives us the cells where the value in accumulator is the same as the dilated version of accumulator space
        for r,c in zip(rows,cols): #zip convert two arrays into list of tuples 
            peaks.append((r,c))
        
        peak_final = []
        used = [False] * len(peaks)
        for i, (r1, c1) in enumerate(peaks):
            if used[i]:
                continue
            group = [(r1, c1)]
            used[i] = True
            for j, (r2, c2) in enumerate(peaks):
                if used[j]:
                    continue
                if abs(rhos[r1] - rhos[r2]) < 20 and abs(thetas[c1] - thetas[c2]) < 0.2: #we cluster the peaks that are close to each other in the accumulator space, we use a threshold of 20 for rho and 0.2 for theta
                    group.append((r2, c2))
                    used[j] = True
            peak_final.append(group[0]) #we take the first element of the group as the representative of the cluster
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	#uncomment to see all the hough lines detected
        # for r, c in peak_final:
        #     rho = rhos[r]
        #     theta = thetas[c]
        #     a = math.cos(theta)
        #     b = math.sin(theta)
        #     x0 = a * rho
        #     y0 = b * rho
        #     x1 = int(x0 + 1000 * (-b))
        #     y1 = int(y0 + 1000 * (a))
        #     x2 = int(x0 - 1000 * (-b))
        #     y2 = int(y0 - 1000 * (a))
        #     cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) #we draw the line on the frame using the rho and theta values from the accumulator space
        # count += 1

        #averaging the lines
        if peak_final:
            rho_vals = [rhos[r_idx] for r_idx, _ in peak_final]
            theta_vals = [thetas[t_idx] for _, t_idx in peak_final]
            rho_avg = sum(rho_vals) / len(rho_vals)
            theta_avg = sum(theta_vals) / len(theta_vals)

            a = math.cos(theta_avg)
            b = math.sin(theta_avg)
            x0 = a * rho_avg
            y0 = b * rho_avg
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("frame", frame)
        k = cv2.waitKey(30) & 0xFF
        if k == 27: #we exit the loop if the user presses the ESC key
            break
    vid.release()
    cv2.destroyAllWindows()
    
main()