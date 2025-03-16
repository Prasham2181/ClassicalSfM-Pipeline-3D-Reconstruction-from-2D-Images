import numpy as np



def disambiguateCameraPose(C_set, R_set, X_set):
    
    
    best_count = -1
    best_index = -1
    for i in range(len(C_set)):
        count = check_cheirality(X_set[i], R_set[i], C_set[i])
        print(f"Candidate {i+1} has {count} points with positive depth.")
        
        # If this candidate has more points with positive depth than previous ones, update best_index.
        if count > best_count:
            best_count = count
            best_index = i
    
    # Retrieve the best candidate's camera pose and triangulated points.
    best_C = C_set[best_index]
    best_R = R_set[best_index]
    best_X = X_set[best_index]
    
    # print(f"\nSelected candidate {best_index+1} with {best_count} valid points.")
    return best_C, best_R, best_X
    
    
def check_cheirality(X ,R , C):
    count = 0   # positive depth count 
    
    for x in X:
        depth1 = x[2]
        x_cam = R@(x-C)
        depth2 = x_cam[2]
        
        if depth1>0 and depth2>0:
            count+=1
            
    return count
    