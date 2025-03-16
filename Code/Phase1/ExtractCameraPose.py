import numpy as np




def extract_camera_pose(E):
    U,S,Vt = np.linalg.svd(E)
    W = np.array([[0,-1,0],
                  [1,0,0],
                  [0,0,1]
                            ])
        
    C1 = U[:,2]
    R1 = U @ W @ Vt
    
    C2 = -U[:,2]
    R2 = U @ W @ Vt
    
    C3 = U[:,2]
    R3 = U @ W.T @ Vt
    
    C4 = -U[:,2]
    R4 = U @ W.T @ Vt
    
    pose = []
    
    for C, R in [(C1,R1), (C2,R2),(C3,R3),(C4,R4)]:
        if np.linalg.det(R)<0:
            R = -R 
            C = -C
            
        pose.append((C,R))
        
        
    return pose 