import numpy as np
import matplotlib.pyplot as pyplot



####### a function which shows you the annotated points ###########
def show_annotation(I, p1, p2):
    plt.figure()
    
    # show the image
    # plot point p1 as a green circle, with markersize 10, and label "tip"
    # plot point p2 as a red circle, with markersize 10, and label "end"
    # plot a line starts at one point and end at another. 
    # Use a suitable color and linewidth for better visualization
    # Add a legend (tip, you can use the "label" keyword when you plot a point)
    
    plt.imshow(I)
    plt_points_tip = plt.plot(p1[0], p1[1], 'ro', markersize = 10, c = 'g', label = 'tip')
    plt_points_end = plt.plot(p2[0], p2[1], 'ro', markersize = 10, c = 'r', label = 'end')
    plt_line = plt.plot([p1[0],p2[0]], [p1[1],p2[1]], linewidth = 2)
    plt.legend()
    plt.show()

######## a function which takes in samples around the annotated points ###########    
def sample_points_around_pen(I, p1, p2):
    Nu = 100 # uniform samples (will mostly be background, and some non-background)
    Nt = 50 # samples at target locations, i.e. near start, end, and middle of pen
    
    target_std_dev = np.array(HALF_WIN_SIZE[:2])/3 # variance to add to locations

    upoints = sample_points_grid(I)
    idxs = np.random.choice(upoints.shape[0], Nu)
    upoints = upoints[idxs,:]
    
    # sample around target locations
    tpoints1 = np.random.randn(Nt,2)
    tpoints1 = tpoints1 * target_std_dev + p1

    tpoints2 = np.random.randn(Nt,2)
    tpoints2 = tpoints2 * target_std_dev + p2

    # sample over length pen
    alpha = np.random.rand(Nt)
    tpoints3 = p1[None,:] * alpha[:,None] + p2[None,:] * (1. - alpha[:,None])
    tpoints3 = tpoints3 + np.random.randn(Nt,2) * target_std_dev
    
    # merge all points
    points = np.vstack((upoints, tpoints1, tpoints2, tpoints3))
    
    # discard points close to border where we can't extract patches
    points = remove_points_near_border(I, points)
    
    return points

################ a function which makes labels from the points (tip, bottom, edge )
def make_labels_for_points(I, p1, p2, points):
    """ Determine the class label (as an integer) on point distance to different parts of the pen """
    num_points = points.shape[0]
    
    # for all points ....
    
    # ... determine their distance to tip of the pen
    dist1 = points - p1
    dist1 = np.sqrt(np.sum(dist1 * dist1, axis=1))
    
    # ... determine their distance to end of the pen
    dist2 = points - p2
    dist2 = np.sqrt(np.sum(dist2 * dist2, axis=1))

    # ... determine distance to pen middle
    alpha = np.linspace(0.2, 0.8, 100)
    midpoints = p1[None,:] * alpha[:,None] + p2[None,:] * (1. - alpha[:,None]) 
    dist3 = scipy.spatial.distance_matrix(midpoints, points)
    dist3 = np.min(dist3, axis=0)
    
    # the class label of a point will be determined by which distance is smallest
    #    and if that distance is at least below `dist_thresh`, otherwise it is background
    dist_thresh = WIN_SIZE[0] * 2./3.

    # store distance to closest point in each class in columns
    class_dist = np.zeros((num_points, 4))
    class_dist[:,0] = dist_thresh
    class_dist[:,1] = dist1
    class_dist[:,2] = dist2
    class_dist[:,3] = dist3
    
    # the class label is now the column with the lowest number
    labels = np.argmin(class_dist, axis=1)
    
    return labels

############ function to extract a single patch from the image ########
def get_patch_at_point(I, p, size):
    # YOUR CODE HER
    left = int(p[0]) - size[0]
    down = int(p[1]) - size[0]
    right = int(p[0]) + size[1]
    up = int(p[1]) + size[1]
    P = I[down:up, left:right, :]    
    return P

################## function to extract patches from the image ##########
def extract_patches(I, p1, p2, size):
    points = sample_points_around_pen(I, p1, p2)
     
    # determine the labels of the points
    labels = make_labels_for_points(I, p1, p2, points)
    
    xs = []
    for p in points:
        P = get_patch_at_point(I, p, size)
        x = patch_to_vec(P,size)
        xs.append(x)
    X = np.array(xs)

    return X, labels, points

######## count the number of labels per classs #############
def count_classes(labels):
    counts = np.array([0,0,0,0])
    for i in labels:
        counts[i] += 1
    return counts

def extract_multiple_images(Is, img_list,annots,size):
    Xs = []
    ys = []
    points = []
    imgids = []

    for idx in img_list:
        I = Is[idx]
        #print(idx)
        #plt.figure()
        #plt.imshow(I)
        I_X, I_y, I_points = extract_patches(I, annots[idx,:2], annots[idx,2:], size)

        classcounts = count_classes(I_y)
        #print(f'image {idx}, class count = {classcounts}')

        Xs.append(I_X)
        ys.append(I_y)
        points.append(I_points)
        imgids.append(np.ones(len(I_y),dtype=int)*idx)

    Xs = np.vstack(Xs)
    ys = np.hstack(ys)
    points = np.vstack(points)
    imgids = np.hstack(imgids)
    
    return Xs, ys, points, imgids

CLASS_NAMES = [
    'background', # class 0
    'tip',        # class 1
    'end',        # class 2
    'middle'      # class 3
]



############## a function for plotting the samples on the image
def plot_samples(Ps, labels,FEAT_SIZE,nsamples):
    uls = np.unique(labels)
    nclasses = len(uls)
    
    plt.figure(figsize=(10,4))
    
    for lidx, label in enumerate(uls):
        idxs = np.where(labels == label)[0]
        idxs = np.random.choice(idxs, nsamples, replace=False)
        
        for j, idx in enumerate(idxs):
            P = Ps[idx,:]
            P = P.reshape(FEAT_SIZE)
            
            plt.subplot(nclasses, nsamples, lidx*nsamples+j+1)
            plt.imshow(P, clim=(0,1))
            plt.axis('off')
            plt.title('label: %d' % label)