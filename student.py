import time
from math import floor
import numpy as np
import cv2
from scipy.sparse import csr_matrix
import util_sweep


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo (or mean albedo for RGB input images) for a pixel is less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- N x 3 array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights. All N images have the same dimension.
                  The images can either be all RGB (height x width x 3), or all grayscale (height x width x 1).

    Output:
        albedo -- float32 image. When the input 'images' are RGB, it should be of dimension height x width x 3,
                  while in the case of grayscale 'images', the dimension should be height x width x 1.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """
    
    # Convert input data to NumPy arrays
    images_np = np.asarray(images)
    lights_np = np.asarray(lights)

    num_images, height, width, num_channels = images_np.shape

    # Flatten images: (height * width, num_images * num_channels) for dot product compatibility
    images_flattened = images_np.transpose(1, 2, 3, 0).reshape(-1, num_images)

    # Compute pseudo-inverse of lights for solving linear system
    lights_pinv = np.linalg.pinv(lights_np)

    # Solution for each pixel: (height * width, 3)
    solution = np.dot(images_flattened, lights_pinv.T)

    # Reshape solution to separate albedo and normals
    albedo = np.linalg.norm(solution.reshape(height, width, num_channels, 3), axis=3)
    normals = solution.reshape(height, width, num_channels, 3) / np.expand_dims(albedo, axis=3)

    # Apply threshold to albedo and set corresponding normals to zero
    albedo[albedo < 1e-7] = 0
    normals[albedo < 1e-7] = 0

    # Average normals across channels if necessary and normalize
    normals_avg = np.mean(normals, axis=2)
    norm = np.linalg.norm(normals_avg, axis=2, keepdims=True)
    normals_avg = np.divide(normals_avg, norm, out=np.zeros_like(normals_avg), where=norm!=0)

    return albedo.astype(np.float32), normals_avg.astype(np.float32)


def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    
    # Combine the intrinsic and extrinsic matrices to form the complete projection matrix.
    projection_matrix = np.dot(K, Rt)

    # Convert 3D points to homogeneous coordinates by appending a column of ones.
    num_points = points.shape[0] * points.shape[1]
    homogeneous_component = np.ones((num_points, 1)).reshape(points.shape[0], points.shape[1], 1)
    homogeneous_points = np.concatenate((points, homogeneous_component), axis=2)
    
    # Apply the projection matrix to the points in homogeneous coordinates.
    projected_points_homogeneous = np.dot(homogeneous_points, projection_matrix.T)

    # Convert from homogeneous coordinates back to 2D by normalizing with the z-component.
    z_component = projected_points_homogeneous[:, :, 2][:, :, np.newaxis]
    projected_points_2D = projected_points_homogeneous / z_component

    # Return only the x and y components, discarding the homogeneous component.
    return projected_points_2D[:, :, :2]


def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc_impl' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc_impl' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x221, x112, x122, x212, x222 ]

    see order argument in np.reshape

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region; assumed to be odd
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    
    # Retrieve the dimensions of the input image
    height, width, num_channels = image.shape

    # Initialize the output array for storing normalized patch vectors
    normalized = np.zeros((height, width, num_channels * ncc_size**2), dtype=np.float32)

    # Calculate the half size of the NCC patch to handle boundaries
    half_ncc = ncc_size // 2

    # Iterate over each pixel in the image, avoiding edges that cannot fully accommodate the NCC patch
    for y in range(half_ncc, height - half_ncc):
        for x in range(half_ncc, width - half_ncc):
            # Initialize a list to hold the flattened and normalized patch vectors for each channel
            patch_vectors = []

            # Process each channel of the image separately
            for channel in range(num_channels):
                # Extract the current patch for this channel
                patch = image[y - half_ncc:y + half_ncc + 1, x - half_ncc:x + half_ncc + 1, channel]
                
                # Subtract the mean of the patch
                patch_mean_subtracted = patch - np.mean(patch)
                
                # Flatten the mean-subtracted patch
                patch_flattened = patch_mean_subtracted.flatten()
                
                # Add the flattened patch to the list of patch vectors
                patch_vectors.append(patch_flattened)

            # Concatenate the flattened patches from all channels into a single vector
            patch_vector = np.concatenate(patch_vectors)
            
            # Compute the L2 norm of the concatenated patch vector
            norm = np.linalg.norm(patch_vector)
            
            # Normalize the patch vector by its norm if the norm is above a threshold, else set it to zero
            if norm > 1e-6:
                patch_vector /= norm
            else:
                patch_vector = np.zeros_like(patch_vector)

            # Store the normalized patch vector in the output array
            normalized[y, x, :] = patch_vector

    return normalized


def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc_impl.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    
    # Ensure the input images are numpy arrays for efficient computation
    image1 = np.asarray(image1)
    image2 = np.asarray(image2)

    # Compute the dot product between corresponding normalized vectors in the two images
    # Resulting NCC map will have the same height and width as the input images
    ncc_map = np.sum(image1 * image2, axis=2)

    return ncc_map