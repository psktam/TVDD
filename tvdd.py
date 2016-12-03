"""
This module contains methods for executing the TVDD filtering algorithm on images.

We're going to try "unraveling" the image into two 1-D vectors, then taking the average of the filtered outputs to
see what it looks like.
"""
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spl


def tvdd_mm(signal, lam, order=1, abs_tol=1e-6, rel_tol=1e-3, max_iters=100):
    """
    Applies 1-D TVDD filtering to the input signal. The algorithm for this function is built around the following cost
    function:

    Args:
        signal (1-D numpy array): a 1-dimensional numpy array, representing the signal you want to clean up
        lam (float): the regularization parameter
        order (int): the order of the filter. This determines how the L1 norm (the regularized term) of the cost
            function is computed
        abs_tol (float): the stop condition for the absolute error. When cost_prev - cost_next dips below this value,
            iteration stops
        rel_tol (float): the stop condition for the relative error. When (cost_prev - cost_next) / cost_prev dips below
            this value, iteration stops.
        max_iters (int): the maximum number of times to iterate. Default is 100
    """
    # First, construct the differentiation operator, based on the filter order provided
    num_points = len(signal)
    diff = sps.eye(num_points)

    for idx in range(order):
        diff_shape = (num_points - (idx +  1), num_points - idx)
        diff = sps.diags([-1, 1], [0, 1], diff_shape).dot(diff)

    # Next, construct the quantities that will remain constant throughout iteration.
    diff_product = diff.dot(diff.T)
    sig_diff = diff.dot(signal)
    next_iteration = curr_iteration = signal  # Initialize

    for _ in range(max_iters):
        iter_diff = diff.dot(curr_iteration)
        inv_mat = 2.0 / lam * sps.diags(np.abs(iter_diff)) + diff_product

        next_iteration = signal - diff.T.dot(spl.cgs(inv_mat, sig_diff[:, np.newaxis], maxiter=100)[0])
        magnitude_of_change = np.linalg.norm(curr_iteration - next_iteration)
        if magnitude_of_change <= abs_tol or magnitude_of_change / np.linalg.norm(curr_iteration) <= rel_tol:
            break
    return next_iteration


def tvdd_sb_itv(image, mu, abs_tol=1e-6, rel_tol=1e-3, max_iters=100):
    """
    Applies 2-D TVDD filtering to the input signal via the split-bregman algorithm. This code was derived by
    back-porting the script written by Benjamin Tremoulheac.
    Args:
        image (2-D numpy array): a 2-D square array representing the image that you want to filter.
        mu (float): the regularization term
        abs_tol (float): the stop condition for the absolute deviation. This is an iterative solver. When the
            term |u_next - u_prev| drops below this value, the iteration stops, where u is the smoothed signal to be
            returned
        rel_tol (float): the stop condition for the relative deviation
        max_iters (int): the maximum number of iterations to do while iterating.
    :return:
    """
    if len(set(image.shape)) > 1:
        raise ValueError("The image array must be square. Got a {} x {} array instead".format(*image.shape))

    image_flat = image.T.ravel()[:, np.newaxis]  # Flatten the image into a column vector
    num_pixels = len(image_flat)
    image_dim = image.shape[0]

    # Now, construct the differentiation operators. Unfortunately, I can't reverse-engineer what the code is, and I
    # haven't had time to read the paper that covers split-Bregman iteration, but I guess you probably already know this
    # This code is ported from the matlab script written by Benjamin Tremoulheac.
    main_diag = [0.0] + [1.0] * (image_dim - 1)
    diags = sps.diags([-1.0, main_diag], [-1, 0], (image_dim, image_dim))

    # Ugh, I really hate these 1-letter variable names
    B = sps.vstack((sps.kron(sps.eye(image_dim), diags), sps.kron(diags, sps.eye(image_dim))))
    Bt = B.T
    BtB = Bt.dot(B)

    b = np.zeros((2 * num_pixels, 1))
    d = b
    solution = image_flat
    err = k = 1.0
    lam = 1.0

    for _ in range(max_iters):
        current_denoised_image = solution
        solution = spl.cgs(sps.eye(num_pixels) + BtB, image_flat - lam * Bt.dot(b - d), maxiter=100)[0][:, np.newaxis]
        Bub = B.dot(solution[:, 0])[:, np.newaxis] + b
        s = np.sqrt((Bub[:num_pixels] ** 2.0) + (Bub[num_pixels:] ** 2.0))
        non_negative_array = np.max([np.zeros(len(s)), s[:, 0] - mu / lam], axis=0)[:, np.newaxis]
        d = np.vstack((non_negative_array * Bub[:num_pixels] / s,
                       non_negative_array * Bub[num_pixels:] / s))
        b = Bub - d
        abs_err = np.linalg.norm(current_denoised_image[:, 0] - solution[:, 0])
        err = np.linalg.norm(current_denoised_image[:, 0] - solution[:, 0]) / np.linalg.norm(solution[:, 0])
        if err <= rel_tol or abs_err <= abs_tol:
            break

    # Make sure to appropriately reshape the solution before returning it.
    reshaped = solution.reshape((image_dim, image_dim)).T
    return reshaped


def striped_tvdd_2d(image, mu, order=1, abs_tol=1e-6, rel_tol=1e-3, max_iters=100):
    """
    Attempts to do TVDD of an image by recasting it as a series of 1-D arrays.

    First, we recast the image as a 1-D array by "zig-zagging" up and down the columns.

        x   x---x   x---x
        |   |   |   |   |
        x   x   x   x   x
        |   |   |   |   |
        x   x   x   x   x
        |   |   |   |   |
        x   x   x   x   x
        |   |   |   |   |
        x---x   x---x   x

    Then we call tvdd_mm on this signal. After TVDDMM is complete, we recast the array back into the original image.
    Then, we do this again with the reconstructed image, this time zig-zagging left and right across the rows:

        x---x---x---x---x
                        |
        x---x---x---x---x
        |
        x---x---x---x---x
                        |
        x---x---x---x---x
        |
        x---x---x---x---x

    We call tvdd_mm on this reconstructed image, reconstruct the array back into its original 2D form, and return it as
    the final, smoothed product.
    """
    # First, resample the matrix, zig-zagging up and down columns
    num_rows, num_cols = image.shape
    flattened_image = []
    for col_idx in range(num_cols):
        direction = -1 if col_idx % 2 else 1
        flattened_image.extend(image[:, col_idx][::direction])
    flattened_image = np.array(flattened_image)

    smoothed_image = tvdd_mm(flattened_image, mu, order, abs_tol, rel_tol, max_iters)

    # Reconstruct, and stripe the other way now
    columns = []
    for col_idx in range(num_cols):
        direction = -1 if col_idx % 2 else 1
        columns.append(smoothed_image[num_rows * col_idx:num_rows * (col_idx + 1)][::direction])
    reconstructed = np.array(columns).T

    # Now, re-stripe the other way
    flattened_image = []
    for row_idx in range(num_rows):
        direction = -1 if row_idx % 2 else 1
        flattened_image.extend(reconstructed[row_idx, :][::direction])
    flattened_image = np.array(flattened_image)

    smoothed_image = tvdd_mm(flattened_image, mu, order, abs_tol, rel_tol, max_iters)
    rows = []
    for row_idx in range(num_rows):
        direction = -1 if row_idx % 2 else 1
        rows.append(smoothed_image[num_cols * row_idx:num_cols * (row_idx + 1)][::direction])
    reconstructed = np.array(rows)

    return reconstructed
