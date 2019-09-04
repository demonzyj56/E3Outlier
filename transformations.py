import abc
import itertools
import numpy as np

from keras.preprocessing.image import apply_affine_transform
from scipy.ndimage.interpolation import rotate as rt
import keras.backend as K


class AffineTransformation(object):
    def __init__(self, flip, tx, ty, k_90_rotate):
        self.flip = flip
        self.tx = tx
        self.ty = ty
        self.k_90_rotate = k_90_rotate

    def __call__(self, x):
        res_x = x
        if self.flip:
            res_x = np.fliplr(res_x)
        if self.tx != 0 or self.ty != 0:
            res_x = apply_affine_transform(res_x, tx=self.tx, ty=self.ty, channel_axis=2, fill_mode='reflect')
        if self.k_90_rotate != 0:
            res_x = np.rot90(res_x, self.k_90_rotate)

        return res_x


class AbstractTransformer(abc.ABC):
    def __init__(self):
        self._transformation_list = None
        self._create_transformation_list()

    @property
    def n_transforms(self):
        return len(self._transformation_list)

    @abc.abstractmethod
    def _create_transformation_list(self):
        return

    def transform_batch(self, x_batch, t_inds):
        assert len(x_batch) == len(t_inds)

        transformed_batch = x_batch.copy()
        for i, t_ind in enumerate(t_inds):
            transformed_batch[i] = self._transformation_list[t_ind](transformed_batch[i])
        return transformed_batch


class RA(AbstractTransformer):
    """Regular Affine Transformation Set."""
    def __init__(self, translation_x=8, translation_y=8):
        self.max_tx = translation_x
        self.max_ty = translation_y
        super().__init__()

    def _create_transformation_list(self):
        transformation_list = []
        for is_flip, tx, ty, k_rotate in itertools.product((False, True),
                                                           (0, -self.max_tx, self.max_tx),
                                                           (0, -self.max_ty, self.max_ty),
                                                           range(4)):
            transformation = AffineTransformation(is_flip, tx, ty, k_rotate)
            transformation_list.append(transformation)

        self._transformation_list = transformation_list


def gen_mask(img, radius=None):
    '''

    :param img: rotated img
    :param radius:
    :return:
    '''
    assert img.shape[0] == img.shape[1]
    length = img.shape[0]
    if radius is None:
        radius = (length-1) / 2
    img_x = np.repeat(np.array(range(length))[:, np.newaxis], length, axis=1)
    center = (length-1) / 2
    dis_x = img_x - center
    dis = dis_x ** 2
    dis = dis + dis.transpose()
    thr = radius ** 2
    mask_rotated = dis <= thr
    img_masked = np.zeros_like(img)
    if len(img.shape) > 2:
        for i in range(img.shape[2]):
            img_masked[:, :, i] = img[:, :, i] * mask_rotated
    else:
        img_masked = img * mask_rotated
    img_masked = img_masked[np.int(center-radius):np.int(center+radius)+1, np.int(center-radius):np.int(center+radius)+1]
    mask_original = mask_rotated[np.int(center-radius):np.int(center+radius)+1, np.int(center-radius):np.int(center+radius)+1]
    if len(img.shape) > 2:
        mask_original = np.repeat(mask_original[:, :, np.newaxis], img.shape[2], axis=2)
    return img_masked, mask_original

def gen_rotate_img(img, degree=0):
    img2 = rt(img, degree)
    r = (img.shape[0] - 1) / 2
    img2, mask = gen_mask(img2, radius=r)
    img3 = img.copy()
    img3[mask] = img2[mask]
    return img3

def gen_simple_rt_img(img, degree=0):
    img2 = rt(img, degree)
    radius = (img.shape[0] - 1) / 2
    center = (img2.shape[0] - 1) / 2
    return img2[np.int(center-radius):np.int(center+radius)+1, np.int(center-radius):np.int(center+radius)+1]

class AnyDegreeTransformation(object):
    def __init__(self, flip, k_rotate, degree_per_rotate):
        self.flip = flip
        self.degree_per_rotate = degree_per_rotate
        self.k_rotate = k_rotate

    def __call__(self, x):
        res_x = x
        if self.flip:
            res_x = np.fliplr(res_x)
        if self.k_rotate != 0:
            if self.k_rotate * self.degree_per_rotate % 90 ==0:
                res_x = np.rot90(res_x, self.k_rotate * self.degree_per_rotate // 90)
            else:
                # res_x = gen_rotate_img(res_x, self.k_rotate * self.degree_per_rotate)
                res_x = gen_simple_rt_img(res_x, self.k_rotate * self.degree_per_rotate)

        return res_x


class RA_IA(AbstractTransformer):
    """Regular affine transformation set + irregular affine transformation set."""
    def __init__(self, translation_x=8, translation_y=8, num_rotation=8):
        self.max_tx = translation_x
        self.max_ty = translation_y
        self.num_rotation = num_rotation
        super().__init__()

    def _create_transformation_list(self):
        transformation_list = []
        degree_per_rotate = 360 / self.num_rotation
        for is_flip, tx, ty, k_rotate in itertools.product((False, True),
                                                           (0, -self.max_tx, self.max_tx),
                                                           (0, -self.max_ty, self.max_ty),
                                                           range(4)):
            transformation = AffineTransformation(is_flip, tx, ty, k_rotate)
            transformation_list.append(transformation)

        # transformations with non 90 degree rotations
        for is_flip, k_rotate in itertools.product((False, True), range(self.num_rotation)):
            if degree_per_rotate * k_rotate % 90 != 0:
                transformation = AnyDegreeTransformation(is_flip, k_rotate, degree_per_rotate)
                transformation_list.append(transformation)

        self._transformation_list = transformation_list

class PatchShuffle(object):
    """Transformation class."""

    def __init__(self, permutation):
        """
        Parameters
        ----------
        permutation: tuple (x, y) of n x n matrix.
            Specify permuted x and y coordinates.
        """
        self.permx, self.permy = self._perm_to_grid(permutation)

    @staticmethod
    def _perm_to_grid(permutation):
        patch_per_row = int(round(np.sqrt(len(permutation))))
        assert patch_per_row ** 2 == len(permutation)
        grid = np.asarray([(x, y) for x in range(patch_per_row) for y in range(patch_per_row)])
        grid = grid[permutation, :]
        y, x = zip(*grid)
        return np.reshape(x, [patch_per_row, -1]), np.reshape(y, [patch_per_row, -1])

    def __call__(self, x):
        # x is always 3-dimensional.
        if K.image_data_format() == 'channels_first':
            H, W = x.shape[1:]
        else:
            H, W = x.shape[:2]
        # The size of x is not known until here, so raise exception if the size
        # of x is not divisive of patch_per_row.
        if (H % self.permx.shape[0] != 0) or (W % self.permx.shape[1] != 0):
            raise ValueError('Invalid permutations')
        steph, stepw = H // self.permx.shape[0], W // self.permx.shape[1]
        out = np.zeros_like(x)
        for i in range(self.permx.shape[0]):
            for j in range(self.permx.shape[1]):
                outx, outy = self.permx[i, j], self.permy[i, j]
                if K.image_data_format() == 'channels_first':
                    out[:, i*steph:(i+1)*steph, j*stepw:(j+1)*stepw] = \
                        x[:, outy*steph:(outy+1)*steph, outx*stepw:(outx+1)*stepw]
                else:
                    out[i*steph:(i+1)*steph, j*stepw:(j+1)*stepw, :] = \
                        x[outy*steph:(outy+1)*steph, outx*stepw:(outx+1)*stepw, :]
        return out

class RA_IA_PR(AbstractTransformer):
    """Regular affine transformation set + irregular affine transformation set + patch re-arranging."""
    def __init__(self, translation_x=8, translation_y=8, num_rotation=8, n_perm=24-1, patch_per_row=2):
        self.max_tx = translation_x
        self.max_ty = translation_y
        self.num_rotation = num_rotation
        self.n_perm = n_perm
        self.patch_per_row = patch_per_row
        self._permutation_list = self._create_permutation_list()
        super().__init__()

    def _create_transformation_list(self):
        transformation_list = []
        degree_per_rotate = 360 / self.num_rotation
        for is_flip, tx, ty, k_rotate in itertools.product((False, True),
                                                           (0, -self.max_tx, self.max_tx),
                                                           (0, -self.max_ty, self.max_ty),
                                                           range(4)):
            transformation = AffineTransformation(is_flip, tx, ty, k_rotate)
            transformation_list.append(transformation)

        # transformations with non 90 degree rotations
        for is_flip, k_rotate in itertools.product((False, True), range(self.num_rotation)):
            if degree_per_rotate * k_rotate % 90 != 0:
                transformation = AnyDegreeTransformation(is_flip, k_rotate, degree_per_rotate)
                transformation_list.append(transformation)

        transformation_list = transformation_list + [
            PatchShuffle(p) for p in self._permutation_list
        ]

        self._transformation_list = transformation_list

    def _create_permutation_list(self):
        if self.patch_per_row <= 2:
            all_perm = list(itertools.permutations(range(self.patch_per_row**2)))
            # inds = np.random.choice(len(all_perm), self.n_perm, replace=False)
            inds = np.array([i for i in range(1, len(all_perm))])
            perms = [list(all_perm[i]) for i in inds]
        else:
        #     perms = []
        #     while len(perms) < self.n_perm:
        #         new = np.random.permutation(self.patch_per_row**2).tolist()
        #         if new not in perms:
        #             perms.append(new)
        # idperm = list(range(self.patch_per_row**2))
        # if idperm not in perms:
        #     perms[0] = idperm
            raise NotImplementedError
        return perms

