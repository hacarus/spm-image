import sys
sys.path.append('../../')

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import resize
from skimage.measure import compare_psnr
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from sklearn.decomposition.dict_learning import sparse_encode
from spmimage.decomposition import KSVD


def make_deficit_image(img: np.ndarray, ratio:float):
	mask = (np.random.rand(img.shape[0], img.shape[1]) > ratio)
	return img * mask, mask

def make_DCT_dict(patch_size: int, dict_size: int):
	A_1D = np.zeros((dict_size, patch_size))
	for k in np.arange(dict_size):
	    for i in np.arange(patch_size):
	        A_1D[k, i] = np.cos(i * k * np.pi / float(dict_size))
	    if k != 0:
	        A_1D[k, :] -= A_1D[k, :].mean()
	return np.kron(A_1D, A_1D)

def sparse_encode_with_mask(img, H, k0, mask, patch_size=8):
    patches = extract_patches_2d(img, (patch_size, patch_size))
    mask_patches = extract_patches_2d(mask, (patch_size, patch_size))

    Y = []
    mask = []
    for p, mp in zip(patches, mask_patches):
    	Y.append(p.flatten())
    	mask.append(mp.flatten())
    Y = np.array(Y)
    mask = np.array(mask)

    W = np.zeros((Y.shape[0], H.shape[0]))
    for idx in range(Y.shape[0]):
        W[idx, :] = sparse_encode(
                    Y[idx, :][mask[idx,:]==1].reshape(1,-1),
                    H[:, mask[idx,:]==1],
                    algorithm='omp', n_nonzero_coefs=k0
                    )
    return W

def ksvd_inpainting(img, n_components, k0, patch_size=8):
    patches = extract_patches_2d(img, (patch_size, patch_size))
    X = []
    for p in patches:
    	X.append(p.flatten())
    X = np.array(X)

    model = KSVD(n_components=n_components, k0=k0, max_iter=100, missing_value=0)
    model.fit(X)
    return model.components_

def reconstruct_image(img, W, H, patch_size=8):
    recon_patches = (np.dot(W, H)).reshape((-1, patch_size, patch_size))
    recon_img = reconstruct_from_patches_2d(recon_patches, img.shape)
    return recon_img

def main():
	img = io.imread("LENNA.bmp")
	img = resize(img, (256,256))
	deficit_img, mask = make_deficit_image(img, 0.7)

	DCT_dict = make_DCT_dict(8, 16)
	code = sparse_encode_with_mask(deficit_img, DCT_dict, 10, mask)
	dct_reconstruct_img = reconstruct_image(deficit_img, code, DCT_dict)

	KSVD_dict = ksvd_inpainting(deficit_img, 30, 10)
	code = sparse_encode_with_mask(deficit_img, KSVD_dict, 10, mask)
	ksvd_reconstruct_img = reconstruct_image(deficit_img, code, KSVD_dict)

	print("psnr of dct reconstruction", compare_psnr(img, dct_reconstruct_img))
	print("psnr of ksvd reconstruction", compare_psnr(img, ksvd_reconstruct_img))

	plt.subplot(2,2,1)
	plt.imshow(img, cmap='gray')
	plt.title("original")
	plt.axis('off')
	plt.subplot(2,2,2)
	plt.imshow(deficit_img, cmap='gray')
	plt.title("70% missing")
	plt.axis('off')
	plt.subplot(2,2,3)
	plt.imshow(dct_reconstruct_img, cmap='gray')
	plt.title("DCT sparse estimation")
	plt.axis('off')
	plt.subplot(2,2,4)
	plt.imshow(ksvd_reconstruct_img, cmap='gray')
	plt.title("KSVD sparse estimation")
	plt.axis('off')
	plt.show()

if __name__ == '__main__':
	main()
