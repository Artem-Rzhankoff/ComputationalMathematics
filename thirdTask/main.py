import numpy as np
from scipy.linalg import svd
from PIL import Image
import argparse

def evaluate_k(m, n, N):
    original_size = m * n * 3
    for k in range(min(m, n) + 1, 1, -1):
        compressed_size = 3 * (m * k + k + k * n) * 8
        if original_size / compressed_size >= N:
            return k
    return min(m, n)


class StandardLibraryAlgorithm:
    def compress(self, image_data, ratio):
        compressed_data = {}
        k = evaluate_k(image_data.shape[0], image_data.shape[1], ratio)
        for channel in range(3):
            channel_data = image_data[: ,:, channel]
            U, s, VT = np.linalg.svd(channel_data,full_matrices=False)
            compressed_data[f'U{channel}'] = U[:, :k]
            compressed_data[f's{channel}'] = s[:k]
            compressed_data[f'V{channel}'] = VT[:k, :]
    
        return compressed_data
    
    def decompress(self, compressed_data):
        original_shape = compressed_data['original_shape']
        reconstucted_data = np.zeros(original_shape, dtype=np.uint8)
        for channel in range(3):
            U, s, VT = compressed_data[f'U{channel}'], compressed_data[f's{channel}'], compressed_data[f'V{channel}']
            print(s[:6])
            reconstucted_data[:, :, channel] = (U @ np.diag(s) @ VT).clip(0, 255).astype(np.uint8)
        return reconstucted_data
    

class PowerMethodAlgorithm:
    @staticmethod
    def power_method(A, num_iterations=1000, tolerance=1e-6):
        v = np.random.rand(A.shape[1])
        v = v / np.linalg.norm(v)

        for _ in range(num_iterations):
            Av = A @ v
            v_new = A.T @ Av
            v_new = v_new / np.linalg.norm(v_new)

            v = v_new
        
        sigma = np.linalg.norm(A @ v)
        return sigma, v
    
    def deflate(self, A, sigma, u, v):
        return A - sigma * np.outer(u, v)

    def compress(self, image_data, ratio):
        compressed_data = {}
        k = evaluate_k(image_data.shape[0], image_data.shape[1], ratio)
        for channel in range(3):
            channel_data = image_data[: ,:, channel]
            U, S, V = [], [], []
            for _ in range(k):
                sigma, v = self.power_method(channel_data)
                u = channel_data @ v
                u /= np.linalg.norm(u)

                U.append(u)
                S.append(sigma)
                V.append(v)

                channel_data = self.deflate(channel_data, sigma, u, v)
            
            compressed_data[f'U{channel}'] = np.array(U)
            compressed_data[f's{channel}'] = np.array(S)
            compressed_data[f'V{channel}'] = np.array(V)
    
        return compressed_data

    def decompress(self, compressed_data):
        original_shape = compressed_data['original_shape']
        reconstucted_data = np.zeros(original_shape, dtype=np.uint8)
        for channel in range(3):
            U, s, VT = compressed_data[f'U{channel}'], compressed_data[f's{channel}'], compressed_data[f'V{channel}']
            print((U.T @ np.diag(s) @ VT))
            reconstucted_data[:, :, channel] = (U.T @ np.diag(s) @ VT).clip(0, 255).astype(np.uint8)
        return reconstucted_data
    

class AdvancedAlgorithm:
    @staticmethod
    def generate_ortonormal_vector(A, k):
        m, n = A.shape[0], A.shape[1]
        
        U = np.eye(m)
        V = np.eye(n)

        B = A.T @ A

        for _ in range(k):
            p, q = np.unravel_index(np.argmax(np.abs(B - np.diag(np.diag(B)))), B.shape)

            if abs(B[p, q]) < 1e-10:
                break


            theta = 0.5 * np.arctan(2 * B[p, q] / (B[p, p] - B[q, q]))
            #theta = 0.5 * np.arctan2(2 * B[p, q], B[q, q] - B[p, p])
            c, s = np.cos(theta), np.sin(theta)

            B_pp, B_pq, B_qp, B_qq = B[p, p], B[p, q], B[q, p], B[q, q]
            B[p, p] = c**2 * B_pp - 2 * c * s * B_pq + s**2 * B_qq
            B[q, q] = s**2 * B_pp + 2 * c * s * B_pq + c**2 * B_qq
            B[p, q] = B[q, p] = (c**2 - s**2) * B_pq + c * s * (B_pp - B_qq)
            B[:, p], B[:, q] = c * B[:, p] - s * B[:, q], s * B[:, p] + c * B[:, q]
            B[p, :], B[q, :] = c * B[p, :] - s * B[q, :], s * B[p, :] + c * B[q, :]

            V[:, p], V[:, q] = c * V[:, p] - s * V[:, q], s * V[:, p] + c * V[:, q]

        S = np.sqrt(np.diag(B))

        zero_mask = (S == 0)
        S[zero_mask] = 1

        U = A @ V / S

        S[zero_mask] = 0

        idx = np.argsort(S)[::-1]
        U, S, V = U[:, idx], S[idx], V[:, idx]

        return U, S, V.T


    def compress(self, image_data, ratio):
        compressed_data = {}
        k = evaluate_k(image_data.shape[0], image_data.shape[1], ratio)

        for channel in range(3):
            channel_data = image_data[: ,:, channel]
            U, S, VT = self.generate_ortonormal_vector(channel_data, k)
            
            compressed_data[f'U{channel}'] = U #np.array(U)
            compressed_data[f's{channel}'] = S#np.array(S)
            compressed_data[f'V{channel}'] = VT#np.array(VT)

        return compressed_data

        

    def decompress(self, compressed_data):
        original_shape = compressed_data['original_shape']
        reconstucted_data = np.zeros(original_shape, dtype=np.uint8)
        for channel in range(3):
            U, s, VT = compressed_data[f'U{channel}'], compressed_data[f's{channel}'], compressed_data[f'V{channel}']
            print(s[:6])
            reconstucted_data[:, :, channel] = (U @ np.diag(s) @ VT).clip(0, 255).astype(np.uint8)
        return reconstucted_data

    
def read_bmp(file_path):
    with Image.open(file_path) as image:
        image_data = np.array(image)
    return image_data

def write_bmp(file_path, image_data):
    image = Image.fromarray(image_data.astype(np.uint8))
    image.save(file_path)

def compress_image(input_file, output_file, compression_ratio, algorithm):
    image_data = read_bmp(input_file)
    compressed_data = algorithm.compress(image_data, compression_ratio)
    compressed_data['original_shape'] = image_data.shape
    np.savez(output_file, **compressed_data)

def decompress_image(input_file, output_file, algorithm):
    compressed_data = np.load(input_file + '.npz')

    reconstructed_data = algorithm.decompress(compressed_data)
    write_bmp(output_file, reconstructed_data)


def main():
    parser = argparse.ArgumentParser(description='BMP Image Compression Utility')
    parser.add_argument('-c', '--compress', action='store_true', help='Compress the image')
    parser.add_argument('-u', '--decompress', action='store_true', help='Decompress the image')
    parser.add_argument('-i', '--input', required=True, help='Input file path')
    parser.add_argument('-o', '--output', required=True, help='Output file path')
    parser.add_argument('-r', '--ratio', type=int, default=2, help='Compression ratio (default: 10)')
    parser.add_argument('-a', '--algorithm', choices=['standard', 'primitive', 'advanced'])

    args = parser.parse_args()

    if args.algorithm == 'standard':
        algorithm = StandardLibraryAlgorithm()
    elif args.algorithm == 'primitive':
        algorithm = PowerMethodAlgorithm()
    else:
        algorithm = AdvancedAlgorithm()

    if args.compress and args.decompress:
        print('Error: cannot use both -c and -u flags simultaneously.')
        return
    
    if not args.compress and not args.decompress:
        print('Error: Either -c or -u flag must be specified.')
        return  
    
    if args.compress:
        compress_image(args.input, args.output, args.ratio, algorithm)
        print('Image compresed successfully.')
    else:
        decompress_image(args.input, args.output, algorithm)


if __name__ == '__main__':
    main()
