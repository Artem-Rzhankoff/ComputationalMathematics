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
            reconstucted_data[:, :, channel] = (U.T @ np.diag(s) @ VT).clip(0, 255).astype(np.uint8)
        return reconstucted_data

def sqr(a):
    return 0.0 if a == 0.0 else a*a

def sign(x):
    return 1 if x >= 0.0 else -1

class AdvancedAlgorithm:

    @staticmethod
    def cancelation(U, S, e, l, k):
        c, s, p = 0.0, 1.0, l-1

        for i in range(l,k+1):
            f, e[i] = s * e[i], c * e[i]

            if abs(f) <= 1.e-6: break
                
            g = S[i]
            ff, gg = np.fabs(f), np.fabs(g)
            if ff > gg:
                h =  ff * np.sqrt(1.0 + sqr(gg/ff))
            else:
                h = gg * np.sqrt(1.0 + sqr(ff/gg))
            S[i], c, s = h, g/h, -f/h

            Y, Z = U[:,p].copy(), U[:,i].copy()
            U[:,p] =  Y * c + Z * s
            U[:,i]  = -Y * s + Z * c

    def testfsplit(self, U, S, e, k):
        goto = False

        for l in np.arange(k+1)[::-1]:
            if abs(e[l]) <= 1.e-6:
                goto = True
                break
            if abs(S[l-1]) <= 1.e-6:
                break

        if goto: return l

        self.cancelation(U, S, e, l, k)
        return l

    def svd(self, A, K):
        k_iter = 30
        # bidiagonalization
        U = np.asarray(A).copy().astype('float64')
        m, n = U.shape

        S, V, e = np.zeros(n), np.zeros((n, n)), np.zeros(n)
        # Householder's reduction

        g, x, scale = 0.0, 0.0, 0.0

        for i in range(n):
            e[i], l = scale * g, i+1
            if i < m:
                scale = U[i:,i].dot(U[i:,i])

                if scale <= 1.e-6:
                    g = 0.0
                else:
                    U[i:,i] = U[i:, i] / scale
                    s = U[i:,i].dot(U[i:,i])
                    f = U[i,i].copy()
                    g = - sign(f) * np.fabs(np.sqrt(s))
                    h = f * g - s
                    U[i,i] = f - g

                    for j in range(l,n):
                        f = U[i:,i].dot(U[i:,j]) / h
                        U[i:,j] += f * U[i:,i]

                    U[i:,i] *= scale
            else:
                g = 0.0

            S[i] = scale * g

            if (i < m) and (i != n-1):
                scale = U[i,l:].dot(U[i,l:])

                if scale <= 1.e-6:
                    g = 0.0
                else:
                    U[i,l:] /= scale
                    s = U[i,l:].dot(U[i,l:])
                    f = U[i,l].copy()
                    g = - sign(f) * np.fabs(np.sqrt(s))
                    h = f * g - s
                    U[i,l] = f - g
                    e[l:] = U[i,l:] / h

                    for j in range(l,m):
                        s = U[j, l:].dot(U[i,l:])
                        U[j,l:] += s * e[l:]

                    U[i,l:] *= scale
            else:
                g = 0.0

        # accumulation of right hand gtransformations
        g, l = 0.0, 0
        for i in np.arange(n)[::-1]:
            if i < n-1:
                if g != 0.0:
                    V[l:,i] = ( U[i,l:] / U[i,l] ) / g
                    for j in range(l,n):
                        s = U[i,l:].dot(V[l:,j])
                        V[l:,j] += s * V[l:,i]

                V[i,l:] = V[l:,i] = 0.0

            V[i,i], g, l = 1.0, e[i], i

        # accumulation of left hand gtransformations
        for i in np.arange(min([m,n]))[::-1]:
            l = i+1
            g = S[i]
            U[i,l:] = 0.0

            if g != 0.0:
                g = 1.0 / g
                for j in range(l,n):
                    f = (U[l:,i].dot(U[l:,j]) / U[i,i]) * g
                    U[i:,j] += f * U[i:,i]

                U[i:,i] *= g

            else:
                U[i:,i] = 0.0

            U[i,i] += 1.0
        
        # Diagonalization of the bidiagonal form

        for k in np.arange(U.shape[1])[::-1]:
            for t in range(k_iter):
                l = self.testfsplit(U, S, e, k)

                if l == k:
                    if S[k] < 0.0:
                        S[k] *= -1
                        V[:,k] *= -1
                    break

                x, y, z = S[l], S[k-1], S[k]
                g, h = e[k-1], e[k]

                f = ((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y)
                aa, bb = np.fabs(f), np.fabs(1.0)
                if aa > bb:
                    g =  aa * np.sqrt(1.0 + sqr(bb/aa))
                else:
                    g = bb * np.sqrt(1.0 + sqr(aa/bb))
                f = ((x-z)*(x+z)+h*((y/(f+(sign(f) * np.fabs(g))))-h))/x

                c = s = 1.0

                for i in range(l+1,k+1):
                    g, y = e[i], S[i]
                    h, g = s*g, c*g

                    aa, bb = np.fabs(f), np.fabs(h)
                    if aa > bb:
                        z = aa * np.sqrt(1.0 + sqr(bb/aa))
                    else:
                        z = bb * np.sqrt(1.0 + sqr(aa/bb))
                    e[i-1] = z

                    c, s = f/z, h/z
                    f, g = x*c+g*s, g*c-x*s
                    h, y = y*s, y*c

                    X, Z = V[:,i-1].copy(), V[:,i].copy()
                    V[:,i-1] = c * X + s * Z
                    V[:,i] = c * Z - s * X

                    aa, bb = np.fabs(f), np.fabs(h)
                    if aa > bb:
                        z = aa * np.sqrt(1.0 + sqr(bb/aa))
                    else:
                        z = bb * np.sqrt(1.0 + sqr(aa/bb))
                    S[i-1] = z

                    if z >= 1.e-6:
                        c, s = f/z, h/z

                    f, x = c*g+s*y, c*y-s*g

                    Y, Z = U[:,i-1].copy(), U[:,i].copy()
                    U[:,i-1] = c * Y + s * Z
                    U[:,i  ] = c * Z - s * Y

                e[l], e[k], S[k] = 0.0, f, x

        idx = np.argsort(-S)

        U = U[:, idx]
        S = S[idx]
        V = np.transpose(V[:, idx])

        U_k = U[:, :K]
        S_k = S[:K]
        V_k = V[:K, :]

        return U_k, S_k, V_k
       


    def compress(self, image_data, ratio):
        compressed_data = {}
        k = evaluate_k(image_data.shape[0], image_data.shape[1], ratio)

        for channel in range(3):
            channel_data = image_data[: ,:, channel]
            U, S, VT = self.svd(channel_data, k)
            
            compressed_data[f'U{channel}'] = U
            compressed_data[f's{channel}'] = S
            compressed_data[f'V{channel}'] = VT

        return compressed_data

        

    def decompress(self, compressed_data):
        original_shape = compressed_data['original_shape']
        reconstucted_data = np.zeros(original_shape, dtype=np.uint8)
        for channel in range(3):
            U, s, VT = compressed_data[f'U{channel}'], compressed_data[f's{channel}'], compressed_data[f'V{channel}']
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
