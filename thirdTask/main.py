import numpy as np
from scipy.linalg import svd
from PIL import Image
import struct
import pickle
import argparse

class StandardLibraryAlgorithm:
    def compress(self, image_data, ratio):
        #compressed_data = []
        compressed_data = {}
        for channel in range(3):
            channel_data = image_data[: ,:, channel]
            U, s, VT = np.linalg.svd(channel_data,full_matrices=False)
            k = int(min(image_data.shape[0], image_data.shape[1]) / ratio)
            #compressed_data.append((U[:, :k], np.diag(s[:k]), VT[:k, :]))
            compressed_data[f'U{channel}'] = U[:, :k]
            compressed_data[f's{channel}'] = s[:k]
            compressed_data[f'V{channel}'] = VT[:k, :]
    
        return compressed_data
    
    def decompress(self, compressed_data, original_shape):
        reconstucted_data = np.zeros(original_shape, dtype=np.uint8)
        for channel in range(3):
            U, s, VT = compressed_data[channel]
            reconstucted_data[:, :, channel] = (U @ np.diag(s) @ VT).clip(0, 255).astype(np.uint8)
        return reconstucted_data
    

class PowerMethodAlgorithm:
    def compress(self, image_data, ratio):
        # Реализация степенного метода для сжатия
        pass

    def decompress(self, compressed_data, original_shape):
        # Реализация степенного метода для восстановления
        pass

    
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
    with open(output_file, 'wb') as file:
        #np.savez(file, (compressed_data, image_data.shape))
        np.savez(file, **compressed_data)

def decompress_image(input_file, output_file, algorithm):
    with open(input_file, 'rb') as file:
        compressed_data, original_shape = pickle.load(file)
    reconstructed_data = algorithm.decompress(compressed_data, original_shape)
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
    else:
        algorithm = PowerMethodAlgorithm()

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
    # мой примитивный алоритм - это сжатие без потерь (losless compression)


if __name__ == '__main__':
    main()
