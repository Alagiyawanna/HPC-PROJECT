import numpy as np
import matplotlib.pyplot as plt

def read_pgm(filename):
    with open(filename, 'rb') as f:
        magic = f.readline().strip()
        assert magic == b'P5', f"Not a P5 PGM file: {magic}"
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()
        width, height = map(int, line.split())
        maxval = int(f.readline().strip())
        data = np.frombuffer(f.read(width * height), dtype=np.uint8)
        return data.reshape((height, width)), maxval

img, _ = read_pgm('output_custom.pgm') 

# Display the image 
plt.figure(figsize=(8, 8))  
plt.imshow(img, cmap='gray')
plt.title('CUDA Convolution Output')
plt.axis('off')
plt.show()