from PIL import Image

# Open the image and convert it to grayscale
img = Image.open('image.jpeg').convert('L')

# Save the image in PGM format
img.save('input.pgm')
print("Image successfully converted to input.pgm!")