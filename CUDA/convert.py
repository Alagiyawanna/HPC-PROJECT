from PIL import Image

# Open the image and convert it to grayscale
img = Image.open('image.png').convert('L')

# Save the image in PGM format
img.save('input_custom.pgm')
print("Image successfully converted to input_custom.pgm!")