from PIL import Image
import pillow_heif


# Register HEIF support with Pillow
pillow_heif.register_heif_opener()

# Open the .HEIC image
image = Image.open("/Users/raghav/Downloads/YES critical foaming/IMG_2232.heic")

# Convert to JPEG format in memory
jpeg_image = image.convert("RGB")

# Display the image
jpeg_image.show()
