from PIL import Image

# # Load your hand image (PNG with transparency preferred)
# input_path = "/Users/mrugank/Documents/Emojis/handImg.png"  # Replace with your image path
# original = Image.open(input_path).convert("RGBA")

# # Parameters
# frames = []
# num_frames = 24  # Adjust for smoothness
# angle_step = 360 // num_frames

# # Generate rotated frames
# for i in range(num_frames):
#     rotated = original.rotate(-i * angle_step, resample=Image.BICUBIC, expand=True)
#     # Optional: crop or resize to maintain uniform size
#     frames.append(rotated)

# # Save as GIF
# frames[0].save(
#     "/Users/mrugank/Documents/Emojis/rotating_hand.gif",
#     save_all=True,
#     append_images=frames[1:],
#     duration=100,  # milliseconds between frames
#     loop=0,        # infinite loop
#     disposal=2
# )

# print("GIF saved as rotating_hand.gif")


# Load original image
original = Image.open("/Users/mrugank/Documents/Emojis/handImg.png").convert("RGBA")

# Define constant canvas size (square canvas helps)
canvas_size = max(original.size) * 2  # Generous padding to fit rotated image
center = (canvas_size // 2, canvas_size // 2)

# Output frames list
frames = []

for angle in range(0, 360, 15):
    # Rotate without expand (keeps original size)
    rotated = original.rotate(angle, resample=Image.BICUBIC, expand=True)

    # Create a black square canvas
    bg = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 255))

    # Center the rotated image on the canvas
    rot_w, rot_h = rotated.size
    top_left = (center[0] - rot_w // 2, center[1] - rot_h // 2)
    bg.paste(rotated, top_left, rotated)

    frames.append(bg)

# Save as GIF
frames[0].save(
    "/Users/mrugank/Documents/Emojis/rotating_hand.gif",
    save_all=True,
    append_images=frames[1:],
    duration=100,
    loop=0
)
