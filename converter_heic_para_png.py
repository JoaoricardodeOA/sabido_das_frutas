from PIL import Image
import pillow_heif
import os


pillow_heif.register_heif_opener()


def convert_heic_to_png(heic_file, output_file):
    try:
        img = Image.open(heic_file)

        img.save(output_file, "PNG")
        print(f"CONVERTIDO {heic_file} PARA {output_file}")
    except Exception as e:
        print(f"Error converting {heic_file}: {e}")


def convert_all_heic_in_directory(input_dir):

    for root, dirs, files in os.walk(input_dir):

        for filename in files:
            if filename.lower().endswith(".heic"):
                heic_file = os.path.join(root, filename)
                
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(input_dir, relative_path)
                
                output_file = os.path.join(output_subdir, os.path.splitext(filename)[0] + ".png")
                
                
                convert_heic_to_png(heic_file, output_file)
                os.remove(heic_file)


input_directory = "TESTE"
convert_all_heic_in_directory(input_directory)