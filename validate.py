# import pydicom
# import numpy as np
# from PIL import Image
#
# tolerance = 1
#
# # Load the DICOM file
# dicom = pydicom.dcmread(r"/home/ubuntu/nlp_project/Code/physionet.org/files/mimic-cxr/2.1.0/files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.dcm")
# dicom_pixels = dicom.pixel_array
#
# # Load the PNG file
# png_image = Image.open(r"/home/ubuntu/nlp_project/Code/physionet.org/files/mimic-cxr/2.1.0/files/p10/p10000032/s50414267/out_png/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.png")
# png_pixels = np.array(png_image)
#
# # Compare shapes
# assert dicom_pixels.shape == png_pixels.shape, "Shapes of DICOM and PNG do not match"
#
# # Compare pixel values
# difference = np.abs(dicom_pixels - png_pixels)
#
# dicom_non_zero_elements = dicom_pixels[dicom_pixels != 0]
# dicom_non_zero_count = len(dicom_non_zero_elements)
#
# # Get non-zero elements in PNG array
# png_non_zero_elements = png_pixels[png_pixels != 0]
# png_non_zero_count = len(png_non_zero_elements)
#
# print(f"Non-zero elements in DICOM: {dicom_non_zero_count}")
# print(f"Non-zero elements in PNG: {png_non_zero_count}")
#
# # If needed, print the actual non-zero elements (be cautious for large arrays)
# print(f"Non-zero elements in DICOM array:\n{dicom_non_zero_elements}")
# print(f"Non-zero elements in PNG array:\n{png_non_zero_elements}")
#
#
# print(f"First 10 non-zero elements in DICOM array:\n{dicom_non_zero_elements[:10]}")
# print(f"First 10 non-zero elements in PNG array:\n{png_non_zero_elements[:10]}")
#
#
# # assert np.all(difference < tolerance), f"Pixel difference exceeds tolerance: {difference.max()}"


import os
import pandas as pd
from tqdm import tqdm
import pydicom
import numpy as np
from PIL import Image


# Function to convert DICOM to PNG
def convert_dicom_to_png(dicom_path, output_path):
    try:
        # Read the DICOM file
        dicom = pydicom.dcmread(dicom_path)
        # Get pixel array
        pixel_array = dicom.pixel_array
        # Normalize pixel values to 0-255
        pixel_array = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255).astype(
            np.uint8)
        # Save as PNG
        image = Image.fromarray(pixel_array)
        image.save(output_path)
    except Exception as e:
        print(f"Error converting {dicom_path} to PNG: {e}")


# Function to extract findings and impressions from a report
def extract_findings_and_impression(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Extract Findings
    findings_start = content.find("FINDINGS:")
    impression_start = content.find("IMPRESSION:")

    findings = ""
    impression = ""

    if findings_start != -1:
        findings = content[findings_start + len("FINDINGS:"):impression_start].strip()

    if impression_start != -1:
        impression = content[impression_start + len("IMPRESSION:"):].strip()

    return findings, impression


# Main logic to create the DataFrame
reports_root_path = input("Enter the root path for reports: ").strip()

# Ensure the path exists
if not os.path.exists(reports_root_path):
    raise FileNotFoundError(f"The specified path does not exist: {reports_root_path}")

data = []

grp_folders = os.listdir(reports_root_path)

for p_grp in grp_folders:
    cxr_path = os.path.join(reports_root_path, p_grp)
    p_files = os.listdir(cxr_path)

    for p in p_files:
        res_path = os.path.join(cxr_path, p)

        if os.path.isdir(res_path):
            dicom_dirs = [d for d in os.listdir(res_path) if os.path.isdir(os.path.join(res_path, d))]
            txt_files = [f for f in os.listdir(res_path) if f.endswith('.txt') and f.startswith('s')]

            for dicom_dir in dicom_dirs:
                dicom_path = os.path.join(res_path, dicom_dir)
                dicom_files = [os.path.join(dicom_path, f) for f in os.listdir(dicom_path) if f.endswith('.dcm')]

                report_file = f"{dicom_dir}.txt"
                if report_file in txt_files:
                    report_path = os.path.join(res_path, report_file)
                    findings, impressions = extract_findings_and_impression(report_path)

                    for dicom_file in dicom_files:
                        dicom_id = os.path.basename(dicom_file)
                        png_path = dicom_file.replace('.dcm', '.png')  # Define the PNG output path

                        # Convert the DICOM to PNG
                        convert_dicom_to_png(dicom_file, png_path)

                        # Append data to the list
                        # data.append({
                        #     "dicom_path": dicom_file,
                        #     "png_path": png_path,
                        #     "dicom_id": dicom_id,
                        #     "findings": findings,
                        #     "impressions": impressions
                        # })

                        data_entry = {
                            "dicom_path": dicom_file,
                            "png_path": png_path,
                            "dicom_id": dicom_id,
                            "findings": findings,
                            "impressions": impressions
                        }

                        data.append(data_entry)

                        print(f"Processed PNG path: {data_entry['png_path']}")

df = pd.DataFrame(data)

print(df.head())
print(f"Total entries: {len(df)}")

# Save the DataFrame to a CSV file
df.to_csv('data_with_png_paths.csv', index=False)
