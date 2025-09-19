from src.dataio.voc_parser import paired_image_xml_list
from pathlib import Path

DATA_ROOT = r"E:\WPT-Project\Data\sized_squares_filled"
for split in ["train", "val", "test"]:
    img_dir = fr"{DATA_ROOT}\{split}"
    xml_dir = fr"{DATA_ROOT}\annotations"
    pairs = paired_image_xml_list(img_dir, xml_dir)
    print(f"{split}: {len(pairs)} pairs")
    for a,b in pairs[:3]:
        print("  ", Path(a).name, "↔", Path(b).name)
