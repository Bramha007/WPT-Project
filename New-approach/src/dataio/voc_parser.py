from pathlib import Path
import xml.etree.ElementTree as ET


def parse_voc(xml_path: str):
    root = ET.parse(xml_path).getroot()
    w = int(root.find("size/width").text)
    h = int(root.find("size/height").text)
    boxes = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        xmin = max(0, int(float(bb.find("xmin").text)))
        ymin = max(0, int(float(bb.find("ymin").text)))
        xmax = int(float(bb.find("xmax").text))
        ymax = int(float(bb.find("ymax").text))
        if xmax > xmin and ymax > ymin:
            boxes.append([xmin, ymin, xmax, ymax])
    return {"width": w, "height": h, "boxes": boxes}


def paired_image_xml_list(img_dir, xml_dir, img_exts={".png", ".bmp", ".jpg", ".jpeg"}):
    img_dir, xml_dir = Path(img_dir), Path(xml_dir)
    items = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() in img_exts:
            xml = xml_dir / (p.stem + ".xml")
            if xml.exists():
                items.append((str(p), str(xml)))
    return items
